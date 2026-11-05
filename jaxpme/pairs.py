"""Pair gathers and reductions for a sorted pair layout.

Layout: `centers` is sorted, and `others` is sorted after applying `others_perm`;
padding pairs come last and point at the padding atom.

Under reverse-mode AD a gather transposes to a scatter-add. XLA runs scatter-adds
with atomic adds that serialise on repeated indices, unless the indices are
flagged sorted or unique. A permutation gather keeps its `unique_indices` flag
through every transpose that matters here, but `indices_are_sorted` does not
survive: JAX transposes a scatter-add into a gather without it, so a second
transpose yields an unflagged scatter-add. The sorted gather and sorted segment
sum are therefore primitives that transpose into each other with the flag kept;
they are linear and have jvp, transpose and batching rules, so any composition of
transformations works, including forward-over-reverse (there is no custom_vjp).

Under `shard_map` an array's type also records the manual mesh axes it varies
over. The primitives keep the type of their operand, and `take`/`segment_sum`
first cast the operand to vary over the axes the indices vary over, as JAX's own
gather does, so an invariant array indexed per device gets its cotangent summed
over the devices.
"""

import jax
import jax.numpy as jnp
from jax.extend.core import Primitive
from jax.interpreters import ad, batching, mlir

from collections import namedtuple
from functools import partial

# idx[perm] (sorted), perm, and its inverse, computed once per index array
Permutation = namedtuple("Permutation", ("sorted", "perm", "inverse"))


def permutation(idx, perm):
    perm = jnp.asarray(perm)
    arange = jnp.arange(perm.shape[0], dtype=perm.dtype)
    inverse = jnp.zeros_like(perm).at[perm].set(arange, unique_indices=True)
    return Permutation(jnp.asarray(idx)[perm], perm, inverse)


def take(x, idx):
    """`x[idx]` for sorted `idx` or a `Permutation`."""
    x = _vary_like(jnp.asarray(x), idx)
    if isinstance(idx, Permutation):
        y = _gather_sorted_p.bind(x, idx.sorted)
        return y.at[idx.inverse].get(unique_indices=True, mode="promise_in_bounds")
    return _gather_sorted_p.bind(x, jnp.asarray(idx))


def segment_sum(x, idx, num_segments):
    """`jax.ops.segment_sum` for sorted `idx` or a `Permutation`."""
    x = _vary_like(jnp.asarray(x), idx)
    if isinstance(idx, Permutation):
        x = x.at[idx.perm].get(unique_indices=True, mode="promise_in_bounds")
        idx = idx.sorted
    return _segment_sum_sorted_p.bind(x, jnp.asarray(idx), num_segments=num_segments)


def _vary_like(x, idx):
    """`x` cast to vary over every manual mesh axis `idx` varies over."""
    idx = idx.sorted if isinstance(idx, Permutation) else jnp.asarray(idx)
    missing = tuple(sorted(_varying(idx) - _varying(x)))
    if not missing:
        return x
    pvary = getattr(jax.lax, "pvary", None)
    if pvary is not None:  # jax < 0.11
        return pvary(x, missing)
    return jax.lax.pcast(x, missing, to="varying")


def _varying(x):
    # outside shard_map, and before jax tracked it, nothing varies
    aval = getattr(x, "aval", None)
    manual = getattr(aval, "manual_axis_type", None)  # jax >= 0.11
    axes = manual.varying if manual is not None else getattr(aval, "vma", ())
    return frozenset(axes)


# -- primitives: registration with JAX's tracing, AD, vmap and lowering machinery --


def _gather_sorted(x, idx):
    return x.at[idx].get(indices_are_sorted=True, mode="promise_in_bounds")


def _segment_sum_sorted(x, idx, num_segments):
    return jax.ops.segment_sum(x, idx, num_segments, indices_are_sorted=True)


def _linear_primitive(name, impl, shape, transpose):
    p = Primitive(name)
    p.def_impl(impl)
    # x's type with the new shape: dtype, weak type and varying mesh axes carry over
    p.def_abstract_eval(lambda x, idx, **kw: x.update(shape=shape(x, idx, **kw)))
    mlir.register_lowering(p, mlir.lower_fun(impl, multiple_results=False))
    ad.defjvp(p, lambda t, x, idx, **kw: p.bind(t, idx, **kw), None)

    def transpose_rule(ct, x, idx, **kw):
        if type(ct) is ad.Zero:
            return [ad.Zero(x.aval), None]
        return [transpose(ct, x.aval, idx, **kw), None]

    def batch_rule(args, dims, **kw):
        (x, idx), (dx, didx) = args, dims
        if didx is not None:
            return jax.vmap(partial(impl, **kw), in_axes=dims)(x, idx), 0
        return p.bind(jnp.moveaxis(x, dx, 1), idx, **kw), 1

    ad.primitive_transposes[p] = transpose_rule
    batching.primitive_batchers[p] = batch_rule
    return p


_gather_sorted_p = _linear_primitive(
    "gather_sorted",
    _gather_sorted,
    lambda x, idx: idx.shape + x.shape[1:],
    lambda ct, x, idx: _segment_sum_sorted_p.bind(ct, idx, num_segments=x.shape[0]),
)
_segment_sum_sorted_p = _linear_primitive(
    "segment_sum_sorted",
    _segment_sum_sorted,
    lambda x, idx, num_segments: (num_segments,) + x.shape[1:],
    lambda ct, x, idx, num_segments: _gather_sorted_p.bind(ct, idx),
)
