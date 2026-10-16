import jax
import jax.numpy as jnp

from functools import partial


def atoms_to_graph(atoms, cutoff, full_list=False):
    import vesin

    positions = jnp.array(atoms.get_positions())
    cell = jnp.array(atoms.get_cell().array)
    if atoms.get_pbc().all():
        periodic = True
    elif not atoms.get_pbc().any():
        periodic = False
    else:
        raise ValueError("no mixed pbc supported")

    nl = vesin.NeighborList(cutoff=cutoff, full_list=full_list)
    i, j, S = nl.compute(points=positions, box=cell, periodic=periodic, quantities="ijS")

    return cell, positions, i, j, S


def compose_cell(batch):
    """The cell the Ewald math consumes.

    `batch.cell` is the raw data cell; `batch.effective_cell` is the cell
    the math needs (for 2D pbc, `prepare` shrinks the vacuum vector — a
    position-dependent convergence trick). `batch.pbc` [S, 3] is the mask
    that picks per lattice vector: True -> raw, False -> effective. The two
    agree on periodic rows, so the values are effective everywhere while
    gradients (stress) reach `batch.cell` through periodic rows only — the
    shrink's cell-gradient is an artifact and is deliberately dropped.
    `effective_cell=None`: `cell` is already effective, pass through.
    """
    if batch.effective_cell is None:
        return batch.cell
    return jnp.where(batch.pbc[:, :, None], batch.cell, batch.effective_cell)


def get_distances(cell, Ra, Rb, cell_shifts):
    # we can't use mic: the unit tests use small cells
    R = Rb - Ra
    if len(cell.shape) == 2:
        # broadcasting single cell over pairs
        R += jnp.einsum("pA,Aa->pa", cell_shifts, cell)
    elif len(cell.shape) == 3:
        # different cell per pair (batched case)
        R += jnp.einsum("pA,pAa->pa", cell_shifts, cell)
    else:
        raise ValueError(f"cell shape {cell.shape} should be 3x3 or pairs x 3x3)")

    return safe_norm(R, axis=-1)


# Pair gathers and reductions for a sorted pair layout: `centers` sorted,
# `others[others_perm]` sorted, padding pairs last on the padding atom.
#
# Under reverse-mode AD a gather transposes to a scatter-add, which XLA executes
# with atomic adds that serialise on repeated indices, unless the indices are
# flagged sorted or unique.
# The flags alone do not survive: JAX transposes a scatter-add into a gather
# without `indices_are_sorted`, so a second transpose yields an unflagged
# scatter-add. The three linear primitives below transpose into each other with
# the flags kept -- sorted gather <-> sorted segment sum, permutation <->
# inverse permutation -- and are closed under jvp, transpose and vmap. There is
# no custom_vjp, so forward-over-reverse differentiation works.


def _linear_primitive(name, impl, shape, transpose, n_idx):
    from jax.extend.core import Primitive
    from jax.interpreters import ad, batching, mlir

    p = Primitive(name)
    p.def_impl(impl)
    p.def_abstract_eval(
        lambda x, *idx, **kw: jax.core.ShapedArray(shape(x, *idx, **kw), x.dtype)
    )
    mlir.register_lowering(p, mlir.lower_fun(impl, multiple_results=False))
    ad.defjvp(p, lambda t, x, *idx, **kw: p.bind(t, *idx, **kw), *[None] * n_idx)

    def transpose_rule(ct, x, *idx, **kw):
        if type(ct) is ad.Zero:
            return [ad.Zero(x.aval)] + [None] * n_idx
        return [transpose(ct, x.aval, *idx, **kw)] + [None] * n_idx

    def batch_rule(args, dims, **kw):
        if any(d is not None for d in dims[1:]):
            return jax.vmap(partial(impl, **kw), in_axes=dims)(*args), 0
        x, *idx = args
        return p.bind(jnp.moveaxis(x, dims[0], 1), *idx, **kw), 1

    ad.primitive_transposes[p] = transpose_rule
    batching.primitive_batchers[p] = batch_rule
    return p


def _gather_sorted(x, idx):
    return x.at[idx].get(indices_are_sorted=True, mode="promise_in_bounds")


def _segment_sum_sorted(x, idx, num_segments):
    return jax.ops.segment_sum(x, idx, num_segments, indices_are_sorted=True)


def _permute(x, perm, inv):
    return x.at[perm].get(unique_indices=True, mode="promise_in_bounds")


gather_sorted_p = _linear_primitive(
    "gather_sorted",
    _gather_sorted,
    lambda x, idx: idx.shape + x.shape[1:],
    lambda ct, x, idx: segment_sum_sorted_p.bind(ct, idx, num_segments=x.shape[0]),
    1,
)
segment_sum_sorted_p = _linear_primitive(
    "segment_sum_sorted",
    _segment_sum_sorted,
    lambda x, idx, num_segments: (num_segments,) + x.shape[1:],
    lambda ct, x, idx, num_segments: gather_sorted_p.bind(ct, idx),
    1,
)
permute_p = _linear_primitive(
    "permute",
    _permute,
    lambda x, perm, inv: x.shape,
    lambda ct, x, perm, inv: permute_p.bind(ct, inv, perm),
    2,
)


def take_sorted(x, idx):
    return gather_sorted_p.bind(jnp.asarray(x), jnp.asarray(idx))


def segment_sum_sorted(x, idx, num_segments):
    return segment_sum_sorted_p.bind(
        jnp.asarray(x), jnp.asarray(idx), num_segments=num_segments
    )


def _inverse(perm):
    perm = jnp.asarray(perm)
    arange = jnp.arange(perm.shape[0], dtype=perm.dtype)
    return perm, jnp.zeros_like(perm).at[perm].set(arange, unique_indices=True)


def take_permuted(x, idx, perm):
    # x[idx] = x[idx[perm]][inverse(perm)]
    perm, inv = _inverse(perm)
    return permute_p.bind(take_sorted(x, jnp.asarray(idx)[perm]), inv, perm)


def segment_sum_permuted(x, idx, perm, num_segments):
    perm, inv = _inverse(perm)
    x = permute_p.bind(jnp.asarray(x), perm, inv)
    return segment_sum_sorted(x, jnp.asarray(idx)[perm], num_segments)


def get_volume(cell):
    return jnp.abs(jnp.linalg.det(cell))


def safe_norm(x, axis=None):
    # derivatives of norm are NaN if x is zero,
    # this fixes that
    x2 = jnp.sum(jax.lax.square(x), axis=axis)
    mask = x2 == 0.0
    masked = jnp.where(mask, 1e-6, x2)
    return jnp.where(mask, 0.0, jnp.sqrt(masked))
