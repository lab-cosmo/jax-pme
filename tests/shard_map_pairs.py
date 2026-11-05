"""The sorted pair layout under `jax.shard_map` against one device, in fp64.

Run in a fresh process with at least two devices, e.g.
`XLA_FLAGS=--xla_force_host_platform_device_count=2 JAX_PLATFORMS=cpu python
tests/shard_map_pairs.py` (`test_pair_layout.py` does). Every device gets its
own batch, as in data-parallel training: the energy, the forces and the
force-training gradient (reverse over reverse, positions and charges) must
match a plain call per batch. An invariant array gathered and reduced through a
per-device permutation must get the sum of the devices' gradients. Exits
non-zero on the first mismatch. Where JAX itself cannot differentiate `erf`
on a per-device array (JAX 0.12.0.1), the batched_tiled part is skipped after
the invariant-operand check and the script prints `SKIP` with the reason.
"""

import numpy as np
import jax
import jax.numpy as jnp
from jax.sharding import Mesh
from jax.sharding import PartitionSpec as P

import sys
from functools import partial

from ase.build import bulk
from ase.io import read
from conftest import REFERENCE_STRUCTURES_DIR

jax.config.update("jax_enable_x64", True)

NUM_K = 400
CUTOFF = 5.0
TOL = 1e-10


def structures(seed):
    """`test_pair_layout.structures`, shifted by much less than any pair's
    distance to the cutoff, so every device's batch has the same shapes."""
    frames = read(REFERENCE_STRUCTURES_DIR / "coulomb_test_frames.xyz", index=":")
    cluster = frames[1].copy()
    cluster.set_pbc(False)
    big = bulk("NaCl", "rocksalt", a=5.64, cubic=True).repeat(2)
    big.rattle(0.1, seed=0)
    big.set_initial_charges(np.where(big.numbers == 11, 1.0, -1.0))
    out = [frames[0], big, cluster, frames[2], frames[3]]
    rng = np.random.default_rng(seed)
    for atoms in out:
        atoms.positions += rng.normal(scale=1e-3, size=atoms.positions.shape)
        atoms.set_initial_charges(atoms.get_initial_charges() * (1 + 0.1 * seed))
    return out


def check(name, got, want):
    got, want = np.asarray(got), np.asarray(want)
    err = np.abs(got - want).max() / np.abs(want).max()
    print(f"  {name}: relative error {err:.1e}", flush=True)
    if not err < TOL:
        sys.exit(f"{name} differs from one device: relative error {err:.3e}")


def _erf_gradient_broken(mesh):
    """Why JAX cannot differentiate `erf` (the real-space Coulomb kernel) on a
    per-device array inside `shard_map`, or None when it can."""
    from jax.scipy.special import erf

    @partial(jax.shard_map, mesh=mesh, in_specs=P("dp"), out_specs=P("dp"))
    def grad_erf(x):
        return jax.grad(lambda y: jnp.sum(erf(y)))(x)

    try:
        jax.jit(grad_erf)(jnp.linspace(0.1, 2.0, 2 * len(mesh.devices)))
    except ValueError as e:
        return f"jax {jax.__version__} fails on grad(erf): {str(e).splitlines()[0][:120]}"
    return None


def main():
    from jaxpme import pairs
    from jaxpme.batched_tiled import Ewald

    devices = jax.devices()
    if len(devices) < 2:
        sys.exit(f"needs >= 2 devices (module docstring), have {devices}")
    mesh = Mesh(np.array(devices), ("dp",))
    n = len(devices)
    print(f"jax {jax.__version__}, {n} devices", flush=True)

    # an invariant array through per-device neighbour lists: its gradient sums the devices'
    m = 64
    rng = np.random.default_rng(0)
    others = rng.integers(0, m, size=(n, 3 * m))  # unsorted, as `others` is
    perm = np.argsort(others, axis=1, kind="stable")
    weights = rng.normal(size=(n, 3 * m))
    x = jnp.asarray(rng.normal(size=m))

    def per_device_loss(x, others, perm, w):
        j = pairs.permutation(others, perm)
        gathered = pairs.take(x, j)
        reduced = pairs.segment_sum(w * gathered, j, m)
        return jnp.sum(w * gathered**2) + jnp.sum(reduced * x)

    @partial(
        jax.shard_map, mesh=mesh, in_specs=(P(), P("dp"), P("dp"), P("dp")), out_specs=P()
    )
    def total(x, others, perm, w):
        return jax.lax.psum(per_device_loss(x, others[0], perm[0], w[0]), "dp")

    value, grad = jax.jit(jax.value_and_grad(total))(x, others, perm, weights)
    ref = [
        jax.value_and_grad(per_device_loss)(x, others[d], perm[d], weights[d])
        for d in range(n)
    ]
    check("invariant operand: value", value, sum(r[0] for r in ref))
    check("invariant operand: gradient", grad, sum(r[1] for r in ref))

    reason = _erf_gradient_broken(mesh)
    if reason:
        print(f"SKIP batched_tiled under shard_map: {reason}", flush=True)
        return

    calc = Ewald()
    batches = [
        calc.prepare(structures(seed), num_k=NUM_K, cutoff=CUTOFF) for seed in range(n)
    ]
    flat = [jax.tree_util.tree_flatten(b) for b in batches]
    treedef = flat[0][1]
    assert all(f[1] == treedef for f in flat), "batches differ in structure"
    dynamic = [i for i, leaf in enumerate(flat[0][0]) if hasattr(leaf, "shape")]
    for i, leaf in enumerate(flat[0][0]):
        if i in dynamic:
            assert all(np.shape(f[0][i]) == np.shape(leaf) for f in flat), (
                "batch shapes differ"
            )
        else:
            assert all(f[0][i] == leaf for f in flat), "batch constants differ"

    def rebuild(arrays):
        leaves = list(flat[0][0])
        for i, x in zip(dynamic, arrays):
            leaves[i] = x
        return jax.tree_util.tree_unflatten(treedef, leaves)

    def derivatives(arrays):
        charges, batch, nopbc, pbc = rebuild(arrays)

        def energy(positions, charges):
            sr = batch._replace(positions=positions)
            return calc.energy(charges, sr, nopbc, pbc).sum()

        forces = jax.grad(energy)

        def force_loss(positions, charges):
            return jnp.sum(forces(positions, charges) ** 2)

        positions = batch.positions
        return (
            energy(positions, charges),
            forces(positions, charges),
            *jax.grad(force_loss, argnums=(0, 1))(positions, charges),
        )

    stacked = [jnp.stack([f[0][i] for f in flat]) for i in dynamic]

    @partial(jax.shard_map, mesh=mesh, in_specs=(P("dp"),), out_specs=P("dp"))
    def sharded(arrays):
        out = derivatives([jnp.squeeze(x, 0) for x in arrays])
        return [x[None] for x in out]

    got = jax.jit(sharded)(stacked)
    want = [jax.jit(derivatives)([f[0][i] for i in dynamic]) for f in flat]
    for k, name in enumerate(
        ("energy", "forces", "d force loss / d positions", "d force loss / d charges")
    ):
        check(name, got[k], np.stack([w[k] for w in want]))

    print("all match", flush=True)


if __name__ == "__main__":
    main()
