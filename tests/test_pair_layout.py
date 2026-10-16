"""Sorted pair layout (issue #37): contract on the batch, and that the sorted
gathers/reductions reproduce the plain ones at every derivative order."""

import numpy as np
import jax
import jax.numpy as jnp

import pytest
from ase.build import bulk
from ase.io import read
from conftest import REFERENCE_STRUCTURES_DIR

jax.config.update("jax_enable_x64", True)

NUM_K = 400
CUTOFF = 5.0


def structures(slab=True):
    frames = read(REFERENCE_STRUCTURES_DIR / "coulomb_test_frames.xyz", index=":")
    cluster = frames[1].copy()
    cluster.set_pbc(False)
    big = bulk("NaCl", "rocksalt", a=5.64, cubic=True).repeat(2)
    big.rattle(0.1, seed=0)
    big.set_initial_charges(np.where(big.numbers == 11, 1.0, -1.0))
    return [frames[0], big, cluster, frames[2]] + ([frames[3]] if slab else [])


def unsorted(batch, nopbc):
    return batch._replace(others_perm=None), nopbc._replace(others_perm=None)


def assert_sorted(x):
    x = np.asarray(x)
    assert np.all(x[1:] >= x[:-1])


def test_vesin_lists_are_centre_sorted():
    from jaxpme.batched_mixed.batching import to_structure

    for atoms in structures():
        s = to_structure(atoms, CUTOFF)
        assert len(s["centers"]) > 0
        assert_sorted(s["centers"])


def test_get_batch_pair_layout():
    from jaxpme.batched_tiled import Ewald

    charges, batch, nopbc, _ = Ewald().prepare(structures(), num_k=NUM_K, cutoff=CUTOFF)
    n_real = int(batch.pair_mask.sum())
    assert n_real < len(batch.centers)  # padding pairs present
    assert np.all(batch.centers[n_real:] == batch.centers.max())
    for pairs in (batch, nopbc):
        assert_sorted(pairs.centers)
        assert_sorted(pairs.others[pairs.others_perm])
        np.testing.assert_array_equal(
            np.sort(pairs.others_perm), np.arange(len(pairs.others))
        )
    assert_sorted(batch.pair_to_structure)
    assert nopbc.pair_mask.sum() > 0


def test_get_batch_rejects_unsorted_centers():
    from jaxpme.batched_tiled.batching import get_batch, prepare

    s = prepare(structures()[0], num_k=NUM_K, cutoff=CUTOFF)
    p = np.random.default_rng(0).permutation(len(s["centers"]))
    s |= {k: s[k][p] for k in ("centers", "others", "cell_shifts")}
    with pytest.raises(ValueError, match="sorted by centers"):
        get_batch([s])


@pytest.mark.parametrize("exponent", [1, 3, 6])
def test_matches_unsorted(exponent):
    from jaxpme.batched_tiled import Ewald

    calc = Ewald(exponent=exponent)
    charges, batch, nopbc, pbc = calc.prepare(
        structures(slab=exponent == 1), num_k=NUM_K, cutoff=CUTOFF
    )

    def everything(charges, batch, nopbc):
        e, f, s = calc.energy_forces_stress(charges, batch, nopbc, pbc)
        dq = jax.grad(lambda q: calc.energy(q, batch, nopbc, pbc).sum())(charges)
        return calc.potentials(charges, batch, nopbc, pbc), e, f, s, dq

    new = everything(charges, batch, nopbc)
    old = everything(charges, *unsorted(batch, nopbc))
    for a, b in zip(new, old):
        assert np.all(np.isfinite(a))
        np.testing.assert_allclose(a, b, atol=1e-12, rtol=1e-12)


def test_higher_order_matches_unsorted():
    from jaxpme.batched_tiled import Ewald

    calc = Ewald()
    charges, batch, nopbc, pbc = calc.prepare(structures(), num_k=NUM_K, cutoff=CUTOFF)

    def derivatives(batch, nopbc):
        def energy(positions, charges):
            sr = batch._replace(positions=positions)
            return calc.energy(charges, sr, nopbc, pbc).sum()

        forces = jax.grad(energy)

        def force_loss(positions, charges):
            return jnp.sum(forces(positions, charges) ** 2)

        # force training (reverse over reverse), Born charges (forward over reverse)
        return (
            jax.jit(jax.grad(force_loss, argnums=(0, 1)))(batch.positions, charges),
            jax.jit(jax.jacfwd(forces, argnums=1))(batch.positions, charges),
        )

    (g_new, bec_new), (g_old, bec_old) = (
        derivatives(batch, nopbc),
        derivatives(*unsorted(batch, nopbc)),
    )
    for a, b in zip((*g_new, bec_new), (*g_old, bec_old)):
        assert np.all(np.isfinite(a)) and np.abs(a).max() > 0
        np.testing.assert_allclose(a, b, atol=1e-11, rtol=1e-11)


@pytest.mark.parametrize("full_neighbor_list", [True, False])
@pytest.mark.parametrize("exclusion_radius", [None, 3.0])
def test_rspace_matches_unsorted(full_neighbor_list, exclusion_radius):
    import vesin

    from jaxpme.potentials import potential
    from jaxpme.solvers import _rspace
    from jaxpme.utils import get_distances

    atoms = structures()[1]
    nl = vesin.NeighborList(cutoff=CUTOFF, full_list=full_neighbor_list, sorted=True)
    i, j, S = nl.compute(atoms.positions, atoms.cell.array, True, quantities="ijS")
    perm = np.argsort(j, kind="stable")
    pot = potential(exclusion_radius=exclusion_radius)
    charges = jnp.asarray(atoms.get_initial_charges())
    cell = jnp.asarray(atoms.cell.array)

    def energy(positions, charges, j_perm):
        r = get_distances(cell, positions[i], positions[j], S)
        V = _rspace(pot, 1.0, charges, r, i, j, full_neighbor_list, j_perm=j_perm)
        return jnp.sum(charges * V)

    positions = jnp.asarray(atoms.positions)
    e, (dr, dq) = jax.value_and_grad(energy, argnums=(0, 1))(positions, charges, perm)
    e_ref, (dr_ref, dq_ref) = jax.value_and_grad(energy, argnums=(0, 1))(
        positions, charges, None
    )
    for a, b in ((e, e_ref), (dr, dr_ref), (dq, dq_ref)):
        np.testing.assert_allclose(a, b, atol=1e-12, rtol=1e-12)


def test_pair_scatters_are_sorted_or_unique():
    # pins the mechanism on what XLA sees: in force training (reverse over
    # reverse) and Born-charge training (reverse over forward over reverse), no
    # scatter over a pair axis may be a plain (contended) scatter-add
    from jaxpme.batched_tiled import Ewald
    from jaxpme.batched_tiled.batching import get_batch, prepare, sample_shapes

    calc = Ewald()
    samples = [prepare(a, num_k=NUM_K, cutoff=CUTOFF) for a in structures()]
    # odd pair-axis sizes, so no other axis is mistaken for one
    n_pairs = sum(sample_shapes(s)["n_pairs"] for s in samples) + 7
    n_nopbc = sum(sample_shapes(s)["n_pairs_nonpbc"] for s in samples) + 3
    charges, batch, nopbc, pbc = get_batch(
        samples, num_pairs=n_pairs, num_pairs_nonpbc=n_nopbc
    )

    def energy(positions, charges):
        return calc.energy(charges, batch._replace(positions=positions), nopbc, pbc).sum()

    forces = jax.grad(energy)

    def force_loss(positions, charges):
        return jnp.sum(forces(positions, charges) ** 2)

    def bec_loss(positions, charges):
        return jnp.sum(jax.jacfwd(forces, argnums=1)(positions, charges) ** 2)

    def ops(op):
        for region in op.regions:
            for block in region.blocks:
                for o in block.operations:
                    yield o
                    yield from ops(o)

    for loss in (force_loss, bec_loss):
        f = jax.jit(jax.grad(loss, argnums=(0, 1)))
        module = f.lower(batch.positions, charges).compiler_ir("stablehlo")
        scatters = [o for o in ops(module.operation) if o.name == "stablehlo.scatter"]
        pair_scatters = [
            o for o in scatters if o.operands[1].type.shape[0] in (n_pairs, n_nopbc)
        ]
        assert len(pair_scatters) > 0
        for o in pair_scatters:
            flags = [str(o.attributes[k]) for k in ("indices_are_sorted", "unique_indices")]
            assert "true" in flags, o
