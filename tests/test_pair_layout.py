"""The batched_tiled pair layout: the contract on prepared batches, and that the
sorted gathers and reductions give the same values and derivatives as plain
indexing."""

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
    """vesin's sorted lists are centre-sorted, and the half list holds exactly
    half the pairs of the full one: `get_batch` and the half-list calculator
    rely on both."""
    from jaxpme.batched_mixed.batching import to_structure

    for atoms in structures():
        full = to_structure(atoms, CUTOFF)
        half = to_structure(atoms, CUTOFF, full_list=False)
        assert len(half["centers"]) > 0
        assert 2 * len(half["centers"]) == len(full["centers"])
        assert_sorted(full["centers"])
        assert_sorted(half["centers"])


def test_get_batch_pair_layout():
    """`get_batch` output satisfies the layout contract for periodic and
    non-periodic pairs, padding included; the sorted ops assume it unchecked."""
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


def test_get_batch_sorts_unsorted_sample():
    """A sample whose pairs are not centre-sorted (assembled outside `prepare`)
    is sorted by `get_batch` and gives the same energies as the sorted one."""
    from jaxpme.batched_tiled import Ewald
    from jaxpme.batched_tiled.batching import get_batch, prepare

    calc = Ewald()
    sample = prepare(structures()[1], num_k=NUM_K, cutoff=CUTOFF)
    shuffled = dict(sample)
    p = np.random.default_rng(0).permutation(len(sample["centers"]))
    shuffled |= {k: sample[k][p] for k in ("centers", "others", "cell_shifts")}
    del shuffled["others_perm"]

    batches = [get_batch([s]) for s in (sample, shuffled)]
    assert_sorted(batches[1][1].centers)
    assert_sorted(batches[1][1].others[batches[1][1].others_perm])
    e, e_shuffled = (calc.energy_forces(*b) for b in batches)
    for a, b in zip(e, e_shuffled):
        np.testing.assert_allclose(a, b, atol=1e-12, rtol=1e-12)


@pytest.mark.parametrize("exponent", [1, 3, 6])
def test_matches_unsorted(exponent):
    """Sorted layout and plain-gather fallback agree on everything the
    calculator returns, for every potential family and boundary condition."""
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
    """Force training (reverse over reverse) and Born charges (forward over
    reverse) differentiate through the primitives' transpose, jvp and batching
    rules; they must agree with plain indexing."""
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
def test_rspace_matches_full_unsorted(full_neighbor_list, exclusion_radius):
    """`_rspace` on half or full lists, sorted or plain, matches the full list
    with plain indexing in value and gradients."""
    import vesin

    from jaxpme import pairs
    from jaxpme.potentials import potential
    from jaxpme.solvers import _rspace
    from jaxpme.utils import get_distances

    atoms = structures()[1]
    pot = potential(exclusion_radius=exclusion_radius)
    charges = jnp.asarray(atoms.get_initial_charges())
    cell = jnp.asarray(atoms.cell.array)
    positions = jnp.asarray(atoms.positions)

    def value_and_grads(full_list, sorted_layout):
        nl = vesin.NeighborList(cutoff=CUTOFF, full_list=full_list, sorted=True)
        i, j, S = nl.compute(atoms.positions, atoms.cell.array, True, quantities="ijS")
        perm = np.argsort(j, kind="stable")
        j_perm = pairs.permutation(j, perm) if sorted_layout else None

        def energy(positions, charges):
            r = get_distances(cell, positions[i], positions[j], S)
            V = _rspace(pot, 1.0, charges, r, i, j, full_list, j_perm=j_perm)
            return jnp.sum(charges * V)

        e, (dr, dq) = jax.value_and_grad(energy, argnums=(0, 1))(positions, charges)
        return e, dr, dq

    reference = value_and_grads(full_list=True, sorted_layout=False)
    for sorted_layout in (True, False):
        for a, b in zip(value_and_grads(full_neighbor_list, sorted_layout), reference):
            np.testing.assert_allclose(a, b, atol=1e-12, rtol=1e-12)


def test_pair_scatters_are_sorted_or_unique():
    """Every scatter over a pair axis in the lowered force-training and
    Born-charge-training graphs is flagged sorted or unique: unflagged ones run
    with contended atomics on GPU, and this is where a regression would show."""
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
