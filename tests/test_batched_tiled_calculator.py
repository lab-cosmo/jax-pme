"""Mirrors test_batched_mixed_calculator.py for the tile-dispatched backend.

`batched_tiled.prepare` requires `num_k` explicitly. We pick `num_k` to match
each test's intended `lr_wavelength` via the analytical inversion
    num_k = prod(L_i) / (2 · lr_wavelength³)
which gives the same convergence as the serial / mixed setups the test cases
were written against.
"""

import numpy as np
import jax

import pytest
from ase.io import read
from conftest import REFERENCE_STRUCTURES_DIR

jax.config.update("jax_enable_x64", True)


def _num_k_for_lr(atoms, lr_wavelength):
    """Pick num_k so that lr_wavelength_for_num_k(cell, num_k) ≈ target."""
    L = np.linalg.norm(atoms.get_cell().array, axis=-1)
    return int(np.prod(L) / (2 * lr_wavelength**3))


@pytest.mark.parametrize("cutoff", [4.0, 5.0, 6.0])
def test_reference_structures(cutoff):
    """Calculator with multiple structures w/ pbc and no-pbc systems."""
    from jaxpme.batched_tiled.calculators import Ewald
    from jaxpme.calculators import Ewald as SerialEwald

    structures = read(REFERENCE_STRUCTURES_DIR / "coulomb_test_frames.xyz", index=":3")
    atoms_no_pbc = structures[-1].copy()
    atoms_no_pbc.set_pbc(False)

    # match the serial config used downstream: smearing = cutoff/4 fixed across
    # the batch; num_k = max per-cell target so every cell gets at least the
    # serial's k-vec count (rectangular K means small cells overshoot slightly).
    lr = cutoff / 8.0
    num_k = max(_num_k_for_lr(a, lr) for a in structures if a.get_pbc().any())

    calculator = Ewald(prefactor=1.0)
    charges, sr_batch, nonperiodic_batch, periodic_batch = calculator.prepare(
        structures + [atoms_no_pbc], num_k=num_k, cutoff=cutoff, smearing=cutoff / 4
    )
    atom_to_structure = sr_batch.atom_to_structure
    potentials = calculator.potentials(charges, sr_batch, nonperiodic_batch, periodic_batch)

    np.testing.assert_(~np.isnan(potentials).any())

    E, F, S = calculator.energy_forces_stress(
        charges, sr_batch, nonperiodic_batch, periodic_batch
    )

    calc = SerialEwald()
    inputs = [
        calc.prepare(atoms, None, cutoff, cutoff / 8, cutoff / 4) for atoms in structures
    ]

    # tiled overshoots K_b for small cells (extra k-vecs barely contribute past
    # convergence at lr=cutoff/8), so per-structure diff is float-noise-level
    # provided smearing matches.
    for i in range(len(structures)):
        pot = calc.potentials(*inputs[i])
        E2, F2, S2 = calc.energy_forces_stress(*inputs[i])

        np.testing.assert_allclose(potentials[atom_to_structure == i], pot, rtol=1e-7)
        np.testing.assert_allclose(E[i], E2, rtol=1e-7)
        np.testing.assert_allclose(F[atom_to_structure == i], F2, rtol=1e-6, atol=1e-10)
        np.testing.assert_allclose(S[i], S2, rtol=1e-6, atol=1e-10)


@pytest.mark.parametrize("cutoff", [4.0, 5.0, 6.0])
def test_mixed(cutoff):
    """Mixed periodic and non-periodic systems."""
    from jaxpme.batched_tiled.calculators import Ewald

    atoms = read(REFERENCE_STRUCTURES_DIR / "coulomb_test_frames.xyz", index="0")
    charges = atoms.get_initial_charges()
    atoms2 = atoms.copy()
    atoms2.set_pbc(False)

    all_distances = atoms2.get_all_distances()
    mask = all_distances != 0.0
    one_over_r = np.where(mask, 1 / np.where(mask, all_distances, 1.0), 0.0)
    bare_potentials = np.einsum("ij,j->ij", one_over_r, charges)
    reference_potentials = bare_potentials.sum(axis=1) / 2

    num_k = _num_k_for_lr(atoms, cutoff / 8.0)

    calculator = Ewald(prefactor=1.0)
    charges_batch, sr_batch, nonperiodic_batch, periodic_batch = calculator.prepare(
        [atoms, atoms2], num_k=num_k, cutoff=cutoff, smearing=cutoff / 4
    )
    atom_to_structure = sr_batch.atom_to_structure
    potentials = calculator.potentials(
        charges_batch, sr_batch, nonperiodic_batch, periodic_batch
    )

    np.testing.assert_allclose(potentials[atom_to_structure == 1], reference_potentials)


@pytest.mark.parametrize("cutoff", [4.0, 5.0, 6.0])
def test_single_system_vs_serial(cutoff):
    """Single periodic system vs serial."""
    from jaxpme.batched_tiled.calculators import Ewald
    from jaxpme.calculators import Ewald as SerialEwald

    atoms = read(REFERENCE_STRUCTURES_DIR / "coulomb_test_frames.xyz", index="0")
    num_k = _num_k_for_lr(atoms, cutoff / 8.0)

    calculator = Ewald(prefactor=1.0)
    charges, sr_batch, nonperiodic_batch, periodic_batch = calculator.prepare(
        [atoms], num_k=num_k, cutoff=cutoff, smearing=cutoff / 4
    )
    potentials = calculator.potentials(charges, sr_batch, nonperiodic_batch, periodic_batch)
    energy = calculator.energy(charges, sr_batch, nonperiodic_batch, periodic_batch)

    calc = SerialEwald()
    inputs = calc.prepare(atoms, None, cutoff, cutoff / 8, cutoff / 4)
    pot_ref = calc.potentials(*inputs)
    energy_ref = calc.energy(*inputs)

    np.testing.assert_allclose(potentials[sr_batch.atom_mask], pot_ref, rtol=1e-7)
    np.testing.assert_allclose(energy[0], energy_ref, rtol=1e-7)


@pytest.mark.parametrize("frame_index", [0, 1, 2, 3, 4, 5])
@pytest.mark.parametrize("cutoff", [4.0, 5.0, 6.0])
def test_single_system_vs_reference(frame_index, cutoff):
    """Single (mixed) PBC system vs reference data."""
    from jaxpme import prefactors
    from jaxpme.batched_tiled.calculators import Ewald

    atoms = read(REFERENCE_STRUCTURES_DIR / "coulomb_test_frames.xyz", index=frame_index)
    num_k = _num_k_for_lr(
        atoms, cutoff / 16.0
    )  # match mixed-test's lr_wavelength=cutoff/16

    calculator = Ewald(prefactor=prefactors.eV_A)
    charges, sr_batch, nonperiodic_batch, periodic_batch = calculator.prepare(
        [atoms], num_k=num_k, cutoff=cutoff, smearing=cutoff / 8
    )
    energy, forces = calculator.energy_forces(
        charges, sr_batch, nonperiodic_batch, periodic_batch
    )

    np.testing.assert_allclose(
        energy[sr_batch.structure_mask], atoms.get_potential_energy(), rtol=1e-4, atol=0
    )
    np.testing.assert_allclose(forces[sr_batch.atom_mask], atoms.get_forces(), rtol=2e-2)


@pytest.mark.parametrize("cutoff", [4.0, 5.0, 6.0])
def test_halfspace_equivalence(cutoff):
    """Halfspace optimization gives the same results as full k-space."""
    from jaxpme.batched_tiled.calculators import Ewald

    structures = read(REFERENCE_STRUCTURES_DIR / "coulomb_test_frames.xyz", index=":3")
    num_k = max(_num_k_for_lr(a, cutoff / 8.0) for a in structures if a.get_pbc().any())

    calc_full = Ewald(prefactor=1.0, halfspace=False)
    ch_f, sr_f, nonp_f, pbc_f = calc_full.prepare(structures, num_k=num_k, cutoff=cutoff)

    calc_half = Ewald(prefactor=1.0, halfspace=True)
    ch_h, sr_h, nonp_h, pbc_h = calc_half.prepare(structures, num_k=num_k, cutoff=cutoff)

    # halfspace should have fewer k-vectors
    k2_full = (pbc_f.k_grid[0] ** 2).sum(axis=-1)
    k2_half = (pbc_h.k_grid[0] ** 2).sum(axis=-1)
    assert (k2_half > 0).sum() < (k2_full > 0).sum()

    # tile-summation order differs between halfspace and full-space (different
    # T1, T2 tile counts) so we get float noise at ~1e-8 level rather than the
    # bit-exact-modulo-sum-order match batched_mixed sees on this test.
    pot_full = calc_full.potentials(ch_f, sr_f, nonp_f, pbc_f)
    pot_half = calc_half.potentials(ch_h, sr_h, nonp_h, pbc_h)
    np.testing.assert_allclose(pot_full, pot_half, rtol=1e-7)

    E_full = calc_full.energy(ch_f, sr_f, nonp_f, pbc_f)
    E_half = calc_half.energy(ch_h, sr_h, nonp_h, pbc_h)
    np.testing.assert_allclose(E_full, E_half, rtol=1e-7)

    E_full2, F_full = calc_full.energy_forces(ch_f, sr_f, nonp_f, pbc_f)
    E_half2, F_half = calc_half.energy_forces(ch_h, sr_h, nonp_h, pbc_h)
    np.testing.assert_allclose(F_full, F_half, rtol=1e-7, atol=1e-12)

    E_full3, F_full3, S_full = calc_full.energy_forces_stress(ch_f, sr_f, nonp_f, pbc_f)
    E_half3, F_half3, S_half = calc_half.energy_forces_stress(ch_h, sr_h, nonp_h, pbc_h)
    np.testing.assert_allclose(S_full, S_half, rtol=1e-7, atol=1e-12)


@pytest.mark.parametrize("cutoff", [4.0, 5.0, 6.0])
def test_iplp_batched_vs_serial(cutoff):
    """Inverse power law potential, batched vs serial energy."""
    from jaxpme.batched_tiled.calculators import Ewald
    from jaxpme.calculators import Ewald as SerialEwald
    from jaxpme.potentials import inverse_power_law

    structures = read(REFERENCE_STRUCTURES_DIR / "coulomb_test_frames.xyz", index=":3")
    iplp = inverse_power_law(1)
    num_k = max(_num_k_for_lr(a, cutoff / 8.0) for a in structures if a.get_pbc().any())

    calculator = Ewald(custom_potential=iplp, prefactor=1.0)
    charges, sr_batch, nonp_batch, pbc_batch = calculator.prepare(
        structures, num_k=num_k, cutoff=cutoff, smearing=cutoff / 4
    )
    energy_batched = calculator.energy(charges, sr_batch, nonp_batch, pbc_batch)

    serial_calc = SerialEwald(custom_potential=iplp)
    for i, atoms in enumerate(structures):
        inputs = serial_calc.prepare(atoms, None, cutoff, cutoff / 8, cutoff / 4)
        energy_serial = serial_calc.energy(*inputs)
        np.testing.assert_allclose(energy_batched[i], energy_serial, rtol=1e-7)


@pytest.mark.parametrize("cutoff", [4.0, 5.0, 6.0])
def test_iplp_halfspace_equivalence(cutoff):
    from jaxpme.batched_tiled.calculators import Ewald
    from jaxpme.potentials import inverse_power_law

    structures = read(REFERENCE_STRUCTURES_DIR / "coulomb_test_frames.xyz", index=":3")
    iplp = inverse_power_law(1)
    num_k = max(_num_k_for_lr(a, cutoff / 8.0) for a in structures if a.get_pbc().any())

    calc_full = Ewald(custom_potential=iplp, halfspace=False)
    ch_f, sr_f, nonp_f, pbc_f = calc_full.prepare(structures, num_k=num_k, cutoff=cutoff)
    E_full = calc_full.energy(ch_f, sr_f, nonp_f, pbc_f)

    calc_half = Ewald(custom_potential=iplp, halfspace=True)
    ch_h, sr_h, nonp_h, pbc_h = calc_half.prepare(structures, num_k=num_k, cutoff=cutoff)
    E_half = calc_half.energy(ch_h, sr_h, nonp_h, pbc_h)

    np.testing.assert_allclose(E_full, E_half, rtol=1e-8)


def test_iplp_2d_pbc_returns_nan():
    """IPLP + 2D PBC returns NaN (correction unsupported)."""
    from ase import Atoms

    from jaxpme.batched_tiled.calculators import Ewald
    from jaxpme.potentials import inverse_power_law

    atoms_2d = Atoms(
        "H2",
        positions=[[5.0, 5.0, 5.0], [5.0, 5.0, 6.0]],
        cell=np.diag([10.0, 10.0, 20.0]),
        pbc=[True, True, False],
    )
    atoms_2d.set_initial_charges([1.0, -1.0])

    iplp = inverse_power_law(1)
    calculator = Ewald(custom_potential=iplp)
    charges, sr_batch, nonp_batch, pbc_batch = calculator.prepare(
        [atoms_2d], num_k=50, cutoff=5.0
    )
    energy = calculator.energy(charges, sr_batch, nonp_batch, pbc_batch)

    assert np.isnan(energy[0])


def test_matches_batched_mixed():
    """Bit-exact equivalence vs batched_mixed at identical params (fp64)."""
    from jaxpme.batched_mixed.calculators import Ewald as MixedEwald
    from jaxpme.batched_tiled.calculators import Ewald as TiledEwald

    structures = read(REFERENCE_STRUCTURES_DIR / "coulomb_test_frames.xyz", index=":3")
    cutoff = 5.0
    num_k = max(_num_k_for_lr(a, cutoff / 8.0) for a in structures if a.get_pbc().any())

    ct = TiledEwald(prefactor=1.0)
    ch_t, sr_t, nopbc_t, pbc_t = ct.prepare(structures, num_k=num_k, cutoff=cutoff)
    E_t, F_t, S_t = ct.energy_forces_stress(ch_t, sr_t, nopbc_t, pbc_t)

    cm = MixedEwald(prefactor=1.0)
    ch_m, sr_m, nopbc_m, pbc_m = cm.prepare(structures, cutoff=cutoff, num_k=num_k)
    E_m, F_m, S_m = cm.energy_forces_stress(ch_m, sr_m, nopbc_m, pbc_m)

    # different array layouts, so mask-extract before comparing
    np.testing.assert_allclose(
        np.array(E_t[sr_t.structure_mask]),
        np.array(E_m[sr_m.structure_mask]),
        rtol=1e-10,
    )
    np.testing.assert_allclose(
        np.array(F_t[sr_t.atom_mask]),
        np.array(F_m[sr_m.atom_mask]),
        rtol=1e-9,
        atol=1e-15,
    )
    np.testing.assert_allclose(
        np.array(S_t[sr_t.structure_mask]),
        np.array(S_m[sr_m.structure_mask]),
        rtol=1e-9,
        atol=1e-15,
    )
