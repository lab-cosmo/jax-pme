import numpy as np
import jax

import pytest
from ase.io import read
from conftest import REFERENCE_STRUCTURES_DIR

jax.config.update("jax_enable_x64", True)


@pytest.mark.parametrize("cutoff", [4.0, 5.0, 6.0])
def test_reference_structures(cutoff):
    """Test calculator with multiple structures w/ pbc and no-pbc systems."""
    from jaxpme.batched_mixed.calculators import Ewald
    from jaxpme.calculators import Ewald as SerialEwald

    structures = read(REFERENCE_STRUCTURES_DIR / "coulomb_test_frames.xyz", index=":3")
    atoms_no_pbc = structures[-1].copy()
    atoms_no_pbc.set_pbc(False)

    calculator = Ewald(prefactor=1.0)
    charges, sr_batch, nonperiodic_batch, periodic_batch = calculator.prepare(
        structures + [atoms_no_pbc], cutoff
    )
    atom_to_structure = sr_batch.atom_to_structure
    potentials = calculator.potentials(charges, sr_batch, nonperiodic_batch, periodic_batch)

    np.testing.assert_(~np.isnan(potentials).any())

    E, F, S = calculator.energy_forces_stress(
        charges, sr_batch, nonperiodic_batch, periodic_batch
    )

    calc = SerialEwald()
    inputs = [
        calc.prepare(
            atoms,
            None,
            cutoff,
            cutoff / 8,
            cutoff / 4,
        )
        for atoms in structures
    ]

    for i in range(len(structures)):
        pot = calc.potentials(*inputs[i])
        E2, F2, S2 = calc.energy_forces_stress(*inputs[i])

        np.testing.assert_allclose(potentials[atom_to_structure == i], pot)
        np.testing.assert_allclose(E[i], E2)
        np.testing.assert_allclose(F[atom_to_structure == i], F2)
        np.testing.assert_allclose(S[i], S2)


@pytest.mark.parametrize("cutoff", [4.0, 5.0, 6.0])
def test_mixed(cutoff):
    """Test calculator with mixed periodic and non-periodic systems."""
    from jaxpme.batched_mixed.calculators import Ewald

    atoms = read(REFERENCE_STRUCTURES_DIR / "coulomb_test_frames.xyz", index="0")
    charges = atoms.get_initial_charges()
    atoms2 = atoms.copy()
    atoms2.set_pbc(False)

    all_distances = atoms2.get_all_distances()
    mask = all_distances != 0.0
    one_over_r = np.where(mask, 1 / np.where(mask, all_distances, 1.0), 0.0)
    bare_potentials = np.einsum("ij,j->ij", one_over_r, charges)
    reference_potentials = bare_potentials.sum(axis=1) / 2

    calculator = Ewald(prefactor=1.0)
    charges_batch, sr_batch, nonperiodic_batch, periodic_batch = calculator.prepare(
        [atoms, atoms2], cutoff
    )
    atom_to_structure = sr_batch.atom_to_structure
    potentials = calculator.potentials(
        charges_batch, sr_batch, nonperiodic_batch, periodic_batch
    )

    np.testing.assert_allclose(potentials[atom_to_structure == 1], reference_potentials)


@pytest.mark.parametrize("cutoff", [4.0, 5.0, 6.0])
def test_single_system_vs_serial(cutoff):
    """Test calculator with a single periodic system."""
    from jaxpme.batched_mixed.calculators import Ewald
    from jaxpme.calculators import Ewald as SerialEwald

    atoms = read(REFERENCE_STRUCTURES_DIR / "coulomb_test_frames.xyz", index="0")

    # Calculate with batched calculator
    calculator = Ewald(prefactor=1.0)
    charges, sr_batch, nonperiodic_batch, periodic_batch = calculator.prepare(
        [atoms], cutoff
    )
    potentials = calculator.potentials(charges, sr_batch, nonperiodic_batch, periodic_batch)
    energy = calculator.energy(charges, sr_batch, nonperiodic_batch, periodic_batch)

    # Compare with serial calculator
    calc = SerialEwald()
    inputs = calc.prepare(atoms, None, cutoff, cutoff / 8, cutoff / 4)
    pot_ref = calc.potentials(*inputs)
    energy_ref = calc.energy(*inputs)

    np.testing.assert_allclose(potentials[sr_batch.atom_mask], pot_ref)
    np.testing.assert_allclose(energy[0], energy_ref)


@pytest.mark.parametrize("frame_index", [0, 1, 2, 3, 4, 5])
@pytest.mark.parametrize("cutoff", [4.0, 5.0, 6.0])
def test_single_system_vs_reference(frame_index, cutoff):
    """Test calculator with a single (mixed) pbc system."""
    from jaxpme import prefactors
    from jaxpme.batched_mixed.calculators import Ewald

    atoms = read(REFERENCE_STRUCTURES_DIR / "coulomb_test_frames.xyz", index=frame_index)

    calculator = Ewald(prefactor=prefactors.eV_A)
    charges, sr_batch, nonperiodic_batch, periodic_batch = calculator.prepare(
        [atoms], cutoff, smearing=cutoff / 8, lr_wavelength=cutoff / 16
    )
    energy, forces = calculator.energy_forces(
        charges, sr_batch, nonperiodic_batch, periodic_batch
    )

    # todo: why are the tolerances so big?
    np.testing.assert_allclose(
        energy[sr_batch.structure_mask], atoms.get_potential_energy(), rtol=1e-4, atol=0
    )
    np.testing.assert_allclose(forces[sr_batch.atom_mask], atoms.get_forces(), rtol=2e-2)


@pytest.mark.parametrize("cutoff", [5.0])
def test_mixed_pbc_calculator(cutoff):
    """Smoke test for mixed PBC: 2D orthorhombic + non-periodic end-to-end."""
    from ase import Atoms

    from jaxpme.batched_mixed.calculators import Ewald

    # 2D orthorhombic structure (needs to be large enough for the cutoff or handled)
    cell_2d = np.diag([10.0, 10.0, 20.0])
    atoms_2d = Atoms(
        "H2",
        positions=[[5.0, 5.0, 5.0], [5.0, 5.0, 6.0]],
        cell=cell_2d,
        pbc=[True, True, False],
    )
    atoms_2d.set_initial_charges([1.0, -1.0])

    # Non-periodic structure
    atoms_0d = Atoms(
        "H2", positions=[[0.0, 0.0, 0.0], [0.0, 0.0, 1.0]], pbc=[False, False, False]
    )
    atoms_0d.set_initial_charges([1.0, -1.0])

    calculator = Ewald(prefactor=1.0)
    charges, sr_batch, nonp_batch, pbc_batch = calculator.prepare(
        [atoms_2d, atoms_0d], cutoff
    )

    # Check everything runs
    potentials = calculator.potentials(charges, sr_batch, nonp_batch, pbc_batch)
    energy = calculator.energy(charges, sr_batch, nonp_batch, pbc_batch)
    energy_f, forces = calculator.energy_forces(charges, sr_batch, nonp_batch, pbc_batch)

    assert not np.isnan(potentials).any()
    assert not np.isnan(energy).any()
    assert not np.isnan(forces).any()

    # Energy should be a vector of length num_structures (padded to power of 2)
    # Here 2 structures + 1 padding -> next power of 2 is 4
    assert energy.shape[0] >= 2
    assert energy[0] != 0
    assert energy[1] != 0
    assert energy[2] == 0  # Padding
    assert energy[3] == 0  # Padding

    # Basic sanity: energy of neutral H2 (0D) with prefactor=1.0
    # should be something reasonable
    assert energy[1] < 0  # Binding energy for opposites


@pytest.mark.parametrize("cutoff", [4.0, 5.0, 6.0])
def test_halfspace_equivalence(cutoff):
    """Test that halfspace optimization gives same results as full k-space."""
    from jaxpme.batched_mixed.calculators import Ewald

    structures = read(REFERENCE_STRUCTURES_DIR / "coulomb_test_frames.xyz", index=":3")

    # Full k-space calculator
    calculator_full = Ewald(prefactor=1.0, halfspace=False)
    charges_full, sr_batch_full, nonp_full, pbc_full = calculator_full.prepare(
        structures, cutoff
    )

    # Halfspace calculator
    calculator_half = Ewald(prefactor=1.0, halfspace=True)
    charges_half, sr_batch_half, nonp_half, pbc_half = calculator_half.prepare(
        structures, cutoff
    )

    # Verify k-vector count is reduced (roughly half)
    # Count non-zero k-vectors (k=0 has G=0, padded k-vectors are also 0)
    k2_full = (pbc_full.k_grid[0] ** 2).sum(axis=-1)
    k2_half = (pbc_half.k_grid[0] ** 2).sum(axis=-1)
    actual_k_full = (k2_full > 0).sum()
    actual_k_half = (k2_half > 0).sum()
    # Half-space should have fewer k-vectors
    assert actual_k_half < actual_k_full

    # Compare potentials (use rtol=1e-8 due to different FP operation ordering)
    pot_full = calculator_full.potentials(charges_full, sr_batch_full, nonp_full, pbc_full)
    pot_half = calculator_half.potentials(charges_half, sr_batch_half, nonp_half, pbc_half)
    np.testing.assert_allclose(pot_full, pot_half, rtol=1e-8)

    # Compare energy
    E_full = calculator_full.energy(charges_full, sr_batch_full, nonp_full, pbc_full)
    E_half = calculator_half.energy(charges_half, sr_batch_half, nonp_half, pbc_half)
    np.testing.assert_allclose(E_full, E_half, rtol=1e-8)

    # Compare forces
    E_full2, F_full = calculator_full.energy_forces(
        charges_full, sr_batch_full, nonp_full, pbc_full
    )
    E_half2, F_half = calculator_half.energy_forces(
        charges_half, sr_batch_half, nonp_half, pbc_half
    )
    np.testing.assert_allclose(F_full, F_half, rtol=1e-8)

    # Compare stress
    E_full3, F_full3, S_full = calculator_full.energy_forces_stress(
        charges_full, sr_batch_full, nonp_full, pbc_full
    )
    E_half3, F_half3, S_half = calculator_half.energy_forces_stress(
        charges_half, sr_batch_half, nonp_half, pbc_half
    )
    np.testing.assert_allclose(S_full, S_half, rtol=1e-8)


@pytest.mark.parametrize("shape", [(4, 4, 4), (5, 5, 5), (8, 8, 8), (7, 9, 11)])
def test_halfspace_kvector_count(shape):
    """Test that count_halfspace_kvectors matches actual generated grid."""
    from jaxpme.batched_mixed.kspace import count_halfspace_kvectors, generate_ewald_k_grid

    k_grid_half = generate_ewald_k_grid(shape, halfspace=True)
    k2 = (k_grid_half**2).sum(axis=-1)
    actual = (k2 > 0).sum()
    expected = count_halfspace_kvectors(shape)
    assert actual == expected


@pytest.mark.parametrize("cutoff", [4.0, 5.0, 6.0])
def test_iplp_batched_vs_serial(cutoff):
    """Test inverse power law potential with batched vs serial calculator."""
    from jaxpme.batched_mixed.calculators import Ewald
    from jaxpme.calculators import Ewald as SerialEwald
    from jaxpme.potentials import inverse_power_law

    structures = read(REFERENCE_STRUCTURES_DIR / "coulomb_test_frames.xyz", index=":3")

    iplp = inverse_power_law(1)

    # Batched calculator
    calculator = Ewald(custom_potential=iplp, prefactor=1.0)
    charges, sr_batch, nonp_batch, pbc_batch = calculator.prepare(structures, cutoff)
    energy_batched = calculator.energy(charges, sr_batch, nonp_batch, pbc_batch)

    # Serial calculator for comparison
    serial_calc = SerialEwald(custom_potential=iplp)
    for i, atoms in enumerate(structures):
        inputs = serial_calc.prepare(atoms, None, cutoff, cutoff / 8, cutoff / 4)
        energy_serial = serial_calc.energy(*inputs)
        np.testing.assert_allclose(energy_batched[i], energy_serial, rtol=1e-5)


@pytest.mark.parametrize("cutoff", [4.0, 5.0, 6.0])
def test_iplp_halfspace_equivalence(cutoff):
    """Test that IPLP halfspace optimization gives same results as full k-space."""
    from jaxpme.batched_mixed.calculators import Ewald
    from jaxpme.potentials import inverse_power_law

    structures = read(REFERENCE_STRUCTURES_DIR / "coulomb_test_frames.xyz", index=":3")
    iplp = inverse_power_law(1)

    # Full k-space
    calc_full = Ewald(custom_potential=iplp, halfspace=False)
    charges_full, sr_full, nonp_full, pbc_full = calc_full.prepare(structures, cutoff)
    E_full = calc_full.energy(charges_full, sr_full, nonp_full, pbc_full)

    # Halfspace
    calc_half = Ewald(custom_potential=iplp, halfspace=True)
    charges_half, sr_half, nonp_half, pbc_half = calc_half.prepare(structures, cutoff)
    E_half = calc_half.energy(charges_half, sr_half, nonp_half, pbc_half)

    np.testing.assert_allclose(E_full, E_half, rtol=1e-8)


def test_iplp_2d_pbc_returns_nan():
    """Test that IPLP with 2D PBC returns NaN (unsupported)."""
    from ase import Atoms

    from jaxpme.batched_mixed.calculators import Ewald
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
    charges, sr_batch, nonp_batch, pbc_batch = calculator.prepare([atoms_2d], cutoff=5.0)
    energy = calculator.energy(charges, sr_batch, nonp_batch, pbc_batch)

    # Energy should be NaN for 2D PBC with IPLP
    assert np.isnan(energy[0])
