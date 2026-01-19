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
