import numpy as np
import jax

import pytest
from ase.io import read

jax.config.update("jax_enable_x64", True)


@pytest.mark.parametrize("cutoff", [4.0, 5.0, 6.0])
def test_reference_structures(cutoff):
    """Test calculator with multiple structures w/ pbc and no-pbc systems."""
    from jaxpme.batched_mixed.calculators import Ewald
    from jaxpme.calculators import Ewald as SerialEwald

    structures = read("reference_structures/coulomb_test_frames.xyz", index=":")
    atoms_no_pbc = structures[-1].copy()
    atoms_no_pbc.set_pbc(False)

    calculator = Ewald(prefactor=1.0)
    charges, sr_batch, nonperiodic_batch, periodic_batch = calculator.prepare(
        structures + [atoms_no_pbc], cutoff
    )
    atom_to_structure = sr_batch.atom_to_structure
    potentials = calculator.potentials(charges, sr_batch, nonperiodic_batch, periodic_batch)

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

    atoms = read("reference_structures/coulomb_test_frames.xyz", index="0")
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
def test_single_periodic_system(cutoff):
    """Test calculator with a single periodic system."""
    from jaxpme.batched_mixed.calculators import Ewald
    from jaxpme.calculators import Ewald as SerialEwald

    atoms = read("reference_structures/coulomb_test_frames.xyz", index="0")

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
