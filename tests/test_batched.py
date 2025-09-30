import numpy as np

import pytest
from ase.io import read


@pytest.mark.parametrize("cutoff", [4.0, 5.0, 6.0])
def test_reference_structures(cutoff):
    from jaxpme.batched.calculators import Ewald

    structures = read("reference_structures/coulomb_test_frames.xyz", index=":")
    atoms_no_pbc = structures[-1].copy()
    atoms_no_pbc.set_pbc(False)

    calculator = Ewald(prefactor=1.0)
    batch = calculator.prepare(structures + [atoms_no_pbc], None, cutoff)
    atom_to_system = batch[5]
    potentials = calculator.potentials(*batch)

    E, F, S = calculator.energy_forces_stress(*batch)

    from jaxpme.calculators import Ewald as SerialEwald

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

        np.testing.assert_allclose(potentials[atom_to_system == i], pot, rtol=2e-5)
        np.testing.assert_allclose(E[i], E2, rtol=2e-5)
        np.testing.assert_allclose(F[atom_to_system == i], F2, rtol=1e-4)
        np.testing.assert_allclose(S[i], S2, rtol=1e-4)


@pytest.mark.parametrize("cutoff", [4.0, 5.0, 6.0])
def test_mixed(cutoff):
    from jaxpme.batched.calculators import Ewald

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
    batch = calculator.prepare([atoms, atoms2], [charges, charges], cutoff)
    atom_to_system = batch[5]
    potentials = calculator.potentials(*batch)

    np.testing.assert_allclose(
        potentials[atom_to_system == 1], reference_potentials, rtol=1e-5
    )
