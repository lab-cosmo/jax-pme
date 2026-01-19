import numpy as np
import jax
import jax.numpy as jnp

import math

import pytest
from ase import Atoms
from ase.io import read
from conftest import REFERENCE_STRUCTURES_DIR

from jaxpme import PME, Ewald

jax.config.update("jax_enable_x64", True)
DTYPE = jnp.float64


def get_calculator(name, potential, prefactor=1.0):
    if name == "ewald":
        if potential == "coulomb":
            return Ewald(prefactor=prefactor)
        else:
            from jaxpme.potentials import inverse_power_law

            iplp = inverse_power_law(1)
            return Ewald(custom_potential=iplp, prefactor=prefactor)
    elif name == "pme":
        if potential == "coulomb":
            return PME(prefactor=prefactor)
        else:
            from jaxpme.potentials import inverse_power_law

            iplp = inverse_power_law(1)
            return PME(custom_potential=iplp, prefactor=prefactor)


def generate_orthogonal_transformations():
    # Generate rotation matrix along x-axis
    def rot_x(phi):
        rot = np.zeros((3, 3), dtype=DTYPE)
        rot[0, 0] = rot[1, 1] = math.cos(phi)
        rot[0, 1] = -math.sin(phi)
        rot[1, 0] = math.sin(phi)
        rot[2, 2] = 1.0

        return rot

    # Generate rotation matrix along z-axis
    def rot_z(theta):
        rot = np.zeros((3, 3), dtype=DTYPE)
        rot[0, 0] = rot[2, 2] = math.cos(theta)
        rot[0, 2] = math.sin(theta)
        rot[2, 0] = -math.sin(theta)
        rot[1, 1] = 1.0

        return rot

    # Generate a few rotation matrices
    rot_1 = rot_z(0.987654)
    rot_2 = rot_z(1.23456) @ rot_x(0.82321)
    transformations = [rot_1, rot_2]

    # make sure that the generated transformations are indeed orthogonal
    for q in transformations:
        id = np.eye(3, dtype=DTYPE)
        id_2 = q.T @ q
        np.testing.assert_allclose(id, id_2, atol=1e-15, rtol=1e-15)

    return transformations


def define_crystal(crystal_name="CsCl"):
    # Define all relevant parameters (atom positions, charges, cell) of the reference
    # crystal structures for which the Madelung constants obtained from the Ewald sums
    # are compared with reference values.
    # see https://www.sciencedirect.com/science/article/pii/B9780128143698000078#s0015
    # More detailed values can be found in https://pubs.acs.org/doi/10.1021/ic2023852

    # Caesium-Chloride (CsCl) structure:
    # - Cubic unit cell
    # - 1 atom pair in the unit cell
    # - Cation-Anion ratio of 1:1
    if crystal_name == "CsCl":
        positions = np.array([[0, 0, 0], [0.5, 0.5, 0.5]], dtype=DTYPE)
        charges = np.array([-1.0, 1.0], dtype=DTYPE)
        cell = np.eye(3, dtype=DTYPE)
        madelung_ref = 2.035361
        num_formula_units = 1

    # Sodium-Chloride (NaCl) structure using a primitive unit cell
    # - non-cubic unit cell (fcc)
    # - 1 atom pair in the unit cell
    # - Cation-Anion ratio of 1:1
    elif crystal_name == "NaCl_primitive":
        positions = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=DTYPE)
        charges = np.array([1.0, -1.0], dtype=DTYPE)
        cell = np.array([[0, 1.0, 1], [1, 0, 1], [1, 1, 0]], dtype=DTYPE)  # fcc
        madelung_ref = 1.74756
        num_formula_units = 1

    # Sodium-Chloride (NaCl) structure using a cubic unit cell
    # - cubic unit cell
    # - 4 atom pairs in the unit cell
    # - Cation-Anion ratio of 1:1
    elif crystal_name == "NaCl_cubic":
        positions = np.array(
            [
                [0.0, 0, 0],
                [1, 0, 0],
                [0, 1, 0],
                [0, 0, 1],
                [1, 1, 0],
                [1, 0, 1],
                [0, 1, 1],
                [1, 1, 1],
            ],
            dtype=DTYPE,
        )
        charges = np.array([+1.0, -1, -1, -1, +1, +1, +1, -1], dtype=DTYPE)
        cell = 2 * np.eye(3, dtype=DTYPE)
        madelung_ref = 1.747565
        num_formula_units = 4

    # ZnS (zincblende) structure
    # - non-cubic unit cell (fcc)
    # - 1 atom pair in the unit cell
    # - Cation-Anion ratio of 1:1
    # Remarks: we use a primitive unit cell which makes the lattice parameter of the
    # cubic cell equal to 2.
    elif crystal_name == "zincblende":
        positions = np.array([[0, 0, 0], [0.5, 0.5, 0.5]], dtype=DTYPE)
        charges = np.array([1.0, -1], dtype=DTYPE)
        cell = np.array([[0, 1, 1], [1, 0, 1], [1, 1, 0]], dtype=DTYPE)
        madelung_ref = 2 * 1.63806 / np.sqrt(3)
        num_formula_units = 1

    # Wurtzite structure
    # - non-cubic unit cell (triclinic)
    # - 2 atom pairs in the unit cell
    # - Cation-Anion ratio of 1:1
    elif crystal_name == "wurtzite":
        u = 3 / 8
        c = np.sqrt(1 / u)
        positions = np.array(
            [
                [0.5, 0.5 / np.sqrt(3), 0.0],
                [0.5, 0.5 / np.sqrt(3), u * c],
                [0.5, -0.5 / np.sqrt(3), 0.5 * c],
                [0.5, -0.5 / np.sqrt(3), (0.5 + u) * c],
            ],
            dtype=DTYPE,
        )
        charges = np.array([1.0, -1, 1, -1], dtype=DTYPE)
        cell = np.array(
            [[0.5, -0.5 * np.sqrt(3), 0], [0.5, 0.5 * np.sqrt(3), 0], [0, 0, c]],
            dtype=DTYPE,
        )
        madelung_ref = 1.64132 / (u * c)
        num_formula_units = 2

    # Fluorite structure (e.g. CaF2 with Ca2+ and F-)
    # - non-cubic (fcc) unit cell
    # - 1 neutral molecule per unit cell
    # - Cation-Anion ratio of 1:2
    elif crystal_name == "fluorite":
        a = 5.463
        a = 1.0
        positions = a * np.array(
            [[1 / 4, 1 / 4, 1 / 4], [3 / 4, 3 / 4, 3 / 4], [0, 0, 0]], dtype=DTYPE
        )
        charges = np.array([-1, -1, 2], dtype=DTYPE)
        cell = np.array([[a, a, 0], [a, 0, a], [0, a, a]], dtype=DTYPE) / 2.0
        madelung_ref = 11.636575
        num_formula_units = 1

    # Copper(I)-Oxide structure (e.g. Cu2O with Cu+ and O2-)
    # - cubic unit cell
    # - 2 neutral molecules per unit cell
    # - Cation-Anion ratio of 2:1
    elif crystal_name == "cu2o":
        a = 1.0
        positions = a * np.array(
            [
                [0, 0, 0],
                [1 / 2, 1 / 2, 1 / 2],
                [1 / 4, 1 / 4, 1 / 4],
                [1 / 4, 3 / 4, 3 / 4],
                [3 / 4, 1 / 4, 3 / 4],
                [3 / 4, 3 / 4, 1 / 4],
            ],
            dtype=DTYPE,
        )
        charges = np.array([-2, -2, 1, 1, 1, 1], dtype=DTYPE)
        cell = a * np.eye(3, dtype=DTYPE)
        madelung_ref = 10.2594570330750
        num_formula_units = 2

    # Wigner crystal in simple cubic structure.
    # Wigner crystals are equivalent to the Jellium or uniform electron gas models.
    # For the purpose of this test, we define them to be structures in which the ion
    # cores form a perfect lattice, while the electrons are uniformly distributed over
    # the cell. In some sources, the role of the positive and negative charges are
    # flipped. These structures are used to test the code for cases in which the total
    # charge of the particles is not zero.
    # Wigner crystal energies are taken from "Zero-Point Energy of an Electron Lattice"
    # by Rosemary A., Coldwell‐Horsfall and Alexei A. Maradudin (1960), eq. (A21).
    elif crystal_name == "wigner_sc":
        positions = np.array([[0, 0, 0]], dtype=DTYPE)
        charges = np.array([1.0], dtype=DTYPE)
        cell = np.array([[1.0, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=DTYPE)

        # Reference value is expressed in terms of the Wigner-Seiz radius, and needs to
        # be rescaled to the case in which the lattice parameter = 1.
        madelung_wigner_seiz = 1.7601188
        wigner_seiz_radius = (3 / (4 * np.pi)) ** (1 / 3)
        madelung_ref = madelung_wigner_seiz / wigner_seiz_radius  # 2.83730
        num_formula_units = 1

    # Wigner crystal in bcc structure (note: this is the most stable structure).
    # See description of "wigner_sc" for a general explanation on Wigner crystals.
    # Used to test the code for cases in which the unit cell has a nonzero net charge.
    elif crystal_name == "wigner_bcc":
        positions = np.array([[0, 0, 0]], dtype=DTYPE)
        charges = np.array([1.0], dtype=DTYPE)
        cell = np.array([[1.0, 0, 0], [0, 1, 0], [1 / 2, 1 / 2, 1 / 2]], dtype=DTYPE)

        # Reference value is expressed in terms of the Wigner-Seiz radius, and needs to
        # be rescaled to the case in which the lattice parameter = 1.
        madelung_wigner_seiz = 1.791860
        wigner_seiz_radius = (3 / (4 * np.pi * 2)) ** (1 / 3)  # 2 atoms per cubic unit cell
        madelung_ref = madelung_wigner_seiz / wigner_seiz_radius  # 3.63924
        num_formula_units = 1

    # Same as above, but now using a cubic unit cell rather than the primitive bcc cell
    elif crystal_name == "wigner_bcc_cubiccell":
        positions = np.array([[0, 0, 0], [1 / 2, 1 / 2, 1 / 2]], dtype=DTYPE)
        charges = np.array([1.0, 1.0], dtype=DTYPE)
        cell = np.array([[1.0, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=DTYPE)

        # Reference value is expressed in terms of the Wigner-Seiz radius, and needs to
        # be rescaled to the case in which the lattice parameter = 1.
        madelung_wigner_seiz = 1.791860
        wigner_seiz_radius = (3 / (4 * np.pi * 2)) ** (1 / 3)  # 2 atoms per cubic unit cell
        madelung_ref = madelung_wigner_seiz / wigner_seiz_radius  # 3.63924
        num_formula_units = 2

    # Wigner crystal in fcc structure
    # See description of "wigner_sc" for a general explanation on Wigner crystals.
    # Used to test the code for cases in which the unit cell has a nonzero net charge.
    elif crystal_name == "wigner_fcc":
        positions = np.array([[0, 0, 0]], dtype=DTYPE)
        charges = np.array([1.0], dtype=DTYPE)
        cell = np.array([[1, 0, 1], [0, 1, 1], [1, 1, 0]], dtype=DTYPE) / 2

        # Reference value is expressed in terms of the Wigner-Seiz radius, and needs to
        # be rescaled to the case in which the lattice parameter = 1.
        madelung_wigner_seiz = 1.791753
        wigner_seiz_radius = (3 / (4 * np.pi * 4)) ** (1 / 3)  # 4 atoms per cubic unit cell
        madelung_ref = madelung_wigner_seiz / wigner_seiz_radius  # 4.58488
        num_formula_units = 1

    # Same as above, but now using a cubic unit cell rather than the primitive fcc cell
    elif crystal_name == "wigner_fcc_cubiccell":
        positions = 0.5 * np.array(
            [[0.0, 0, 0], [1, 0, 1], [1, 1, 0], [0, 1, 1]], dtype=DTYPE
        )
        charges = np.array([1.0, 1, 1, 1], dtype=DTYPE)
        cell = np.eye(3, dtype=DTYPE)

        # Reference value is expressed in terms of the Wigner-Seiz radius, and needs to
        # be rescaled to the case in which the lattice parameter = 1.
        madelung_wigner_seiz = 1.791753
        wigner_seiz_radius = (3 / (4 * np.pi * 4)) ** (1 / 3)  # 4 atoms per cubic unit cell
        madelung_ref = madelung_wigner_seiz / wigner_seiz_radius  # 4.58488
        num_formula_units = 4

    else:
        raise ValueError(f"crystal_name = {crystal_name} is not supported!")

    madelung_ref = np.array(madelung_ref, dtype=DTYPE)
    charges = charges.reshape((-1, 1))

    return positions, charges, cell, madelung_ref, num_formula_units


@pytest.mark.parametrize(
    "crystal_name",
    [
        "CsCl",
        "NaCl_primitive",
        "NaCl_cubic",
        "zincblende",
        "wurtzite",
        "cu2o",
        "fluorite",
    ],
)
@pytest.mark.parametrize("scaling_factor", [1 / 2.0353610, 1.0, 3.4951291])
@pytest.mark.parametrize("calc_name", ["ewald", "pme"])
@pytest.mark.parametrize("potential", ["coulomb", "iplp"])
def test_madelung(crystal_name, scaling_factor, calc_name, potential):
    """
    Check that the Madelung constants obtained from the Ewald sum calculator matches
    the reference values.
    In this test, only the charge-neutral crystal systems are chosen for which the
    potential converges relatively quickly, while the systems with a net charge are
    treated separately below.
    The structures cover a broad range of simple crystals, with cells ranging from cubic
    to triclinic, as well as cation-anion ratios of 1:1, 1:2 and 2:1.
    """
    # Get input parameters and adjust to account for scaling
    pos, charges, cell, madelung_ref, num_units = define_crystal(crystal_name)
    pos *= scaling_factor
    cell *= scaling_factor
    madelung_ref /= scaling_factor

    charges = jnp.array(charges).flatten()
    atoms = Atoms(positions=pos, cell=cell, pbc=True)

    calc = get_calculator(calc_name, potential)

    # Define calculator and tolerances
    if calc_name == "ewald":
        sr_cutoff = scaling_factor
        smearing = sr_cutoff / 5.0
        lr_wavelength = 0.5 * smearing
        rtol = 4e-6

        inputs = calc.prepare(atoms, charges, sr_cutoff, lr_wavelength, smearing)
        calculate = calc.energy
    elif calc_name == "pme":
        sr_cutoff = 2 * scaling_factor
        smearing = sr_cutoff / 5.0
        rtol = 9e-4

        inputs = calc.prepare(atoms, charges, sr_cutoff, smearing / 8, smearing)
        calculate = calc.energy

    # Compute potential and compare against target value using default hypers
    energy = calculate(*inputs)
    madelung = -energy / num_units

    np.testing.assert_allclose(madelung, madelung_ref, atol=0.0, rtol=rtol)


# Since structures without charge neutrality show slower convergence, these
# structures are tested separately.
@pytest.mark.parametrize(
    "crystal_name",
    [
        "wigner_sc",
        "wigner_fcc",
        "wigner_fcc_cubiccell",
        "wigner_bcc",
        "wigner_bcc_cubiccell",
    ],
)
@pytest.mark.parametrize("scaling_factor", [0.4325, 1.0, 2.0353610])
def test_wigner(crystal_name, scaling_factor):
    """
    Check that the energy of a Wigner solid obtained from the Ewald sum calculator
    matches the reference values.
    In this test, the Wigner solids are defined by placing arranging positively charged
    point particles on a bcc lattice, leading to a net charge of the unit cell if we
    only look at the ions. This charge is compensated by a homogeneous neutral back-
    ground charge of opposite sign (physically: completely delocalized electrons).

    The presence of a net charge (due to the particles but without background) leads
    to numerically slower convergence of the relevant sums.
    """
    # Get parameters defining atomic positions, cell and charges
    positions, charges, cell, madelung_ref, _ = define_crystal(crystal_name)
    positions *= scaling_factor
    cell *= scaling_factor
    madelung_ref /= scaling_factor

    cell_dimensions = jnp.linalg.norm(cell, axis=-1)
    cutoff = float(jnp.min(cell_dimensions) / 2 - 1e-6)

    atoms = Atoms(positions=positions, cell=cell, pbc=True)

    # The first value of 0.1 corresponds to what would be
    # chosen by default for the "wigner_sc" or "wigner_bcc_cubiccell" structure.
    smearings = [0.1, 0.06, 0.019]
    for smearing in smearings:
        # Readjust smearing parameter to match nearest neighbor distance
        if crystal_name in ["wigner_fcc", "wigner_fcc_cubiccell"]:
            smeareff = float(smearing) / np.sqrt(2)
        elif crystal_name in ["wigner_bcc_cubiccell", "wigner_bcc"]:
            smeareff = float(smearing) * np.sqrt(3) / 2
        elif crystal_name == "wigner_sc":
            smeareff = float(smearing)
        smeareff *= scaling_factor

        lr_wavelength = smeareff / 2

        calc = Ewald()

        inputs = calc.prepare(atoms, charges, cutoff, lr_wavelength, smeareff)
        potentials = calc.potentials(*inputs)

        energies = potentials * charges
        energies_ref = -jnp.ones_like(energies) * madelung_ref / 2
        np.testing.assert_allclose(energies, energies_ref, atol=0.0, rtol=4.2e-6)


@pytest.mark.parametrize("sr_cutoff", [5.54, 6.01])
@pytest.mark.parametrize("frame_index", [0, 1, 2])
@pytest.mark.parametrize("scaling_factor", [0.4325, 1.3353610])
@pytest.mark.parametrize("ortho", generate_orthogonal_transformations())
@pytest.mark.parametrize("calc_name", ["pme", "ewald"])
@pytest.mark.parametrize("potential", ["coulomb", "iplp"])
@pytest.mark.parametrize("padded", [False, True])
def test_random_structure(
    sr_cutoff, frame_index, scaling_factor, ortho, calc_name, potential, padded
):
    """
    Verify that energy, forces and stress agree with GROMACS.

    Structures consisting of 4 Na and 4 Cl atoms placed randomly in cubic cells of
    varying sizes.

    GROMACS values are computed with SPME and parameters as defined in the manual:
    https://manual.gromacs.org/documentation/current/user-guide/mdp-options.html#ewald
    """
    # coulombtype = PME fourierspacing = 0.01 ; 1/nm
    # pme_order = 8
    # rcoulomb = 0.3  ; nm
    from jaxpme import prefactors

    frame = read(REFERENCE_STRUCTURES_DIR / "coulomb_test_frames.xyz", frame_index)

    energy_target = jnp.array(frame.get_potential_energy(), dtype=DTYPE) / scaling_factor
    forces_target = jnp.array(frame.get_forces(), dtype=DTYPE) / scaling_factor**2
    stress_target = (
        jnp.array(frame.get_stress(voigt=False, include_ideal_gas=False), dtype=DTYPE)
        / scaling_factor
    )
    stress_target *= 2.0  # convert from GROMACS "virial"

    positions = scaling_factor * (jnp.array(frame.positions, dtype=DTYPE) @ ortho)

    cell = scaling_factor * jnp.array(np.array(frame.cell), dtype=DTYPE) @ ortho
    charges = jnp.array([1, 1, 1, 1, -1, -1, -1, -1], dtype=DTYPE)
    sr_cutoff = scaling_factor * sr_cutoff
    smearing = sr_cutoff / 6.0

    atoms = Atoms(positions=positions, cell=cell, pbc=True)

    calc = get_calculator(calc_name, potential, prefactor=prefactors.eV_A)
    if calc_name == "ewald":
        inputs = calc.prepare(atoms, charges, sr_cutoff, smearing / 2, smearing)

        rtol_e = 2e-5
        rtol_f = 3.5e-3
    elif calc_name == "pme":
        inputs = calc.prepare(atoms, charges, sr_cutoff, smearing / 8, smearing)

        rtol_e = 4.5e-3
        rtol_f = 5.0e-3

    if not padded:
        energy, forces, stress = calc.energy_forces_stress(*inputs)

        np.testing.assert_allclose(energy, energy_target, atol=0.0, rtol=rtol_e)
        np.testing.assert_allclose(forces, forces_target @ ortho, atol=0.0, rtol=rtol_f)

        stress_target = jnp.einsum("ab,aA,bB->AB", stress_target, ortho, ortho)
        np.testing.assert_allclose(stress, stress_target, atol=0.0, rtol=2e-3)

    else:
        num_extra_atoms = 3
        num_extra_pairs = 5
        N = positions.shape[0]

        charges, cell, positions, i, j, S, *args = inputs
        extra_charges = jnp.zeros(num_extra_atoms, dtype=charges.dtype)
        extra_positions = jnp.zeros((num_extra_atoms, 3), dtype=positions.dtype)
        extra_idx = jnp.array([N] * num_extra_pairs, dtype=i.dtype).flatten()
        extra_S = jnp.ones((num_extra_pairs, 3), dtype=S.dtype)

        charges = jnp.concatenate((charges, extra_charges))
        positions = jnp.concatenate((positions, extra_positions))
        i = jnp.concatenate((i, extra_idx))
        j = jnp.concatenate((j, extra_idx))
        S = jnp.concatenate((S, extra_S))

        pair_mask = i != N
        atom_mask = jnp.arange(positions.shape[0]) < N

        inputs = (charges, cell, positions, i, j, S, *args)

        energy, forces, stress = calc.energy_forces_stress(
            *inputs, atom_mask=atom_mask, pair_mask=pair_mask
        )

        forces = forces[:N]

        np.testing.assert_allclose(energy, energy_target, atol=0.0, rtol=rtol_e)
        np.testing.assert_allclose(forces, forces_target @ ortho, atol=0.0, rtol=rtol_f)

        stress_target = jnp.einsum("ab,aA,bB->AB", stress_target, ortho, ortho)
        np.testing.assert_allclose(stress, stress_target, atol=0.0, rtol=2e-3)
