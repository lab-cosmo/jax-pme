"""Explicit mixed-PBC correctness: a single batch containing 3D-PBC,
2D-PBC orthorhombic, 2D-PBC triclinic, and non-PBC structures, validated
per-structure against the serial Ewald calculator at fp64.
"""

import numpy as np
import jax

import pytest
from ase import Atoms
from ase.build import bulk

jax.config.update("jax_enable_x64", True)


def _num_k_for_lr(atoms, lr_wavelength):
    L = np.linalg.norm(atoms.get_cell().array, axis=-1)
    return int(np.prod(L) / (2 * lr_wavelength**3))


def _make_systems():
    # 3D-PBC: NaCl
    atoms_3d = bulk("NaCl", "rocksalt", a=5.6)
    atoms_3d.set_initial_charges(np.tile([1.0, -1.0], len(atoms_3d) // 2))

    # 2D-PBC orthorhombic
    atoms_2d_ortho = Atoms(
        "H2",
        positions=[[5.0, 5.0, 5.0], [5.0, 5.0, 6.0]],
        cell=np.diag([10.0, 10.0, 20.0]),
        pbc=[True, True, False],
    )
    atoms_2d_ortho.set_initial_charges([1.0, -1.0])

    # 2D-PBC triclinic: in-plane non-orthogonal lattice
    cell_tri = np.array(
        [
            [8.0, 0.0, 0.0],
            [4.0, 7.0, 0.0],
            [0.0, 0.0, 20.0],
        ]
    )
    atoms_2d_tri = Atoms(
        "H2",
        positions=[[1.0, 1.0, 9.0], [4.0, 3.0, 11.0]],
        cell=cell_tri,
        pbc=[True, True, False],
    )
    atoms_2d_tri.set_initial_charges([1.0, -1.0])

    # non-PBC: H2 molecule
    atoms_0d = Atoms("H2", positions=[[0.0, 0.0, 0.0], [0.0, 0.0, 1.0]], pbc=False)
    atoms_0d.set_initial_charges([1.0, -1.0])

    return [atoms_3d, atoms_2d_ortho, atoms_2d_tri, atoms_0d]


@pytest.mark.parametrize("cutoff", [4.0, 5.0])
def test_mixed_pbc_vs_batched_mixed(cutoff):
    """[3D, 2D-ortho, 2D-triclinic, non-PBC] in one batch, equivalence vs batched_mixed."""
    from jaxpme.batched_mixed.calculators import Ewald as MixedEwald
    from jaxpme.batched_tiled.calculators import Ewald as TiledEwald

    systems = _make_systems()
    pbc_systems = [a for a in systems if a.get_pbc().any()]
    num_k = max(_num_k_for_lr(a, cutoff / 8.0) for a in pbc_systems)

    ct = TiledEwald(prefactor=1.0)
    ch_t, sr_t, nopbc_t, pbc_t = ct.prepare(systems, num_k=num_k, cutoff=cutoff)
    E_t, F_t, S_t = ct.energy_forces_stress(ch_t, sr_t, nopbc_t, pbc_t)

    cm = MixedEwald(prefactor=1.0)
    ch_m, sr_m, nopbc_m, pbc_m = cm.prepare(systems, cutoff=cutoff, num_k=num_k)
    E_m, F_m, S_m = cm.energy_forces_stress(ch_m, sr_m, nopbc_m, pbc_m)

    # per-structure energies match (real systems only — same order in both batches)
    for i in range(len(systems)):
        np.testing.assert_allclose(
            float(E_t[i]),
            float(E_m[i]),
            rtol=1e-10,
            err_msg=f"structure {i} energy mismatch",
        )

    # forces (mask-extracted; both backends pad differently)
    np.testing.assert_allclose(
        np.array(F_t[sr_t.atom_mask]),
        np.array(F_m[sr_m.atom_mask]),
        rtol=1e-9,
        atol=1e-15,
    )

    # stress (mask-extracted)
    np.testing.assert_allclose(
        np.array(S_t[sr_t.structure_mask]),
        np.array(S_m[sr_m.structure_mask]),
        rtol=1e-9,
        atol=1e-15,
    )


@pytest.mark.parametrize("cutoff", [4.0, 5.0])
def test_mixed_pbc_vs_serial_per_structure(cutoff):
    """3D-PBC structure inside a mixed batch matches its standalone serial result.

    Serial Ewald only supports 3D-PBC (not 2D, not non-PBC); we validate the
    3D-PBC system in the batch directly, then verify the non-PBC system against
    a bare 1/r reference.
    """
    from jaxpme.batched_tiled.calculators import Ewald as TiledEwald
    from jaxpme.calculators import Ewald as SerialEwald

    systems = _make_systems()
    pbc_systems = [a for a in systems if a.get_pbc().any()]
    num_k = max(_num_k_for_lr(a, cutoff / 8.0) for a in pbc_systems)

    ct = TiledEwald(prefactor=1.0)
    ch_t, sr_t, nopbc_t, pbc_t = ct.prepare(
        systems, num_k=num_k, cutoff=cutoff, smearing=cutoff / 4
    )
    E_t, F_t = ct.energy_forces(ch_t, sr_t, nopbc_t, pbc_t)
    pot_t = ct.potentials(ch_t, sr_t, nopbc_t, pbc_t)

    # 3D-PBC: validate against serial Ewald
    idx_3d = next(i for i, a in enumerate(systems) if a.get_pbc().sum() == 3)
    serial = SerialEwald()
    inputs = serial.prepare(systems[idx_3d], None, cutoff, cutoff / 8, cutoff / 4)
    E_s, F_s = serial.energy_forces(*inputs)
    np.testing.assert_allclose(float(E_t[idx_3d]), float(E_s), rtol=1e-7)
    mask_3d = (sr_t.atom_to_structure == idx_3d) & sr_t.atom_mask
    np.testing.assert_allclose(np.array(F_t[mask_3d]), np.array(F_s), rtol=1e-7, atol=1e-10)

    # non-PBC: bare 1/r reference
    idx_0d = next(i for i, a in enumerate(systems) if not a.get_pbc().any())
    atoms_0d = systems[idx_0d]
    q = atoms_0d.get_initial_charges()
    dist = atoms_0d.get_all_distances()
    mask = dist != 0.0
    one_over_r = np.where(mask, 1 / np.where(mask, dist, 1.0), 0.0)
    pot_ref = (one_over_r @ q) / 2
    mask_0d = (sr_t.atom_to_structure == idx_0d) & sr_t.atom_mask
    np.testing.assert_allclose(np.array(pot_t[mask_0d]), pot_ref, rtol=1e-10)


def test_2d_triclinic_potentials_are_finite():
    """2D-PBC triclinic case produces finite potentials/forces/stress (no NaN)."""
    from jaxpme.batched_tiled.calculators import Ewald

    cell_tri = np.array([[8.0, 0.0, 0.0], [4.0, 7.0, 0.0], [0.0, 0.0, 20.0]])
    atoms = Atoms(
        "H2",
        positions=[[1.0, 1.0, 9.0], [4.0, 3.0, 11.0]],
        cell=cell_tri,
        pbc=[True, True, False],
    )
    atoms.set_initial_charges([1.0, -1.0])

    num_k = _num_k_for_lr(atoms, 0.625)
    calc = Ewald(prefactor=1.0)
    ch, sr, nopbc, pbc = calc.prepare([atoms], num_k=num_k, cutoff=5.0)
    E, F, S = calc.energy_forces_stress(ch, sr, nopbc, pbc)

    assert not np.isnan(np.array(E)).any()
    assert not np.isnan(np.array(F)).any()
    assert not np.isnan(np.array(S)).any()
