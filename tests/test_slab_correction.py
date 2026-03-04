import numpy as np
import jax
import jax.numpy as jnp

import math

import pytest
from ase import Atoms
from ase.build import make_supercell
from ase.io import read
from conftest import REFERENCE_STRUCTURES_DIR

jax.config.update("jax_enable_x64", True)
DTYPE = jnp.float64


# -- unit tests for correction_pbc directly --


def test_orthorhombic_regression():
    """New general correction matches the old orthorhombic formula exactly."""
    from jaxpme.potentials import coulomb

    pot = coulomb()

    cell = jnp.array([[10.0, 0, 0], [0, 12.0, 0], [0, 0, 20.0]], dtype=DTYPE)
    positions = jnp.array(
        [[1.0, 2.0, 3.0], [4.0, 5.0, 7.0], [2.0, 8.0, 10.0], [6.0, 3.0, 15.0]],
        dtype=DTYPE,
    )
    charges = jnp.array([1.0, -1.0, 0.5, -0.5], dtype=DTYPE)
    pbc = jnp.array([True, True, False])

    result = pot.correction_pbc(positions, cell, charges, pbc)

    # manual calculation using the original orthorhombic formula
    z_i = positions[:, 2]
    basis_len = 20.0
    charge_tot = jnp.sum(charges)
    M_axis = jnp.sum(charges * z_i)
    M_axis_sq = jnp.sum(charges * (z_i**2))
    expected = (4.0 * jnp.pi) * (
        z_i * M_axis
        - 0.5 * (M_axis_sq + charge_tot * (z_i**2))
        - (charge_tot / 12.0) * (basis_len**2)
    )

    np.testing.assert_allclose(result, expected, atol=1e-12, rtol=0)


def test_analytic_dipole():
    """Analytic verification for a simple +1/-1 dipole along non-periodic axis."""
    from jaxpme.potentials import coulomb

    pot = coulomb()

    cell = jnp.array([[10.0, 0, 0], [0, 10.0, 0], [0, 0, 20.0]], dtype=DTYPE)
    z1, z2 = 9.0, 11.0
    positions = jnp.array([[5.0, 5.0, z1], [5.0, 5.0, z2]], dtype=DTYPE)
    charges = jnp.array([1.0, -1.0], dtype=DTYPE)
    pbc = jnp.array([True, True, False])

    result = pot.correction_pbc(positions, cell, charges, pbc)

    # charge_tot=0, so formula simplifies:
    # E_slab_2d = 4pi * (z_i * M_z - 0.5 * Q_zz)
    M_z = 1.0 * z1 + (-1.0) * z2
    Q_zz = 1.0 * z1**2 + (-1.0) * z2**2
    expected = (
        4.0
        * jnp.pi
        * jnp.array([z1 * M_z - 0.5 * Q_zz, z2 * M_z - 0.5 * Q_zz], dtype=DTYPE)
    )

    np.testing.assert_allclose(result, expected, atol=1e-12, rtol=0)


def test_3d_pbc_gives_zero():
    """correction_pbc returns zero for 3D PBC."""
    from jaxpme.potentials import coulomb

    pot = coulomb()

    cell = jnp.array([[10.0, 0, 0], [0, 10.0, 0], [0, 0, 10.0]], dtype=DTYPE)
    positions = jnp.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=DTYPE)
    charges = jnp.array([1.0, -1.0], dtype=DTYPE)
    pbc = jnp.array([True, True, True])

    result = pot.correction_pbc(positions, cell, charges, pbc)
    np.testing.assert_allclose(result, 0.0, atol=1e-15)


@pytest.mark.parametrize(
    "cell",
    [
        jnp.array([[10.0, 0, 0], [0, 10.0, 0], [0, 0, 20.0]], dtype=DTYPE),
        jnp.array([[10.0, 3.0, 0], [0, 10.0, 0], [1.0, 2.0, 20.0]], dtype=DTYPE),
    ],
    ids=["orthorhombic", "triclinic"],
)
def test_jit_compatibility(cell):
    """correction_pbc JITs cleanly, including gradients."""
    from jaxpme.potentials import coulomb

    pot = coulomb()

    positions = jnp.array([[1.0, 2.0, 3.0], [4.0, 5.0, 7.0]], dtype=DTYPE)
    charges = jnp.array([1.0, -1.0], dtype=DTYPE)
    pbc = jnp.array([True, True, False])

    # JIT the raw correction
    result_eager = pot.correction_pbc(positions, cell, charges, pbc)
    result_jit = jax.jit(pot.correction_pbc)(positions, cell, charges, pbc)
    np.testing.assert_allclose(result_jit, result_eager, atol=1e-14)

    # grad through JIT (correction depends on positions and cell)
    def loss(pos, c):
        return jnp.sum(pot.correction_pbc(pos, c, charges, pbc))

    grads = jax.jit(jax.grad(loss, argnums=(0, 1)))(positions, cell)
    assert jnp.all(jnp.isfinite(grads[0]))
    assert jnp.all(jnp.isfinite(grads[1]))


@pytest.mark.parametrize(
    "pbc,perm",
    [
        ([True, True, False], [0, 1, 2]),
        ([True, False, True], [0, 2, 1]),
        ([False, True, True], [2, 0, 1]),
    ],
)
def test_pbc_orientations_correction_pbc(pbc, perm):
    """Same system with different pbc axes gives same correction (unit level)."""
    from jaxpme.potentials import coulomb

    pot = coulomb()

    base_cell = jnp.array([[10.0, 0, 0], [0, 12.0, 0], [0, 0, 25.0]], dtype=DTYPE)
    base_pos = jnp.array([[2.0, 3.0, 5.0], [7.0, 4.0, 8.0]], dtype=DTYPE)
    charges = jnp.array([1.0, -1.0], dtype=DTYPE)

    cell = base_cell[jnp.array(perm)][:, jnp.array(perm)]
    positions = base_pos[:, jnp.array(perm)]

    result = pot.correction_pbc(positions, cell, charges, jnp.array(pbc))

    # reference: canonical [T,T,F] orientation
    ref = pot.correction_pbc(base_pos, base_cell, charges, jnp.array([True, True, False]))

    np.testing.assert_allclose(result, ref, atol=1e-12, rtol=0)


# -- helpers for batched calculator tests --


def _make_slab_atoms(cell, positions, charges, pbc):
    """Helper to create ASE Atoms with initial charges for 2D PBC."""
    symbols = ["Na" if q > 0 else "Cl" for q in charges]
    atoms = Atoms(symbols, positions=positions, cell=cell, pbc=pbc)
    atoms.set_initial_charges(charges)
    return atoms


def _batched_energy(atoms_list, cutoff=5.0):
    """Compute Ewald energy using the batched_mixed calculator."""
    from jaxpme.batched_mixed.calculators import Ewald

    calculator = Ewald(prefactor=1.0)
    charges, sr_batch, nonp_batch, pbc_batch = calculator.prepare(atoms_list, cutoff=cutoff)
    return calculator.energy(charges, sr_batch, nonp_batch, pbc_batch)


def _batched_energy_forces(atoms_list, cutoff=5.0):
    """Compute Ewald energy and forces using the batched_mixed calculator."""
    from jaxpme.batched_mixed.calculators import Ewald

    calculator = Ewald(prefactor=1.0)
    charges, sr_batch, nonp_batch, pbc_batch = calculator.prepare(atoms_list, cutoff=cutoff)
    return calculator.energy_forces(charges, sr_batch, nonp_batch, pbc_batch)


# -- tiling consistency (key test) --


@pytest.mark.parametrize(
    "tiling_name,P",
    [
        ("primitive", np.eye(3, dtype=int)),
        ("sheared", np.array([[1, 1, 0], [0, 1, 0], [0, 0, 1]])),
        ("2x1", np.array([[2, 0, 0], [0, 1, 0], [0, 0, 1]])),
        ("rotated_45", np.array([[1, 1, 0], [-1, 1, 0], [0, 0, 1]])),
        ("2x2", np.array([[2, 0, 0], [0, 2, 0], [0, 0, 1]])),
        ("sheared_2x1", np.array([[2, 1, 0], [0, 1, 0], [0, 0, 1]])),
    ],
)
def test_tiling_square_lattice(tiling_name, P):
    """Same square 2D lattice with different unit cells gives same energy/f.u."""
    a = 8.0
    L = 25.0

    cell_prim = np.array([[a, 0, 0], [0, a, 0], [0, 0, L]], dtype=float)
    positions_prim = np.array([[0, 0, 5.0], [a / 2, a / 2, 8.0]], dtype=float)
    charges_prim = [1.0, -1.0]

    atoms_prim = _make_slab_atoms(
        cell_prim, positions_prim, charges_prim, [True, True, False]
    )

    atoms_sc = make_supercell(atoms_prim, P)
    n_fu = abs(int(round(np.linalg.det(P))))

    E_prim = _batched_energy([atoms_prim])
    E_sc = _batched_energy([atoms_sc])

    # Tolerance 5e-4: different unit cells get different shrunk cell heights
    # (h_min depends on L_max), leading to different slab correction residuals
    # ~exp(-G_min * gap). The residual is ~1e-4 for α=1.5.
    np.testing.assert_allclose(
        float(E_sc[0]) / n_fu,
        float(E_prim[0]),
        rtol=5e-4,
        err_msg=f"Tiling '{tiling_name}' (n_fu={n_fu}) failed",
    )


@pytest.mark.parametrize(
    "tiling_name,P",
    [
        ("primitive_hex", np.eye(3, dtype=int)),
        ("rectangular", np.array([[1, 0, 0], [1, 2, 0], [0, 0, 1]])),
    ],
)
def test_tiling_hexagonal_lattice(tiling_name, P):
    """Same hexagonal 2D lattice with different unit cells gives same energy/f.u."""
    a = 8.0
    L = 25.0

    cell_prim = np.array(
        [[a, 0, 0], [a / 2, a * math.sqrt(3) / 2, 0], [0, 0, L]], dtype=float
    )
    positions_prim = np.array(
        [[0, 0, 5.0], [a / 2, a * math.sqrt(3) / 6, 8.0]], dtype=float
    )
    charges_prim = [1.0, -1.0]

    atoms_prim = _make_slab_atoms(
        cell_prim, positions_prim, charges_prim, [True, True, False]
    )

    atoms_sc = make_supercell(atoms_prim, P)
    n_fu = abs(int(round(np.linalg.det(P))))

    E_prim = _batched_energy([atoms_prim])
    E_sc = _batched_energy([atoms_sc])

    # See test_tiling_square_lattice for tolerance rationale.
    np.testing.assert_allclose(
        float(E_sc[0]) / n_fu,
        float(E_prim[0]),
        rtol=5e-4,
        err_msg=f"Tiling '{tiling_name}' (n_fu={n_fu}) failed",
    )


# -- rotation invariance --


def _rotation_matrix(theta, phi):
    """Rotation: Rz(theta) @ Rx(phi)."""
    ct, st = math.cos(theta), math.sin(theta)
    cp, sp = math.cos(phi), math.sin(phi)
    Rz = np.array([[ct, -st, 0], [st, ct, 0], [0, 0, 1]], dtype=float)
    Rx = np.array([[1, 0, 0], [0, cp, -sp], [0, sp, cp]], dtype=float)
    return Rz @ Rx


@pytest.mark.parametrize(
    "cell",
    [
        np.diag([10.0, 10.0, 25.0]),
        np.array([[10.0, 3.0, 0.0], [0.0, 12.0, 0.0], [0.0, 0.0, 25.0]]),
    ],
    ids=["orthorhombic", "triclinic"],
)
def test_rotation_invariance(cell):
    """Energy is invariant under rotation of (cell, positions) for 2D PBC."""
    positions = np.array(
        [[2.0, 3.0, 5.0], [7.0, 4.0, 8.0], [5.0, 8.0, 15.0], [3.0, 6.0, 20.0]],
        dtype=float,
    )
    charges = [1.0, -1.0, 0.5, -0.5]

    R = _rotation_matrix(0.987654, 0.82321)
    cell_rot = cell @ R
    positions_rot = positions @ R

    atoms_orig = _make_slab_atoms(cell, positions, charges, [True, True, False])
    atoms_rot = _make_slab_atoms(cell_rot, positions_rot, charges, [True, True, False])

    E_orig = _batched_energy([atoms_orig])
    E_rot = _batched_energy([atoms_rot])

    np.testing.assert_allclose(float(E_rot[0]), float(E_orig[0]), rtol=1e-5)


# -- force consistency --


@pytest.mark.parametrize(
    "cell",
    [
        np.diag([10.0, 10.0, 25.0]),
        np.array([[10.0, 3.0, 0.0], [0.0, 10.0, 0.0], [1.0, 2.0, 25.0]]),
    ],
    ids=["orthorhombic", "triclinic"],
)
def test_force_consistency(cell):
    """Forces from autodiff match finite-difference energy derivatives."""
    from jaxpme.batched_mixed.calculators import Ewald

    positions_np = np.array(
        [[2.0, 3.0, 5.0], [7.0, 4.0, 8.0], [5.0, 8.0, 15.0]], dtype=float
    )
    charges_list = [1.0, -1.0, 0.5]
    cutoff = 5.0

    atoms = _make_slab_atoms(cell, positions_np, charges_list, [True, True, False])

    calculator = Ewald(prefactor=1.0)
    charges, sr_batch, nonp_batch, pbc_batch = calculator.prepare([atoms], cutoff=cutoff)
    energy, forces = calculator.energy_forces(charges, sr_batch, nonp_batch, pbc_batch)

    # finite differences: perturb positions in the pre-prepared batch
    # (not re-preparing, which would change the k-grid via slab thickness)
    N = len(positions_np)
    eps = 1e-5
    forces_fd = np.zeros((N, 3))
    for i in range(N):
        for d in range(3):
            pos_p = sr_batch.positions.copy()
            pos_p[i, d] += eps
            Ep = float(
                calculator.energy(
                    charges, sr_batch._replace(positions=pos_p), nonp_batch, pbc_batch
                )[0]
            )

            pos_m = sr_batch.positions.copy()
            pos_m[i, d] -= eps
            Em = float(
                calculator.energy(
                    charges, sr_batch._replace(positions=pos_m), nonp_batch, pbc_batch
                )[0]
            )

            forces_fd[i, d] = -(Ep - Em) / (2 * eps)

    # only compare real atoms (not padding)
    np.testing.assert_allclose(np.array(forces[:N]), forces_fd, atol=1e-6, rtol=1e-4)


# -- batched calculator specific tests --


def test_batched_jit():
    """Batched calculator JITs cleanly for 2D PBC with non-orthorhombic cell."""
    from jaxpme.batched_mixed.calculators import Ewald

    cell = np.array([[10.0, 3.0, 0.0], [0.0, 12.0, 0.0], [0.0, 0.0, 25.0]], dtype=float)
    positions = np.array([[2.0, 3.0, 5.0], [7.0, 4.0, 8.0]], dtype=float)

    atoms = _make_slab_atoms(cell, positions, [1.0, -1.0], [True, True, False])

    calculator = Ewald(prefactor=1.0)
    charges, sr_batch, nonp_batch, pbc_batch = calculator.prepare([atoms], cutoff=5.0)

    energy_eager = calculator.energy(charges, sr_batch, nonp_batch, pbc_batch)
    energy_jit = jax.jit(calculator.energy)(charges, sr_batch, nonp_batch, pbc_batch)
    np.testing.assert_allclose(energy_jit, energy_eager, atol=1e-14)


# -- k-grid reduction for large vacuum --


def test_cell_shrink_large_vacuum():
    """Non-periodic cell vector is shrunk to reduce k-grid for large vacuums."""
    from jaxpme.batched_mixed.batching import prepare

    # Slab with 3 Å thickness but 100 Å vacuum cell
    cell = np.diag([10.0, 10.0, 100.0])
    positions = np.array([[5.0, 5.0, 50.0], [5.0, 5.0, 53.0]], dtype=float)
    cutoff = 5.0

    atoms = _make_slab_atoms(cell, positions, [1.0, -1.0], [True, True, False])
    structure = prepare(atoms, cutoff=cutoff)

    # Cell should be shrunk: h_min = 3 + 1.5*10 = 18
    h_shrunk = abs(structure["cell"][2, 2])
    assert h_shrunk < 100.0, "cell should have been shrunk"
    assert h_shrunk >= 18.0 - 0.01, "cell should be at least h_min"

    # k-grid should be much smaller than with original cell
    lr_wavelength = cutoff / 8.0
    ns_original = int(np.ceil(100.0 / lr_wavelength))
    kz_max = int(np.abs(structure["lr"].k_grid[:, 2]).max())
    assert kz_max < ns_original // 2, "k-grid should be much smaller than original"


def test_cell_shrink_nonorthorhombic():
    """Cell shrinking works for non-orthorhombic slab cells."""
    from jaxpme.batched_mixed.batching import prepare

    cell = np.array([[10.0, 3.0, 0.0], [0.0, 10.0, 0.0], [0.0, 0.0, 80.0]])
    positions = np.array(
        [[5.0, 5.0, 40.0], [5.0, 5.0, 42.0], [5.0, 5.0, 45.0]], dtype=float
    )

    atoms = _make_slab_atoms(cell, positions, [1.0, -0.5, -0.5], [True, True, False])
    structure = prepare(atoms, cutoff=5.0)

    # Normal is z-hat for this cell, thickness = 5
    # L_max = ||[10,3,0]|| ≈ 10.44, h_min = 5 + 1.5*10.44 ≈ 20.66
    h_shrunk = np.linalg.norm(structure["cell"][2])
    assert h_shrunk < 80.0
    assert h_shrunk >= 20.0 - 0.01


def test_cell_shrink_preserves_direction():
    """Shrinking preserves the direction of the non-periodic cell vector."""
    from jaxpme.batched_mixed.batching import prepare

    # Tilted non-periodic vector
    cell = np.array([[10.0, 0.0, 0.0], [0.0, 10.0, 0.0], [2.0, 1.0, 100.0]])
    positions = np.array([[5.0, 5.0, 50.0], [5.0, 5.0, 53.0]], dtype=float)

    atoms = _make_slab_atoms(cell, positions, [1.0, -1.0], [True, True, False])
    structure = prepare(atoms, cutoff=5.0)

    original_dir = cell[2] / np.linalg.norm(cell[2])
    shrunk_dir = structure["cell"][2] / np.linalg.norm(structure["cell"][2])
    np.testing.assert_allclose(shrunk_dir, original_dir, atol=1e-12)


def test_cell_no_shrink_when_small():
    """Cell is not shrunk when vacuum is already small."""
    from jaxpme.batched_mixed.batching import prepare

    cell = np.diag([10.0, 10.0, 12.0])
    positions = np.array([[5.0, 5.0, 5.0], [5.0, 5.0, 8.0]], dtype=float)

    atoms = _make_slab_atoms(cell, positions, [1.0, -1.0], [True, True, False])
    prepare(atoms, cutoff=5.0)  # just verify it doesn't crash

    # h_min = 3 + 1.5*10 = 18, h_cell = 9 < h_min, so no shrink
    cell2 = np.diag([10.0, 10.0, 9.0])
    atoms2 = _make_slab_atoms(cell2, positions, [1.0, -1.0], [True, True, False])
    structure2 = prepare(atoms2, cutoff=5.0)
    assert structure2["cell"][2, 2] == 9.0  # unchanged


@pytest.mark.parametrize("vacuum", [20.0, 50.0, 100.0, 200.0])
def test_energy_independent_of_vacuum(vacuum):
    """Energy is independent of vacuum size for 2D PBC (cell shrinking handles this)."""
    cell = np.diag([10.0, 10.0, vacuum])
    positions = np.array([[5.0, 5.0, vacuum / 2], [5.0, 5.0, vacuum / 2 + 3.0]])

    atoms = _make_slab_atoms(cell, positions, [1.0, -1.0], [True, True, False])
    energy = float(_batched_energy([atoms])[0])

    # Reference: smallest reasonable vacuum
    cell_ref = np.diag([10.0, 10.0, 20.0])
    pos_ref = np.array([[5.0, 5.0, 10.0], [5.0, 5.0, 13.0]])
    atoms_ref = _make_slab_atoms(cell_ref, pos_ref, [1.0, -1.0], [True, True, False])
    energy_ref = float(_batched_energy([atoms_ref])[0])

    # All vacuum sizes shrink to the same h_min, so results are bitwise identical.
    np.testing.assert_allclose(energy, energy_ref, rtol=1e-10)


def test_num_k_with_large_vacuum():
    """num_k path produces sane cutoff/lr_wavelength for large-vacuum slabs."""
    from jaxpme.batched_mixed.batching import prepare

    cell = np.diag([10.0, 10.0, 100.0])
    positions = np.array([[5.0, 5.0, 50.0], [5.0, 5.0, 53.0]], dtype=float)
    atoms = _make_slab_atoms(cell, positions, [1.0, -1.0], [True, True, False])

    # With original 100 Å cell, num_k=200 gives lr_wavelength ≈ 3.7, cutoff ≈ 29
    # With shrunk cell (~9 Å), lr_wavelength and cutoff are much smaller
    structure = prepare(atoms, num_k=200)

    # Cell should be shrunk
    assert abs(structure["cell"][2, 2]) < 100.0

    # The derived smearing should be reasonable (not inflated by vacuum)
    assert structure["smearing"] < 5.0  # would be ~7.4 without shrinking


# -- MAD-1.5 2D reference structures --

MAD_2D_FRAMES = read(str(REFERENCE_STRUCTURES_DIR / "mad15_2d_subset.xyz"), index=":")


@pytest.mark.parametrize("frame_idx", range(len(MAD_2D_FRAMES)))
def test_mad15_shrink_vs_noshrink(frame_idx):
    """Energy with shrunk vacuum matches unshrunk 100 A cell on real structures."""
    from unittest.mock import patch

    atoms = MAD_2D_FRAMES[frame_idx]
    cutoff = 5.0

    E_shrunk = float(_batched_energy([atoms], cutoff=cutoff)[0])

    with patch("jaxpme.batched_mixed.batching.shrink_2d_cell", lambda c, p, pos: c):
        E_noshrink = float(_batched_energy([atoms], cutoff=cutoff)[0])

    np.testing.assert_allclose(E_shrunk, E_noshrink, rtol=5e-4)


@pytest.mark.parametrize("frame_idx", range(len(MAD_2D_FRAMES)))
def test_mad15_finite_forces(frame_idx):
    """Forces are finite and sum to ~zero on real 2D structures."""
    atoms = MAD_2D_FRAMES[frame_idx]
    _, forces = _batched_energy_forces([atoms], cutoff=5.0)
    N = len(atoms)
    forces = np.array(forces[:N])

    assert np.all(np.isfinite(forces))
    np.testing.assert_allclose(forces.sum(axis=0), 0.0, atol=1e-8)


@pytest.mark.parametrize("frame_idx", range(len(MAD_2D_FRAMES)))
def test_mad15_nonneutral_finite(frame_idx):
    """Non-neutral 2D slabs produce finite energies and forces."""
    atoms = MAD_2D_FRAMES[frame_idx].copy()
    charges = atoms.get_initial_charges().copy()
    charges[0] += 0.5  # break neutrality
    atoms.set_initial_charges(charges)

    energy = float(_batched_energy([atoms], cutoff=5.0)[0])
    assert np.isfinite(energy)

    _, forces = _batched_energy_forces([atoms], cutoff=5.0)
    forces = np.array(forces[: len(atoms)])
    assert np.all(np.isfinite(forces))
    np.testing.assert_allclose(forces.sum(axis=0), 0.0, atol=1e-8)
