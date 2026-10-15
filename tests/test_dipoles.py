import numpy as np
import jax
import jax.numpy as jnp

import pytest
from ase import Atoms
from ase.io import read
from conftest import REFERENCE_STRUCTURES_DIR

from jaxpme import Ewald, EwaldDipole
from jaxpme.prefactors import prefactors

jax.config.update("jax_enable_x64", True)


def random_system(n=4, seed=0, triclinic=True):
    rng = np.random.default_rng(seed)
    cell = np.diag([6.0, 7.0, 8.0])
    if triclinic:
        cell = cell + 0.3 * rng.normal(size=(3, 3))
    positions = rng.uniform(0, 1, (n, 3)) @ cell
    charges = rng.normal(size=n)
    charges -= charges.mean()
    dipoles = rng.normal(size=(n, 3))
    return cell, positions, charges, dipoles


def direct_sum(positions, charges, dipoles):
    # true (unhalved) potential and field, non-periodic
    n = len(positions)
    phi = np.zeros(n)
    field = np.zeros((n, 3))
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            x = positions[i] - positions[j]
            r = np.linalg.norm(x)
            u = x / r
            phi[i] += charges[j] / r + dipoles[j] @ x / r**3
            field[i] += (
                charges[j] * x / r**3 + (3 * u * (u @ dipoles[j]) - dipoles[j]) / r**3
            )
    return phi, field


def direct_sum_2d(cell, positions, charges, dipoles, N):
    # true (unhalved) potential and field of a 2D-periodic slab: direct sum over
    # the (2N+1)^2 in-plane image cells, truncation error ~ 1/N
    n = len(positions)
    nx, ny = np.meshgrid(np.arange(-N, N + 1), np.arange(-N, N + 1), indexing="ij")
    shifts = nx.ravel()[:, None] * cell[0] + ny.ravel()[:, None] * cell[1]
    phi = np.zeros(n)
    field = np.zeros((n, 3))
    for i in range(n):
        for j in range(n):
            x = positions[i] - positions[j] + shifts
            if i == j:
                x = x[np.any(shifts != 0, axis=-1)]
            r = np.linalg.norm(x, axis=-1)
            mx = x @ dipoles[j]
            phi[i] += np.sum(charges[j] / r + mx / r**3)
            field[i] += np.sum(
                (charges[j] / r**3 + 3 * mx / r**5)[:, None] * x
                - dipoles[j] / r[:, None] ** 3,
                axis=0,
            )
    return phi, field


# -- references: ESPResSo (via torch-pme) --
# dipoles_test_frames.xyz is torch-pme's examples/dipoles_test_frames.xyz (BSD-3)

frames = read(REFERENCE_STRUCTURES_DIR / "dipoles_test_frames.xyz", ":3")
cutoffs = [3.9986718930, 4.0000000000, 4.7363281250]
alphas = [0.8819831493, 0.8956299559, 0.7215211182]


@pytest.mark.parametrize("frame, cutoff, alpha", zip(frames, cutoffs, alphas))
def test_espresso_crystal(frame, cutoff, alpha):
    calculator = EwaldDipole(prefactor=prefactors.eV_A)
    inputs = calculator.prepare(
        frame,
        np.zeros(len(frame)),
        None,
        cutoff,
        lr_wavelength=0.3,
        smearing=1 / (np.sqrt(2) * alpha),
    )
    energy, forces = calculator.energy_forces(*inputs)

    np.testing.assert_allclose(energy, frame.get_potential_energy(), rtol=1e-6)
    np.testing.assert_allclose(forces, frame.get_forces(), atol=1e-6)


def parallel_dipoles(pbc):
    positions = np.array([[0.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 4.0, 0.0]])
    dipoles = np.array([[1.0, 1.0, 0.0]] * 3)
    atoms = Atoms("H3", positions=positions, cell=10 * np.eye(3), pbc=pbc)
    return atoms, dipoles


def test_parallel_dipoles_direct():
    atoms, dipoles = parallel_dipoles(pbc=False)
    calculator = EwaldDipole()
    inputs = calculator.prepare(atoms, np.zeros(3), dipoles, 6.0)
    np.testing.assert_allclose(calculator.energy(*inputs), -0.265625)


def test_parallel_dipoles_espresso():
    atoms, dipoles = parallel_dipoles(pbc=True)
    calculator = EwaldDipole()
    inputs = calculator.prepare(
        atoms, np.zeros(3), dipoles, 6.0, lr_wavelength=0.3, smearing=1 / np.sqrt(2)
    )
    # ESPResSo DipolarP3M, alpha = 1, mesh 64
    np.testing.assert_allclose(calculator.energy(*inputs), -0.30848574939287954, rtol=1e-5)


@pytest.mark.parametrize("smearing", [0.8, 1.2, 1.6])
def test_single_dipole_cubic(smearing):
    # cubic lattice sum of parallel dipoles vanishes (spherical summation), so
    # with tin-foil boundary only -2π|μ|²/(3V) remains; pins the self term
    L = 5.0
    mu = np.array([[0.3, -0.7, 0.5]])
    atoms = Atoms("H", positions=[[1.0, 2.0, 3.0]], cell=L * np.eye(3), pbc=True)
    calculator = EwaldDipole()
    inputs = calculator.prepare(
        atoms, np.zeros(1), mu, 8 * smearing, lr_wavelength=smearing / 3, smearing=smearing
    )
    expected = -2 * np.pi * np.sum(mu**2) / (3 * L**3)
    np.testing.assert_allclose(calculator.energy(*inputs), expected, rtol=1e-8)


# -- internal consistency --


def test_nonperiodic_direct_sum():
    _, positions, charges, dipoles = random_system(n=6, seed=1)
    atoms = Atoms("H6", positions=positions, pbc=False)
    calculator = EwaldDipole()
    phi, field = calculator.potentials(*calculator.prepare(atoms, charges, dipoles, 6.0))

    phi_ref, field_ref = direct_sum(positions, charges, dipoles)
    np.testing.assert_allclose(2 * phi, phi_ref, atol=1e-12)
    np.testing.assert_allclose(2 * field, field_ref, atol=1e-12)


@pytest.mark.parametrize("full_neighbor_list", [False, True])
@pytest.mark.parametrize("exclusion_radius", [None, 3.0])
def test_zero_dipoles_is_ewald(full_neighbor_list, exclusion_radius):
    cell, positions, charges, _ = random_system()
    atoms = Atoms("H4", positions=positions, cell=cell, pbc=True)
    kwargs = dict(full_neighbor_list=full_neighbor_list, exclusion_radius=exclusion_radius)

    dipole = EwaldDipole(**kwargs)
    inputs = dipole.prepare(atoms, charges, np.zeros((4, 3)), 6.0, smearing=1.2)
    phi, _ = dipole.potentials(*inputs)
    ewald = Ewald(**kwargs)
    ref_inputs = ewald.prepare(atoms, charges, 6.0, smearing=1.2)
    ref = ewald.potentials(*ref_inputs)

    np.testing.assert_allclose(phi, ref, atol=1e-12)
    for a, b in zip(
        dipole.energy_forces_stress(*inputs), ewald.energy_forces_stress(*ref_inputs)
    ):
        np.testing.assert_allclose(a, b, atol=1e-12)


def test_charge_pair_limit():
    # a dipole is the limit of +q/-q at ±d/2 with q d = μ; checks the relative
    # sign of q and i k·μ in the structure factor
    cell, positions, charges, dipoles = random_system()
    atoms = Atoms("H4", positions=positions, cell=cell, pbc=True)
    calculator = EwaldDipole()
    phi, _ = calculator.potentials(
        *calculator.prepare(atoms, charges, dipoles, 6.0, lr_wavelength=0.5, smearing=1.2)
    )

    d = 1e-4
    pair_positions = np.concatenate(
        [positions, positions + dipoles * d / 2, positions - dipoles * d / 2]
    )
    pair_charges = np.concatenate([charges, np.full(4, 1 / d), np.full(4, -1 / d)])
    pairs = Atoms("H12", positions=pair_positions, cell=cell, pbc=True)
    ewald = Ewald()
    ref = ewald.potentials(
        *ewald.prepare(pairs, pair_charges, 6.0, lr_wavelength=0.5, smearing=1.2)
    )

    np.testing.assert_allclose(phi, ref[:4], atol=1e-7)


def test_smearing_independence():
    cell, positions, charges, dipoles = random_system()
    atoms = Atoms("H4", positions=positions, cell=cell, pbc=True)
    calculator = EwaldDipole()

    energies = [
        calculator.energy(
            *calculator.prepare(
                atoms, charges, dipoles, 12.0, lr_wavelength=0.25, smearing=smearing
            )
        )
        for smearing in [0.9, 1.2, 1.5]
    ]
    np.testing.assert_allclose(energies, energies[0], rtol=1e-10)


@pytest.mark.parametrize("pbc", [[True, True, True], [True, True, False]])
def test_halved_convention(pbc):
    cell, positions, charges, dipoles = random_system()
    charges += 0.1  # non-neutral, exercises background and slab terms
    cell[2] *= 4
    atoms = Atoms("H4", positions=positions, cell=cell, pbc=True)
    calculator = EwaldDipole()
    *inputs, _ = calculator.prepare(atoms, charges, dipoles, 6.0, smearing=1.2)
    inputs = (*inputs, np.array(pbc))

    phi, field = calculator.potentials(*inputs)
    dq, dmu = jax.grad(calculator.energy, argnums=(0, 1))(*inputs)

    np.testing.assert_allclose(dq, 2 * phi, atol=1e-12)
    np.testing.assert_allclose(dmu, -2 * field, atol=1e-12)


def test_forces_finite_differences():
    cell, positions, charges, dipoles = random_system()
    atoms = Atoms("H4", positions=positions, cell=cell, pbc=True)
    calculator = EwaldDipole()
    charges, dipoles, cell, positions, *rest = calculator.prepare(
        atoms, charges, dipoles, 6.0, smearing=1.2
    )
    _, forces = calculator.energy_forces(charges, dipoles, cell, positions, *rest)

    h = 1e-5
    fd = np.zeros_like(positions)
    for a in range(len(positions)):
        for b in range(3):
            dp = np.zeros_like(positions)
            dp[a, b] = h
            ep = calculator.energy(charges, dipoles, cell, positions + dp, *rest)
            em = calculator.energy(charges, dipoles, cell, positions - dp, *rest)
            fd[a, b] = -(ep - em) / (2 * h)

    np.testing.assert_allclose(forces, fd, atol=1e-7)


def test_stress_is_strain_derivative():
    # stress[a, b] = ∂E/∂ε_ab for r_ib -> r_ib + ε_ab r_ia (same for the cell rows);
    # dipoles are held fixed under strain, so it is not symmetric
    cell, positions, charges, dipoles = random_system()
    atoms = Atoms("H4", positions=positions, cell=cell, pbc=True)
    calculator = EwaldDipole()
    charges, dipoles, cell, positions, *rest = calculator.prepare(
        atoms, charges, dipoles, 6.0, smearing=1.2
    )
    _, _, stress = calculator.energy_forces_stress(charges, dipoles, cell, positions, *rest)

    def strained_energy(eps):
        return calculator.energy(
            charges, dipoles, cell + cell @ eps, positions + positions @ eps, *rest
        )

    ref = jax.grad(strained_energy)(jnp.zeros((3, 3)))
    assert not np.allclose(stress, stress.T, atol=1e-3)
    np.testing.assert_allclose(stress, ref, atol=1e-12)


def test_higher_order_derivatives():
    # induction μ <- αE inside a model, then forces and the BEC Jacobian
    # need third derivatives through the solver
    cell, positions, charges, dipoles = random_system()
    atoms = Atoms("H4", positions=positions, cell=cell, pbc=True)
    calculator = EwaldDipole()
    charges, dipoles, cell, positions, *rest = calculator.prepare(
        atoms, charges, dipoles, 6.0, smearing=1.2
    )

    def energy(positions, external_field):
        _, field = calculator.potentials(charges, dipoles, cell, positions, *rest)
        mu = dipoles + 0.1 * (2 * field + external_field)
        return calculator.energy(charges, mu, cell, positions, *rest)

    bec = jax.jit(jax.jacfwd(jax.grad(energy, argnums=0), argnums=1))(
        positions, jnp.zeros(3)
    )
    assert bec.shape == (4, 3, 3)
    assert np.all(np.isfinite(bec))


# -- 2D slab --


@pytest.mark.parametrize("triclinic", [False, True])
def test_slab_charge_pair_limit(triclinic):
    cell, positions, charges, dipoles = random_system(triclinic=triclinic)
    cell[2] = [0.0, 0.0, 30.0] if not triclinic else cell[2] * 30.0 / cell[2, 2]
    pbc = np.array([True, True, False])

    def potentials(calculator, atoms, *args):
        # 3D neighbor list is exact for the slab since the vacuum exceeds the cutoff
        atoms.pbc = True
        inputs = calculator.prepare(atoms, *args, 6.0, lr_wavelength=0.5, smearing=1.2)
        return calculator.potentials(*inputs[:-1], pbc=pbc)

    atoms = Atoms("H4", positions=positions, cell=cell)
    phi, field = potentials(EwaldDipole(), atoms, charges, dipoles)

    d = 1e-4
    pair_positions = np.concatenate(
        [positions, positions + dipoles * d / 2, positions - dipoles * d / 2]
    )
    pair_charges = np.concatenate([charges, np.full(4, 1 / d), np.full(4, -1 / d)])
    pairs = Atoms("H12", positions=pair_positions, cell=cell)
    ref = potentials(Ewald(), pairs, pair_charges)

    np.testing.assert_allclose(phi, ref[:4], atol=1e-7)


def test_slab_vacuum_convergence():
    cell, positions, charges, dipoles = random_system(triclinic=False)
    pbc = np.array([True, True, False])
    calculator = EwaldDipole()

    results = []
    for height in [30.0, 45.0]:
        cell[2] = [0.0, 0.0, height]
        atoms = Atoms("H4", positions=positions, cell=cell, pbc=True)
        inputs = calculator.prepare(
            atoms, charges, dipoles, 9.0, lr_wavelength=0.4, smearing=1.5
        )
        results.append(calculator.potentials(*inputs[:-1], pbc=pbc))

    np.testing.assert_allclose(results[0][0], results[1][0], atol=1e-4)
    np.testing.assert_allclose(results[0][1], results[1][1], atol=1e-4)


@pytest.mark.parametrize("with_charges", [False, True])
def test_slab_direct_sum(with_charges):
    # the slab correction with dipoles is only otherwise pinned by reciprocity
    # (test_halved_convention) and vacuum convergence; this checks its magnitude
    cell, positions, charges, dipoles = random_system(seed=3)
    cell[2] = [0.0, 0.0, 40.0]
    positions[:, 2] = 10.0 + 3.0 * (positions[:, 2] / cell[2, 2])
    if not with_charges:
        charges = np.zeros(4)
    pbc = np.array([True, True, False])

    atoms = Atoms("H4", positions=positions, cell=cell, pbc=True)
    calculator = EwaldDipole()
    inputs = calculator.prepare(
        atoms, charges, dipoles, 9.0, lr_wavelength=0.4, smearing=1.5
    )
    phi, field = calculator.potentials(*inputs[:-1], pbc=pbc)

    # truncation error ~ 1/N: Richardson-extrapolate two windows
    coarse = direct_sum_2d(cell, positions, charges, dipoles, 100)
    fine = direct_sum_2d(cell, positions, charges, dipoles, 200)
    phi_ref, field_ref = (2 * f - c for c, f in zip(coarse, fine))

    np.testing.assert_allclose(2 * phi, phi_ref, rtol=1e-4, atol=1e-5)
    np.testing.assert_allclose(2 * field, field_ref, rtol=1e-4, atol=1e-5)
