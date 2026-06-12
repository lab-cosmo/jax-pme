import numpy as np
import jax
import jax.numpy as jnp

import pytest
from ase import Atoms
from scipy.special import exp1 as scipy_exp1
from scipy.special import gamma as scipy_gamma
from scipy.special import gammaincc as scipy_gammaincc

from jaxpme import P3M, PME, Ewald
from jaxpme.potentials import inverse_power_law, potential

jax.config.update("jax_enable_x64", True)

EXPONENTS = [1, 2, 3, 4, 5, 6]
SMEARINGS = [0.2, 1.0, 1.56]


def upper_gamma(a, x):
    # Gamma(a, x) for a <= 0 via Gamma(a, x) = (Gamma(a+1, x) - x**a exp(-x)) / a
    if a > 0:
        return scipy_gammaincc(a, x) * scipy_gamma(a)
    if a == 0:
        return scipy_exp1(x)
    return (upper_gamma(a + 1, x) - x**a * np.exp(-x)) / a


def cscl():
    atoms = Atoms(
        "CsCl",
        positions=[[0.0, 0.0, 0.0], [0.5, 0.5, 0.5]],
        cell=np.eye(3),
        pbc=True,
    )
    charges = np.array([1.0, -1.0])
    return atoms, charges


def charged():
    # net charge: exercises the k=0 term (background for p < 3, lr_k0 for p > 3)
    atoms = Atoms("Na", positions=[[0.0, 0.0, 0.0]], cell=np.eye(3), pbc=True)
    charges = np.array([1.0])
    return atoms, charges


def direct_lattice_sum(atoms, charges, exponent, n_images):
    # brute-force E = 1/2 sum_{ij, n}' q_i q_j / |r_ij + nL|^p over a cube of images
    positions = atoms.get_positions()
    cell = atoms.get_cell().array

    grid = np.arange(-n_images, n_images + 1)
    shifts = np.stack(np.meshgrid(grid, grid, grid), axis=-1).reshape(-1, 3) @ cell

    rij = positions[None, :, :] - positions[:, None, :]
    d = np.linalg.norm(rij[:, :, None, :] + shifts[None, None, :, :], axis=-1)
    d = np.where(d > 0, d, np.inf)
    qq = (charges[:, None] * charges[None, :])[:, :, None]

    return 0.5 * np.sum(qq * d ** (-float(exponent)))


@pytest.mark.parametrize("exponent", EXPONENTS)
@pytest.mark.parametrize("smearing", SMEARINGS)
def test_lr_k2_reference(exponent, smearing):
    # closed forms vs an independent scipy evaluation of Gamma(peff, x)/x**peff
    pot = inverse_power_law(exponent)

    k2 = np.geomspace(1e-3, 3e2, 100)
    x = 0.5 * smearing**2 * k2

    peff = (3 - exponent) / 2
    prefac = np.pi**1.5 / scipy_gamma(exponent / 2) * (2 * smearing**2) ** peff
    ref = prefac * upper_gamma(peff, x) / x**peff

    # atol covers cancellation noise in the (vanishing) large-x tail
    np.testing.assert_allclose(
        np.asarray(pot.lr_k2(smearing, jnp.asarray(k2))), ref, rtol=1e-10, atol=1e-40
    )


@pytest.mark.parametrize("exponent", EXPONENTS)
def test_lr_k0(exponent):
    # the k=0 term must be finite: zero (background) for p <= 3,
    # continuous with k -> 0 for p > 3
    pot = potential(exponent=exponent)
    values = pot.lr(1.0, jnp.array([0.0, 1e-10]))

    assert np.all(np.isfinite(values))
    if exponent <= 3:
        assert values[0] == 0.0
    else:
        np.testing.assert_allclose(values[0], values[1], rtol=1e-4)


# rtol reflects the truncation error of the cube-summed reference, which
# vanishes only as ~n_images**(3 - exponent)
@pytest.mark.parametrize("exponent,rtol", [(4, 3e-4), (5, 2e-5), (6, 1e-6)])
def test_ewald_vs_direct_sum(exponent, rtol):
    # for p > 3 the lattice sum converges absolutely -> brute force comparison
    atoms, charges = cscl()

    calc = Ewald(exponent=exponent)
    energy = calc.energy(*calc.prepare(atoms, charges, 6.0))

    reference = direct_lattice_sum(atoms, charges, exponent, n_images=16)

    np.testing.assert_allclose(float(energy), reference, rtol=rtol)


# for p > 3 the lattice sum of a charged cell converges absolutely, but only
# if the finite k=0 term (lr_k0) is included -> exercises it end-to-end.
# p = 4 is omitted: the cube-summed reference converges too slowly (~1/N)
@pytest.mark.parametrize("exponent,rtol", [(5, 3e-3), (6, 1e-4)])
def test_ewald_vs_direct_sum_charged(exponent, rtol):
    atoms, charges = charged()

    calc = Ewald(exponent=exponent)
    energy = calc.energy(*calc.prepare(atoms, charges, 6.0))

    reference = direct_lattice_sum(atoms, charges, exponent, n_images=24)

    np.testing.assert_allclose(float(energy), reference, rtol=rtol)


@pytest.mark.parametrize("system", [cscl, charged])
@pytest.mark.parametrize("exponent", EXPONENTS)
def test_ewald_parameter_independence(exponent, system):
    # the result must not depend on the smearing / cutoff / k-grid choice.
    # for a charged cell this requires the correct k=0 handling: background
    # correction for p < 3, lr_k0 for p > 3. p = 3 is excluded there, since
    # the energy of a charged cell diverges
    if system is charged and exponent == 3:
        pytest.skip("charged cell energy diverges for p = 3")

    atoms, charges = system()
    calc = Ewald(exponent=exponent)

    energy_a = calc.energy(*calc.prepare(atoms, charges, 5.0, 0.5, 0.9))
    energy_b = calc.energy(*calc.prepare(atoms, charges, 8.0, 0.4, 1.4))

    np.testing.assert_allclose(float(energy_a), float(energy_b), rtol=1e-6)


@pytest.mark.parametrize("calculator", [PME, P3M])
@pytest.mark.parametrize("exponent", [4, 5, 6])
def test_mesh_calculators_vs_ewald(calculator, exponent):
    # the mesh k-grid contains k=0 once, so charged cells pick up lr_k0
    # through the FFT path; rtol is set by mesh discretization error
    for system in (cscl, charged):
        atoms, charges = system()

        calc = calculator(exponent=exponent)
        energy = calc.energy(*calc.prepare(atoms, charges, 6.0, 0.125, 1.0))

        serial = Ewald(exponent=exponent)
        reference = serial.energy(*serial.prepare(atoms, charges, 6.0))

        np.testing.assert_allclose(float(energy), float(reference), rtol=2e-4)


@pytest.mark.parametrize("halfspace", [False, True])
@pytest.mark.parametrize("exponent", EXPONENTS)
def test_batched_mixed_vs_serial(exponent, halfspace):
    # mixed-size batch -> zero-padded k-grids; the charged structure
    # exercises the explicit k=0 term in the solver for both halfspace modes
    from jaxpme.batched_mixed.calculators import Ewald as BatchedEwald

    systems = [cscl, charged] if exponent != 3 else [cscl]
    structures = []
    for system in systems:
        atoms, charges = system()
        atoms.set_initial_charges(charges)
        structures.append(atoms)

    calc = BatchedEwald(custom_potential=inverse_power_law(exponent), halfspace=halfspace)
    energies = calc.energy(*calc.prepare(structures, 6.0))

    serial = Ewald(exponent=exponent)
    for k, atoms in enumerate(structures):
        reference = serial.energy(*serial.prepare(atoms, None, 6.0))
        np.testing.assert_allclose(float(energies[k]), float(reference), rtol=1e-12)


@pytest.mark.parametrize("exponent", EXPONENTS)
def test_batched_flat_vs_serial(exponent):
    from jaxpme.batched_flat.calculators import Ewald as FlatEwald

    systems = [cscl, charged] if exponent != 3 else [cscl]
    structures, all_charges = zip(*[system() for system in systems])

    calc = FlatEwald(exponent=exponent)
    batch = calc.prepare(list(structures), list(all_charges), 6.0)
    energies = calc.energy(*batch)

    serial = Ewald(exponent=exponent)
    for k, (atoms, charges) in enumerate(zip(structures, all_charges)):
        reference = serial.energy(*serial.prepare(atoms, charges, 6.0))
        np.testing.assert_allclose(float(energies[k]), float(reference), rtol=1e-12)


@pytest.mark.parametrize("exponent", EXPONENTS)
def test_finite_energy_forces_stress(exponent):
    # regression for issue #20: p >= 3 produced NaNs
    atoms, charges = cscl()
    calc = Ewald(exponent=exponent)

    inputs = calc.prepare(atoms, charges, 6.0)
    energy, forces, stress = calc.energy_forces_stress(*inputs)

    assert np.isfinite(energy)
    assert np.all(np.isfinite(forces))
    assert np.all(np.isfinite(stress))


def test_unsupported_exponent():
    with pytest.raises(ValueError, match="Unsupported exponent"):
        inverse_power_law(7)

    with pytest.raises(ValueError, match="Unsupported exponent"):
        inverse_power_law(2.5)
