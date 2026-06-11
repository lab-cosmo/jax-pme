import numpy as np
import jax
import jax.numpy as jnp

import pytest
from ase import Atoms
from scipy.special import exp1 as scipy_exp1
from scipy.special import gamma as scipy_gamma
from scipy.special import gammaincc as scipy_gammaincc

from jaxpme import Ewald
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


@pytest.mark.parametrize("exponent", EXPONENTS)
def test_ewald_parameter_independence(exponent):
    # the result must not depend on the smearing / cutoff / k-grid choice
    atoms, charges = cscl()
    calc = Ewald(exponent=exponent)

    energy_a = calc.energy(*calc.prepare(atoms, charges, 5.0, 0.5, 0.9))
    energy_b = calc.energy(*calc.prepare(atoms, charges, 8.0, 0.4, 1.4))

    np.testing.assert_allclose(float(energy_a), float(energy_b), rtol=1e-6)


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
