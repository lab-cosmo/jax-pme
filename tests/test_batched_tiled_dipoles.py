import numpy as np
import jax
import jax.numpy as jnp

import pytest
from ase import Atoms
from ase.io import read
from conftest import REFERENCE_STRUCTURES_DIR

from jaxpme import EwaldDipole as SerialEwaldDipole
from jaxpme.batched_tiled import EwaldDipole

jax.config.update("jax_enable_x64", True)


def _num_k_for_lr(atoms, lr_wavelength):
    L = np.linalg.norm(atoms.get_cell().array, axis=-1)
    return int(np.prod(L) / (2 * lr_wavelength**3))


def structures():
    rng = np.random.default_rng(0)

    coulomb = read(REFERENCE_STRUCTURES_DIR / "coulomb_test_frames.xyz", index=":2")
    dipolar = read(REFERENCE_STRUCTURES_DIR / "dipoles_test_frames.xyz", index="0")
    dipolar.set_initial_charges(np.zeros(len(dipolar)))

    cluster = Atoms("H5", positions=rng.uniform(0, 5, (5, 3)), pbc=False)
    cluster.set_initial_charges(rng.normal(size=5))

    slab = Atoms(
        "H6",
        positions=rng.uniform(0, 1, (6, 3)) * [6.0, 6.5, 3.0] + [0, 0, 10.0],
        cell=[[6.0, 0.0, 0.0], [0.5, 6.5, 0.0], [0.0, 0.0, 40.0]],
        pbc=[True, True, False],
    )
    charges = rng.normal(size=6)
    slab.set_initial_charges(charges - charges.mean())

    atomss = [*coulomb, dipolar, cluster, slab]
    for atoms in atomss:
        if atoms is not dipolar:
            atoms.set_array("dipoles", rng.normal(size=(len(atoms), 3)))
    return atomss


def serial_reference(atoms, cutoff, smearing):
    calculator = SerialEwaldDipole()
    pbc = atoms.get_pbc()
    if pbc.sum() == 2:
        # serial has no 2D neighbor list; the vacuum exceeds the cutoff, so a
        # 3D list is exact, and the slab correction is switched on via pbc
        atoms = atoms.copy()
        atoms.pbc = True
    inputs = calculator.prepare(
        atoms, None, None, cutoff, lr_wavelength=cutoff / 8, smearing=smearing
    )
    inputs = (*inputs[:-1], pbc)
    phi, field = calculator.potentials(*inputs)
    energy, forces, stress = calculator.energy_forces_stress(*inputs)
    return phi, field, energy, forces, stress


@pytest.mark.parametrize("BM, BK", [(32, 128), (8, 32)])
def test_vs_serial(BM, BK):
    cutoff = 6.0
    smearing = 1.5
    atomss = structures()
    num_k = max(_num_k_for_lr(a, cutoff / 8) for a in atomss if a.get_pbc().all())

    calculator = EwaldDipole()
    inputs = calculator.prepare(
        atomss, num_k, cutoff=cutoff, smearing=smearing, BM=BM, BK=BK
    )
    phi, field = calculator.potentials(*inputs)
    energy, forces, stress = calculator.energy_forces_stress(*inputs)
    atom_to_structure = inputs[2].atom_to_structure

    for b, atoms in enumerate(atomss):
        ref_phi, ref_field, ref_energy, ref_forces, ref_stress = serial_reference(
            atoms, cutoff, smearing
        )
        sel = atom_to_structure == b
        # the slab differs by the vacuum shrink (residual ~ exp(-3π))
        is_slab = atoms.get_pbc().sum() == 2
        tol = dict(rtol=3e-5, atol=3e-5) if is_slab else dict(atol=1e-8)
        np.testing.assert_allclose(phi[sel], ref_phi, **tol)
        np.testing.assert_allclose(field[sel], ref_field, **tol)
        np.testing.assert_allclose(energy[b], ref_energy, **tol)
        np.testing.assert_allclose(forces[sel], ref_forces, **tol)
        if atoms.get_pbc().all():
            np.testing.assert_allclose(stress[b], ref_stress, **tol)


def test_halved_convention():
    atomss = structures()
    calculator = EwaldDipole()
    charges, flat_dipoles, *batch = calculator.prepare(atomss, 2000)

    phi, field = calculator.potentials(charges, flat_dipoles, *batch)
    dq, dmu = jax.grad(
        lambda q, mu: calculator.energy(q, mu, *batch).sum(), argnums=(0, 1)
    )(charges, flat_dipoles)

    np.testing.assert_allclose(dq, 2 * phi, atol=1e-12)
    np.testing.assert_allclose(dmu, -2 * field, atol=1e-12)


def test_higher_order_derivatives():
    atomss = structures()
    calculator = EwaldDipole()
    charges, flat_dipoles, batch, *rest = calculator.prepare(atomss, 1000)

    def energy(positions, external_field):
        b = batch._replace(positions=positions)
        _, field = calculator.potentials(charges, flat_dipoles, b, *rest)
        mu = flat_dipoles + 0.1 * (2 * field + external_field)
        return calculator.energy(charges, mu, b, *rest).sum()

    bec = jax.jit(jax.jacfwd(jax.grad(energy, argnums=0), argnums=1))(
        batch.positions, np.zeros(3)
    )
    assert np.all(np.isfinite(bec))


def test_stress_is_strain_derivative():
    # stress[a, b] = ∂E/∂ε_ab per structure for r_ib -> r_ib + ε_ab r_ia (same for the
    # cell rows), dipoles held fixed. For 2D the non-periodic row has no cell term
    # by design (see jaxpme.utils.compose_cell), consistent with straining `batch.cell`
    atomss = structures()
    calculator = EwaldDipole()
    charges, flat_dipoles, batch, *rest = calculator.prepare(atomss, 1000)
    _, _, stress = calculator.energy_forces_stress(charges, flat_dipoles, batch, *rest)

    def strained_energies(eps):
        b = batch._replace(
            positions=batch.positions + batch.positions @ eps,
            cell=batch.cell + batch.cell @ eps,
        )
        return calculator.energy(charges, flat_dipoles, b, *rest)

    ref = jax.jacrev(strained_energies)(jnp.zeros((3, 3)))
    np.testing.assert_allclose(stress, ref, atol=1e-12)
