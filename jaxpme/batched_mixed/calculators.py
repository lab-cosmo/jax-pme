import jax
import jax.numpy as jnp

from jaxpme.calculators import Calculator
from jaxpme.kspace import get_reciprocal
from jaxpme.potentials import potential
from jaxpme.utils import safe_norm

from .kspace import generate_ewald_kvectors
from .utils import get_distances, get_volume


def Ewald(
    exponent=1,
    prefactor=1.0,
    custom_potential=None,
):
    from jaxpme.solvers import ewald

    pot = potential(
        exponent=exponent,
        exclusion_radius=None,
        custom_potential=custom_potential,
    )
    solver = ewald(pot, full_neighbor_list=True)

    def potentials_fn(
        charges,
        batch,
        batch_nopbc,
        batch_pbc,
    ):
        N_all = charges.shape[0]

        pbc_mask = batch.pbc_mask[batch.atom_to_structure]

        charges *= batch.atom_mask

        if batch.distances is None:
            r = get_distances(
                batch.cell[batch.pair_to_structure],
                batch.positions[batch.centers],
                batch.positions[batch.others],
                batch.cell_shifts,
            )
            r *= batch.pair_mask
        else:
            r = batch.distances

        extra_r = (
            safe_norm(
                batch.positions[batch_nopbc.others] - batch.positions[batch_nopbc.centers],
                axis=-1,
            )
            * batch_nopbc.pair_mask
        )

        # real-space, no-pbc
        real_space = jax.ops.segment_sum(
            charges[batch_nopbc.others] * pot.real(extra_r),
            batch_nopbc.centers,
            num_segments=N_all,
        )

        real_space += jax.ops.segment_sum(
            charges[batch_nopbc.centers] * pot.real(extra_r),
            batch_nopbc.others,
            num_segments=N_all,
        )

        real_space = (real_space * batch.atom_mask * ~pbc_mask) / 2

        # real-space, pbc
        real_space += (
            solver.rspace(
                batch.smearing[batch.pair_to_structure],
                charges,
                r,
                batch.centers,
                batch.others,
            )
            * batch.atom_mask
            * pbc_mask
        )

        # k-space, pbc

        cell = batch.cell[batch_pbc.structure_to_structure]
        reciprocal_cell = jax.vmap(get_reciprocal)(cell)
        volume = jax.vmap(get_volume)(cell)
        pbc = batch_pbc.pbc

        kvectors = jax.vmap(generate_ewald_kvectors)(reciprocal_cell, batch_pbc.k_grid)

        k_space = jax.vmap(solver.kspace)(
            batch.smearing[batch_pbc.structure_to_structure],
            charges[batch_pbc.atom_to_atom],
            kvectors,
            batch.positions[batch_pbc.atom_to_atom],
            volume,
            cell,
            pbc,
        )

        k_space = (
            jax.ops.segment_sum(
                k_space.flatten(), batch_pbc.atom_to_atom.flatten(), num_segments=N_all
            )
            * batch.atom_mask
            * pbc_mask
        )

        return real_space + k_space

    def prepare_fn(atomss, cutoff, lr_wavelength=None, smearing=None):
        from .batching import get_batch, prepare

        return get_batch(
            [
                prepare(
                    atoms,
                    cutoff,
                    lr_wavelength=lr_wavelength,
                    smearing=smearing,
                )
                for atoms in atomss
            ]
        )

    def energy_fn(
        charges,
        batch,
        batch_nopbc,
        batch_pbc,
    ):
        potentials = potentials_fn(charges, batch, batch_nopbc, batch_pbc)
        energies = charges * potentials

        return (
            jax.ops.segment_sum(
                energies, batch.atom_to_structure, num_segments=batch.cell.shape[0]
            )
            * batch.structure_mask
        )

    def energy_and_forces_fn(
        charges,
        batch,
        batch_nopbc,
        batch_pbc,
    ):
        # if you provide pre-computed distances, you need to deal with grads yourself
        assert batch.distances is None

        def total_energy_fn(charges, batch, batch_nopbc, batch_pbc):
            energies = energy_fn(charges, batch, batch_nopbc, batch_pbc)
            return energies.sum(), energies

        (_, energy), grads = jax.value_and_grad(
            total_energy_fn, argnums=1, has_aux=True, allow_int=True
        )(charges, batch, batch_nopbc, batch_pbc)
        forces = -grads.positions * batch.atom_mask[:, None]

        return energy, forces

    def energy_and_forces_and_stress_fn(
        charges,
        batch,
        batch_nopbc,
        batch_pbc,
    ):
        # if you provide pre-computed distances, you need to deal with grads yourself
        assert batch.distances is None

        def total_energy_fn(charges, batch, batch_nopbc, batch_pbc):
            energies = energy_fn(charges, batch, batch_nopbc, batch_pbc)
            return energies.sum(), energies

        (_, energy), batch_grads = jax.value_and_grad(
            total_energy_fn, argnums=1, has_aux=True, allow_int=True
        )(charges, batch, batch_nopbc, batch_pbc)

        forces = -batch_grads.positions * batch.atom_mask[:, None]

        stress = (
            jax.ops.segment_sum(
                jnp.einsum("ia,ib->iab", batch.positions, batch_grads.positions),
                batch.atom_to_structure,
                num_segments=batch.cell.shape[0],
            )
            + jnp.einsum("sAa,sAb->sab", batch.cell, batch_grads.cell)
        ) * batch.structure_mask[:, None, None]

        return energy, forces, stress

    return Calculator(
        prepare_fn,
        potentials_fn,
        energy_fn,
        energy_and_forces_fn,
        energy_and_forces_and_stress_fn,
    )
