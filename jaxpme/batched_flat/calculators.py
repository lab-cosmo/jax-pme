import jax
import jax.numpy as jnp

from jaxpme.calculators import Calculator
from jaxpme.kspace import get_reciprocal
from jaxpme.utils import get_distances, get_volume, safe_norm

from .potentials import potential


def Ewald(
    exponent=1,
    prefactor=1.0,
    custom_potential=None,
):
    from .solvers import ewald

    pot = potential(
        exponent=exponent,
        exclusion_radius=None,
        custom_potential=custom_potential,
    )
    solver = ewald(pot)

    def potentials_fn(
        # per system
        cell,
        smearing,
        charges,
        system_mask,
        # per atom
        positions,
        atom_to_system,
        atom_mask,
        periodic_atom_mask,  # True if periodic
        # per pair (within cutoff)
        distances,
        cell_shifts,  # used for distances if those are missing
        i,
        j,
        pair_to_system,
        pair_mask,
        # per pair (outside cutoff, non-pbc)
        extra_i,
        extra_j,
        extra_pair_mask,
        # per k-vector
        kgrid,
        k_to_system,
        k_mask,
        # per k,i (trig)
        trig_to_atom,
        trig_to_k,
    ):
        reciprocal_cell = jax.vmap(get_reciprocal)(cell)
        volume = jax.vmap(get_volume)(cell)
        charges *= atom_mask

        if distances is None:
            r = get_distances(cell[pair_to_system], positions[i], positions[j], cell_shifts)
            r *= pair_mask
        else:
            r = distances

        extra_r = (
            safe_norm(positions[extra_j] - positions[extra_i], axis=-1) * extra_pair_mask
        )

        return prefactor * solver(
            reciprocal_cell,
            volume,
            smearing,
            charges,
            system_mask,
            positions,
            atom_to_system,
            atom_mask,
            periodic_atom_mask,
            r,
            i,
            j,
            pair_to_system,
            extra_r,
            extra_i,
            extra_j,
            extra_pair_mask,
            kgrid,
            k_to_system,
            k_mask,
            trig_to_atom,
            trig_to_k,
        )

    def prepare_fn(atomss, chargess, cutoff, lr_wavelength=None, smearing=None):
        from .batching import to_batch
        from .preparation import prepare

        if chargess is None:
            chargess = [None] * len(atomss)

        return to_batch(
            [
                prepare(
                    atoms,
                    cutoff,
                    charges=charges,
                    lr_wavelength=lr_wavelength,
                    smearing=smearing,
                )
                for (atoms, charges) in zip(atomss, chargess)
            ]
        )

    def energy_fn(*args):
        cell = args[0]
        charges = args[2]
        system_mask = args[3]
        atom_to_system = args[5]

        potentials = potentials_fn(*args)
        energies = charges * potentials

        return (
            jax.ops.segment_sum(energies, atom_to_system, num_segments=cell.shape[0])
            * system_mask
        )

    def energy_and_forces_fn(*args):
        atom_mask = args[6]
        distances = args[8]

        # if you provide pre-computed distances, you need to deal with grads yourself
        assert distances is None

        def total_energy_fn(*args):
            energies = energy_fn(*args)
            return energies.sum(), energies

        (_, energy), grads = jax.value_and_grad(total_energy_fn, argnums=4, has_aux=True)(
            *args
        )
        return energy, -grads * atom_mask[:, None]

    def energy_and_forces_and_stress_fn(*args):
        cell = args[0]
        system_mask = args[3]
        positions = args[4]
        atom_to_system = args[5]
        atom_mask = args[6]
        distances = args[8]

        # if you provide pre-computed distances, you need to deal with grads yourself
        assert distances is None

        def total_energy_fn(*args):
            energies = energy_fn(*args)
            return energies.sum(), energies

        (_, energy), (cell_grads, pos_grads) = jax.value_and_grad(
            total_energy_fn, argnums=(0, 4), has_aux=True
        )(*args)

        forces = -pos_grads * atom_mask[:, None]

        stress = (
            jax.ops.segment_sum(
                jnp.einsum("ia,ib->iab", positions, pos_grads),
                atom_to_system,
                num_segments=cell.shape[0],
            )
            + jnp.einsum("sAa,sAb->sab", cell, cell_grads)
        ) * system_mask[:, None, None]

        return energy, forces, stress

    return Calculator(
        prepare_fn,
        potentials_fn,
        energy_fn,
        energy_and_forces_fn,
        energy_and_forces_and_stress_fn,
    )
