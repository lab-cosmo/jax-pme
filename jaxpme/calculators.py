import jax
import jax.numpy as jnp

from collections import namedtuple

from .kspace import generate_kvectors, get_reciprocal
from .potentials import potential
from .solvers import ewald, pme
from .utils import atoms_to_graph, get_distances

Calculator = namedtuple(
    "Calculator",
    ("prepare", "potentials", "energy", "energy_forces", "energy_forces_stress"),
)


def Ewald(exponent=1, exclusion_radius=None, prefactor=1.0, custom_potential=None):
    # instantiate here so we have access in the prepare_fn (not used yet)
    pot = potential(
        exponent=exponent,
        exclusion_radius=exclusion_radius,
        custom_potential=custom_potential,
    )
    solver = ewald(pot)

    def potentials_fn(
        charges,
        cell,
        positions,
        i,
        j,
        cell_shifts,
        k_grid,
        smearing,
        atom_mask=None,
        pair_mask=None,
    ):
        reciprocal_cell = get_reciprocal(cell)

        r = get_distances(cell, positions[i], positions[j], cell_shifts)
        if pair_mask is not None:
            r *= pair_mask

        volume = jnp.abs(jnp.linalg.det(cell))
        kvectors = generate_kvectors(
            reciprocal_cell, k_grid.shape, dtype=positions.dtype, for_ewald=True
        )

        if atom_mask is not None:
            charges *= atom_mask

        rspace = solver.rspace(smearing, charges, r, i, j)
        kspace = solver.kspace(smearing, charges, kvectors, positions, volume)

        potentials = rspace + kspace

        if atom_mask is not None:
            potentials *= atom_mask

        return prefactor * potentials

    def prepare_fn(atoms, charges, cutoff, lr_wavelength, smearing):
        from .kspace import get_kgrid_ewald

        # todo: insert tuning logic

        graph = atoms_to_graph(atoms, cutoff)

        if charges is None:
            charges = jnp.array([-1.0, 1.0])
            charges = jnp.tile(charges, len(atoms) // 2)

        else:
            charges = jnp.array(charges).flatten()

        k_grid = get_kgrid_ewald(jnp.array(atoms.get_cell().array), lr_wavelength)

        return charges, *graph, k_grid, smearing

    return Calculator(prepare_fn, potentials_fn, *get_calculate_functions(potentials_fn))


def PME(
    exponent=1,
    exclusion_radius=None,
    prefactor=1.0,
    interpolation_nodes=4,
    custom_potential=None,
):
    # instantiate here so we have access in the prepare_fn (not used yet)
    pot = potential(
        exponent=exponent,
        exclusion_radius=exclusion_radius,
        custom_potential=custom_potential,
    )
    solver = pme(pot, interpolation_nodes=interpolation_nodes)

    def potentials_fn(
        charges,
        cell,
        positions,
        i,
        j,
        cell_shifts,
        k_grid,
        smearing,
        atom_mask=None,
        pair_mask=None,
    ):
        inverse_cell = jnp.linalg.inv(cell)

        r = get_distances(cell, positions[i], positions[j], cell_shifts)
        if pair_mask is not None:
            r *= pair_mask

        volume = jnp.abs(jnp.linalg.det(cell))
        kvectors = generate_kvectors(
            inverse_cell.T * 2 * jnp.pi,
            k_grid.shape,
            dtype=positions.dtype,
            for_ewald=False,
        )


        if atom_mask is not None:
            charges *= atom_mask

        rspace = solver.rspace(smearing, charges, r, i, j)
        kspace = solver.kspace(
            smearing, charges, inverse_cell, k_grid, kvectors, positions, volume
        )

        potentials = rspace + kspace

        if atom_mask is not None:
            potentials *= atom_mask

        return prefactor * potentials

    def prepare_fn(atoms, charges, cutoff, mesh_spacing, smearing):
        from .kspace import get_kgrid_mesh

        # todo: insert tuning logic

        graph = atoms_to_graph(atoms, cutoff)

        if charges is None:
            charges = jnp.array([-1.0, 1.0])
            charges = jnp.tile(charges, len(atoms) // 2)

        else:
            charges = jnp.array(charges).flatten()

        k_grid = get_kgrid_mesh(jnp.array(atoms.get_cell().array), mesh_spacing)

        return charges, *graph, k_grid, smearing

    return Calculator(prepare_fn, potentials_fn, *get_calculate_functions(potentials_fn))


# -- potentials -> energy & derivarives --
def get_calculate_functions(potentials_fn):
    def energy_fn(charges, cell, positions, *args, **kwargs):
        potentials = potentials_fn(charges, cell, positions, *args, **kwargs)

        if "atom_mask" in kwargs:
            if kwargs["atom_mask"] is not None:
                energies = jnp.where(kwargs["atom_mask"], charges * potentials, 0.0)
        else:
            energies = charges * potentials

        return jnp.sum(energies)

    def energy_and_forces_fn(charges, cell, positions, *args, **kwargs):
        energy, grads = jax.value_and_grad(energy_fn, argnums=2)(
            charges, cell, positions, *args, **kwargs
        )

        return energy, -grads

    def energy_and_forces_and_stress_fn(charges, cell, positions, *args, **kwargs):
        energy, (cell_grads, pos_grads) = jax.value_and_grad(energy_fn, argnums=(1, 2))(
            charges, cell, positions, *args, **kwargs
        )

        forces = -pos_grads

        stress = jnp.einsum("ia,ib->ab", positions, pos_grads) + jnp.einsum(
            "Aa,Ab->ab", cell, cell_grads
        )

        return energy, forces, stress

    return (
        energy_fn,
        energy_and_forces_fn,
        energy_and_forces_and_stress_fn,
    )
