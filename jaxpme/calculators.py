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


def Ewald(exponent=1, exclusion_radius=None):
    # instantiate here so we have access in the prepare_fn (not used yet)
    pot = potential(exponent=exponent, exclusion_radius=exclusion_radius)
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
    ):
        reciprocal_cell = get_reciprocal(cell)
        r = get_distances(cell, positions[i], positions[j], cell_shifts)

        volume = jnp.abs(jnp.linalg.det(cell))
        kvectors = generate_kvectors(
            reciprocal_cell, k_grid.shape, dtype=positions.dtype, for_ewald=True
        )

        rspace = solver.rspace(smearing, charges, r, i, j)
        kspace = solver.kspace(smearing, charges, kvectors, positions, volume)

        return rspace + kspace

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


def PME(exponent=1, exclusion_radius=None, interpolation_nodes=4):
    # instantiate here so we have access in the prepare_fn (not used yet)
    pot = potential(exponent=exponent, exclusion_radius=exclusion_radius)
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
    ):
        inverse_cell = jnp.linalg.inv(cell)
        r = get_distances(cell, positions[i], positions[j], cell_shifts)

        volume = jnp.abs(jnp.linalg.det(cell))
        kvectors = generate_kvectors(
            inverse_cell.T * 2 * jnp.pi,
            k_grid.shape,
            dtype=positions.dtype,
            for_ewald=False,
        )

        rspace = solver.rspace(smearing, charges, r, i, j)
        kspace = solver.kspace(
            smearing, charges, inverse_cell, k_grid, kvectors, positions, volume
        )

        return rspace + kspace

    def prepare_fn(atoms, charges, cutoff, mesh_spacing, smearing):
        from .kspace import get_kgrid_mesh

        # todo: inset tuning logic

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
    def energy_fn(charges, cell, positions, *args):
        potentials = potentials_fn(charges, cell, positions, *args)

        return jnp.sum(charges * potentials)

    def energy_and_forces_fn(charges, cell, positions, *args):
        energy, grads = jax.value_and_grad(energy_fn, argnums=2)(
            charges, cell, positions, *args
        )

        return energy, -grads

    def energy_and_forces_and_stress_fn(charges, cell, positions, *args):
        energy, (pos_grads, cell_grads) = jax.value_and_grad(energy_fn, argnums=(1, 2))(
            charges, cell, positions, *args
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
