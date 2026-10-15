import numpy as np
import jax
import jax.numpy as jnp

from collections import namedtuple

from .kspace import generate_kvectors, get_reciprocal
from .potentials import potential
from .solvers import ewald, ewald_dipole, p3m, pme
from .utils import atoms_to_graph, get_displacements, get_distances

# calculators: high-level interface
#
# As discussed in the readme, the main user-facing part of this library is defined here:
# Different Calculators that expose a prepare function, which puts inputs together, and
# various compute functions that consume these inputs. They are all derived from the
# potentials_fn, which computes the potential at each position. All compute functions
# are fully compatible with JAX function transformations.
#
# For now, we accept some duplication between Ewald and PME to avoid overly abstracting
# too early. This will be revisited later; we could probably use plain classes here.
#
# Thus:
Calculator = namedtuple(
    "Calculator",
    ("prepare", "potentials", "energy", "energy_forces", "energy_forces_stress"),
)


def Ewald(
    exponent=1,
    exclusion_radius=None,
    prefactor=1.0,
    custom_potential=None,
    full_neighbor_list=False,
):
    pot = potential(
        exponent=exponent,
        exclusion_radius=exclusion_radius,
        custom_potential=custom_potential,
    )
    solver = ewald(pot, full_neighbor_list=full_neighbor_list)

    def potentials_fn(
        charges,
        cell,
        positions,
        i,
        j,
        cell_shifts,
        k_grid,
        smearing,
        pbc=None,
        atom_mask=None,
        pair_mask=None,
        distances=None,
    ):
        if distances is None:
            r = get_distances(cell, positions[i], positions[j], cell_shifts)
            if pair_mask is not None:
                r *= pair_mask
        else:
            r = distances

        if atom_mask is not None:
            charges *= atom_mask

        if pbc is not None and not np.any(pbc):
            # non-periodic: no Ewald splitting -- there is no periodic image to sum,
            # and the reciprocal block divides by the (zero) volume. prepare()
            # supplies the all-pairs list; smearing=None sums the bare potential.
            potentials = solver.rspace(None, charges, r, i, j)
        else:
            reciprocal_cell = get_reciprocal(cell)
            volume = jnp.abs(jnp.linalg.det(cell))
            kvectors = generate_kvectors(
                reciprocal_cell, k_grid.shape, dtype=positions.dtype, for_ewald=True
            )

            rspace = solver.rspace(smearing, charges, r, i, j)
            kspace = solver.kspace(
                smearing, charges, kvectors, positions, volume, cell, pbc
            )
            potentials = rspace + kspace

        if atom_mask is not None:
            potentials *= atom_mask

        return prefactor * potentials

    def prepare_fn(atoms, charges, cutoff, lr_wavelength=None, smearing=None):
        return _prepare_ewald(
            atoms,
            charges,
            cutoff,
            lr_wavelength,
            smearing,
            exclusion_radius,
            full_neighbor_list,
        )

    return Calculator(prepare_fn, potentials_fn, *get_calculate_functions(potentials_fn))


def _prepare_ewald(
    atoms, charges, cutoff, lr_wavelength, smearing, exclusion_radius, full_neighbor_list
):
    from .kspace import get_kgrid_ewald

    if lr_wavelength is None:
        lr_wavelength = cutoff / 8.0

    if smearing is None:
        smearing = cutoff / 4.0

    pbc = np.asarray(atoms.get_pbc())
    cell = np.asarray(atoms.get_cell().array)

    if charges is None:
        charges = np.asarray(atoms.get_initial_charges())
    else:
        charges = np.asarray(charges).flatten()

    if not pbc.any():
        # non-periodic: all-pairs list (no cutoff -- a finite cluster has no screening)
        if exclusion_radius is not None:
            # the long-/short-range split is undefined without periodicity
            raise NotImplementedError(
                "exclusion_radius is not supported for non-periodic structures"
            )
        positions = np.asarray(atoms.get_positions())
        i, j = np.triu_indices(len(positions), k=1)
        if full_neighbor_list:
            i, j = np.concatenate([i, j]), np.concatenate([j, i])
        cell_shifts = np.zeros((len(i), 3), dtype=int)
        graph = (cell, positions, i, j, cell_shifts)
    else:
        graph = atoms_to_graph(atoms, cutoff, full_list=full_neighbor_list)

    k_grid = get_kgrid_ewald(cell, lr_wavelength)

    return charges, *graph, k_grid, smearing, pbc


def EwaldDipole(
    exclusion_radius=None,
    prefactor=1.0,
    full_neighbor_list=False,
):
    """Ewald for point charges and point dipoles (Coulomb, tin-foil boundary).

    `potentials` returns (φ, F) with φ = Φ/2 and F = E/2 (halved convention),
    so the energy is Σ q φ - Σ μ·F. The physical field at the atoms is 2F.
    """
    pot = potential(exponent=1, exclusion_radius=exclusion_radius)
    solver = ewald_dipole(pot, full_neighbor_list=full_neighbor_list)

    def potentials_fn(
        charges,
        dipoles,
        cell,
        positions,
        i,
        j,
        cell_shifts,
        k_grid,
        smearing,
        pbc=None,
        atom_mask=None,
        pair_mask=None,
    ):
        R = get_displacements(cell, positions[i], positions[j], cell_shifts)
        if pair_mask is not None:
            R *= pair_mask[:, None]

        if atom_mask is not None:
            charges *= atom_mask
            dipoles *= atom_mask[:, None]

        if pbc is not None and not np.any(pbc):
            phi, field = solver.rspace(None, charges, dipoles, R, i, j)
        else:
            reciprocal_cell = get_reciprocal(cell)
            volume = jnp.abs(jnp.linalg.det(cell))
            kvectors = generate_kvectors(
                reciprocal_cell, k_grid.shape, dtype=positions.dtype, for_ewald=True
            )

            phi, field = solver.rspace(smearing, charges, dipoles, R, i, j)
            phi_k, field_k = solver.kspace(
                smearing, charges, dipoles, kvectors, positions, volume, cell, pbc
            )
            phi += phi_k
            field += field_k

        if atom_mask is not None:
            phi *= atom_mask
            field *= atom_mask[:, None]

        return prefactor * phi, prefactor * field

    def prepare_fn(atoms, charges, dipoles, cutoff, lr_wavelength=None, smearing=None):
        if dipoles is None:
            dipoles = atoms.get_array("dipoles")
        dipoles = np.asarray(dipoles).reshape(-1, 3)

        charges, *rest = _prepare_ewald(
            atoms,
            charges,
            cutoff,
            lr_wavelength,
            smearing,
            exclusion_radius,
            full_neighbor_list,
        )
        return charges, dipoles, *rest

    def energy_fn(charges, dipoles, cell, positions, *args, **kwargs):
        phi, field = potentials_fn(charges, dipoles, cell, positions, *args, **kwargs)
        return jnp.sum(charges * phi) - jnp.sum(dipoles * field)

    def energy_and_forces_fn(charges, dipoles, cell, positions, *args, **kwargs):
        energy, grads = jax.value_and_grad(energy_fn, argnums=3)(
            charges, dipoles, cell, positions, *args, **kwargs
        )
        return energy, -grads

    def energy_and_forces_and_stress_fn(charges, dipoles, cell, positions, *args, **kwargs):
        # dipoles are held fixed under strain, so the stress is not symmetric
        energy, (cell_grads, pos_grads) = jax.value_and_grad(energy_fn, argnums=(2, 3))(
            charges, dipoles, cell, positions, *args, **kwargs
        )
        stress = jnp.einsum("ia,ib->ab", positions, pos_grads) + jnp.einsum(
            "Aa,Ab->ab", cell, cell_grads
        )
        return energy, -pos_grads, stress

    return Calculator(
        prepare_fn,
        potentials_fn,
        energy_fn,
        energy_and_forces_fn,
        energy_and_forces_and_stress_fn,
    )


def PME(
    exponent=1,
    exclusion_radius=None,
    prefactor=1.0,
    interpolation_nodes=4,
    custom_potential=None,
    full_neighbor_list=False,
):
    pot = potential(
        exponent=exponent,
        exclusion_radius=exclusion_radius,
        custom_potential=custom_potential,
    )
    solver = pme(
        pot, interpolation_nodes=interpolation_nodes, full_neighbor_list=full_neighbor_list
    )

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
        pbc=None,
    ):
        reciprocal_cell = get_reciprocal(cell)

        r = get_distances(cell, positions[i], positions[j], cell_shifts)
        if pair_mask is not None:
            r *= pair_mask

        volume = jnp.abs(jnp.linalg.det(cell))
        kvectors = generate_kvectors(
            reciprocal_cell,
            k_grid.shape,
            dtype=positions.dtype,
            for_ewald=False,
        )

        if atom_mask is not None:
            charges *= atom_mask

        rspace = solver.rspace(smearing, charges, r, i, j)
        kspace = solver.kspace(
            smearing,
            charges,
            reciprocal_cell,
            k_grid,
            kvectors,
            positions,
            volume,
            cell,
            pbc,
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
            charges = np.asarray(atoms.get_initial_charges())
        else:
            charges = np.asarray(charges).flatten()

        k_grid = get_kgrid_mesh(np.asarray(atoms.get_cell().array), mesh_spacing)

        return charges, *graph, k_grid, smearing

    return Calculator(prepare_fn, potentials_fn, *get_calculate_functions(potentials_fn))


def P3M(
    exponent=1,
    exclusion_radius=None,
    prefactor=1.0,
    interpolation_nodes=4,
    custom_potential=None,
    full_neighbor_list=False,
):
    """Particle-Particle Particle-Mesh calculator.

    Like PME but uses B-spline interpolation with an optimized influence
    function that corrects for the smoothing introduced by mesh assignment.
    Supports interpolation_nodes = 1, 2, 3, 4, 5.
    """
    pot = potential(
        exponent=exponent,
        exclusion_radius=exclusion_radius,
        custom_potential=custom_potential,
    )
    solver = p3m(
        pot, interpolation_nodes=interpolation_nodes, full_neighbor_list=full_neighbor_list
    )

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
        pbc=None,
    ):
        reciprocal_cell = get_reciprocal(cell)

        r = get_distances(cell, positions[i], positions[j], cell_shifts)
        if pair_mask is not None:
            r *= pair_mask

        volume = jnp.abs(jnp.linalg.det(cell))
        kvectors = generate_kvectors(
            reciprocal_cell,
            k_grid.shape,
            dtype=positions.dtype,
            for_ewald=False,
        )

        if atom_mask is not None:
            charges *= atom_mask

        rspace = solver.rspace(smearing, charges, r, i, j)
        kspace = solver.kspace(
            smearing,
            charges,
            reciprocal_cell,
            k_grid,
            kvectors,
            positions,
            volume,
            cell,
            pbc,
        )

        potentials = rspace + kspace

        if atom_mask is not None:
            potentials *= atom_mask

        return prefactor * potentials

    def prepare_fn(atoms, charges, cutoff, mesh_spacing, smearing):
        from .kspace import get_kgrid_mesh

        graph = atoms_to_graph(atoms, cutoff)

        if charges is None:
            charges = np.asarray(atoms.get_initial_charges())
        else:
            charges = np.asarray(charges).flatten()

        k_grid = get_kgrid_mesh(np.asarray(atoms.get_cell().array), mesh_spacing)

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
