import jax.numpy as jnp

from collections import namedtuple

from .ewald import ewald
from .kspace import generate_kvectors, get_reciprocal
from .pme import pme
from .potential import coulomb

Interface = namedtuple("Interface", ("prepare", "potentials", "energy"))


# -- helpers --
def atoms_to_graph(atoms, cutoff):
    import vesin

    positions = jnp.array(atoms.get_positions())
    cell = jnp.array(atoms.get_cell().array)

    nl = vesin.NeighborList(cutoff=cutoff, full_list=False)
    i, j, S = nl.compute(points=positions, box=cell, periodic=True, quantities="ijS")

    return cell, positions, i, j, S


def get_distances(cell, Ra, Rb, cell_shifts):
    # we can't use mic: the unit tests use small cells
    R = Rb - Ra
    R += jnp.einsum("pA,Aa->pa", cell_shifts, cell)
    return jnp.linalg.norm(R, axis=-1)


# -- Ewald interface --
def ewald_potentials(
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

    calculator = ewald(coulomb(smearing))
    rspace = calculator.rspace(charges, r, i, j)
    kspace = calculator.kspace(charges, kvectors, positions, volume)

    return rspace + kspace


def ewald_energy(
    charges,
    cell,
    positions,
    i,
    j,
    cell_shifts,
    k_grid,
    smearing,
):
    potentials = ewald_potentials(
        charges,
        cell,
        positions,
        i,
        j,
        cell_shifts,
        k_grid,
        smearing,
    )

    return jnp.sum(charges * potentials)


def ewald_prepare(atoms, charges, cutoff, lr_wavelength, smearing):
    from .kspace import get_kgrid_ewald

    graph = atoms_to_graph(atoms, cutoff)

    if charges is None:
        charges = jnp.array([-1.0, 1.0])
        charges = jnp.tile(charges, len(atoms) // 2)

    else:
        charges = jnp.array(charges).flatten()

    k_grid = get_kgrid_ewald(jnp.array(atoms.get_cell().array), lr_wavelength)

    return charges, *graph, k_grid, smearing


Ewald = Interface(ewald_prepare, ewald_potentials, ewald_energy)

# -- PME --


def pme_potentials(
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

    calculator = pme(coulomb(smearing))
    rspace = calculator.rspace(charges, r, i, j)
    kspace = calculator.kspace(charges, inverse_cell, k_grid, kvectors, positions, volume)

    return rspace + kspace


def pme_energy(
    charges,
    cell,
    positions,
    i,
    j,
    cell_shifts,
    k_grid,
    smearing,
):
    potentials = pme_potentials(
        charges,
        cell,
        positions,
        i,
        j,
        cell_shifts,
        k_grid,
        smearing,
    )
    return jnp.sum(charges * potentials)


def pme_prepare(atoms, charges, cutoff, mesh_spacing, smearing):
    from .kspace import get_kgrid_mesh

    graph = atoms_to_graph(atoms, cutoff)

    if charges is None:
        charges = jnp.array([-1.0, 1.0])
        charges = jnp.tile(charges, len(atoms) // 2)

    else:
        charges = jnp.array(charges).flatten()

    k_grid = get_kgrid_mesh(jnp.array(atoms.get_cell().array), mesh_spacing)

    return charges, *graph, k_grid, smearing


PME = Interface(pme_prepare, pme_potentials, pme_energy)
