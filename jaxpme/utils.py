import jax
import jax.numpy as jnp


def atoms_to_graph(atoms, cutoff, full_list=False):
    import vesin

    positions = jnp.array(atoms.get_positions())
    cell = jnp.array(atoms.get_cell().array)
    if atoms.get_pbc().all():
        periodic = True
    elif not atoms.get_pbc().any():
        periodic = False
    else:
        raise ValueError("no mixed pbc supported")

    nl = vesin.NeighborList(cutoff=cutoff, full_list=full_list)
    i, j, S = nl.compute(points=positions, box=cell, periodic=periodic, quantities="ijS")

    return cell, positions, i, j, S


def get_distances(cell, Ra, Rb, cell_shifts):
    # we can't use mic: the unit tests use small cells
    R = Rb - Ra
    if len(cell.shape) == 2:
        # broadcasting single cell over pairs
        R += jnp.einsum("pA,Aa->pa", cell_shifts, cell)
    elif len(cell.shape) == 3:
        # different cell per pair (batched case)
        R += jnp.einsum("pA,pAa->pa", cell_shifts, cell)
    else:
        raise ValueError(f"cell shape {cell.shape} should be 3x3 or pairs x 3x3)")

    return safe_norm(R, axis=-1)


def get_volume(cell):
    return jnp.abs(jnp.linalg.det(cell))


def safe_norm(x, axis=None):
    # derivatives of norm are NaN if x is zero,
    # this fixes that
    x2 = jnp.sum(jax.lax.square(x), axis=axis)
    mask = x2 == 0.0
    masked = jnp.where(mask, 1e-6, x2)
    return jnp.where(mask, 0.0, jnp.sqrt(masked))
