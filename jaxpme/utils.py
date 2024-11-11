import jax
import jax.numpy as jnp


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
    return safe_norm(R, axis=-1)


def safe_norm(x, axis=None):
    # derivatives of norm are NaN if x is zero,
    # this fixes that
    x2 = jnp.sum(jax.lax.square(x), axis=axis)
    mask = x2 == 0.0
    masked = jnp.where(mask, 1e-6, x2)
    return jnp.where(mask, 0.0, jnp.sqrt(masked))
