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
    return jnp.linalg.norm(R, axis=-1)
