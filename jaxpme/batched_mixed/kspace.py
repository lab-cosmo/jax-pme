import numpy as np
import jax.numpy as jnp

# split the k-grid generation into two parts:
# (a) computing the *prefactors* for the reciprocal cell
# (b) actually applying them to the cell.
#
# (a) is preprocessing, it doesn't have to be in the compute graph;
# (b) must be to have correct cell gradients.
#
# we generate the k-grid in "flattened" form, so it's more efficient
# to batch across very uneven k-grids.


def generate_ewald_k_grid(shape, size=None):
    num_k = shape[0] * shape[1] * shape[2]
    if size is None:
        size = num_k

    k_grid = np.zeros((size, 3), dtype=float)

    # this is just a manual outer product: we want all combinations of x,y,z which
    # we accomplish by systematically mixing tile and repeat
    k_grid[:num_k, 0] = shape[0] * np.fft.fftfreq(shape[0]).repeat(shape[1] * shape[2])
    k_grid[:num_k, 1] = np.tile(
        shape[1] * np.fft.fftfreq(shape[1]).repeat(shape[0]), shape[2]
    )
    k_grid[:num_k, 2] = np.tile(shape[2] * np.fft.fftfreq(shape[2]), shape[0] * shape[1])

    return k_grid


def generate_ewald_kvectors(reciprocal_cell, k_grid):
    return (
        jnp.einsum("k,a->ka", k_grid[:, 0], reciprocal_cell[0])
        + jnp.einsum("k,a->ka", k_grid[:, 1], reciprocal_cell[1])
        + jnp.einsum("k,a->ka", k_grid[:, 2], reciprocal_cell[2])
    )
