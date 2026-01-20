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

    fx = np.fft.fftfreq(shape[0]) * shape[0]
    fy = np.fft.fftfreq(shape[1]) * shape[1]
    fz = np.fft.fftfreq(shape[2]) * shape[2]

    kx, ky, kz = np.meshgrid(fx, fy, fz)
    k_grid[:num_k, 0] = kx.reshape(-1)
    k_grid[:num_k, 1] = ky.reshape(-1)
    k_grid[:num_k, 2] = kz.reshape(-1)

    return k_grid


def generate_ewald_kvectors(reciprocal_cell, k_grid):
    return (
        jnp.einsum("k,a->ka", k_grid[:, 0], reciprocal_cell[0])
        + jnp.einsum("k,a->ka", k_grid[:, 1], reciprocal_cell[1])
        + jnp.einsum("k,a->ka", k_grid[:, 2], reciprocal_cell[2])
    )
