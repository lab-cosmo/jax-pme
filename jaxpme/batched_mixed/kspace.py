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


def generate_ewald_k_grid(shape, size=None, halfspace=False):
    fx = np.fft.fftfreq(shape[0]) * shape[0]
    fy = np.fft.fftfreq(shape[1]) * shape[1]
    fz = np.fft.fftfreq(shape[2]) * shape[2]

    kx, ky, kz = np.meshgrid(fx, fy, fz, indexing="ij")
    kx, ky, kz = kx.flatten(), ky.flatten(), kz.flatten()

    if halfspace:
        # Positive half-space: lexicographically first of each (k, -k) pair
        # This excludes k=0 and Nyquist frequencies (which have negligible contribution
        # for well-converged Ewald sums due to exponential damping).
        # All included k-vectors have implicit weight 2.
        mask = (kx > 0) | ((kx == 0) & (ky > 0)) | ((kx == 0) & (ky == 0) & (kz > 0))
        kx, ky, kz = kx[mask], ky[mask], kz[mask]

    num_k = len(kx)
    if size is None:
        size = num_k

    k_grid = np.zeros((size, 3), dtype=float)
    n = min(num_k, size)
    k_grid[:n, 0], k_grid[:n, 1], k_grid[:n, 2] = kx[:n], ky[:n], kz[:n]

    return k_grid


def count_halfspace_kvectors(shape):
    """Count k-vectors in the positive half-space (excluding k=0 and Nyquist)."""
    fx = np.fft.fftfreq(shape[0]) * shape[0]
    fy = np.fft.fftfreq(shape[1]) * shape[1]
    fz = np.fft.fftfreq(shape[2]) * shape[2]

    kx, ky, kz = np.meshgrid(fx, fy, fz, indexing="ij")
    kx, ky, kz = kx.flatten(), ky.flatten(), kz.flatten()

    mask = (kx > 0) | ((kx == 0) & (ky > 0)) | ((kx == 0) & (ky == 0) & (kz > 0))
    return mask.sum()


def generate_ewald_kvectors(reciprocal_cell, k_grid):
    return (
        jnp.einsum("k,a->ka", k_grid[:, 0], reciprocal_cell[0])
        + jnp.einsum("k,a->ka", k_grid[:, 1], reciprocal_cell[1])
        + jnp.einsum("k,a->ka", k_grid[:, 2], reciprocal_cell[2])
    )
