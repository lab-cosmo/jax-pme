import jax
import jax.numpy as jnp

from collections import namedtuple

Calculator = namedtuple("Calculator", ("rspace", "kspace"))


def pme(potential, interpolation_nodes=4):
    from .mesh import lagrange

    compute_weights, points_to_mesh, mesh_to_points = lagrange(interpolation_nodes)

    def rspace(charges, r, i, j):
        N = charges.shape[0]

        potentials_bare = potential.sr_r(r)

        pot = jax.ops.segment_sum(charges[j] * potentials_bare, i, num_segments=N)
        pot += jax.ops.segment_sum(charges[i] * potentials_bare, j, num_segments=N)

        pot /= 2

        return pot

    def kspace(charges, inverse_cell, kgrid, kvectors, positions, volume):
        k2 = jax.lax.square(kvectors).sum(axis=-1)
        mesh = compute_weights(inverse_cell, positions, kgrid)

        rho_mesh = points_to_mesh(charges, kgrid, mesh)
        mesh_hat = jnp.fft.rfftn(rho_mesh, norm="backward", s=rho_mesh.shape)
        kernel = potential.lr_k2(k2)

        filter_hat = mesh_hat * kernel

        potential_mesh = jnp.fft.irfftn(filter_hat, norm="forward", s=rho_mesh.shape)
        pot = mesh_to_points(potential_mesh, kgrid, mesh) / volume

        pot -= charges * potential.corr_self

        charge_tot = jnp.sum(charges)
        prefac = potential.corr_background
        pot -= 2.0 * prefac * charge_tot / volume

        pot /= 2

        return pot

    return Calculator(rspace, kspace)
