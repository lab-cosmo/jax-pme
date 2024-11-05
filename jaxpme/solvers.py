import jax
import jax.numpy as jnp

from collections import namedtuple
from functools import partial

# note: kspace has *different* signatures
Solver = namedtuple("Solver", ("rspace", "kspace"))


# -- shared real-space part --
def _rspace(potential, smearing, charges, r, i, j):
    N = charges.shape[0]

    potentials_bare = potential.sr(smearing, r)

    pot = jax.ops.segment_sum(charges[j] * potentials_bare, i, num_segments=N)
    pot += jax.ops.segment_sum(charges[i] * potentials_bare, j, num_segments=N)

    pot /= 2

    return pot


# -- different solvers for reciprocal-space part --


def pme(potential, interpolation_nodes=4):
    from .mesh import lagrange

    compute_weights, points_to_mesh, mesh_to_points = lagrange(interpolation_nodes)

    rspace = partial(_rspace, potential)

    def kspace(smearing, charges, inverse_cell, kgrid, kvectors, positions, volume):
        k2 = jax.lax.square(kvectors).sum(axis=-1)

        mesh = compute_weights(inverse_cell, positions, kgrid)

        rho_mesh = points_to_mesh(charges, kgrid, mesh)
        mesh_hat = jnp.fft.rfftn(rho_mesh, norm="backward", s=rho_mesh.shape)
        kernel = potential.lr(smearing, k2)

        filter_hat = mesh_hat * kernel

        potential_mesh = jnp.fft.irfftn(filter_hat, norm="forward", s=rho_mesh.shape)
        pot = mesh_to_points(potential_mesh, kgrid, mesh) / volume

        pot += potential.correction(smearing, charges, volume)
        return pot / 2

    return Solver(rspace, kspace)


def ewald(potential):
    rspace = partial(_rspace, potential)

    def kspace(smearing, charges, kvectors, positions, volume):
        k2 = jax.lax.square(kvectors).sum(axis=-1)

        G = potential.lr(smearing, k2)  # -> [k]
        trig_args = kvectors @ (positions.T)  # -> [k, i]

        cos_all = jnp.cos(trig_args)
        sin_all = jnp.sin(trig_args)
        cos_summed = jnp.sum(cos_all * charges, axis=-1)
        sin_summed = jnp.sum(sin_all * charges, axis=-1)

        def potential_f(c, s):
            return jnp.sum(G * c * cos_summed + G * s * sin_summed)

        pot = jax.vmap(potential_f, in_axes=(1, 1))(cos_all, sin_all)

        pot /= volume

        pot += potential.correction(smearing, charges, volume)
        return pot / 2

    return Solver(rspace, kspace)