import jax
import jax.numpy as jnp

from collections import namedtuple
from functools import partial

# solvers: actual method implementations
#
# This is the core of the package: We implement the actual methods that are
# exposed externally through calculators. Each solver has a real-space part,
# which is always the same: We just compute the real-space part of the potential
# for all the pairs and sum up in a big segment_sum. The reciprocal-space part
# differs from method to method, Ewald just sums over k-vectors and PME does
# everything on a reciprocal-space grid using FFTs. To reflect this, the functions
# do *not* share the same arguments. This complexity is hidden from the user in the
# calculators, but it is exposed here.
#
# Therefore:
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

    def kspace(smearing, charges, reciprocal_cell, kgrid, kvectors, positions, volume):
        k2 = jax.lax.square(kvectors).sum(axis=-1)

        # got reciprocal_cell as input, like ewald, but the mesh needs the inverse
        mesh = compute_weights(reciprocal_cell.T / (2 * jnp.pi), positions, kgrid)

        rho_mesh = points_to_mesh(charges, mesh)
        mesh_hat = jnp.fft.rfftn(rho_mesh, norm="backward", s=rho_mesh.shape)

        kernel = potential.lr(smearing, k2)
        filter_hat = mesh_hat * kernel

        potential_mesh = jnp.fft.irfftn(filter_hat, norm="forward", s=rho_mesh.shape)
        pot = mesh_to_points(potential_mesh, mesh) / volume

        pot += potential.correction(smearing, charges, volume)
        return pot / 2

    return Solver(rspace, kspace)


def ewald(potential):
    rspace = partial(_rspace, potential)

    def kspace(smearing, charges, kvectors, positions, volume):
        # kvectors : [k, 3]
        # positions: [i, 3]
        k2 = jax.lax.square(kvectors).sum(axis=-1)  # -> [k]

        G = potential.lr(smearing, k2)
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
