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
def _rspace(potential, smearing, charges, r, i, j, full_neighbor_list=False):
    N = charges.shape[0]

    if smearing is None:
        # no range separation: sum the bare potential (non-periodic case)
        potentials_bare = potential.real(r)
    else:
        potentials_bare = potential.sr(smearing, r)

    pot = jax.ops.segment_sum(charges[j] * potentials_bare, i, num_segments=N)

    if not full_neighbor_list:
        # "reverse" pairs
        pot += jax.ops.segment_sum(charges[i] * potentials_bare, j, num_segments=N)

    pot /= 2

    return pot


# -- different solvers for reciprocal-space part --


def pme(potential, interpolation_nodes=4, full_neighbor_list=False):
    from .mesh import lagrange

    compute_weights, points_to_mesh, mesh_to_points = lagrange(interpolation_nodes)

    rspace = partial(_rspace, potential, full_neighbor_list=full_neighbor_list)

    def kspace(
        smearing, charges, reciprocal_cell, kgrid, kvectors, positions, volume, cell, pbc
    ):
        k2 = jax.lax.square(kvectors).sum(axis=-1)

        # got reciprocal_cell as input, like ewald, but the mesh needs the inverse
        mesh = compute_weights(reciprocal_cell.T / (2 * jnp.pi), positions, kgrid)

        rho_mesh = points_to_mesh(charges, mesh)
        mesh_hat = jnp.fft.rfftn(rho_mesh, norm="backward", s=rho_mesh.shape)

        kernel = potential.lr(smearing, k2)
        filter_hat = mesh_hat * kernel

        potential_mesh = jnp.fft.irfftn(filter_hat, norm="forward", s=rho_mesh.shape)
        pot = mesh_to_points(potential_mesh, mesh) / volume

        pot += potential.correction(smearing, charges, volume, positions, cell, pbc)
        return pot / 2

    return Solver(rspace, kspace)


def p3m(potential, interpolation_nodes=4, full_neighbor_list=False):
    from .kspace import p3m_influence
    from .mesh import bspline

    compute_weights, points_to_mesh, mesh_to_points = bspline(interpolation_nodes)

    rspace = partial(_rspace, potential, full_neighbor_list=full_neighbor_list)

    def kspace(
        smearing, charges, reciprocal_cell, kgrid, kvectors, positions, volume, cell, pbc
    ):
        k2 = jax.lax.square(kvectors).sum(axis=-1)

        # got reciprocal_cell as input, like ewald, but the mesh needs the inverse
        mesh = compute_weights(reciprocal_cell.T / (2 * jnp.pi), positions, kgrid)

        rho_mesh = points_to_mesh(charges, mesh)
        mesh_hat = jnp.fft.rfftn(rho_mesh, norm="backward", s=rho_mesh.shape)

        # P3M: multiply kernel by influence function to correct for interpolation
        ns = jnp.array(kgrid.shape)
        influence = p3m_influence(kvectors, cell, ns, interpolation_nodes)
        kernel = potential.lr(smearing, k2) * influence

        filter_hat = mesh_hat * kernel

        potential_mesh = jnp.fft.irfftn(filter_hat, norm="forward", s=rho_mesh.shape)
        pot = mesh_to_points(potential_mesh, mesh) / volume

        pot += potential.correction(smearing, charges, volume, positions, cell, pbc)
        return pot / 2

    return Solver(rspace, kspace)


def ewald(potential, full_neighbor_list=False, halfspace=False):
    rspace = partial(_rspace, potential, full_neighbor_list=full_neighbor_list)

    # halfspace: sum over positive half of k-space and multiply by 2
    # to account for (k, -k) symmetry
    g_factor = 2 if halfspace else 1

    def kspace(smearing, charges, kvectors, positions, volume, cell, pbc):
        # kvectors : [k, 3]
        # positions: [i, 3]
        k2 = jax.lax.square(kvectors).sum(axis=-1)  # -> [k]

        # k=0 is excluded from the sum and added back once below: halfspace
        # grids drop it, full grids contain it, and batching pads k-grids with
        # zero vectors -- this treats all three uniformly
        G = jnp.where(k2 == 0.0, 0.0, potential.lr(smearing, k2) * g_factor)
        trig_args = kvectors @ (positions.T)  # -> [k, i]

        cos_all = jnp.cos(trig_args)
        sin_all = jnp.sin(trig_args)
        cos_summed = jnp.sum(cos_all * charges, axis=-1)
        sin_summed = jnp.sum(sin_all * charges, axis=-1)

        def potential_f(c, s):
            return jnp.sum(G * c * cos_summed + G * s * sin_summed)

        pot = jax.vmap(potential_f, in_axes=(1, 1))(cos_all, sin_all)

        # the k=0 term, finite for some potentials (1/r^p with p > 3)
        pot += potential.lr(smearing, jnp.zeros(())) * jnp.sum(charges)

        pot /= volume

        pot += potential.correction(smearing, charges, volume, positions, cell, pbc)
        return pot / 2

    return Solver(rspace, kspace)


# -- point dipoles (Coulomb only) --
#
# A dipole is the charge density -μ·∇δ, so a source (q, μ) at r_j contributes
# (q - μ·∇)ψ to the potential at r_i, and minus its gradient to the field. In
# reciprocal space this is the structure factor S(k) = Σ_j (q_j + i k·μ_j) e^{ik·r_j}.
# Outputs follow the halved convention: φ = Φ/2, F = E/2, so that the energy is
# Σ q φ - Σ μ·F, dE/dq = 2φ and dE/dμ = -2F.


def _rspace_dipole(
    potential, smearing, charges, dipoles, R, i, j, full_neighbor_list=False
):
    from .utils import safe_norm

    N = charges.shape[0]

    r = safe_norm(R, axis=-1)
    psi, d1, d2 = potential.sr_derivatives(smearing, r)

    # x_hat points from the source j to the target i
    r_safe = jnp.where(r == 0.0, 1.0, r)
    x_hat = -R / r_safe[:, None]
    d1_over_r = d1 / r_safe

    def from_source(q, mu, sign):
        # sign = -1 for the reverse pair (j <- i), which flips x_hat
        mu_x = jnp.sum(mu * x_hat, axis=-1)
        phi = q * psi - sign * d1 * mu_x
        field = (
            -sign * (q * d1)[:, None] * x_hat
            + ((d2 - d1_over_r) * mu_x)[:, None] * x_hat
            + d1_over_r[:, None] * mu
        )
        return phi, field

    phi_ij, field_ij = from_source(charges[j], dipoles[j], 1.0)
    phi = jax.ops.segment_sum(phi_ij, i, num_segments=N)
    field = jax.ops.segment_sum(field_ij, i, num_segments=N)

    if not full_neighbor_list:
        phi_ji, field_ji = from_source(charges[i], dipoles[i], -1.0)
        phi += jax.ops.segment_sum(phi_ji, j, num_segments=N)
        field += jax.ops.segment_sum(field_ji, j, num_segments=N)

    return phi / 2, field / 2


def ewald_dipole(potential, full_neighbor_list=False, halfspace=False):
    from .kspace import slab_energy_per_atom, slab_field_per_atom, slab_geometry
    from .potentials import dipole_self_field

    rspace = partial(_rspace_dipole, potential, full_neighbor_list=full_neighbor_list)

    g_factor = 2 if halfspace else 1

    def kspace(smearing, charges, dipoles, kvectors, positions, volume, cell, pbc):
        k2 = jax.lax.square(kvectors).sum(axis=-1)  # -> [k]

        # k=0 carries no Coulomb contribution (tin-foil, neutralizing background)
        G = jnp.where(k2 == 0.0, 0.0, potential.lr(smearing, k2) * g_factor)
        trig_args = kvectors @ (positions.T)  # -> [k, i]

        c = jnp.cos(trig_args)
        s = jnp.sin(trig_args)
        k_mu = kvectors @ dipoles.T  # -> [k, i]

        S_re = jnp.sum(charges * c - k_mu * s, axis=-1)
        S_im = jnp.sum(charges * s + k_mu * c, axis=-1)

        a = G * S_re
        b = G * S_im
        phi = (a @ c + b @ s) / volume
        field = (s.T * a - c.T * b) @ kvectors / volume

        phi += potential.correction(smearing, charges, volume)
        field += dipole_self_field(smearing) * dipoles

        if pbc is not None:
            n_hat, basis_len = slab_geometry(cell, pbc)
            z = positions @ n_hat
            mu_z = dipoles @ n_hat
            charge_tot = jnp.sum(charges)
            M_axis = jnp.sum(charges * z + mu_z)
            M_axis_sq = jnp.sum(charges * z**2 + 2 * mu_z * z)

            is_2d = jnp.sum(pbc) == 2
            slab_phi = slab_energy_per_atom(z, M_axis, M_axis_sq, charge_tot, basis_len)
            slab_field = slab_field_per_atom(z, M_axis, charge_tot)[:, None] * n_hat
            phi += is_2d * slab_phi / volume
            field += is_2d * slab_field / volume

        return phi / 2, field / 2

    return Solver(rspace, kspace)
