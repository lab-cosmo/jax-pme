import jax
import jax.numpy as jnp

from collections import namedtuple

Calculator = namedtuple("Calculator", ("rspace", "kspace"))


def ewald(potential):
    def rspace(charges, r, i, j):
        N = charges.shape[0]

        potentials_bare = potential.sr_r(r)

        pot = jax.ops.segment_sum(charges[j] * potentials_bare, i, num_segments=N)
        pot += jax.ops.segment_sum(charges[i] * potentials_bare, j, num_segments=N)

        pot /= 2

        return pot

    def kspace(charges, kvectors, positions, volume):
        k2 = jax.lax.square(kvectors).sum(axis=-1)
        G = potential.lr_k2(k2)  # -> [k]
        trig_args = kvectors @ (positions.T)  # -> [k, i]

        cos_all = jnp.cos(trig_args)
        sin_all = jnp.sin(trig_args)
        cos_summed = jnp.sum(cos_all * charges, axis=-1)
        sin_summed = jnp.sum(sin_all * charges, axis=-1)

        def potential_f(c, s):
            return jnp.sum(G * c * cos_summed + G * s * sin_summed)

        pot = jax.vmap(potential_f, in_axes=(1, 1))(cos_all, sin_all)

        pot /= volume

        pot -= charges * potential.corr_self

        charge_tot = jnp.sum(charges)
        prefac = potential.corr_background
        pot -= 2.0 * prefac * charge_tot / volume

        pot /= 2

        return pot

    return Calculator(rspace, kspace)
