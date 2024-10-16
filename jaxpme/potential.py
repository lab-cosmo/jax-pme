import jax
import jax.numpy as jnp

from collections import namedtuple

Potential = namedtuple(
    "Potential", ("sr_r", "lr_r", "lr_k2", "corr_background", "corr_self")
)


def coulomb(smearing):
    def lr_k2(k2):
        return jnp.where(
            k2 == 0.0,
            0.0,
            4 * jnp.pi * jnp.exp(-0.5 * smearing**2 * k2) / k2,
        )

    def lr_r(r):
        return jax.scipy.special.erf(r / (smearing * jnp.sqrt(2.0))) / r

    def sr_r(r):
        return 1.0 / r - lr_r(r)

    corr_background = jnp.pi * smearing**2
    corr_self = jnp.sqrt(2.0 / jnp.pi) / smearing

    return Potential(sr_r, lr_r, lr_k2, corr_background, corr_self)
