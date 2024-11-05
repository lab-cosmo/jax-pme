import jax
import jax.numpy as jnp

from collections import namedtuple

# -- high-level interface --
Potential = namedtuple("Potential", ("sr", "lr", "correction"))


def potential(exponent=1, exclusion_radius=None):
    if exponent == 1:
        pot = coulomb()
    else:
        raise ValueError("we do not yet support exponent != 1")

    if exclusion_radius is not None:

        def cutoff_function(r):
            # todo: double masking
            return jnp.where(
                r < exclusion_radius,
                (1 + jnp.cos(r * (jnp.pi / exclusion_radius))) * 0.5,
                0.0,
            )

    def sr(smearing, r):
        if exclusion_radius is not None:
            return -pot.lr_r(smearing) * cutoff_function(r)
        else:
            return pot.sr_r(smearing, r)

    def lr(smearing, k2):
        return pot.lr_k2(smearing, k2)

    def correction(smearing, charges, volume):
        c = -charges * pot.correction_self(smearing)

        charge_tot = jnp.sum(charges)
        prefac = pot.correction_background(smearing)
        c -= 2 * prefac * charge_tot / volume

        return c

    return Potential(sr, lr, correction)


# -- low-level implementation of potentials --
RawPotential = namedtuple(
    "RawPotential", ("sr_r", "lr_r", "lr_k2", "correction_background", "correction_self")
)


def coulomb():
    def lr_k2(smearing, k2):
        # todo: double masking
        return jnp.where(
            k2 == 0.0,
            0.0,
            4 * jnp.pi * jnp.exp(-0.5 * smearing**2 * k2) / k2,
        )

    def lr_r(smearing, r):
        return jax.scipy.special.erf(r / (smearing * jnp.sqrt(2.0))) / r

    def sr_r(smearing, r):
        return 1.0 / r - lr_r(smearing, r)

    def correction_background(smearing):
        return jnp.pi * smearing**2

    def correction_self(smearing):
        return jnp.sqrt(2.0 / jnp.pi) / smearing

    return RawPotential(sr_r, lr_r, lr_k2, correction_background, correction_self)
