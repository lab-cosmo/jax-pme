import jax
import jax.numpy as jnp

from collections import namedtuple

# -- high-level interface --
Potential = namedtuple("Potential", ("sr", "lr", "correction"))


def potential(exponent=1, exclusion_radius=None, custom_potential=None):
    if custom_potential is None:
        if exponent == 1:
            pot = coulomb()
        else:
            pot = inverse_power_law(exponent)
    else:
        pot = custom_potential

    if exclusion_radius is not None:

        def cutoff_function(r):
            return jnp.where(
                r < exclusion_radius,
                (1 + jnp.cos(r * (jnp.pi / exclusion_radius))) * 0.5,
                0.0,
            )

    def sr(smearing, r):
        mask = r == 0.0
        masked = jnp.where(mask, 1e-6, r)
        if exclusion_radius is not None:
            return jnp.where(
                mask, 0.0, cutoff_function(masked) * -pot.lr_r(smearing, masked)
            )
        else:
            return jnp.where(mask, 0.0, pot.sr_r(smearing, masked))

    def lr(smearing, k2):
        mask = k2 == 0.0
        masked = jnp.where(mask, 1e-6, k2)
        return jnp.where(mask, 0.0, pot.lr_k2(smearing, masked))

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
        return 4 * jnp.pi * jnp.exp(-0.5 * smearing**2 * k2) / k2

    def lr_r(smearing, r):
        return jax.scipy.special.erf(r / (smearing * jnp.sqrt(2.0))) / r

    def sr_r(smearing, r):
        return 1.0 / r - lr_r(smearing, r)

    def correction_background(smearing):
        return jnp.pi * smearing**2

    def correction_self(smearing):
        return jnp.sqrt(2.0 / jnp.pi) / smearing

    return RawPotential(sr_r, lr_r, lr_k2, correction_background, correction_self)


def inverse_power_law(exponent):
    from jax.scipy.special import gammainc, gammaincc, gammaln

    def gamma(x):
        return jnp.exp(gammaln(x))

    def lr_k2(smearing, k2):
        peff = (3 - exponent) / 2
        factor = jnp.pi**1.5 / gamma(exponent / 2) * (2 * smearing**2) ** peff
        x = 0.5 * smearing**2 * k2

        return (factor * gammaincc(peff, x) / x**peff) * gamma(peff)

    def lr_r(smearing, r):
        x = 0.5 * r**2 / smearing**2
        peff = exponent / 2
        factor = 1.0 / (2 * smearing**2) ** peff
        return factor * gammainc(peff, x) / x**peff

    def sr_r(smearing, r):
        return r ** (-exponent) - lr_r(smearing, r)

    def correction_background(smearing):
        factor = jnp.pi**1.5 * (2 * smearing**2) ** ((3 - exponent) / 2)
        factor /= (3 - exponent) * gamma(exponent / 2)
        return factor

    def correction_self(smearing):
        phalf = exponent / 2
        return 1 / gamma(phalf + 1) / (2 * smearing**2) ** phalf

    return RawPotential(sr_r, lr_r, lr_k2, correction_background, correction_self)
