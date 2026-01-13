import jax
import jax.numpy as jnp

from collections import namedtuple

# -- high-level interface --
Potential = namedtuple("Potential", ("sr", "lr", "real", "correction"))


def potential(exponent=1, exclusion_radius=None, custom_potential=None):
    # potential: higher-level interface to be consumed by solvers
    #
    # This instantiates a bundle of functions that evaluate a potential
    # in both real (.sr) and reciprocal (.lr) space, with corrections
    # for self-interactions and background charges (.correction).
    #
    # For low-level implementations of the physical potentials see below;
    # this is the common functionality, including the removal of all contributions
    # from inside the exclusion_radius (commonly the real-space cutoff).
    #
    # We have draft support for custom potentials: Anything that implements the
    # RawPotential interface (see below) can be wrapped in this Potential interface
    # and pushed through solvers and then calculators.
    #
    # Note that we careful to mask out zero distance r and |k|**2 to avoid NaNs,
    # even in the backward pass, see https://github.com/jax-ml/jax/issues/1052.
    #
    # We do this here to keep the raw potentials simple: We can just write out
    # the actual formulas like 1/r and not worry about details like this.

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

    def real(r):
        mask = r == 0
        masked = jnp.where(mask, 1e-6, r)
        return jnp.where(mask, 0.0, pot.real(masked))

    def correction(
        smearing,
        charges,
        volume,
        positions=None,
        cell=None,
        pbc=None,
    ):
        c = -charges * pot.correction_self(smearing)

        charge_tot = jnp.sum(charges)
        prefac = pot.correction_background(smearing)
        c -= 2 * prefac * charge_tot / volume

        if (positions is not None) and (cell is not None):
            c = c + pot.correction_pbc(positions, cell, charges, pbc)

        return c

    return Potential(sr, lr, real, correction)


# -- low-level implementation of potentials --
RawPotential = namedtuple(
    "RawPotential",
    (
        "sr_r",
        "lr_r",
        "lr_k2",
        "real",
        "correction_background",
        "correction_self",
        "correction_pbc",
    ),
)


def coulomb():
    def lr_k2(smearing, k2):
        return 4 * jnp.pi * jnp.exp(-0.5 * smearing**2 * k2) / k2

    def lr_r(smearing, r):
        return jax.scipy.special.erf(r / (smearing * jnp.sqrt(2.0))) / r

    def sr_r(smearing, r):
        return real(r) - lr_r(smearing, r)

    def real(r):
        return 1.0 / r

    def correction_background(smearing):
        return jnp.pi * smearing**2

    def correction_self(smearing):
        return jnp.sqrt(2.0 / jnp.pi) / smearing

    def correction_pbc(
        positions,
        cell,
        charges,
        pbc=None,
    ):
        # "2D periodicity" correction for 1/r potential
        if pbc is None:
            pbc = jnp.array([True, True, True])

        n_periodic = jnp.sum(pbc)
        is_2d = n_periodic == 2

        axis = jnp.argmax(
            jnp.where(
                jnp.expand_dims(is_2d, -1),
                jnp.logical_not(pbc).astype(jnp.int64),
                jnp.zeros_like(pbc, dtype=jnp.int64),
            ),
            axis=-1,
        )

        E_slab = jnp.zeros_like(charges)

        # gather z_i along the non-periodic axis
        idx = jnp.expand_dims(jnp.full((positions.shape[0],), axis, dtype=jnp.int32), 1)
        z_i = jnp.take_along_axis(positions, idx, axis=1)

        # gather basis length for that axis
        cell_norms = jnp.linalg.norm(cell, axis=-1)  # shape (3,)
        basis_len = cell_norms[axis]

        V = jnp.abs(jnp.linalg.det(cell))
        charge_tot = jnp.sum(charges, axis=0)
        M_axis = jnp.sum(charges * z_i, axis=0)
        M_axis_sq = jnp.sum(charges * (z_i**2), axis=0)

        E_slab_2d = (4.0 * jnp.pi / V) * (
            z_i * M_axis
            - 0.5 * (M_axis_sq + charge_tot * (z_i**2))
            - (charge_tot / 12.0) * (basis_len**2)
        )

        return jnp.where(jnp.expand_dims(is_2d, -1), E_slab_2d, E_slab)

    return RawPotential(
        sr_r, lr_r, lr_k2, real, correction_background, correction_self, correction_pbc
    )


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
        return real(r) - lr_r(smearing, r)

    def real(r):
        return r ** (-exponent)

    def correction_background(smearing):
        factor = jnp.pi**1.5 * (2 * smearing**2) ** ((3 - exponent) / 2)
        factor /= (3 - exponent) * gamma(exponent / 2)
        return factor

    def correction_self(smearing):
        phalf = exponent / 2
        return 1 / gamma(phalf + 1) / (2 * smearing**2) ** phalf

    def correction_pbc(positions, cell, charges, pbc=None):
        raise NotImplementedError(
            "Mixed PBC correction is not implemented for this potential."
        )

    return RawPotential(
        sr_r, lr_r, lr_k2, real, correction_background, correction_self, correction_pbc
    )
