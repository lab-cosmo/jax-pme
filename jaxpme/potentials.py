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
        lr_k0 = getattr(pot, "lr_k0", None)
        k0 = 0.0 if lr_k0 is None else lr_k0(smearing)
        return jnp.where(mask, k0, pot.lr_k2(smearing, masked))

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

        if pbc is not None:
            c += pot.correction_pbc(positions, cell, charges, pbc) / volume

        return c

    return Potential(sr, lr, real, correction)


# -- low-level implementation of potentials --
#
# lr_k0 gives the k->0 limit of lr_k2 (used in place of the masked k=0 term);
# it is optional (None means 0, i.e. a neutralizing background).
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
        "lr_k0",
    ),
    defaults=(None,),
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

    def correction_pbc(positions, cell, charges, pbc):
        # 2D slab correction (Yeh-Berkowitz / Pan-Hu)
        # generalized to arbitrary (triclinic) cells.
        # ref: https://doi.org/10.1063/1.3216473

        is_2d = jnp.sum(pbc) == 2
        nonpbc = ~pbc

        # non-periodic axis index and two periodic lattice vectors
        k = jnp.argmax(nonpbc.astype(jnp.int32))
        v1 = cell[(k + 1) % 3]
        v2 = cell[(k + 2) % 3]

        # normal to the periodic plane
        n = jnp.cross(v1, v2)
        n_hat = n / jnp.linalg.norm(n)

        # project positions and cell height onto the normal
        z_i = positions @ n_hat
        basis_len = jnp.abs(jnp.dot(cell[k], n_hat))

        charge_tot = jnp.sum(charges)
        M_axis = jnp.sum(charges * z_i)
        M_axis_sq = jnp.sum(charges * (z_i**2))

        E_slab_2d = (4.0 * jnp.pi) * (
            z_i * M_axis
            - 0.5 * (M_axis_sq + charge_tot * (z_i**2))
            - (charge_tot / 12.0) * (basis_len**2)
        )

        return is_2d * E_slab_2d

    return RawPotential(
        sr_r, lr_r, lr_k2, real, correction_background, correction_self, correction_pbc
    )


@jax.custom_jvp
def _exp1(x):
    # exponential integral E1, x > 0: power series below 1, continued fraction
    # above (same scheme as scipy/torch-pme). jax.scipy.special.exp1 is avoided
    # since it takes ~20s to compile and its jvp leaks tracers (jax v0.4.30).
    euler = 0.577215664901532860606512090082402431

    x_small = jnp.minimum(x, 1.0)
    x_large = jnp.maximum(x, 1.0)

    def series_step(k, carry):
        e1, r = carry
        r = -r * k * x_small / (k + 1.0) ** 2
        return e1 + r, r

    ones = jnp.ones_like(x)
    e1, _ = jax.lax.fori_loop(1, 26, series_step, (ones, ones))
    small = -euler - jnp.log(x_small) + x_small * e1

    def cf_step(i, t0):
        k = 100 - i
        return k / (1.0 + k / (x_large + t0))

    t0 = jax.lax.fori_loop(0, 100, cf_step, jnp.zeros_like(x))
    large = jnp.exp(-x_large) / (x_large + t0)

    return jnp.where(x <= 1.0, small, large)


@_exp1.defjvp
def _exp1_jvp(primals, tangents):
    (x,) = primals
    (dx,) = tangents
    return _exp1(x), -jnp.exp(-x) / x * dx


def inverse_power_law(exponent):
    from jax.scipy.special import erfc, gammainc, gammaln

    # The reciprocal-space kernel is Gamma(peff, x) / x**peff with
    # peff = (3 - exponent)/2, which the generic gammaincc only handles for
    # peff > 0 (exponent < 3). Following torch-pme, we use closed forms per
    # integer exponent instead.
    if exponent not in (1, 2, 3, 4, 5, 6):
        raise ValueError(f"Unsupported exponent: {exponent} (must be an integer in 1..6)")

    def gamma(x):
        return jnp.exp(gammaln(x))

    def gammaincc_over_powerlaw(x):
        # Gamma((3 - exponent)/2, x) / x**((3 - exponent)/2)
        if exponent == 1:
            return jnp.exp(-x) / x
        if exponent == 2:
            return jnp.sqrt(jnp.pi / x) * erfc(jnp.sqrt(x))
        if exponent == 3:
            return _exp1(x)
        if exponent == 4:
            return 2 * (jnp.exp(-x) - jnp.sqrt(jnp.pi * x) * erfc(jnp.sqrt(x)))
        if exponent == 5:
            return jnp.exp(-x) - x * _exp1(x)
        if exponent == 6:
            return (
                (2 - 4 * x) * jnp.exp(-x) + 4 * jnp.sqrt(jnp.pi * x**3) * erfc(jnp.sqrt(x))
            ) / 3

    def lr_prefactor(smearing):
        peff = (3 - exponent) / 2
        return jnp.pi**1.5 / gamma(exponent / 2) * (2 * smearing**2) ** peff

    def lr_k2(smearing, k2):
        x = 0.5 * smearing**2 * k2
        return lr_prefactor(smearing) * gammaincc_over_powerlaw(x)

    def lr_k0(smearing):
        # for exponent > 3 the kernel has a finite k->0 limit; for <= 3 the
        # divergent k=0 term is dropped (neutralizing background instead)
        if exponent <= 3:
            return jnp.zeros_like(smearing)
        return lr_prefactor(smearing) * 2 / (exponent - 3)

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
        # diverges at exponent 3 and is not needed beyond
        # (see SI of https://doi.org/10.48550/arXiv.2412.03281)
        if exponent >= 3:
            return jnp.zeros_like(smearing)
        factor = jnp.pi**1.5 * (2 * smearing**2) ** ((3 - exponent) / 2)
        factor /= (3 - exponent) * gamma(exponent / 2)
        return factor

    def correction_self(smearing):
        phalf = exponent / 2
        return 1 / gamma(phalf + 1) / (2 * smearing**2) ** phalf

    def correction_pbc(positions, cell, charges, pbc):
        # 2D PBC correction not implemented for inverse power law!
        # -> returns 0 for 3D PBC, NaN for 2D PBC.
        is_3d = jnp.sum(pbc) == 3
        zeros = jnp.zeros(positions.shape[0], dtype=positions.dtype)
        nans = jnp.full(positions.shape[0], jnp.nan)
        return jax.lax.cond(is_3d, lambda: zeros, lambda: nans)

    return RawPotential(
        sr_r,
        lr_r,
        lr_k2,
        real,
        correction_background,
        correction_self,
        correction_pbc,
        lr_k0,
    )
