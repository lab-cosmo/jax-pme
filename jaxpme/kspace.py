import jax
import jax.numpy as jnp

from functools import partial


def p3m_influence(kvectors, cell, ns, interpolation_nodes):
    """P3M influence function 1/U²(k), correcting for B-spline charge assignment."""
    # phase advance per mesh cell: kh[i] = (k · cell[i]) / n[i]
    kh = jnp.einsum("...j,ij->...i", kvectors, cell) / ns

    # U²(k) = Π_axis sinc(kh_axis / 2π)^(2n)
    sinc_vals = jnp.sinc(kh / (2 * jnp.pi))
    u_squared = jnp.prod(sinc_vals ** (2 * interpolation_nodes), axis=-1)
    u_squared = jnp.where(u_squared == 0, 1.0, u_squared)

    return 1.0 / u_squared


def get_reciprocal(cell):
    # note: reciprocal is in rows (like cell)
    return jnp.linalg.inv(cell).T * 2 * jnp.pi


def get_kgrid_ewald_shape(cell, lr_wavelength):
    # note: this seems odd, but is correct -- we have to consider the
    #       real-space length of the wave vectors
    ns = jnp.ceil(jnp.linalg.norm(cell, axis=-1) / lr_wavelength)

    return (int(ns[0]), int(ns[1]), int(ns[2]))


def lr_wavelength_for_num_k(cell, num_k):
    """Compute lr_wavelength that yields approximately num_k halfspace k-vectors.

    Analytical inversion of get_kgrid_ewald_shape assuming halfspace ≈ N_total/2.
    The ceil in get_kgrid_ewald_shape means the actual count will be slightly above
    num_k (conservative in accuracy).
    """
    lengths = jnp.linalg.norm(cell, axis=-1)
    return float((jnp.prod(lengths) / (2 * num_k)) ** (1 / 3))


def get_kgrid_mesh_shape(cell, mesh_spacing):
    start = jnp.array(get_kgrid_ewald_shape(cell, mesh_spacing))
    actual = 2 * start + 1
    # todo: revisit need for padding to powers of 2
    ns = jnp.array(2) ** (jnp.ceil(jnp.log2(actual)))

    return (int(ns[0]), int(ns[1]), int(ns[2]))


def get_kgrid_ewald(cell, lr_wavelength):
    # in principle, ShapeDtypeStruct would suffice here, but this
    # does not play well with batching -- it can't be reshaped
    return jnp.ones(get_kgrid_ewald_shape(cell, lr_wavelength))


def get_kgrid_mesh(cell, mesh_spacing):
    return jnp.ones(get_kgrid_mesh_shape(cell, mesh_spacing))


@partial(jax.jit, static_argnums=(1, 2, 3))
def generate_kvectors(reciprocal_cell, shape, for_ewald=True, dtype=jnp.float32):
    # The frequencies from the fftfreq function  are of the form [0, 1/n, 2/n, ...]
    # These are then converted to [0, 1, 2, ...] by multiplying with n.
    kxs = (reciprocal_cell[0] * shape[0]) * jnp.fft.fftfreq(shape[0], dtype=dtype)[
        ..., None
    ]
    kys = (reciprocal_cell[1] * shape[1]) * jnp.fft.fftfreq(shape[1], dtype=dtype)[
        ..., None
    ]

    if for_ewald:
        kzs = (reciprocal_cell[2] * shape[2]) * jnp.fft.fftfreq(shape[2], dtype=dtype)[
            ..., None
        ]
    else:
        kzs = (reciprocal_cell[2] * shape[2]) * jnp.fft.rfftfreq(shape[2], dtype=dtype)[
            ..., None
        ]

    result = kxs[:, None, None] + kys[None, :, None] + kzs[None, None, :]

    if for_ewald:
        return result.reshape(-1, 3)
    else:
        return result
