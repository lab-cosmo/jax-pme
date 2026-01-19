import jax
import jax.numpy as jnp

from functools import partial


def p3m_influence(kvectors, cell, ns, interpolation_nodes):
    """Compute P3M influence function 1/U²(k).

    The influence function corrects for the smoothing introduced by the
    B-spline charge assignment. U²(k) is the Fourier transform of the
    assignment function squared.

    U²(k) = Π_{axis} [sinc(k_axis * h_axis / 2π)]^(2n)

    where h_axis is the actual mesh spacing along each axis and n is
    the number of interpolation nodes.

    Args:
        kvectors: k-vectors, shape [..., 3]
        cell: unit cell matrix, shape [3, 3]
        ns: mesh shape as array [nx, ny, nz]
        interpolation_nodes: number of B-spline nodes (1-5)

    Returns:
        influence: 1/U²(k), same shape as kvectors[..., 0]
    """
    # Actual mesh spacing per axis: h = |cell_axis| / n_axis
    cell_lengths = jnp.linalg.norm(cell, axis=-1)
    mesh_spacing = cell_lengths / ns

    # kh / (2π) = k * h / (2π) for each axis
    # sinc in numpy is sin(πx)/(πx), so sinc(kh/2π) = sin(kh/2) / (kh/2)
    # We need: sinc(kh / (2π)) where sinc(x) = sin(πx)/(πx)
    # So the argument to jnp.sinc is kh / (2π)
    kh_over_2pi = kvectors * mesh_spacing / (2 * jnp.pi)

    # U²(k) = Π_axis sinc(kh_axis / 2π)^(2n)
    # jnp.sinc(x) = sin(πx)/(πx)
    sinc_vals = jnp.sinc(kh_over_2pi)  # shape [..., 3]
    u_squared = jnp.prod(sinc_vals ** (2 * interpolation_nodes), axis=-1)

    # Avoid division by zero at k=0
    # At k=0, sinc(0)=1 so U²(0)=1, but we mask it out anyway in the kernel
    u_squared = jnp.where(u_squared == 0, 1.0, u_squared)

    return 1.0 / u_squared


def get_reciprocal(cell):
    # note: reciprocal is in rows (like cell)
    return jnp.linalg.inv(cell).T * 2 * jnp.pi


def get_kgrid_ewald(cell, lr_wavelength):
    # note: this seems odd, but is correct -- we have to consider the
    #       real-space length of the wave vectors
    ns = jnp.ceil(jnp.linalg.norm(cell, axis=-1) / lr_wavelength)

    shape = (int(ns[0]), int(ns[1]), int(ns[2]))

    # in principle, ShapeDtypeStruct would suffice here, but this
    # does not play well with batching -- it can't be reshaped
    return jnp.ones(shape)


def get_kgrid_mesh(cell, mesh_spacing):
    start = jnp.array(get_kgrid_ewald(cell, mesh_spacing).shape)
    actual = 2 * start + 1
    # todo: revisit need for padding to powers of 2
    ns = jnp.array(2) ** (jnp.ceil(jnp.log2(actual)))

    shape = (int(ns[0]), int(ns[1]), int(ns[2]))

    return jnp.ones(shape)


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
