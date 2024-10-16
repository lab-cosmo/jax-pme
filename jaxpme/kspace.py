import jax
import jax.numpy as jnp

from functools import partial

# conventions:
# - reciprocal_cell contains (like cell) the *reciprocal* basis in rows


def get_reciprocal(cell):
    return jnp.linalg.inv(cell).T * 2 * jnp.pi


def get_kgrid_ewald(cell, lr_wavelength):
    # copied from torch-pme -- don't understand why
    ns = jnp.ceil(jnp.linalg.norm(cell, axis=-1) / lr_wavelength)

    shape = (int(ns[0]), int(ns[1]), int(ns[2]))

    # return jax.ShapeDtypeStruct(shape=shape, dtype=jnp.float32)
    return jnp.ones(shape)


def get_kgrid_mesh(cell, mesh_spacing):
    start = jnp.array(get_kgrid_ewald(cell, mesh_spacing).shape)
    actual = 2 * start + 1
    ns = jnp.array(2) ** (jnp.ceil(jnp.log2(actual)))

    shape = (int(ns[0]), int(ns[1]), int(ns[2]))

    # return jax.ShapeDtypeStruct(shape=shape, dtype=jnp.float32)
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

    # then take the cartesian product (all possible combinations, same as meshgrid)
    result = kxs[:, None, None] + kys[None, :, None] + kzs[None, None, :]

    if for_ewald:
        return result.reshape(-1, 3)
    else:
        return result