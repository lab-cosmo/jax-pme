import jax.numpy as jnp

from jaxpme.utils import safe_norm


def get_distances(cell, Ra, Rb, cell_shifts):
    R = Rb - Ra
    R += jnp.einsum("pA,pAa->pa", cell_shifts, cell)
    return safe_norm(R, axis=-1)


def get_volume(cell):
    return jnp.abs(jnp.linalg.det(cell))
