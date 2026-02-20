import jax.numpy as jnp

import pytest

from jaxpme.kspace import (
    get_kgrid_ewald,
    get_kgrid_ewald_shape,
    get_kgrid_mesh,
    get_kgrid_mesh_shape,
)

CUBIC_CELL = jnp.eye(3) * 10.0
ORTHORHOMBIC_CELL = jnp.diag(jnp.array([8.0, 10.0, 12.0]))


@pytest.mark.parametrize("cell", [CUBIC_CELL, ORTHORHOMBIC_CELL])
def test_kgrid_shape_functions(cell):
    """Shape functions return correct tuples matching grid shapes."""
    spacing = 2.0

    ewald_shape = get_kgrid_ewald_shape(cell, spacing)
    mesh_shape = get_kgrid_mesh_shape(cell, spacing)

    assert ewald_shape == get_kgrid_ewald(cell, spacing).shape
    assert mesh_shape == get_kgrid_mesh(cell, spacing).shape
    assert all(s & (s - 1) == 0 for s in mesh_shape)  # powers of 2
