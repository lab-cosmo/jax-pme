import jax.numpy as jnp


def lagrange(interpolation_nodes=4):
    assert interpolation_nodes == 4, "cannot be bothered"
    even = interpolation_nodes % 2 == 0

    def compute_weights(inverse_cell, positions, ns):
        ns = jnp.array(ns.shape)
        positions_rel = ns * jnp.einsum("na, aA->nA", positions, inverse_cell)

        if even:
            # For Lagrange interpolation, when the order is odd, the relative position
            # of a charge is the midpoint of the two nearest gridpoints. For P3M, the
            # same is true for even orders.
            positions_rel_idx = jnp.floor(positions_rel)
            offsets = positions_rel - (positions_rel_idx + 1 / 2)
        else:
            # For Lagrange interpolation, when the order is even, the relative position
            # of a charge is the nearest gridpoint. For P3M, the same is true for
            # odd orders.
            positions_rel_idx = jnp.round(positions_rel)
            offsets = positions_rel - positions_rel_idx

        # time to interpolate
        x = offsets
        x2 = x * x
        x3 = x * x2
        if interpolation_nodes == 4:
            interpolation_weights = jnp.stack(
                [
                    1 / 48 * (-3 + 2 * x + 12 * x2 - 8 * x3),
                    1 / 48 * (27 - 54 * x - 12 * x2 + 24 * x3),
                    1 / 48 * (27 + 54 * x - 12 * x2 - 24 * x3),
                    1 / 48 * (-3 - 2 * x + 12 * x2 + 8 * x3),
                ]
            )

        indices_to_interpolate = jnp.stack(
            [
                (positions_rel_idx + i) % ns
                for i in range(
                    1 - (interpolation_nodes + 1) // 2,
                    1 + interpolation_nodes // 2,
                )
            ],
            axis=0,
        ).astype(int)

        # Generate shifts for x, y, z axes and flatten for indexing
        x_shifts, y_shifts, z_shifts = jnp.meshgrid(
            jnp.arange(interpolation_nodes),
            jnp.arange(interpolation_nodes),
            jnp.arange(interpolation_nodes),
            indexing="ij",
        )
        x_shifts = x_shifts.flatten()
        y_shifts = y_shifts.flatten()
        z_shifts = z_shifts.flatten()

        # Generate a flattened representation of all the indices
        # of the mesh points on which we wish to interpolate the
        # density.
        x_indices = indices_to_interpolate[x_shifts, :, 0]
        y_indices = indices_to_interpolate[y_shifts, :, 1]
        z_indices = indices_to_interpolate[z_shifts, :, 2]

        return (
            interpolation_weights,
            x_shifts,
            y_shifts,
            z_shifts,
            x_indices,
            y_indices,
            z_indices,
        )

    def points_to_mesh(particle_weights, ns, mesh):
        # particle_weights is [n], compatible with interpolation_weights
        # ns is s ShapeStruct
        # mesh is the result of compute_weights

        (
            interpolation_weights,
            x_shifts,
            y_shifts,
            z_shifts,
            x_indices,
            y_indices,
            z_indices,
        ) = mesh

        # Update mesh values by combining particle weights and interpolation weights
        nx = ns.shape[0]
        ny = ns.shape[1]
        nz = ns.shape[2]
        rho_mesh = jnp.zeros(
            (nx, ny, nz),
            dtype=particle_weights.dtype,
        )

        rho_mesh = rho_mesh.at[x_indices, y_indices, z_indices].add(
            particle_weights
            * interpolation_weights[x_shifts, :, 0]
            * interpolation_weights[y_shifts, :, 1]
            * interpolation_weights[z_shifts, :, 2]
        )

        return rho_mesh

    def mesh_to_points(mesh_vals, ns, mesh):
        # mesh_vals is [nx, ny, nz]

        (
            interpolation_weights,
            x_shifts,
            y_shifts,
            z_shifts,
            x_indices,
            y_indices,
            z_indices,
        ) = mesh

        tmp = (
            mesh_vals[x_indices, y_indices, z_indices]
            * interpolation_weights[x_shifts, :, 0]
            * interpolation_weights[y_shifts, :, 1]
            * interpolation_weights[z_shifts, :, 2]
        )

        return tmp.sum(axis=0)

    return compute_weights, points_to_mesh, mesh_to_points
