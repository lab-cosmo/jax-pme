import jax.numpy as jnp


def lagrange(interpolation_nodes=4):
    # mesh interpolation (lagrage method)
    #
    # We define a bundle of tightly-coupled functions that perform the tasks of mapping
    # real-space positions <=> fixed-size grid in a stateless way.
    #
    # compute_weights: Performs the "setup" of the grid; for each real-space point,
    #   we determine where in the grid a non-zero contribution is recorded and which
    #   weight is assigned. We do *not* actually instantiate the grid, we just return
    #   its specification. This returns a "state" of the grid that would otherwise be
    #   managed by a stateful class.
    # points_to_mesh: Uses this information to interpolate a scalar at each position
    #   onto the grid. This instantiates an "actual" grid.
    # mesh_to_points: Does the reverse, interpolating back to positions.
    #
    # Naturally, the latter two functions require the mesh "specification" produced
    # by compute_weights to function!

    assert interpolation_nodes == 4, "interpolation_nodes != not yet implemented"
    even = interpolation_nodes % 2 == 0

    def compute_weights(inverse_cell, positions, ns):
        # inverse_cell: *not* the reciprocal cell, just inv(cell)
        # positions: real-space positions where we want to interpolate. points_to_mesh
        #   must be called with a matching shape!
        # ns: the shape of this array determines the grid shape. Values are not used.

        ns_array = jnp.array(ns.shape)
        positions_rel = ns_array * jnp.einsum("na, aA->nA", positions, inverse_cell)

        if even:
            # For Lagrange interpolation, when the order is odd, the relative position
            # of a charge is the midpoint of the two nearest gridpoints.
            # For P3M, the same is true for even orders.
            positions_rel_idx = jnp.floor(positions_rel)
            offsets = positions_rel - (positions_rel_idx + 1 / 2)
        else:
            # For Lagrange interpolation, when the order is even, the relative position
            # of a charge is the nearest gridpoint. For P3M, the same is true for
            # odd orders.
            positions_rel_idx = jnp.round(positions_rel)
            offsets = positions_rel - positions_rel_idx

        # compute interpolation weights
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
                (positions_rel_idx + i) % ns_array
                for i in range(
                    1 - (interpolation_nodes + 1) // 2,
                    1 + interpolation_nodes // 2,
                )
            ],
            axis=0,
        ).astype(int)

        # generate shifts for x, y, z axes and flatten for indexing
        x_shifts, y_shifts, z_shifts = jnp.meshgrid(
            jnp.arange(interpolation_nodes),
            jnp.arange(interpolation_nodes),
            jnp.arange(interpolation_nodes),
            indexing="ij",
        )
        x_shifts = x_shifts.flatten()
        y_shifts = y_shifts.flatten()
        z_shifts = z_shifts.flatten()

        # generate a flattened representation of all the indices
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
            ns,
        )

    def points_to_mesh(particle_weights, mesh):
        # particle_weights: length == len(interpolation_weights)
        # mesh: output of compute_weights

        (
            interpolation_weights,
            x_shifts,
            y_shifts,
            z_shifts,
            x_indices,
            y_indices,
            z_indices,
            ns,
        ) = mesh

        rho_mesh = jnp.zeros(
            ns.shape,
            dtype=particle_weights.dtype,
        )
        rho_mesh = rho_mesh.at[x_indices, y_indices, z_indices].add(
            particle_weights
            * interpolation_weights[x_shifts, :, 0]
            * interpolation_weights[y_shifts, :, 1]
            * interpolation_weights[z_shifts, :, 2]
        )

        return rho_mesh

    def mesh_to_points(mesh_vals, mesh):
        # mesh_vals: shape matching ns.shape
        # mesh: output of compute_weights

        (
            interpolation_weights,
            x_shifts,
            y_shifts,
            z_shifts,
            x_indices,
            y_indices,
            z_indices,
            _,
        ) = mesh

        return (
            mesh_vals[x_indices, y_indices, z_indices]
            * interpolation_weights[x_shifts, :, 0]
            * interpolation_weights[y_shifts, :, 1]
            * interpolation_weights[z_shifts, :, 2]
        ).sum(axis=0)

    return compute_weights, points_to_mesh, mesh_to_points
