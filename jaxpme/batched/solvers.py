import jax
import jax.numpy as jnp


def ewald(potential):
    def _ewald(
        # per system
        reciprocal_cell,
        volume,
        smearing,
        charges,
        system_mask,
        # per atom
        positions,
        atom_to_system,
        atom_mask,
        periodic_atom_mask,  # True if periodic
        # per pair (within cutoff)
        r,
        i,
        j,
        pair_to_system,
        # per pair (outside cutoff, non-pbc)
        extra_r,
        extra_i,
        extra_j,
        extra_pair_mask,
        # per k-vector
        kgrid,
        k_to_system,
        k_mask,
        # per k,i (trig)
        trig_to_atom,
        trig_to_k,
    ):
        N = charges.shape[0]
        K = k_mask.shape[0]

        # sr contributions for non-periodic systems
        non_periodic = jax.ops.segment_sum(
            charges[j] * potential.real(r),
            i,
            num_segments=N,
        )

        # lr contributions for non-periodic systems
        non_periodic += jax.ops.segment_sum(
            charges[extra_j] * potential.real(extra_r) * extra_pair_mask,
            extra_i,
            num_segments=N,
        )

        non_periodic *= ~periodic_atom_mask * atom_mask

        # real space contributions for periodic systems
        periodic = jax.ops.segment_sum(
            charges[j] * potential.sr(smearing[pair_to_system], r),
            i,
            num_segments=N,
        )

        # reciprocal-space contributions for periodic systems
        k_vectors = jnp.zeros_like(kgrid)
        for i in range(3):
            k_vectors += kgrid[:, i, None] * reciprocal_cell[k_to_system][:, i]

        k2 = jax.lax.square(k_vectors).sum(axis=-1)
        G = jax.vmap(potential.lr)(smearing[k_to_system], k2)

        trigs = (k_vectors[trig_to_k] * positions[trig_to_atom]).sum(axis=-1)

        c = jnp.cos(trigs)
        s = jnp.sin(trigs)

        C = jax.ops.segment_sum(
            c * charges[trig_to_atom],
            trig_to_k,
            num_segments=K,
        )
        S = jax.ops.segment_sum(
            s * charges[trig_to_atom],
            trig_to_k,
            num_segments=K,
        )

        pots = jax.ops.segment_sum(
            (G * C)[trig_to_k] * c + (G * S)[trig_to_k] * s,
            trig_to_atom,
            num_segments=N,
        )

        pots /= volume[atom_to_system]
        pots += potential.batched_correction(
            smearing, charges, volume, atom_to_system, system_mask
        )
        periodic += pots

        periodic *= periodic_atom_mask * atom_mask

        return (non_periodic + periodic) / 2

    return _ewald
