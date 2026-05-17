"""Batched Ewald calculator backed by a tile-dispatched XLA kernel.

Same API surface as `jaxpme.batched_mixed.calculators.Ewald`: returns a
`Calculator` namedtuple with `prepare`, `potentials`, `energy`, `energy_forces`,
`energy_forces_stress`. The reciprocal sum routes through
`phi_recip_xla_vmap` over the sum-padded flat-atom layout produced by
`batching.get_batch`. Stress flows naturally through autograd (no custom_vjp,
no `stop_gradient` on `kvec` / `W`).
"""

import jax
import jax.numpy as jnp

from jaxpme.calculators import Calculator
from jaxpme.kspace import (
    get_reciprocal,
    slab_energy_per_atom,
    slab_geometry,
)
from jaxpme.potentials import potential
from jaxpme.utils import get_distances, get_volume, safe_norm

from .kernel import phi_recip_xla_vmap
from .kspace import generate_ewald_kvectors


def Ewald(
    exponent=1,
    prefactor=1.0,
    custom_potential=None,
    halfspace=True,
):
    """Tile-dispatched batched Ewald calculator.

    Tile sizes BM, BK live on `prepare`, not here — they are batch-construction
    parameters, not calculator parameters.
    """
    from jaxpme.solvers import ewald

    pot = potential(
        exponent=exponent,
        exclusion_radius=None,
        custom_potential=custom_potential,
    )
    solver = ewald(pot, full_neighbor_list=True, halfspace=halfspace)
    g_factor = 2 if halfspace else 1

    # raw potential for correction-side dispatch (Coulomb gets 2D slab,
    # inverse-power-law does not — returns NaN for 2D PBC)
    from jaxpme.potentials import coulomb, inverse_power_law

    if custom_potential is not None:
        _raw = custom_potential
        _is_coulomb = False
    elif exponent == 1:
        _raw = coulomb()
        _is_coulomb = True
    else:
        _raw = inverse_power_law(exponent)
        _is_coulomb = False

    def _kspace_setup(sr_batch, batch_pbc):
        """Materialise per-system (kvec [B, K_pad, 3], W [B, K_pad]) plus the
        per-system metadata used by the kernel and corrections. No
        `stop_gradient` — `kvec`/`W` depend on `cell` via `reciprocal_cell`,
        and we want stress to flow through.
        """
        cell_pbc = sr_batch.cell[batch_pbc.structure_to_structure]
        reciprocal_cell = jax.vmap(get_reciprocal)(cell_pbc)
        volume_pbc = jax.vmap(get_volume)(cell_pbc)
        smearing_pbc = sr_batch.smearing[batch_pbc.structure_to_structure]
        pbc_pbc = batch_pbc.pbc

        kvec = jax.vmap(generate_ewald_kvectors)(reciprocal_cell, batch_pbc.k_grid)
        k2 = (kvec**2).sum(axis=-1)
        G = jax.vmap(pot.lr)(smearing_pbc, k2)
        W = G * g_factor / volume_pbc[:, None]

        # BM/BK survive the prefetch / device_put pipeline as dummy-array
        # shape metadata (see `batched_tiled.batching.get_batch`). Reading
        # `.shape[0]` recovers the static Python int needed by the tiled
        # kernel's `lax.dynamic_slice` size arguments.
        n_kvec_tiles = batch_pbc.k_grid.shape[1] // batch_pbc.BK.shape[0]
        atom_off = jnp.asarray(batch_pbc.pbc_atom_off)
        pass1_flat = jnp.asarray(batch_pbc.pass1_flat)
        pass2_flat = jnp.asarray(batch_pbc.pass2_flat)

        return (
            kvec,
            W,
            cell_pbc,
            volume_pbc,
            smearing_pbc,
            pbc_pbc,
            atom_off,
            pass1_flat,
            pass2_flat,
            batch_pbc.BM.shape[0],
            batch_pbc.BK.shape[0],
            n_kvec_tiles,
        )

    def _kspace_corrections(
        charges,
        sr_batch,
        batch_pbc,
        cell_pbc,
        volume_pbc,
        smearing_pbc,
        pbc_pbc,
        N_all,
    ):
        """Self + background + 2D-slab corrections, on the flat sum-padded layout."""
        B_pbc = batch_pbc.structure_mask.shape[0]
        seg = batch_pbc.pbc_segment_atom
        mask = batch_pbc.pbc_atom_mask
        q_pbc = charges[batch_pbc.pbc_to_flat] * mask
        r_pbc = sr_batch.positions[batch_pbc.pbc_to_flat] * mask[:, None]

        cs_per_b = _raw.correction_self(smearing_pbc)
        c_self = -q_pbc * cs_per_b[seg]

        cb_per_b = _raw.correction_background(smearing_pbc)
        charge_tot = jax.ops.segment_sum(q_pbc, seg, num_segments=B_pbc)
        c_back = -2.0 * cb_per_b[seg] * (charge_tot / volume_pbc)[seg]

        is_2d = jnp.sum(pbc_pbc, axis=-1) == 2
        if _is_coulomb:
            n_hat, basis_len = jax.vmap(slab_geometry)(cell_pbc, pbc_pbc)  # [B, 3], [B]
            z_atom = jnp.sum(r_pbc * n_hat[seg], axis=-1)
            M_axis = jax.ops.segment_sum(q_pbc * z_atom, seg, num_segments=B_pbc)
            M_axis_sq = jax.ops.segment_sum(q_pbc * z_atom**2, seg, num_segments=B_pbc)
            E_slab = slab_energy_per_atom(
                z_atom, M_axis[seg], M_axis_sq[seg], charge_tot[seg], basis_len[seg]
            )
            slab_term = jnp.where(is_2d[seg], E_slab / volume_pbc[seg], 0.0)
        else:
            # IPLP / custom: 2D-PBC correction is not implemented -> NaN
            slab_term = jnp.where(is_2d[seg], jnp.full_like(c_self, jnp.nan), 0.0)

        c_total_pbc = (c_self + c_back + slab_term) * mask
        return jax.ops.segment_sum(c_total_pbc, batch_pbc.pbc_to_flat, num_segments=N_all)

    def kspace_fn(charges, sr_batch, batch_pbc):
        """Full kspace contribution per atom (kernel + corrections), /2."""
        N_all = charges.shape[0]
        (
            kvec,
            W,
            cell_pbc,
            volume_pbc,
            smearing_pbc,
            pbc_pbc,
            atom_off,
            pass1_flat,
            pass2_flat,
            BM_use,
            BK_use,
            n_kvec_tiles,
        ) = _kspace_setup(sr_batch, batch_pbc)

        r_pbc = sr_batch.positions[batch_pbc.pbc_to_flat]
        q_pbc = charges[batch_pbc.pbc_to_flat] * batch_pbc.pbc_atom_mask

        phi_pbc = phi_recip_xla_vmap(
            r_pbc,
            q_pbc,
            kvec,
            W,
            atom_off,
            pass1_flat,
            pass2_flat,
            BM=BM_use,
            BK=BK_use,
            n_kvec_tiles=n_kvec_tiles,
        )
        phi_pbc = phi_pbc * batch_pbc.pbc_atom_mask
        phi_full = jax.ops.segment_sum(phi_pbc, batch_pbc.pbc_to_flat, num_segments=N_all)
        corr = _kspace_corrections(
            charges,
            sr_batch,
            batch_pbc,
            cell_pbc,
            volume_pbc,
            smearing_pbc,
            pbc_pbc,
            N_all,
        )
        return (phi_full + corr) / 2

    def potentials_fn(charges, batch, batch_nopbc, batch_pbc):
        N_all = charges.shape[0]
        pbc_mask_atom = batch.pbc_mask[batch.atom_to_structure]
        charges = charges * batch.atom_mask

        if batch.distances is None:
            r = get_distances(
                batch.cell[batch.pair_to_structure],
                batch.positions[batch.centers],
                batch.positions[batch.others],
                batch.cell_shifts,
            )
            r = r * batch.pair_mask
        else:
            r = batch.distances

        extra_r = (
            safe_norm(
                batch.positions[batch_nopbc.others] - batch.positions[batch_nopbc.centers],
                axis=-1,
            )
            * batch_nopbc.pair_mask
        )

        # real-space, non-PBC: bare 1/r between every pair
        real_space = jax.ops.segment_sum(
            charges[batch_nopbc.others] * pot.real(extra_r),
            batch_nopbc.centers,
            num_segments=N_all,
        )
        real_space += jax.ops.segment_sum(
            charges[batch_nopbc.centers] * pot.real(extra_r),
            batch_nopbc.others,
            num_segments=N_all,
        )
        real_space = (real_space * batch.atom_mask * ~pbc_mask_atom) / 2

        # real-space, PBC: range-separated short-range from neighbor list
        real_space += (
            solver.rspace(
                batch.smearing[batch.pair_to_structure],
                charges,
                r,
                batch.centers,
                batch.others,
            )
            * batch.atom_mask
            * pbc_mask_atom
        )

        # reciprocal-space
        k_space = kspace_fn(charges, batch, batch_pbc) * batch.atom_mask * pbc_mask_atom

        return (real_space + k_space) * prefactor

    def energy_fn(charges, batch, batch_nopbc, batch_pbc):
        potentials = potentials_fn(charges, batch, batch_nopbc, batch_pbc)
        energies = charges * potentials
        return (
            jax.ops.segment_sum(
                energies, batch.atom_to_structure, num_segments=batch.cell.shape[0]
            )
            * batch.structure_mask
        )

    def energy_and_forces_fn(charges, batch, batch_nopbc, batch_pbc):
        assert batch.distances is None

        def total_energy_fn(charges, batch, batch_nopbc, batch_pbc):
            energies = energy_fn(charges, batch, batch_nopbc, batch_pbc)
            return energies.sum(), energies

        (_, energy), grads = jax.value_and_grad(
            total_energy_fn, argnums=1, has_aux=True, allow_int=True
        )(charges, batch, batch_nopbc, batch_pbc)
        forces = -grads.positions * batch.atom_mask[:, None]
        return energy, forces

    def energy_and_forces_and_stress_fn(charges, batch, batch_nopbc, batch_pbc):
        assert batch.distances is None

        def total_energy_fn(charges, batch, batch_nopbc, batch_pbc):
            energies = energy_fn(charges, batch, batch_nopbc, batch_pbc)
            return energies.sum(), energies

        (_, energy), grads = jax.value_and_grad(
            total_energy_fn, argnums=1, has_aux=True, allow_int=True
        )(charges, batch, batch_nopbc, batch_pbc)
        forces = -grads.positions * batch.atom_mask[:, None]
        # σ_αβ = (1/V) Σ_i r_iα ∂E/∂r_iβ + (cell ∂E/∂cell)_αβ — the standard
        # virial form, split into per-atom and per-cell pieces.
        stress = (
            jax.ops.segment_sum(
                jnp.einsum("ia,ib->iab", batch.positions, grads.positions),
                batch.atom_to_structure,
                num_segments=batch.cell.shape[0],
            )
            + jnp.einsum("sAa,sAb->sab", batch.cell, grads.cell)
        ) * batch.structure_mask[:, None, None]
        return energy, forces, stress

    def prepare_fn(
        atomss,
        num_k,
        cutoff,
        smearing=None,
        BM=32,
        BK=128,
    ):
        if not halfspace:
            # full-k-space path still works; flag kept for parity with
            # batched_mixed where `num_k` was halfspace-only.
            pass

        from .batching import get_batch, prepare

        return get_batch(
            [
                prepare(
                    atoms,
                    num_k=num_k,
                    cutoff=cutoff,
                    smearing=smearing,
                    halfspace=halfspace,
                )
                for atoms in atomss
            ],
            BM=BM,
            BK=BK,
        )

    return Calculator(
        prepare_fn,
        potentials_fn,
        energy_fn,
        energy_and_forces_fn,
        energy_and_forces_and_stress_fn,
    )
