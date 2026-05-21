"""Reciprocal-Ewald kernel: pure JAX, tile-dispatched, vmap + segment_sum / reshape-sum.

Two passes over (BM × BK) work tiles.

Pass 1 builds structure factors per k-tile:

    S^r_k = Σ_j q_j cos(k · r_j)
    S^i_k = Σ_j q_j sin(k · r_j)

Pass 2 builds per-atom potentials:

    φ_i = Σ_k W_k [cos(k · r_i) · S^r_k + sin(k · r_i) · S^i_k]

The dispatch table `dispatch_table [T, 3]` rows are `(b, m_tile, k_tile)`,
ordered outer-m_tile / inner-k_tile. This enumerates the on-diagonal blocks
of the block-diagonal atom×kvec work matrix (one block per (system, m_tile,
k_tile) triple, with variable atom-tile count per system from sum-padding).
See `batching.py:_build_dispatch_table` for construction.

Cross-system blocks (system A's atoms × system B's k-vecs) are never enumerated.
The k-axis is rectangular: every system has the same `n_kvec_tiles = K_pad/BK`
tiles, so per-system k-vec offset is implicit (`b · K_pad`).

`vmap(per_triple)` produces per-tile partials. Pass 1 reduces those partials
across atom-tiles within each `(b, k_tile)` segment via `segment_sum` (segment
ids are unsorted under this layout but that's fine — `segment_sum` is
order-agnostic). Pass 2 reduces across k-tiles within each `(b, m_tile)`
segment via `reshape + sum(axis=1)`, since the table's outer-m_tile ordering
makes those segments exactly `n_kvec_tiles`-row contiguous chunks. Stress flows
through naturally because `kvec` and `W` cotangents are not blocked.
"""

import jax
import jax.numpy as jnp
from jax import lax


def phi_recip_xla_vmap(
    r,
    q,
    kvec,
    W,
    atom_off,
    dispatch_table,
    *,
    BM,
    BK,
    n_kvec_tiles,
):
    """Per-atom reciprocal Ewald potential on the sum-padded flat layout.

    Inputs:
      r [N_total, 3]              flat atom positions (sum-padded)
      q [N_total]                 flat charges (already masked)
      kvec [B, K_pad, 3]          per-system k-vectors (rectangular, padding rows = 0)
      W [B, K_pad]                per-system weights G·g_factor/V (padding rows = 0)
      atom_off [B+1]              prefix sum over per-system padded atom counts
      dispatch_table [T, 3]       (b, m_tile, k_tile) triples, pass-2 ordered
      BM, BK, n_kvec_tiles        compile-time tile sizes / k-tile count

    Returns:
      phi [N_total]               per-atom reciprocal potential (no 1/2 factor)
    """
    dtype = r.dtype
    N_total = r.shape[0]
    idx_dtype = atom_off.dtype
    ZERO = jnp.asarray(0, idx_dtype)
    BMc = jnp.asarray(BM, idx_dtype)
    BKc = jnp.asarray(BK, idx_dtype)

    K_total = kvec.shape[0] * kvec.shape[1]  # = B * K_pad

    # ---- Pass 1: structure factors ----
    def per_triple1(triple):
        b, mt, kt = triple[0], triple[1], triple[2]
        a_offset = atom_off[b] + mt * BMc
        kv = lax.dynamic_slice(kvec[b], (kt * BKc, ZERO), (BK, 3))
        rt = lax.dynamic_slice(r, (a_offset, ZERO), (BM, 3))
        qt = lax.dynamic_slice(q, (a_offset,), (BM,))
        theta = rt @ kv.T
        c = jnp.cos(theta)
        s = jnp.sin(theta)
        return ((qt[:, None] * c).sum(0), (qt[:, None] * s).sum(0))

    Sr_per, Si_per = jax.vmap(per_triple1)(dispatch_table)  # [T, BK] each

    # Pass 1 segments by (b, kt). Rectangular K → segment id = b·n_kvec_tiles + kt
    # without a kvec_off gather. Ids are unsorted under pass-2 ordering, but
    # `segment_sum` is order-agnostic.
    K_TILES = K_total // BK
    group_p1 = (dispatch_table[:, 0] * n_kvec_tiles + dispatch_table[:, 2]).astype(
        jnp.int32
    )
    Sr_grouped = jax.ops.segment_sum(Sr_per, group_p1, num_segments=K_TILES)
    Si_grouped = jax.ops.segment_sum(Si_per, group_p1, num_segments=K_TILES)
    Sr = Sr_grouped.reshape(kvec.shape[0], kvec.shape[1])  # [B, K_pad]
    Si = Si_grouped.reshape(kvec.shape[0], kvec.shape[1])

    # ---- Pass 2: per-atom potentials ----
    def per_triple2(triple):
        b, mt, kt = triple[0], triple[1], triple[2]
        a_offset = atom_off[b] + mt * BMc
        kv = lax.dynamic_slice(kvec[b], (kt * BKc, ZERO), (BK, 3))
        Wt = lax.dynamic_slice(W[b], (kt * BKc,), (BK,))
        Sr_t = lax.dynamic_slice(Sr[b], (kt * BKc,), (BK,))
        Si_t = lax.dynamic_slice(Si[b], (kt * BKc,), (BK,))
        rt = lax.dynamic_slice(r, (a_offset, ZERO), (BM, 3))
        theta = rt @ kv.T
        c = jnp.cos(theta)
        s = jnp.sin(theta)
        return (c * (Wt * Sr_t)[None, :] + s * (Wt * Si_t)[None, :]).sum(1)

    phi_per = jax.vmap(per_triple2)(dispatch_table)  # [T, BM]
    # Outer-m_tile / inner-k_tile ordering → each global m_tile is exactly
    # `n_kvec_tiles` consecutive rows. Reshape + sum, no segment_sum.
    M_TILES = N_total // BM
    phi_grouped = phi_per.reshape(M_TILES, n_kvec_tiles, BM).sum(axis=1)
    return phi_grouped.reshape(N_total).astype(dtype)
