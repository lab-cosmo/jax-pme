"""Reciprocal-Ewald kernel: pure JAX, tile-dispatched, vmap + segment_sum.

Two passes over (BM × BK) work tiles.

Pass 1 builds structure factors per k-tile:

    S^r_k = Σ_j q_j cos(k · r_j)
    S^i_k = Σ_j q_j sin(k · r_j)

Pass 2 builds per-atom potentials:

    φ_i = Σ_k W_k [cos(k · r_i) · S^r_k + sin(k · r_i) · S^i_k]

The dispatch tables `pass1_flat[T1, 3]` (b, kt, j_atom) and `pass2_flat[T2, 3]`
(b, mt, j_kvec) enumerate the on-diagonal (BM × BK) blocks of the block-diagonal
atom×kvec work matrix — one block per (system, k-tile, atom-tile) triple,
varying in count per system because atoms are sum-padded with variable N_b.
See `batching.py:_build_dispatch_tables` for the construction.

Cross-system blocks (system A's atoms × system B's k-vecs) are never enumerated.
The k-axis is rectangular: every system has the same `n_kvec_tiles = K_pad/BK`
tiles, so per-system k-vec offset is implicit (`b · K_pad`).

`vmap(per_triple)` produces per-tile partials; `segment_sum` glues them into
the global `S^r/S^i` and `φ` arrays. Stress flows through naturally because
`kvec` and `W` cotangents are not blocked.
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
    pass1_flat,
    pass2_flat,
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
      pass1_flat [T1, 3]          (b, kt, j_atom_tile) triples
      pass2_flat [T2, 3]          (b, mt, j_kvec_tile) triples
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
        b, kt, j = triple[0], triple[1], triple[2]
        a_offset = atom_off[b] + j * BMc
        # kvec[b]: [K_pad, 3] — gather then dynamic-slice the k-tile
        kv = lax.dynamic_slice(kvec[b], (kt * BKc, ZERO), (BK, 3))
        rt = lax.dynamic_slice(r, (a_offset, ZERO), (BM, 3))
        qt = lax.dynamic_slice(q, (a_offset,), (BM,))
        theta = rt @ kv.T
        c = jnp.cos(theta)
        s = jnp.sin(theta)
        return ((qt[:, None] * c).sum(0), (qt[:, None] * s).sum(0))

    Sr_per, Si_per = jax.vmap(per_triple1)(pass1_flat)  # [T1, BK] each

    # Group by global k-tile id (= b * n_kvec_tiles + kt). With rectangular K
    # this is one multiply-add instead of a kvec_off gather.
    K_TILES = K_total // BK
    group_p1 = (pass1_flat[:, 0] * n_kvec_tiles + pass1_flat[:, 1]).astype(jnp.int32)
    Sr_grouped = jax.ops.segment_sum(Sr_per, group_p1, num_segments=K_TILES)
    Si_grouped = jax.ops.segment_sum(Si_per, group_p1, num_segments=K_TILES)
    Sr = Sr_grouped.reshape(kvec.shape[0], kvec.shape[1])  # [B, K_pad]
    Si = Si_grouped.reshape(kvec.shape[0], kvec.shape[1])

    # ---- Pass 2: per-atom potentials ----
    def per_triple2(triple):
        b, mt, j = triple[0], triple[1], triple[2]
        a_offset = atom_off[b] + mt * BMc
        kv = lax.dynamic_slice(kvec[b], (j * BKc, ZERO), (BK, 3))
        Wt = lax.dynamic_slice(W[b], (j * BKc,), (BK,))
        Sr_t = lax.dynamic_slice(Sr[b], (j * BKc,), (BK,))
        Si_t = lax.dynamic_slice(Si[b], (j * BKc,), (BK,))
        rt = lax.dynamic_slice(r, (a_offset, ZERO), (BM, 3))
        theta = rt @ kv.T
        c = jnp.cos(theta)
        s = jnp.sin(theta)
        return (c * (Wt * Sr_t)[None, :] + s * (Wt * Si_t)[None, :]).sum(1)

    phi_per = jax.vmap(per_triple2)(pass2_flat)  # [T2, BM]
    M_TILES = N_total // BM
    group_p2 = (atom_off[pass2_flat[:, 0]] // BMc + pass2_flat[:, 1]).astype(jnp.int32)
    phi_grouped = jax.ops.segment_sum(phi_per, group_p2, num_segments=M_TILES)
    return phi_grouped.reshape(N_total).astype(dtype)
