"""Host-side batching for `batched_tiled`.

Atoms are sum-padded per system to `⌈N_b/BM⌉·BM`, concatenated into a flat
array of length `N_pbc_total` (bucket-rounded for JIT cache stability, then
rounded up to a multiple of BM). K-vectors are rectangular `[B_pbc, K_pad, 3]`
with the same K_pad for every system — `num_k` is required in `prepare`, which
sets the per-cell K target via `lr_wavelength_for_num_k`.

The dispatch table `dispatch_table[T, 3]` enumerates the on-diagonal
`(BM × BK)` blocks of the block-diagonal atom×kvec work matrix — one triple
`(b, m_tile, k_tile)` per block, sorted in pass-2 order (outer m_tile, inner
k_tile) so consecutive `n_kvec_tiles` rows form one `(b, m_tile)` group. This
lets pass 2 collapse to `reshape + sum(axis=1)` instead of `segment_sum`; pass
1 uses `segment_sum` on segment ids `b·n_kvec_tiles + kt` (unsorted, but
`segment_sum` is order-agnostic for correctness). See `_build_dispatch_table`.

`Batch` and `NonPeriodic` are re-imported from `batched_mixed.batching`; only
the `Periodic` layout differs.
"""

import numpy as np

from collections import namedtuple

from jaxpme.batched_mixed.batching import (
    Batch,
    NonPeriodic,
    next_size,
    shrink_2d_cell,
    to_lr,
    to_structure,
)

Periodic = namedtuple(
    "Periodic",
    (
        # rectangular k-grid: [B_pbc, K_pad, 3]
        "k_grid",
        # per-pbc-system metadata
        "structure_to_structure",  # [B_pbc]   sr structure index
        "structure_mask",  # [B_pbc]   real vs padding system
        "pbc",  # [B_pbc, 3]
        # flat per-system-padded atoms (atoms padded to multiples of BM)
        "pbc_atom_off",  # [B_pbc + 1]  prefix sum
        "pbc_segment_atom",  # [N_pbc_total] system id per flat slot
        "pbc_to_flat",  # [N_pbc_total] sr atom index
        "pbc_atom_mask",  # [N_pbc_total] real vs padding atom
        # dispatch table (b, m_tile, k_tile), pass-2 ordered
        "dispatch_table",  # [T, 3]
        # tile sizes baked in at batch construction
        "BM",
        "BK",
    ),
)


__all__ = ["Batch", "NonPeriodic", "Periodic", "get_batch", "prepare", "sample_shapes"]


def sample_shapes(structure, BM=32):
    """Per-sample size contributions to a `get_batch` call with tile size BM.

    `get_batch`'s counting loop runs on this, so external batch-size planners
    (accumulating samples against a budget) can rely on the same accounting.
    Minimal valid batch sizes for a set of samples are the sums (`num_k`: max)
    plus the reserve: +1 on `num_structures` / `num_atoms` / `num_pairs`
    (padding structure, padding atom, guaranteed padding pair), +1 then
    BM-rounded on `num_atoms_pbc` (i.e. exactly sum + BM), nothing on
    `num_pairs_nonpbc` / `num_k` / `num_structures_pbc`. Note `next_size`
    clamps its minimum to 1, so integer sizes must be >= 1 even where the
    true need is 0.
    """
    lr = structure["lr"]
    is_pbc = hasattr(lr, "k_grid")
    n_atoms = len(structure["positions"])
    return {
        "n_atoms": n_atoms,
        "n_pairs": len(structure["centers"]),
        "is_pbc": is_pbc,
        "n_atoms_pbc": int(np.ceil(n_atoms / BM) * BM) if is_pbc else 0,
        "n_pairs_nonpbc": 0 if is_pbc else len(lr.centers),
        "num_k": lr.k_grid.shape[0] if is_pbc else 0,
    }


def _build_dispatch_table(pbc_atom_off, n_kvec_tiles, BM):
    """Enumerate the on-diagonal (BM × BK) blocks of the block-diagonal
    atom×kvec work matrix. Returns `dispatch_table [T, 3]` int32 with rows
    `(b, m_tile, k_tile)` in pass-2 order — outer `m_tile`, inner `k_tile`
    — so consecutive `n_kvec_tiles` rows share the same `(b, m_tile)` group.

    Pass 2 uses this directly: reshape `[T, BM] -> [M_TILES, n_kvec_tiles, BM]`
    and sum along axis 1. Pass 1 uses `segment_sum` with segment ids
    `b·n_kvec_tiles + kt` — unsorted under this layout, but `segment_sum`
    handles arbitrary orderings correctly.
    """
    B = len(pbc_atom_off) - 1
    n_mt_per_b = ((pbc_atom_off[1:] - pbc_atom_off[:-1]) // BM).astype(np.int32)  # [B]

    counts = (n_kvec_tiles * n_mt_per_b).astype(np.int32)  # triples per system
    T = int(counts.sum())

    # offset[b] = start of system b's triples in the flat table
    cumcounts = np.concatenate([[0], np.cumsum(counts[:-1])]).astype(np.int32)

    # b column: repeat each b by its triple count
    b_col = np.repeat(np.arange(B, dtype=np.int32), counts)
    # inner index within system b: ranges over [0, n_kvec_tiles · n_mt_b)
    inner = np.arange(T, dtype=np.int32) - np.repeat(cumcounts, counts)

    # outer m_tile, inner k_tile
    mt_col = inner // n_kvec_tiles
    kt_col = inner % n_kvec_tiles

    return np.stack([b_col, mt_col, kt_col], axis=1)


def get_batch(
    samples,
    num_structures=None,
    num_structures_pbc=None,
    num_atoms=None,
    num_atoms_pbc=None,
    num_pairs=None,
    num_pairs_nonpbc=None,
    num_k=None,
    BM=32,
    BK=128,
    dtype=None,
    int_dtype=int,
    strategy="powers_of_2",
):
    """Build a batch from `prepare`d samples.

    BM, BK are tile sizes baked into the batch (Python ints). They drive both
    the per-system atom padding (`⌈N_b/BM⌉·BM`) and the K_pad alignment
    (multiple of BK). Production defaults: BM=32, BK=128.

    `dtype` is the float dtype (default: inferred from `samples[0]`; required
    explicit for `samples=[]`, which yields a pure-padding batch at the given
    sizes). `int_dtype` is the dtype of the neighbor-list index arrays
    (`centers`/`others`/`cell_shifts`/`*_to_structure`); the kernel-internal
    flat-layout arrays stay int32 regardless. Per-sample accounting lives in
    `sample_shapes` (incl. the size-reserve contract).
    """
    shapes = [sample_shapes(structure, BM=BM) for structure in samples]

    num_structures = num_structures if num_structures is not None else strategy
    num_structures_pbc = num_structures_pbc if num_structures_pbc is not None else strategy
    num_atoms = num_atoms if num_atoms is not None else strategy
    num_atoms_pbc = num_atoms_pbc if num_atoms_pbc is not None else strategy
    num_pairs = num_pairs if num_pairs is not None else strategy
    num_pairs_nonpbc = num_pairs_nonpbc if num_pairs_nonpbc is not None else strategy
    num_k_strat = num_k if num_k is not None else strategy

    _num_structures = len(samples)
    _total_atoms = sum(s["n_atoms"] for s in shapes)
    _total_pairs = sum(s["n_pairs"] for s in shapes)
    _max_k = max((s["num_k"] for s in shapes), default=0)
    _total_pairs_nonpbc = sum(s["n_pairs_nonpbc"] for s in shapes)
    _total_pbc = sum(s["is_pbc"] for s in shapes)

    # outer sr_batch sizing (same scheme as batched_mixed)
    n_structures = next_size(_num_structures + 1, strategy=num_structures)
    n_atoms = next_size(_total_atoms + 1, strategy=num_atoms)
    n_pairs = next_size(_total_pairs + 1, strategy=num_pairs)
    n_pairs_nonpbc = next_size(_total_pairs_nonpbc, strategy=num_pairs_nonpbc)

    # K_pad: bucket-rounded max-K, then rounded up to a multiple of BK so the
    # kernel sees a clean tile grid. Padding rows have k=0 -> W=0, contribute
    # nothing (halfspace excludes k=0; for full-space coulomb.lr_k2(s, 0) is
    # zeroed via the coulomb() wrapper in potentials.py).
    k_size = next_size(_max_k, strategy=num_k_strat)
    K_pad = int(np.ceil(k_size / BK) * BK)

    B_pbc_padded = max(1, next_size(_total_pbc, strategy=num_structures_pbc))

    # Per-pbc-system atom slots, padded to multiples of BM.
    pbc_n_padded = [s["n_atoms_pbc"] for s in shapes if s["is_pbc"]]
    while len(pbc_n_padded) < B_pbc_padded:
        pbc_n_padded.append(0)
    N_pbc_total_min = int(sum(pbc_n_padded))
    N_pbc_total = next_size(N_pbc_total_min + 1, strategy=num_atoms_pbc)
    # ensure multiple of BM so the kernel's M_TILES = N_pbc_total // BM is exact
    N_pbc_total = int(np.ceil(N_pbc_total / BM) * BM)
    # stash slack in the last (always-padding) pbc system
    pbc_n_padded[-1] += N_pbc_total - N_pbc_total_min

    pbc_atom_off = np.zeros(B_pbc_padded + 1, dtype=np.int32)
    pbc_atom_off[1:] = np.cumsum(pbc_n_padded)
    assert int(pbc_atom_off[-1]) == N_pbc_total

    padding_atom_idx = _total_atoms
    padding_structure_idx = n_structures - 1
    if dtype is None:
        if not samples:
            raise ValueError("empty `samples` requires an explicit `dtype`")
        dtype = samples[0]["positions"].dtype

    # sr_batch arrays
    charges = np.zeros(n_atoms, dtype=dtype)
    positions = np.zeros((n_atoms, 3), dtype=dtype)
    cell = np.zeros((n_structures, 3, 3), dtype=dtype)
    cell[:] = np.eye(3)
    effective_cell = cell.copy()
    pbc_rows = np.zeros((n_structures, 3), dtype=bool)
    smearing = np.ones(n_structures, dtype=dtype)
    centers = np.full(n_pairs, padding_atom_idx, dtype=int_dtype)
    others = np.full(n_pairs, padding_atom_idx, dtype=int_dtype)
    cell_shifts = np.zeros((n_pairs, 3), dtype=int_dtype)
    atom_to_structure = np.full(n_atoms, padding_structure_idx, dtype=int_dtype)
    pair_to_structure = np.full(n_pairs, padding_structure_idx, dtype=int_dtype)
    structure_mask = np.zeros(n_structures, dtype=bool)
    pbc_mask = np.zeros(n_structures, dtype=bool)
    atom_mask = np.zeros(n_atoms, dtype=bool)
    pair_mask = np.zeros(n_pairs, dtype=bool)

    # nonperiodic
    nonpbc_centers = np.full(n_pairs_nonpbc, padding_atom_idx, dtype=int_dtype)
    nonpbc_others = np.full(n_pairs_nonpbc, padding_atom_idx, dtype=int_dtype)
    nonpbc_pair_mask = np.zeros(n_pairs_nonpbc, dtype=bool)

    # periodic
    pbc_kgrid = np.zeros((B_pbc_padded, K_pad, 3), dtype=dtype)
    pbc_structure_to_structure = np.full(
        B_pbc_padded, padding_structure_idx, dtype=int_dtype
    )
    pbc_structure_mask = np.zeros(B_pbc_padded, dtype=bool)
    pbc_vectors = np.zeros((B_pbc_padded, 3), dtype=bool)
    pbc_segment_atom = np.zeros(N_pbc_total, dtype=np.int32)
    pbc_to_flat = np.full(N_pbc_total, padding_atom_idx, dtype=np.int32)
    pbc_atom_mask_flat = np.zeros(N_pbc_total, dtype=bool)
    # system id per slot — assigned even on padding rows for self-consistency
    for b in range(B_pbc_padded):
        s, e = int(pbc_atom_off[b]), int(pbc_atom_off[b + 1])
        pbc_segment_atom[s:e] = b

    atom_offset = 0
    pair_offset = 0
    nonpbc_offset = 0
    pbc_idx = 0
    for idx, structure in enumerate(samples):
        lr = structure["lr"]
        is_periodic = hasattr(lr, "k_grid")

        num_n = len(structure["positions"])
        num_p = len(structure["centers"])

        atom_slice = slice(atom_offset, atom_offset + num_n)
        pair_slice = slice(pair_offset, pair_offset + num_p)

        charges[atom_slice] = structure["charges"]
        positions[atom_slice] = structure["positions"]
        cell[idx] = structure["cell"]
        effective_cell[idx] = structure.get("effective_cell", structure["cell"])
        pbc_rows[idx] = structure["pbc"]
        centers[pair_slice] = structure["centers"] + atom_offset
        others[pair_slice] = structure["others"] + atom_offset
        cell_shifts[pair_slice] = structure["cell_shifts"]

        atom_to_structure[atom_slice] = idx
        pair_to_structure[pair_slice] = idx
        structure_mask[idx] = True
        atom_mask[atom_slice] = True
        pair_mask[pair_slice] = True

        if is_periodic:
            pbc_mask[idx] = True
            smearing[idx] = structure["smearing"]
            pbc_structure_to_structure[pbc_idx] = idx
            pbc_structure_mask[pbc_idx] = True
            pbc_vectors[pbc_idx] = structure["pbc"]

            n_k = lr.k_grid.shape[0]
            pbc_kgrid[pbc_idx, :n_k] = lr.k_grid

            s = int(pbc_atom_off[pbc_idx])
            pbc_to_flat[s : s + num_n] = np.arange(atom_offset, atom_offset + num_n)
            pbc_atom_mask_flat[s : s + num_n] = True

            pbc_idx += 1
        else:
            num_nonpbc = len(lr.centers)
            nonpbc_slice = slice(nonpbc_offset, nonpbc_offset + num_nonpbc)
            smearing[idx] = 1.0
            nonpbc_centers[nonpbc_slice] = lr.centers + atom_offset
            nonpbc_others[nonpbc_slice] = lr.others + atom_offset
            nonpbc_pair_mask[nonpbc_slice] = True
            nonpbc_offset += num_nonpbc

        atom_offset += num_n
        pair_offset += num_p

    n_kvec_tiles = K_pad // BK
    dispatch_table = _build_dispatch_table(pbc_atom_off, n_kvec_tiles, BM)

    sr_batch = Batch(
        positions=positions,
        cell=cell,
        effective_cell=effective_cell,
        pbc=pbc_rows,
        smearing=smearing,
        centers=centers,
        others=others,
        cell_shifts=cell_shifts,
        atom_mask=atom_mask,
        pair_mask=pair_mask,
        structure_mask=structure_mask,
        pbc_mask=pbc_mask,
        atom_to_structure=atom_to_structure,
        pair_to_structure=pair_to_structure,
        distances=None,
    )
    periodic_batch = Periodic(
        k_grid=pbc_kgrid,
        structure_to_structure=pbc_structure_to_structure,
        structure_mask=pbc_structure_mask,
        pbc=pbc_vectors,
        pbc_atom_off=pbc_atom_off,
        pbc_segment_atom=pbc_segment_atom,
        pbc_to_flat=pbc_to_flat,
        pbc_atom_mask=pbc_atom_mask_flat,
        dispatch_table=dispatch_table,
        # BM/BK are encoded as the shape of a 1-D dummy array so the values
        # survive `jax.device_put` / pytree machinery as static shape
        # metadata (Python ints stored in a namedtuple field would be
        # promoted to 0-D jax arrays by `device_put`, and the tiled kernel
        # needs them as static `lax.dynamic_slice` size arguments). The
        # values are never read — only `.shape[0]` matters — so `bool` is
        # the smallest and least-surprising dtype choice.
        BM=np.zeros(BM, dtype=bool),
        BK=np.zeros(BK, dtype=bool),
    )
    nonperiodic_batch = NonPeriodic(
        centers=nonpbc_centers,
        others=nonpbc_others,
        pair_mask=nonpbc_pair_mask,
    )

    return charges, sr_batch, nonperiodic_batch, periodic_batch


def prepare(atoms, num_k, cutoff=None, smearing=None, halfspace=True, dtype=np.float64):
    """Per-structure preprocessing.

    `num_k` fixes the reciprocal grid via `lr_wavelength_for_num_k` (K_b counts
    vary slightly across cells from axis-rounding in `get_kgrid_ewald_shape`;
    the batcher max-pads to a common K_pad). The real-space `cutoff` and
    `smearing` follow it, matching `batched_mixed`: when omitted,
    `cutoff = lr_wavelength · 8` and `smearing = lr_wavelength · 2`, keeping real
    and reciprocal space balanced. An explicit `cutoff` overrides only the
    real-space radius — `smearing` still tracks `num_k` — so a cutoff below
    `lr_wavelength · 8` under-converges the real-space sum unless `smearing` is
    pinned too.
    """
    from jaxpme.kspace import lr_wavelength_for_num_k

    cell = atoms.get_cell().array.astype(dtype)
    pbc = atoms.get_pbc()

    # for 2D PBC: shrink non-periodic cell vector before deriving Ewald params
    if pbc.sum() == 2:
        positions = atoms.get_positions().astype(dtype)
        effective_cell = shrink_2d_cell(cell, pbc, positions)
    else:
        effective_cell = cell

    if pbc.any():
        lr_wavelength = lr_wavelength_for_num_k(effective_cell, num_k)
        if cutoff is None:
            cutoff = lr_wavelength * 8.0
        if smearing is None:
            smearing = lr_wavelength * 2.0
        structure = to_structure(atoms, cutoff, dtype=dtype)
    else:
        # non-pbc real space is the bare 1/r sum over *all* pairs (built in
        # to_lr), so the cutoff neighbor list would only be masked off here.
        # Skip it (cutoff=None -> empty list) instead of carrying dead pairs.
        lr_wavelength = None
        structure = to_structure(atoms, cutoff=None, dtype=dtype)
    # cell stays raw; the shrunk cell travels alongside (jaxpme.utils.compose_cell)
    if pbc.any():
        structure["effective_cell"] = effective_cell

    smearing_out, lr = to_lr(structure, lr_wavelength, smearing, halfspace=halfspace)

    structure["lr"] = lr
    if smearing_out is not None:
        structure["smearing"] = smearing_out

    return structure
