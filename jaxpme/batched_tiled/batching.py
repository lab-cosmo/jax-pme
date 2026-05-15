"""Host-side batching for `batched_tiled`.

Atoms are sum-padded per system to `⌈N_b/BM⌉·BM`, concatenated into a flat
array of length `N_pbc_total` (bucket-rounded for JIT cache stability, then
rounded up to a multiple of BM). K-vectors are rectangular `[B_pbc, K_pad, 3]`
with the same K_pad for every system — `num_k` is required in `prepare`, which
sets the per-cell K target via `lr_wavelength_for_num_k`.

The dispatch tables `pass1_flat[T, 3]` and `pass2_flat[T, 3]` enumerate the
on-diagonal `(BM × BK)` blocks of the block-diagonal atom×kvec work matrix —
one triple `(b, k-tile, atom-tile)` per block. See `_build_dispatch_tables`.

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
        # dispatch tables (b, k-tile, atom-tile) / (b, atom-tile, k-tile)
        "pass1_flat",  # [T, 3]
        "pass2_flat",  # [T, 3]
        # tile sizes baked in at batch construction
        "BM",
        "BK",
    ),
)


__all__ = ["Batch", "NonPeriodic", "Periodic", "get_batch", "prepare"]


def _build_dispatch_tables(pbc_atom_off, n_kvec_tiles, BM):
    """Enumerate the on-diagonal (BM × BK) blocks of the block-diagonal
    atom×kvec work matrix. Returns `(pass1_flat, pass2_flat)`, each `[T, 3]`
    int32 with rows `(b, k-tile, atom-tile)` and `(b, atom-tile, k-tile)`.

    Same set of triples in both tables, reordered for the segment_sum group
    layout of each pass:
      pass 1 segments by `(b, kt)` so contributions from different atom-tiles
        within the same (system, k-tile) group reduce to S^r/S^i;
      pass 2 segments by `(b, mt)` so contributions from different k-tiles
        within the same (system, atom-tile) group reduce to φ.
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

    n_mt_each = np.repeat(n_mt_per_b, counts)
    # pass1 within-system order: outer kt, inner j_atom
    kt_col = inner // n_mt_each
    j_atom_col = inner % n_mt_each
    pass1_flat = np.stack([b_col, kt_col, j_atom_col], axis=1)

    # pass2 within-system order: outer mt, inner j_kvec
    mt_col = inner // n_kvec_tiles
    j_kvec_col = inner % n_kvec_tiles
    pass2_flat = np.stack([b_col, mt_col, j_kvec_col], axis=1)

    return pass1_flat, pass2_flat


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
    strategy="powers_of_2",
):
    """Build a batch from `prepare`d samples.

    BM, BK are tile sizes baked into the batch (Python ints). They drive both
    the per-system atom padding (`⌈N_b/BM⌉·BM`) and the K_pad alignment
    (multiple of BK). Production defaults: BM=32, BK=128.
    """
    _num_structures = len(samples)
    _num_atoms = []
    _num_pairs = []
    _num_pairs_nonpbc = []
    _is_pbc = []
    _num_k = []

    num_structures = num_structures if num_structures is not None else strategy
    num_structures_pbc = num_structures_pbc if num_structures_pbc is not None else strategy
    num_atoms = num_atoms if num_atoms is not None else strategy
    num_atoms_pbc = num_atoms_pbc if num_atoms_pbc is not None else strategy
    num_pairs = num_pairs if num_pairs is not None else strategy
    num_pairs_nonpbc = num_pairs_nonpbc if num_pairs_nonpbc is not None else strategy
    num_k_strat = num_k if num_k is not None else strategy

    for structure in samples:
        lr = structure["lr"]
        _num_atoms.append(len(structure["positions"]))
        _num_pairs.append(len(structure["centers"]))
        if hasattr(lr, "k_grid"):
            _is_pbc.append(True)
            _num_k.append(lr.k_grid.shape[0])
        else:
            _is_pbc.append(False)
            _num_pairs_nonpbc.append(len(lr.centers))

    _num_atoms = np.array(_num_atoms)
    _num_pairs = np.array(_num_pairs)
    _num_pairs_nonpbc = np.array(_num_pairs_nonpbc)
    _is_pbc = np.array(_is_pbc)
    _num_k = np.array(_num_k)

    _total_atoms = int(_num_atoms.sum())
    _total_pairs = int(_num_pairs.sum())
    _max_k = int(_num_k.max()) if len(_num_k) > 0 else 0
    _total_pairs_nonpbc = int(_num_pairs_nonpbc.sum()) if len(_num_pairs_nonpbc) > 0 else 0
    _total_pbc = int(_is_pbc.sum())

    # outer sr_batch sizing (same scheme as batched_mixed)
    n_structures = next_size(_num_structures + 1, strategy=num_structures)
    n_atoms = next_size(_total_atoms + 1, strategy=num_atoms)
    n_pairs = next_size(_total_pairs + 1, strategy=num_pairs)
    n_pairs_nonpbc = next_size(_total_pairs_nonpbc, strategy=num_pairs_nonpbc)

    # K_pad: bucket-rounded max-K, then rounded up to a multiple of BK so the
    # kernel sees a clean tile grid. Padding rows have k=0 -> W=0, contribute
    # nothing (halfspace excludes k=0; for full-space coulomb.lr_k2(s, 0) is
    # zeroed via the coulomb() wrapper in potentials.py).
    if _max_k > 0:
        k_size = next_size(_max_k, strategy=num_k_strat)
        K_pad = int(np.ceil(k_size / BK) * BK)
    else:
        K_pad = BK  # one tile, even if no periodic systems

    B_pbc_padded = max(1, next_size(_total_pbc, strategy=num_structures_pbc))

    # Per-pbc-system atom slots, padded to multiples of BM.
    pbc_n_padded = []
    for ip, structure in enumerate(samples):
        if _is_pbc[ip]:
            n = int(_num_atoms[ip])
            pbc_n_padded.append(int(np.ceil(n / BM) * BM))
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
    dtype = samples[0]["positions"].dtype

    # sr_batch arrays
    charges = np.zeros(n_atoms, dtype=dtype)
    positions = np.zeros((n_atoms, 3), dtype=dtype)
    cell = np.zeros((n_structures, 3, 3), dtype=dtype)
    cell[:] = np.eye(3)
    smearing = np.ones(n_structures, dtype=dtype)
    centers = np.full(n_pairs, padding_atom_idx, dtype=int)
    others = np.full(n_pairs, padding_atom_idx, dtype=int)
    cell_shifts = np.zeros((n_pairs, 3), dtype=int)
    atom_to_structure = np.full(n_atoms, padding_structure_idx, dtype=int)
    pair_to_structure = np.full(n_pairs, padding_structure_idx, dtype=int)
    structure_mask = np.zeros(n_structures, dtype=bool)
    pbc_mask = np.zeros(n_structures, dtype=bool)
    atom_mask = np.zeros(n_atoms, dtype=bool)
    pair_mask = np.zeros(n_pairs, dtype=bool)

    # nonperiodic
    nonpbc_centers = np.full(n_pairs_nonpbc, padding_atom_idx, dtype=int)
    nonpbc_others = np.full(n_pairs_nonpbc, padding_atom_idx, dtype=int)
    nonpbc_pair_mask = np.zeros(n_pairs_nonpbc, dtype=bool)

    # periodic
    pbc_kgrid = np.zeros((B_pbc_padded, K_pad, 3), dtype=dtype)
    pbc_structure_to_structure = np.full(B_pbc_padded, padding_structure_idx, dtype=int)
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
    pass1_flat, pass2_flat = _build_dispatch_tables(pbc_atom_off, n_kvec_tiles, BM)

    sr_batch = Batch(
        positions=positions,
        cell=cell,
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
        pass1_flat=pass1_flat,
        pass2_flat=pass2_flat,
        BM=BM,
        BK=BK,
    )
    nonperiodic_batch = NonPeriodic(
        centers=nonpbc_centers,
        others=nonpbc_others,
        pair_mask=nonpbc_pair_mask,
    )

    return charges, sr_batch, nonperiodic_batch, periodic_batch


def prepare(atoms, num_k, cutoff, smearing=None, halfspace=True, dtype=np.float64):
    """Per-structure preprocessing.

    `num_k` is required: it sets the per-cell K target via
    `lr_wavelength_for_num_k`. Actual K_b counts vary slightly across cells
    because of axis-rounding in `get_kgrid_ewald_shape`; the batcher max-pads
    them to a common K_pad. `cutoff` is the real-space neighbor list radius.
    `smearing` defaults to `lr_wavelength · 2`.
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
        if smearing is None:
            smearing = lr_wavelength * 2.0
    else:
        lr_wavelength = None

    structure = to_structure(atoms, cutoff, dtype=dtype)
    structure["cell"] = effective_cell

    smearing_out, lr = to_lr(structure, lr_wavelength, smearing, halfspace=halfspace)

    structure["lr"] = lr
    if smearing_out is not None:
        structure["smearing"] = smearing_out

    return structure
