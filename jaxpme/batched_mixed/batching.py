import numpy as np

from collections import namedtuple

from .kspace import generate_ewald_k_grid

Batch = namedtuple(
    "Batch",
    (
        "positions",
        "centers",
        "others",
        "cell_shifts",
        "distances",
        "cell",
        "smearing",
        "atom_mask",
        "pair_mask",
        "structure_mask",
        "pbc_mask",
        "atom_to_structure",
        "pair_to_structure",
    ),
)
Periodic = namedtuple(
    "Periodic",
    ("k_grid", "atom_to_atom", "structure_to_structure", "atom_mask", "structure_mask"),
    # atom_to_atom: indices into SR batch positions [num_pbc, num_atoms_pbc]
    #               -> positions[atom_to_atom]
    # structure_to_structure: indices into SR batch structures [num_pbc]
    #                   -> cell[structure_to_structure]
    # structure_mask: mask for valid periodic structures [num_pbc]
    #              -> True for real structures, False for padding
)
NonPeriodic = namedtuple("NonPeriodic", ("centers", "others", "pair_mask"))


def get_batch(
    samples,
    num_structures=None,
    num_structures_pbc=None,
    num_atoms=None,
    num_atoms_pbc=None,
    num_pairs=None,
    num_pairs_nonpbc=None,
    num_k=None,
    strategy="powers_of_2",
):
    _num_structures = len(samples)
    _num_atoms = []
    _num_pairs = []
    _num_pairs_nonpbc = []
    _is_pbc = []
    _num_k = []

    for structure in samples:
        lr = structure["lr"]
        _num_atoms.append(len(structure["positions"]))
        _num_pairs.append(len(structure["centers"]))
        if hasattr(lr, "k_grid"):
            _is_pbc.append(True)
            _num_k.append(lr.k_grid.shape[0] * lr.k_grid.shape[1] * lr.k_grid.shape[2])
        else:
            _is_pbc.append(False)
            _num_pairs_nonpbc.append(len(lr.centers))

    _num_atoms = np.array(_num_atoms)
    _num_pairs = np.array(_num_pairs)
    _num_pairs_nonpbc = np.array(_num_pairs_nonpbc)
    _is_pbc = np.array(_is_pbc)
    _num_k = np.array(_num_k)

    _total_atoms = _num_atoms.sum()
    _total_pairs = _num_pairs.sum()
    _max_atoms_pbc = _num_atoms[_is_pbc].max() if _is_pbc.any() else 0
    _max_k = _num_k.max() if len(_num_k) > 0 else 0
    _total_pairs_nonpbc = _num_pairs_nonpbc.sum() if len(_num_pairs_nonpbc) > 0 else 0
    _total_pbc = _is_pbc.sum()

    num_structures = get_size(num_structures, _num_structures, strategy=strategy)
    num_atoms = get_size(num_atoms, _total_atoms, strategy=strategy)
    num_pairs = get_size(num_pairs, _total_pairs, strategy=strategy)
    num_atoms_pbc = get_size(num_atoms_pbc, _max_atoms_pbc, strategy=strategy)
    num_k = get_size(num_k, _max_k, strategy=strategy)
    num_pairs_nonpbc = get_size(num_pairs_nonpbc, _total_pairs_nonpbc, strategy=strategy)
    num_pbc = get_size(num_structures_pbc, _total_pbc, strategy=strategy)

    padding_atom_idx = _total_atoms
    padding_structure_idx = num_structures - 1

    dtype = samples[0]["positions"].dtype

    charges = np.zeros(num_atoms, dtype=dtype)
    positions = np.zeros((num_atoms, 3), dtype=dtype)
    cell = np.zeros((num_structures, 3, 3), dtype=dtype)
    cell[:] = np.eye(3)
    smearing = np.ones(num_structures, dtype=dtype)
    centers = np.ones(num_pairs, dtype=int) * padding_atom_idx
    others = np.ones(num_pairs, dtype=int) * padding_atom_idx
    cell_shifts = np.zeros((num_pairs, 3), dtype=int)
    atom_to_structure = np.ones(num_atoms, dtype=int) * padding_structure_idx
    pair_to_structure = np.ones(num_pairs, dtype=int) * padding_structure_idx
    structure_mask = np.zeros(num_structures, dtype=bool)
    pbc_mask = np.zeros(num_structures, dtype=bool)
    atom_mask = np.zeros(num_atoms, dtype=bool)
    pair_mask = np.zeros(num_pairs, dtype=bool)

    nonpbc_centers = np.ones(num_pairs_nonpbc, dtype=int) * padding_atom_idx
    nonpbc_others = np.ones(num_pairs_nonpbc, dtype=int) * padding_atom_idx
    nonpbc_pair_mask = np.zeros(num_pairs_nonpbc, dtype=bool)

    pbc_kgrid = np.zeros((num_pbc, num_k, 3), dtype=dtype)
    pbc_atom_to_atom = np.ones((num_pbc, num_atoms_pbc), dtype=int) * padding_atom_idx
    pbc_structure_to_structure = np.ones(num_pbc, dtype=int) * padding_structure_idx
    pbc_atom_mask = np.zeros((num_pbc, num_atoms_pbc), dtype=bool)
    pbc_structure_mask = np.zeros(num_pbc, dtype=bool)

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
            pbc_atom_to_atom[pbc_idx, :num_n] = np.arange(atom_offset, atom_offset + num_n)

            k_grid_shape = lr.k_grid.shape
            pbc_kgrid[pbc_idx] = generate_ewald_k_grid(k_grid_shape, size=num_k)

            pbc_atom_mask[pbc_idx, :num_n] = True
            pbc_structure_mask[pbc_idx] = True
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
        atom_to_atom=pbc_atom_to_atom,
        structure_to_structure=pbc_structure_to_structure,
        atom_mask=pbc_atom_mask,
        structure_mask=pbc_structure_mask,
    )

    nonperiodic_batch = NonPeriodic(
        centers=nonpbc_centers,
        others=nonpbc_others,
        pair_mask=nonpbc_pair_mask,
    )

    return charges, sr_batch, nonperiodic_batch, periodic_batch


def prepare(atoms, cutoff, lr_wavelength=None, smearing=None):
    structure = to_structure(atoms, cutoff)

    structure["charges"] = atoms.get_initial_charges()

    if lr_wavelength is None:
        lr_wavelength = cutoff / 8.0

    if smearing is None:
        smearing = cutoff / 4.0

    smearing, lr = to_lr(atoms, structure, lr_wavelength, smearing)

    structure["lr"] = lr
    if smearing is not None:
        structure["smearing"] = smearing

    return structure


def to_lr(atoms, structure, lr_wavelength, smearing):
    if atoms.pbc.all():
        cell = atoms.get_cell().array
        k_grid = get_kgrid_ewald(cell, lr_wavelength)
        return smearing, Periodic(
            k_grid=k_grid,
            atom_to_atom=None,
            structure_to_structure=None,
            atom_mask=None,
            structure_mask=None,
        )
    elif not atoms.pbc.all():
        N = len(atoms)
        full_i = np.arange(N).repeat(N)
        full_j = np.tile(np.arange(N), N)

        return None, NonPeriodic(
            centers=full_i,
            others=full_j,
            pair_mask=None,
        )

    else:
        raise ValueError("no mixed pbc yet")


def get_kgrid_ewald(cell, lr_wavelength):
    ns = np.ceil(np.linalg.norm(cell, axis=-1) / lr_wavelength)
    return np.ones((int(ns[0]), int(ns[1]), int(ns[2])))


def to_structure(atoms, cutoff, dtype=np.float32):
    from vesin import ase_neighbor_list as neighbor_list

    structure = {}
    structure["cell"] = atoms.get_cell().array.astype(dtype)
    structure["positions"] = atoms.get_positions().astype(dtype)
    structure["atomic_numbers"] = atoms.get_atomic_numbers().astype(int)
    structure["charges"] = atoms.get_initial_charges().astype(dtype)

    if atoms.pbc.all():
        centers, others, D, S = neighbor_list("ijDS", atoms, cutoff)
    elif atoms.pbc.any():
        raise ValueError  # not supported here
    else:
        assert not atoms.pbc.any()

        centers, others, D = neighbor_list("ijD", atoms, cutoff)
        S = np.zeros((len(centers), 3), dtype=int)
        if (structure["cell"] == 0).all():
            structure["cell"] = np.eye(3)

    structure["centers"] = centers
    structure["others"] = others
    structure["cell_shifts"] = S
    structure["displacements"] = D.astype(dtype)
    structure["pbc"] = atoms.get_pbc()

    return structure


def get_size(proposed, actual, strategy="powers_of_2"):
    if proposed is not None:
        if isinstance(proposed, str):
            return _get_size(actual + 1, strategy=proposed)
        elif isinstance(proposed, int):
            assert proposed > actual
            return proposed
        else:
            msg = "could not determine size.\n"
            msg += f"proposed: {proposed} actual: {actual} strategy: {strategy}"
            raise ValueError(msg)
    else:
        return _get_size(actual + 1, strategy=strategy)


def _get_size(n, strategy="powers_of_2"):
    if strategy == "multiples":
        return multiples(n)

    prefix = "powers_of_"
    if strategy.startswith(prefix):
        exponent = int(strategy[len(prefix) :])
        return next_power(n, exponent)

    prefix = "multiples_of_"
    if strategy.startswith(prefix):
        x = int(strategy[len(prefix) :])
        return next_multiple(n, x)

    raise ValueError(f"unknown padding size strategy {strategy}")


def next_multiple(val, n):
    return n * (1 + int(val // n))


def next_power(val, x):
    return int(x ** np.ceil(np.log(val) / np.log(x)))


def multiples(val):
    if val <= 32:
        return next_multiple(val, 4)

    if val <= 64:
        return next_multiple(val, 16)

    if val <= 256:
        return next_multiple(val, 64)

    if val <= 1024:
        return next_multiple(val, 256)

    if val <= 4096:
        return next_multiple(val, 1024)

    if val <= 32768:
        return next_multiple(val, 4096)

    if val <= 65536:
        return next_multiple(val, 16384)

    return next_power(val, 2)


## test ##


assert (
    _get_size(
        13,
        strategy="powers_of_2",
    )
    == 16
)

assert (
    _get_size(
        33,
        strategy="powers_of_4",
    )
    == 64
)

assert (
    _get_size(
        11,
        strategy="multiples",
    )
    == 12
)


# ---------------------------------------------------------------------------
# Incremental / generator-style batcher
# ---------------------------------------------------------------------------


def _sample_totals(sample):
    """Compute basic count statistics for a single sample.

    Returns a dict containing (non-prefixed) counts:
      atoms: number of atoms in the structure
      pairs: number of short-range neighbor pairs (centers)
      pairs_nonpbc: number of long-range pairs for NonPeriodic samples
      pbc: 1 if periodic, 0 otherwise

    The caller is responsible for prefixing with 'total_' when aggregating.
    """
    lr = sample["lr"]
    num_atoms = len(sample["positions"])
    num_pairs = len(sample["centers"])
    if hasattr(lr, "k_grid"):
        num_pairs_nonpbc = 0
        pbc = 1
    else:
        # NonPeriodic: lr.centers exists
        num_pairs_nonpbc = len(lr.centers)
        pbc = 0
    return {
        "atoms": num_atoms,
        "pairs": num_pairs,
        "pairs_nonpbc": num_pairs_nonpbc,
        "pbc": pbc,
    }


def _empty_totals():
    return {
        "total_structures": 0,
        "total_atoms": 0,
        "total_pairs": 0,
        "total_pairs_nonpbc": 0,
        "total_pbc": 0,
    }


def _accumulate(totals, sample_counts):
    """Return a NEW totals dict after adding sample_counts."""
    new = dict(totals)  # shallow copy
    new["total_structures"] += 1
    new["total_atoms"] += sample_counts["atoms"]
    new["total_pairs"] += sample_counts["pairs"]
    new["total_pairs_nonpbc"] += sample_counts["pairs_nonpbc"]
    new["total_pbc"] += sample_counts["pbc"]
    return new


def iter_batches_by_totals(samples, should_yield=None):
    """Yield batches of samples guided by aggregate "total_*" size stats.

    Parameters
    ----------
    samples : iterable
        Iterable of prepared sample dicts (as produced by `prepare`).
    should_yield : callable(next_stats, current_stats) -> bool, optional
        Decision function consulted BEFORE adding the next sample. It receives:
          next_stats: totals dict AS IF the next sample were added.
          current_stats: current totals dict (state of the ongoing batch).
        If it returns True, the current batch is yielded *before* the sample is
        added, then a fresh batch starts.

        If None, a default strategy that never yields early is used; all
        samples appear in one batch.

    Yields
    ------
    (batch, totals) : (list[dict], dict)
        batch: list of sample dicts in the batch.
        totals: final aggregate stats for the yielded batch.

    Notes
    -----
    - This generator does not perform padding or allocation; it only groups
      samples to make it easier to implement complex sizing / splitting logic.
    - The separation of computing `next_stats` first allows sophisticated
      threshold or heuristic logic (e.g., model-driven cost prediction) to be
      implemented externally without making this loop more complex.
    """
    if should_yield is None:

        def should_yield(_next, _current):  # type: ignore
            return False

    batch = []
    current_totals = _empty_totals()

    for sample in samples:
        counts = _sample_totals(sample)
        next_totals = _accumulate(current_totals, counts)

        # Decide if we should flush BEFORE incorporating this sample.
        if batch and should_yield(next_totals, current_totals):
            # Yield current batch and reset.
            yield batch, current_totals
            batch = []
            current_totals = _empty_totals()
            # Recompute next_totals in the fresh context.
            next_totals = _accumulate(current_totals, counts)

        # Incorporate sample.
        batch.append(sample)
        current_totals = next_totals

    if batch:
        yield batch, current_totals


def debug_total_stats(samples):
    """Convenience helper: materialize batches with a trivial size strategy.

    This is intended for quick interactive exploration; it simply consumes the
    iterator and returns a list of (totals) for each batch (one batch if no
    splitting occurs).
    """
    return [totals for _batch, totals in iter_batches_by_totals(samples)]
