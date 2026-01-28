import numpy as np

from jaxpme.batched_mixed.batching import (
    Batch,
    NonPeriodic,
    Periodic,
    next_size,
    prepare,
)
from jaxpme.batched_mixed.kspace import count_halfspace_kvectors, generate_ewald_k_grid

# Re-export for convenience
__all__ = ["prepare", "get_batch", "Batch", "NonPeriodic", "Periodic"]


def assign_to_slots(num_k, num_atoms, slot_num_k, slot_num_atoms):
    """Assign PBC samples to slots based on explicit slot shapes.

    Samples are assigned to the first (smallest) slot that can fit them.

    Args:
        num_k: array of k-vector counts per PBC sample
        num_atoms: array of atom counts per PBC sample
        slot_num_k: list of k capacity per slot (ascending order recommended)
        slot_num_atoms: list of atom capacity per slot

    Returns:
        slot_assignments: array of slot index for each PBC sample
    """
    num_samples = len(num_k)
    num_slots = len(slot_num_k)
    slot_assignments = np.full(num_samples, -1, dtype=int)

    for i in range(num_samples):
        for slot_idx in range(num_slots):
            if (
                num_k[i] <= slot_num_k[slot_idx]
                and num_atoms[i] <= slot_num_atoms[slot_idx]
            ):
                slot_assignments[i] = slot_idx
                break

    # Samples that don't fit in any slot get assigned to the last slot
    # (they will be truncated or cause issues - user should size slots appropriately)
    slot_assignments[slot_assignments == -1] = num_slots - 1

    return slot_assignments


def get_batch(
    samples,
    slot_num_k=None,
    slot_num_atoms_pbc=None,
    slot_num_pbc=None,
    num_structures=None,
    num_atoms=None,
    num_pairs=None,
    num_pairs_nonpbc=None,
    strategy="powers_of_2",
    halfspace=True,
):
    """Create batched data structures with slot support for k-space.

    Args:
        samples: List of prepared samples
        slot_num_k: List of k-vector capacities per slot, e.g., [4096, 32768].
                    Samples assigned to first slot that fits. None = auto-size.
        slot_num_atoms_pbc: List of atom capacities per slot. Must match
                            len(slot_num_k). None = auto-size to max.
        slot_num_pbc: List of PBC structure capacities per slot. None = auto.
        ... (other args same as batched_mixed.get_batch)

    Returns:
        charges, sr_batch, nonperiodic_batch, periodic_batches
        where periodic_batches is a tuple of Periodic namedtuples (one per slot)
    """
    # Gather statistics
    _num_structures = len(samples)
    _num_atoms = []
    _num_pairs = []
    _num_pairs_nonpbc = []
    _is_pbc = []
    _num_k = []
    _pbc_sample_indices = []

    for idx, structure in enumerate(samples):
        lr = structure["lr"]
        _num_atoms.append(len(structure["positions"]))
        _num_pairs.append(len(structure["centers"]))
        if hasattr(lr, "k_grid") and lr.k_grid is not None:
            _is_pbc.append(True)
            _pbc_sample_indices.append(idx)
            k_grid_shape = lr.k_grid.shape
            if halfspace:
                _num_k.append(count_halfspace_kvectors(k_grid_shape))
            else:
                _num_k.append(k_grid_shape[0] * k_grid_shape[1] * k_grid_shape[2])
        else:
            _is_pbc.append(False)
            _num_pairs_nonpbc.append(len(lr.centers))

    _num_atoms = np.array(_num_atoms)
    _num_pairs = np.array(_num_pairs)
    _num_pairs_nonpbc = np.array(_num_pairs_nonpbc)
    _is_pbc = np.array(_is_pbc)
    _num_k = np.array(_num_k)
    _pbc_sample_indices = np.array(_pbc_sample_indices)
    _pbc_num_atoms = (
        _num_atoms[_pbc_sample_indices] if len(_pbc_sample_indices) > 0 else np.array([])
    )

    # Determine number of slots and their sizes
    if slot_num_k is None:
        # Single slot, auto-sized (equivalent to batched_mixed)
        num_slots = 1
        _max_k = _num_k.max() if len(_num_k) > 0 else 1
        _max_atoms_pbc = _pbc_num_atoms.max() if len(_pbc_num_atoms) > 0 else 1
        slot_num_k = [next_size(_max_k, strategy=strategy)]
        slot_num_atoms_pbc = [next_size(_max_atoms_pbc, strategy=strategy)]
    else:
        num_slots = len(slot_num_k)
        if slot_num_atoms_pbc is None:
            # Use same atom capacity for all slots (auto-sized to max)
            _max_atoms_pbc = _pbc_num_atoms.max() if len(_pbc_num_atoms) > 0 else 1
            slot_num_atoms_pbc = [next_size(_max_atoms_pbc, strategy=strategy)] * num_slots
        else:
            assert len(slot_num_atoms_pbc) == num_slots

    # Assign PBC samples to slots
    if len(_num_k) > 0:
        slot_assignments = assign_to_slots(
            _num_k, _pbc_num_atoms, slot_num_k, slot_num_atoms_pbc
        )
    else:
        slot_assignments = np.array([], dtype=int)

    # Count samples per slot
    slot_pbc_counts = [np.sum(slot_assignments == s) for s in range(num_slots)]

    # Determine slot_num_pbc (structure count per slot)
    if slot_num_pbc is None:
        slot_num_pbc = [next_size(max(c, 1), strategy=strategy) for c in slot_pbc_counts]
    else:
        assert len(slot_num_pbc) == num_slots

    # Compute sizes for real-space batch (unchanged from batched_mixed)
    num_structures = num_structures if num_structures is not None else strategy
    num_atoms = num_atoms if num_atoms is not None else strategy
    num_pairs = num_pairs if num_pairs is not None else strategy
    num_pairs_nonpbc = num_pairs_nonpbc if num_pairs_nonpbc is not None else strategy

    _total_atoms = _num_atoms.sum()
    _total_pairs = _num_pairs.sum()
    _total_pairs_nonpbc = _num_pairs_nonpbc.sum() if len(_num_pairs_nonpbc) > 0 else 0

    num_structures = next_size(_num_structures + 1, strategy=num_structures)
    num_atoms = next_size(_total_atoms + 1, strategy=num_atoms)
    num_pairs = next_size(_total_pairs + 1, strategy=num_pairs)
    num_pairs_nonpbc = next_size(_total_pairs_nonpbc + 1, strategy=num_pairs_nonpbc)

    padding_atom_idx = _total_atoms
    padding_structure_idx = num_structures - 1

    dtype = samples[0]["positions"].dtype

    # Initialize real-space arrays
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

    # Initialize per-slot arrays
    slot_kgrids = [
        np.zeros((slot_num_pbc[s], slot_num_k[s], 3), dtype=dtype) for s in range(num_slots)
    ]
    slot_atom_to_atom = [
        np.ones((slot_num_pbc[s], slot_num_atoms_pbc[s]), dtype=int) * padding_atom_idx
        for s in range(num_slots)
    ]
    slot_structure_to_structure = [
        np.ones(slot_num_pbc[s], dtype=int) * padding_structure_idx
        for s in range(num_slots)
    ]
    slot_atom_masks = [
        np.zeros((slot_num_pbc[s], slot_num_atoms_pbc[s]), dtype=bool)
        for s in range(num_slots)
    ]
    slot_structure_masks = [np.zeros(slot_num_pbc[s], dtype=bool) for s in range(num_slots)]
    slot_pbc_vectors = [
        np.zeros((slot_num_pbc[s], 3), dtype=bool) for s in range(num_slots)
    ]
    slot_counters = [0] * num_slots

    # Fill arrays
    atom_offset = 0
    pair_offset = 0
    nonpbc_offset = 0
    pbc_idx = 0

    for idx, structure in enumerate(samples):
        lr = structure["lr"]
        is_periodic = hasattr(lr, "k_grid") and lr.k_grid is not None

        num_n = len(structure["positions"])
        num_p = len(structure["centers"])

        atom_slice = slice(atom_offset, atom_offset + num_n)
        pair_slice = slice(pair_offset, pair_offset + num_p)

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

            # Determine slot for this PBC sample
            slot_idx = slot_assignments[pbc_idx]
            slot_pos = slot_counters[slot_idx]

            slot_structure_to_structure[slot_idx][slot_pos] = idx
            slot_atom_to_atom[slot_idx][slot_pos, :num_n] = np.arange(
                atom_offset, atom_offset + num_n
            )

            k_grid_shape = lr.k_grid.shape
            slot_kgrids[slot_idx][slot_pos] = generate_ewald_k_grid(
                k_grid_shape, size=slot_num_k[slot_idx], halfspace=halfspace
            )

            slot_atom_masks[slot_idx][slot_pos, :num_n] = True
            slot_structure_masks[slot_idx][slot_pos] = True
            slot_pbc_vectors[slot_idx][slot_pos] = structure["pbc"]

            slot_counters[slot_idx] += 1
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

    nonperiodic_batch = NonPeriodic(
        centers=nonpbc_centers,
        others=nonpbc_others,
        pair_mask=nonpbc_pair_mask,
    )

    # Create tuple of Periodic batches (one per slot)
    periodic_batches = tuple(
        Periodic(
            k_grid=slot_kgrids[s],
            atom_to_atom=slot_atom_to_atom[s],
            structure_to_structure=slot_structure_to_structure[s],
            atom_mask=slot_atom_masks[s],
            structure_mask=slot_structure_masks[s],
            pbc=slot_pbc_vectors[s],
        )
        for s in range(num_slots)
    )

    charges = np.zeros(num_atoms, dtype=dtype)
    for idx, structure in enumerate(samples):
        start = (atom_to_structure == idx).argmax()
        end = start + len(structure["charges"])
        charges[start:end] = structure["charges"]

    return charges, sr_batch, nonperiodic_batch, periodic_batches
