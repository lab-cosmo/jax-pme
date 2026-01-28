# batched_mixed_slots

Slot-based batching for improved GPU occupancy in k-space computations.

## Motivation

In `batched_mixed`, all PBC samples share one `(num_k, num_atoms_pbc)` shape, padded to the maximum. For datasets with outliers, this causes poor occupancy:

```
Example (MAD dataset):
- 97% of samples have k <= 4096
- 3% outliers require k up to 131072
- Result: all samples padded to k=131072 → 23.5% occupancy
```

With slots, samples are partitioned by size into separate arrays with different padding targets:

```
Slot 0: k=4096, atoms=64   → handles 97% of samples
Slot 1: k=32768, atoms=192 → handles 3% of samples
Result: ~74% average occupancy (3x improvement)
```

## Usage

```python
from jaxpme.batched_mixed_slots import Ewald

calculator = Ewald(prefactor=1.0)

# Specify slot capacities (samples assigned to first slot that fits)
charges, batch, batch_nopbc, batch_pbc = calculator.prepare(
    atoms_list,
    cutoff,
    slot_num_k=[4096, 32768],        # k-vector capacity per slot
    slot_num_atoms_pbc=[64, 192],    # atom capacity per slot
)

# batch_pbc is now tuple[Periodic, ...] with one entry per slot
energy = calculator.energy(charges, batch, batch_nopbc, batch_pbc)
```

Without slot parameters, behaves identically to `batched_mixed` (single auto-sized slot).

## How It Works

1. **Partitioning**: Each PBC sample is assigned to the first slot where both `num_k <= slot_num_k` and `num_atoms <= slot_num_atoms_pbc`

2. **Batching**: Each slot gets its own `Periodic` namedtuple with shape `(num_pbc_slot, num_k_slot, ...)` and `(num_pbc_slot, num_atoms_slot, ...)`

3. **Computation**: The calculator iterates over slots (Python loop, unrolled at JAX trace time), vmaps k-space per slot, and scatters results back to atom indices

4. **Empty slots**: Allocated with minimum shape and `structure_mask=False` (no special handling needed)

## Parameters

- `slot_num_k`: List of k-vector capacities, e.g., `[4096, 32768]` creates 2 slots
- `slot_num_atoms_pbc`: List of atom capacities per slot (must match length of `slot_num_k`)
- `slot_num_pbc`: Optional list of structure capacities per slot (auto-sized if not provided)

## Design Notes

- Real-space computation unchanged (single flat batch)
- Only k-space uses slots
- Number of slots inferred from tuple length at trace time
- Slots should be ordered by size (smallest first) for correct assignment
