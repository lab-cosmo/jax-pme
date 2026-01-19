# Batched Ewald: Mixed Approach

Batched Ewald summation using a hybrid strategy: real-space uses flat arrays with `segment_sum`, reciprocal-space uses `vmap` over padded k-grids. This naturally handles mixed PBC/non-PBC batches.

## Usage

```python
from jaxpme.batched_mixed import Ewald

calculator = Ewald(prefactor=1.0)

# atoms_list: list of ASE Atoms objects (can mix PBC and non-PBC)
charges, batch, batch_nopbc, batch_pbc = calculator.prepare(atoms_list, cutoff=5.0)

potentials = calculator.potentials(charges, batch, batch_nopbc, batch_pbc)
energies = calculator.energy(charges, batch, batch_nopbc, batch_pbc)
energies, forces = calculator.energy_forces(charges, batch, batch_nopbc, batch_pbc)
energies, forces, stress = calculator.energy_forces_stress(charges, batch, batch_nopbc, batch_pbc)
```

## Batch Structures

`prepare()` returns `(charges, Batch, NonPeriodic, Periodic)`:

- **`Batch`**: Main batch with all atoms/pairs flattened. Key fields:
  - `positions`, `cell`: padded arrays
  - `atom_mask`, `pair_mask`, `structure_mask`: boolean masks for valid entries
  - `atom_to_structure`, `pair_to_structure`: indices mapping atoms/pairs → structures
  - `pbc_mask`: which atoms belong to periodic structures
  - `distances`: `None` by default (computed on-the-fly), or precomputed

- **`NonPeriodic`**: Extra pairs for non-PBC structures (bare 1/r, no Ewald splitting)

- **`Periodic`**: K-space data for periodic structures, vmapped over structures

## Contracts

1. **Forces/stress require `batch.distances = None`** — distances are recomputed inside `energy_forces*` to enable autodiff through positions.

2. **Outputs are padded** — use masks to extract real values:
   ```python
   real_energies = energies[batch.structure_mask]
   real_forces = forces[batch.atom_mask]
   ```

3. **Charges from ASE** — reads `atoms.get_initial_charges()`. Pass same charges array back to methods.

## Padding Strategy

By default, dimensions are padded to powers of 2 for JIT stability across batches. Override via `get_batch()` kwargs: `num_atoms=128`, `num_pairs="powers_of_2"`, etc.
