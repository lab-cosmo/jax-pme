# Batched Ewald: Tiled Approach

A batched Ewald backend designed for **heterogeneous** atom counts across the batch. Atoms are sum-padded per system (each system padded individually to a multiple of `BM`), and the reciprocal sum runs through a pure-JAX tile-dispatched kernel (`vmap + segment_sum` in pass 1, `vmap + reshape-sum` in pass 2). Stress works through autograd — no `custom_vjp`, no `stop_gradient` on `kvec/W`.

## Usage

```python
from jaxpme.batched_tiled import Ewald

calculator = Ewald(prefactor=1.0)

# num_k is REQUIRED — sets per-cell K-vec target via lr_wavelength_for_num_k.
# cutoff is still required for the real-space neighbor list.
charges, batch, batch_nopbc, batch_pbc = calculator.prepare(
    atoms_list, num_k=200, cutoff=5.0
)

potentials = calculator.potentials(charges, batch, batch_nopbc, batch_pbc)
energies = calculator.energy(charges, batch, batch_nopbc, batch_pbc)
energies, forces = calculator.energy_forces(charges, batch, batch_nopbc, batch_pbc)
energies, forces, stress = calculator.energy_forces_stress(charges, batch, batch_nopbc, batch_pbc)
```

`Batch` and `NonPeriodic` are re-imported from `batched_mixed` (same shapes). Only `Periodic` differs.

## How it works

The standard Ewald reciprocal sum is the structure-factor form:

```
S_k = Σ_j q_j exp(i k·r_j)          [structure factor]
φ_i = Σ_k W_k Re(exp(-i k·r_i) S_k) [per-atom potential]
```

The compute is shaped like a `[N_atoms, K_kvecs]` outer-product of trig values, summed in two passes (first over atoms, then over k-vecs). For a batch of systems, we want **per-system blocks** of this matrix — atoms from system A pair only with k-vecs from system A.

### Block-diagonal work matrix

In a batch of three systems with `(N_0, K_0), (N_1, K_1), (N_2, K_2)` shapes, the conceptual work matrix is **block-diagonal**:

```
              kvecs_0     kvecs_1     kvecs_2
              ┌─────────┬─────────┬─────────┐
   atoms_0    │   real  │   ZERO  │   ZERO  │
              ├─────────┼─────────┼─────────┤
   atoms_1    │   ZERO  │   real  │   ZERO  │
              ├─────────┼─────────┼─────────┤
   atoms_2    │   ZERO  │   ZERO  │   real  │
              └─────────┴─────────┴─────────┘
```

`batched_tiled` only computes the on-diagonal blocks. The dispatch table enumerates which `(BM × BK)` sub-tiles cover each diagonal block.

### Two padding axes, two different strategies

**Atoms — sum-padded.** Each system's atoms are padded to `⌈N_b/BM⌉·BM` and concatenated into one flat array of length `N_pbc_total`:

```
   sys 0 (5 atoms, BM=4)   sys 1 (3 atoms)   sys 2 (9 atoms)
   ┌───────────────┐       ┌───────┐         ┌───────────────────────┐
   │aaaaa--- ────  │       │aaa-   │         │aaaaaaaaa--- ──── ──── │
   └───────────────┘       └───────┘         └───────────────────────┘
   |←  8 slots    →|       |← 4   →|         |←     12 slots        →|
                          flat array:
                  ┌────────┬────┬───────────────┐
                  │ 8 (s0) │ 4  │   12 (s2)     │     N_pbc_total = 24
                  └────────┴────┴───────────────┘
   pbc_atom_off:  0       8    12              24
   pbc_atom_mask: 11111000-1110-111111111000000000000000
```

Variable n_atom_tiles per system: `s0`→2 tiles, `s1`→1 tile, `s2`→3 tiles.

This is the heterogeneous-batch win vs `batched_mixed`'s max-padding (which would give every system 12 slots → 36 total).

**K-vectors — rectangular.** All systems share `K_pad` (the bucket-rounded max-K, snapped to a multiple of BK). Variable per-system K_b is paid as a small max-padding overhead — cheap because `num_k` in `prepare` keeps per-cell K targets uniform. Each system has the same `n_kvec_tiles = K_pad / BK`.

```
   k_grid: [3, 16, 3]  (BK=8, K_pad=16, n_kvec_tiles=2 for every system)
   sys 0:  kkkkkkkkkk------     (10 real, 6 padding rows -> W=0)
   sys 1:  kkkkkkkkkk------
   sys 2:  kkkkkkkkkk------
```

### Dispatch table

For each system's `(n_atom_tiles_b × n_kvec_tiles)` on-diagonal block, `get_batch` enumerates one triple per `(BM × BK)` sub-tile (vectorised numpy, not in the JIT trace). The resulting array is a plain `jnp` array at the kernel call site — prefetched to device alongside positions, cell, and the rest of the batch. One table, in pass-2 order (outer `m_tile`, inner `k_tile`):

```
dispatch_table: [T, 3] — (b, m_tile, k_tile)

  T = (K_pad / BK) · (N_pbc_total / BM)
    = sum over systems of (n_atom_tiles_b · n_kvec_tiles)
```

The kernel `jax.vmap`s `per_triple(...)` over the table; each call computes a `[BM, BK]` trig block (`cos(r @ k.T)` and `sin(r @ k.T)`) and reduces it. Two different reductions glue the partials into global arrays:

- **Pass 1** uses `jax.ops.segment_sum` on segment ids `b · n_kvec_tiles + kt` to build `S^r`, `S^i [B, K_pad]`. Ids are unsorted under pass-2 ordering, but `segment_sum` is order-agnostic.
- **Pass 2** exploits the outer-m_tile ordering: each `(b, m_tile)` segment is exactly `n_kvec_tiles` consecutive rows, so `phi_per.reshape(M_TILES, n_kvec_tiles, BM).sum(axis=1)` does the reduction without any `segment_sum` at all — better XLA fusion.

### JIT cache stability

The dispatch table has **shape `[T, 3]`** that depends only on the bucket-rounded `N_pbc_total` and `K_pad` (via `next_size`) — not on per-system N_b. Contents are runtime data; JIT keys on shape, not values. Same-bucket batches reuse the JIT cache regardless of how atoms are distributed across systems. Compile only happens when a batch bumps to a new bucket. (The "built on host" point above is about not constructing it inside the JIT trace — at runtime it lives on device like the other inputs.)

## Contracts

1. **`num_k` is required.** No cutoff-only auto-derivation. `num_k` sets the per-cell K target; the batcher max-pads to one rectangular K_pad. Tune Ewald α (smearing) to shift work between real and reciprocal space.

2. **Tile sizes `(BM, BK)` baked at `prepare` time.** Defaults `BM=32, BK=128`. They control:
   - per-system atom padding (`⌈N_b/BM⌉·BM`)
   - K_pad alignment (multiple of BK)
   - the trig-block shape inside the kernel
   Different `(BM, BK)` → different JIT cache.

3. **Forces/stress require `batch.distances = None`** — distances recomputed inside `energy_forces*` for autodiff through positions.

4. **Outputs are padded** — extract with masks:
   ```python
   real_energies = energies[batch.structure_mask]
   real_forces = forces[batch.atom_mask]
   ```

5. **Charges from ASE** — `atoms.get_initial_charges()`, pass back to methods.

6. **Coulomb only for 2D-PBC slab correction.** Inverse-power-law + 2D-PBC returns NaN (unsupported correction — same as `batched_mixed`).

## Padding strategy

`next_size` buckets every padded dimension to powers of 2 by default for JIT cache stability. Override via `get_batch()` kwargs: `num_atoms_pbc="powers_of_2"`, etc. K_pad is then snapped up to a multiple of BK; N_pbc_total is snapped up to a multiple of BM. Slack in the atom total is absorbed by the last (padding) pbc-system.
