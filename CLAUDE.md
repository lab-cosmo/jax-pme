# Claude Notes: jax-pme

## What This Is
JAX implementation of Ewald summation, PME, and P3M for long-range electrostatic interactions. Used in molecular simulations. Version 0.1.0-alpha.1 — API not finalized.

## Structure
```
jaxpme/
├── calculators.py      # Main API: Ewald(), PME(), P3M() factory functions
├── solvers.py          # Real-space and reciprocal-space solvers (ewald, pme, p3m)
├── potentials.py       # Coulomb, inverse power-law, range-separated potentials
├── kspace.py           # K-space: numpy preprocessing (grids) + jax.numpy JIT ops (p3m_influence, kvectors)
├── mesh.py             # Mesh interpolation: lagrange() and bspline() (n=1-5)
├── utils.py            # Neighbor lists, cell handling
├── prefactors.py       # Unit conversion constants (eV_A, etc.)
├── batched_flat/       # Batched Ewald (flat padding strategy)
└── batched_mixed/      # Batched Ewald (mixed real/reciprocal batching)
    ├── calculators.py  # batched_mixed.Ewald() — main batched API
    └── batching.py     # Batch preparation utilities

tests/
├── conftest.py         # REFERENCE_STRUCTURES_DIR constant
├── test_ewald.py       # Core tests (Madelung constants, random structures)
├── test_kspace.py      # K-space computation tests (half-space optimization, etc.)
├── test_inverse_power_law.py  # 1/r^p potentials: closed forms, direct-sum references
├── test_slab_correction.py  # 2D PBC: tiling, rotation, forces, MAD-1.5 references
├── test_batched_*.py   # Batched implementation tests
```

## Key APIs
- **Serial**: `Ewald()`, `PME()`, or `P3M()` → `.prepare()` → `.energy()/.potentials()/.energy_forces()/.energy_forces_stress()`
- **Batched**: `jaxpme.batched_mixed.Ewald()` → `.prepare([atoms_list], cutoff)` → same methods but batched

### Potential convention (matches torch-pme)
`V_i = (1/2) Σ_{j≠i} q_j v(r_ij)` — the 1/2 is absorbed into the potential so `energy = Σ_i q_i V_i` (no extra 1/2). Both PBC and non-PBC paths (serial and batched) return this halved potential. If comparing against a textbook `V_i = Σ q_j/r_ij` reference, expect a factor of 2.

### PME vs P3M
- **PME**: Lagrange interpolation (4-node). Faster but forces less smooth.
- **P3M**: B-spline interpolation (n=1-5) with influence function correction. Smoother forces, better for MD.

### P3M Influence Function
The `p3m_influence()` function in `kspace.py` computes 1/U²(k) to correct for B-spline smoothing. Key detail: for non-orthogonal cells, k-vectors must be **projected onto cell axes** (dot product), not multiplied component-wise. The formula is `kh[i] = (k · cell[i]) / n[i]`.

## Design Notes
- All preprocessing (k-grid shape, charges, cell) uses **numpy**, not jax.numpy, to avoid unnecessary device allocation. Only JIT-traced compute paths use jax.numpy.
- `shrink_2d_cell` is only called when `pbc.sum() == 2`

## Current Limitations
- PME only supports 4-node Lagrange interpolation
- Mixed PBC (2D) only works in batched implementations, not serial
- Serial `PME`/`P3M` still return `NaN` for non-PBC structures (volume == 0);
  only serial `Ewald` and the batched calculators handle non-PBC
- Power-law potentials return `NaN` for mixed PBC corrections (no slab correction)
- Power-law exponents limited to integers 1-6 (`ValueError` otherwise)
- `calculators.py` has TODO for PME/P3M parameter tuning logic

## Inverse power-law potentials (1/r^p)
- Integer exponents 1-6 supported (issue #20, ported from torch-pme):
  the k-space kernel Γ(peff, x)/x^peff with peff = (3-p)/2 uses closed forms
  per exponent, since generic `gammaincc` only handles peff > 0 (p < 3)
- For p > 3 the kernel has a finite k→0 limit, supplied via the optional
  `lr_k0` field on `RawPotential` (defaults to `None` = 0 = neutralizing
  background); the background correction is zero for p ≥ 3
- `_exp1` in `potentials.py` is a custom E1 implementation:
  `jax.scipy.special.exp1` takes ~20s to compile and its jvp leaks tracers

## Non-periodic (0D) structures
- Serial `Ewald` routes `pbc=[F,F,F]` to a bare 1/r sum over *all* pairs
  (no cutoff, no Ewald splitting), mirroring the batched `NonPeriodic` contract
- The all-pairs list respects `full_neighbor_list` (half by default, both
  directions if `True`), so the returned graph stays usable for SR models
- Serial `Ewald.prepare` therefore returns `charges, *graph, k_grid, smearing, pbc`
  (one more element than `PME`/`P3M`, which return `..., smearing`)

## 2D PBC (slab correction)
- Supports arbitrary triclinic cells (not just orthorhombic)
- `correction_pbc` in `potentials.py` projects onto the plane normal via cross product
- `shrink_2d_cell` in `batching.py` reduces the non-periodic cell vector before deriving Ewald parameters, preventing large vacuum from inflating the k-grid
- Vacuum gap formula: `h_min = thickness + 1.5 * L_max` (residual ≈ exp(-3π) ≈ 7e-5)

## Testing
Run from the package root: `python -m pytest tests/ -v`
- Reference structures in `tests/reference_structures/coulomb_test_frames.xyz`
- Tests validate against Madelung constants and GROMACS reference data

## Code Style
- Functional JAX style, minimal docstrings ("sharp tool" philosophy)
- Comments purely for understanding tricky things, not explaining code
- Format with `ruff format . && ruff check --fix .`

## Branch Status
- `main`: stable release (includes batched implementations from PR #12 and P3M calculator from PR #13)

## Contributing

- Always make a PR and do not break CI
- **Update CLAUDE.md and README.md before opening or merging a PR** — keep docs in sync with code changes
