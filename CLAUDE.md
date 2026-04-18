# Claude Notes: jax-pme

## What This Is
JAX implementation of Ewald summation, PME, and P3M for long-range electrostatic interactions. Used in molecular simulations. Version 0.1.0-alpha.1 — API not finalized.

## Structure
```
jaxpme/
├── calculators.py      # Main API: Ewald(), PME(), P3M() factory functions
├── solvers.py          # Real-space and reciprocal-space solvers (ewald, pme, p3m)
├── potentials.py       # Coulomb, inverse power-law, range-separated potentials
├── kspace.py           # K-space computations + p3m_influence() function
├── mesh.py             # Mesh interpolation: lagrange() and bspline() (n=1-5)
├── utils.py            # Neighbor lists, cell handling (no mixed PBC)
├── prefactors.py       # Unit conversion constants (eV_A, etc.)
├── batched_flat/       # Batched Ewald (flat padding strategy)
└── batched_mixed/      # Batched Ewald (mixed real/reciprocal batching)
    ├── calculators.py  # batched_mixed.Ewald() — main batched API
    └── batching.py     # Batch preparation utilities

tests/
├── conftest.py         # REFERENCE_STRUCTURES_DIR constant
├── test_ewald.py       # Core tests (Madelung constants, random structures)
├── test_kspace.py      # K-space computation tests (half-space optimization, etc.)
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

## Current Limitations
- PME only supports 4-node Lagrange interpolation
- Mixed PBC (2D) only works in batched implementations, not serial
- Power-law potentials raise `NotImplementedError` for mixed PBC corrections
- `calculators.py` has TODO for PME/P3M parameter tuning logic

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
