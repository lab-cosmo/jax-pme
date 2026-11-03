# Changelog

All notable changes to `jax-pme` are documented here. The API is not finalised and may change without notice between minor versions.

## Unreleased

### Added
- `EwaldDipole` (serial and `batched_tiled`): potentials and fields of point charges and dipoles, experimental (#38)

### Changed
- `batched_tiled` uses a half neighbour list by default; `Ewald(full_neighbor_list=True)` for full lists (#40)
- `batched_tiled` uses a sorted pair layout, making forces and force-training gradients much faster on GPU (#41)
- `batched_tiled.batching.get_batch` returns `charges, dipoles, sr_batch, nonperiodic, periodic` (#38)

## v0.1.0 (2026-09-24)

First release of jax-pme on PyPI: Ewald summation and particle-mesh methods in JAX.

### Calculators
- Serial `Ewald`, `PME` (Lagrange interpolation) and `P3M` (B-spline interpolation with influence-function correction) calculators for energies, potentials, forces and stress (#13)
- Batched Ewald backends: `batched_flat`, `batched_mixed` (fixed k-grid batcher, #12, #21) and `batched_tiled` for heterogeneous batches with tile-dispatched reciprocal kernel (#29)
- Half-space k-vector optimisation for batched Ewald (#15)

### Physics
- Coulomb and inverse power-law potentials with integer exponents 1 to 6 (#31)
- 2D PBC slab correction for arbitrary triclinic cells with automatic vacuum shrinking (#23, #17)
- Non-PBC structures handled via a bare 1/r sum in serial and batched Ewald, with documented potential convention (#27, #30)
- Support for masked atoms and pairs, and for explicitly passed distances and neighbor lists

### Packaging and infrastructure
- Published to PyPI with the version derived from git tags via setuptools_scm; MIT license file added (#36)
- CI for tests (Python 3.10 and 3.12, minimum jax 0.4.30, Linux and macOS) and linting (#19)
