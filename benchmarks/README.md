# Benchmarks

Timing of `jaxpme.batched_tiled.Ewald` on fixed batches of real structures, to compare revisions and catch performance regressions (including ones caused by JAX upgrades). Not run in CI: timings need a GPU.

## Data

`pool.json` describes a structure pool sampled once, with a fixed seed, from three public datasets:

| key | dataset | split | structures | atoms |
|---|---|---|---|---|
| `omol25` | OMol25 [1] | `train_4M` | 2000 | 112k, all non-periodic, charged |
| `omat24` | OMat24 [2] | `train` | 5000 | 93k, all 3D periodic |
| `mad16` | MAD [3] | v1.6 `train` | 5000 | 95k, mixed 3D / 2D slabs / non-periodic |

The pool itself (three npz files, ~6.5 MB) is not in the repository; `pool.load` checks each file against the sha256 in `pool.json`. Positions, numbers, cells and pbc come from the datasets; charges are synthetic (per-atom N(0, 0.4), shifted to each structure's total charge, which is the OMol25 charge and 0 otherwise), generated from a seed stored with each batch.

`batches.json` freezes 16 batches drawn from the pool by `make_batches.py`: sets `omol`, `omat`, `mad` and `joint` (equal atom shares of the three), each as `small` (2048 atoms, 3 draws) and `large` (24576 atoms, 1 draw). Members are stored explicitly, so the batches don't change with the script. Add a new batch set rather than regenerating this one.

## Running

```bash
python benchmarks/run.py --pool POOL_DIR --out results.jsonl --label main --rev <sha>
python benchmarks/compare.py main.jsonl branch.jsonl > comparison.md
```

`run.py` times whichever `jaxpme` is importable (use `PYTHONPATH` to select a checkout; `--rev` records its revision when `jaxpme.__version__` can't). Defaults: `num_k=1024` with derived cutoff and smearing, float32, matmul precision `highest`, 10 repeats after a compiling first call. Ops: `energy`, `energy_forces`, `energy_forces_stress` (batches with periodic structures), and the gradients of ‖F‖² w.r.t. positions (`dr`) and charges (`dq`). Each JSONL row records the revision, JAX version, device, memory fraction, settings, pool and batch checksums, pair counts, padded sizes, first-call (compile) time, median and min time, peak device memory, and an output fingerprint. `--smoke` runs one small batch per set on any device, as a check that the script works.

`compare.py` prints one markdown table per op and device, with the first file as reference: median ms per arm, the ratio reference / arm (> 1: faster), and `dev`, the largest relative deviation of the output fingerprints (~1e-7 in float32 when arms agree). Results from different devices are never put in one table; compare arms run on the same node.

## References

1. D. S. Levine et al., The Open Molecules 2025 (OMol25) Dataset, Evaluations, and Models, arXiv:2505.08762 (2025).
2. L. Barroso-Luque et al., The Open Materials 2024 (OMat24) inorganic materials dataset and models, Nat. Comput. Sci. (2026), arXiv:2410.12771.
3. A. Mazitov et al., Massive Atomic Diversity: a compact universal dataset for atomistic machine learning, Sci. Data (2025), arXiv:2506.19674.
