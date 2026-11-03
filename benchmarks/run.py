"""Time `batched_tiled.Ewald` on the frozen benchmark batches; JSONL row per measurement.

    python benchmarks/run.py --pool DIR --out results.jsonl --label main [--rev SHA]

Imports whichever `jaxpme` is on the path, so the same script times any revision
that has the `batched_tiled` calculator API. Compare files with `compare.py`.
"""

import numpy as np

import argparse
import hashlib
import json
import os
import platform
import time
from datetime import datetime
from pathlib import Path

HERE = Path(__file__).parent
OPS = ("energy", "energy_forces", "energy_forces_stress", "dr", "dq")

p = argparse.ArgumentParser()
p.add_argument("--pool", required=True, help="directory with the pool npz files")
p.add_argument("--out", required=True, help="JSONL file, appended to")
p.add_argument("--label", required=True, help="arm name, e.g. main")
p.add_argument(
    "--rev", default=None, help="jax-pme revision, if __version__ doesn't know it"
)
p.add_argument("--num-k", type=int, default=1024)
p.add_argument("--dtype", default="float32", choices=["float32", "float64"])
p.add_argument("--matmul-precision", default="highest")
p.add_argument("--repeats", type=int, default=10)
p.add_argument("--sets", nargs="+", default=None)
p.add_argument("--sizes", nargs="+", default=None)
p.add_argument("--ops", nargs="+", default=list(OPS), choices=OPS)
p.add_argument(
    "--smoke", action="store_true", help="first small batch per set, 1 repeat, CPU ok"
)
args = p.parse_args()

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402

jax.config.update("jax_enable_x64", args.dtype == "float64")
jax.config.update("jax_default_matmul_precision", args.matmul_precision)

from pool import MANIFEST, build, load  # noqa: E402

import jaxpme  # noqa: E402
from jaxpme.batched_tiled import Ewald  # noqa: E402

device = jax.devices()[0]
if not args.smoke:
    assert device.platform == "gpu", f"timing runs need a GPU, got {device}"

batches_path = HERE / "batches.json"
batches = json.loads(batches_path.read_text())["batches"]
if args.smoke:
    seen = set()
    batches = [
        b
        for b in batches
        if b["size"] == "small" and not (b["set"] in seen or seen.add(b["set"]))
    ]
    args.repeats = 1
batches = [
    b
    for b in batches
    if (args.sets is None or b["set"] in args.sets)
    and (args.sizes is None or b["size"] in args.sizes)
]

stats = device.memory_stats() or {}
meta = {
    "label": args.label,
    "rev": args.rev,
    "jaxpme_version": jaxpme.__version__,
    "jaxpme_file": jaxpme.__file__,
    "jax": jax.__version__,
    "device": device.device_kind,
    "platform": device.platform,
    "bytes_limit": stats.get("bytes_limit"),
    "mem_fraction": os.environ.get("XLA_PYTHON_CLIENT_MEM_FRACTION"),
    "dtype": args.dtype,
    "matmul_precision": args.matmul_precision,
    "num_k": args.num_k,
    "repeats": args.repeats,
    "host": platform.node(),
    "date": datetime.now().isoformat(timespec="seconds"),
    "pool": {k: v["sha256"] for k, v in MANIFEST["datasets"].items()},
    "batches_sha256": hashlib.sha256(batches_path.read_bytes()).hexdigest(),
}
out = Path(args.out)


def emit(row):
    with out.open("a") as f:
        f.write(json.dumps(meta | row) + "\n")


def fingerprint(x):
    # order-independent summary of an output, to check that arms agree
    x = np.asarray(x, dtype=np.float64)
    w = np.random.default_rng(1).normal(size=x.shape)
    return [float(x.sum()), float(np.abs(x).sum()), float((x * w).sum())]


def peak_bytes():
    return (device.memory_stats() or {}).get("peak_bytes_in_use")


def ops(calc, has_pbc):
    def energy(q, b, n, p):
        return calc.energy(q, b, n, p)

    def forces(q, b, n, p):
        def e(r):
            return calc.energy(q, b._replace(positions=r), n, p).sum()

        return -jax.grad(e)(b.positions)

    def dr(q, b, n, p):
        def loss(r):
            return jnp.sum(forces(q, b._replace(positions=r), n, p) ** 2)

        return jax.grad(loss)(b.positions)

    def dq(q, b, n, p):
        return jax.grad(lambda q: jnp.sum(forces(q, b, n, p) ** 2))(q)

    fns = {
        "energy": energy,
        "energy_forces": calc.energy_forces,
        "energy_forces_stress": calc.energy_forces_stress,
        "dr": dr,
        "dq": dq,
    }
    return {
        k: v
        for k, v in fns.items()
        if k in args.ops and (has_pbc or k != "energy_forces_stress")
    }


def measure(fn, inputs):
    row = {}
    try:
        fn = jax.jit(fn)
        t0 = time.perf_counter()
        res = jax.block_until_ready(fn(*inputs))
        row["first_call_s"] = time.perf_counter() - t0
        leaves = jax.tree.leaves(res)
        row["fingerprint"] = [fingerprint(x) for x in leaves]
        row["finite"] = all(bool(np.isfinite(x).all()) for x in leaves)
        times = []
        for _ in range(args.repeats):
            t0 = time.perf_counter()
            jax.block_until_ready(fn(*inputs))
            times.append(time.perf_counter() - t0)
        row |= {
            "median_s": float(np.median(times)),
            "min_s": float(np.min(times)),
            "times_s": times,
        }
    except Exception as e:
        if "RESOURCE_EXHAUSTED" not in str(e):
            raise
        row |= {"oom": True, "error": str(e).splitlines()[0][:300]}
        jax.clear_caches()
    row["peak_bytes"] = peak_bytes()
    return row


pool = load(args.pool)
calc = Ewald()
dtype = np.dtype(args.dtype)

for batch in batches:
    atoms = build(pool, batch)
    for a in atoms:
        a.positions = a.positions.astype(dtype)
        a.cell = a.cell.array.astype(dtype)
    t0 = time.perf_counter()
    charges, sr, nopbc, pbc = calc.prepare(atoms, num_k=args.num_k)
    prepare_s = time.perf_counter() - t0
    inputs = jax.device_put((jnp.asarray(charges, dtype), sr, nopbc, pbc))
    info = {
        "batch": batch["name"],
        "set": batch["set"],
        "size": batch["size"],
        "atoms": batch["atoms"],
        "structures": batch["structures"],
        "pairs": int(np.asarray(sr.pair_mask).sum()),
        "pairs_nonpbc": int(np.asarray(nopbc.pair_mask).sum()),
        "padded": {
            "atoms": len(charges),
            "pairs": len(sr.centers),
            "pairs_nonpbc": len(nopbc.centers),
        },
        "prepare_s": prepare_s,
    }
    has_pbc = any("T" in k for k in batch["pbc"])
    for op, fn in ops(calc, has_pbc).items():
        row = info | {"op": op} | measure(fn, inputs)
        emit(row)
        med = "OOM" if row.get("oom") else f"{1e3 * row['median_s']:9.3f} ms"
        first = row.get("first_call_s", 0)
        print(
            f"{args.label:8} {batch['name']:16} {op:22} {med}  1st {first:5.1f} s",
            flush=True,
        )
