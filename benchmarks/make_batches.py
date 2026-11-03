"""Freeze benchmark batches from the structure pool into `batches.json`.

Run once. `batches.json` is committed and stores members explicitly, so batches
stay fixed even if this script changes; rerun only to define a new batch set.

    python benchmarks/make_batches.py --pool DIR [--out benchmarks/batches.json]
"""

import numpy as np

import argparse
import json
from collections import Counter
from pathlib import Path

from pool import load

SEED = 20261104
SIZES = {"small": (2048, 3), "large": (24576, 1)}  # atom budget, draws
SETS = {
    "omol": ["omol25"],
    "omat": ["omat24"],
    "mad": ["mad16"],
    "joint": ["omol25", "omat24", "mad16"],  # equal atom shares
}


def fill(n_atoms, budget, rng, patience=20):
    """Seeded greedy fill: add shuffled structures while they fit the budget."""
    chosen, total, misses = [], 0, 0
    for i in rng.permutation(len(n_atoms)):
        if total + n_atoms[i] <= budget:
            chosen.append(int(i))
            total += int(n_atoms[i])
            misses = 0
        else:
            misses += 1
            if misses >= patience:
                break
    return chosen


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--pool", required=True)
    p.add_argument("--out", default=Path(__file__).parent / "batches.json")
    args = p.parse_args()

    pool = load(args.pool)
    batches = []
    for s, (set_name, names) in enumerate(SETS.items()):
        for z, (size, (budget, draws)) in enumerate(SIZES.items()):
            for d in range(draws):
                rng = np.random.default_rng([SEED, s, z, d])
                members = []
                for name in names:
                    share = budget // len(names)
                    members += [[name, i] for i in fill(pool[name].n_atoms, share, rng)]
                n = [int(pool[name].n_atoms[i]) for name, i in members]
                pbc = Counter(
                    "".join("T" if x else "F" for x in pool[name].pbc[i])
                    for name, i in members
                )
                batches.append(
                    {
                        "name": f"{set_name}-{size}-{d}",
                        "set": set_name,
                        "size": size,
                        "budget": budget,
                        "atoms": sum(n),
                        "structures": len(members),
                        "pbc": dict(sorted(pbc.items())),
                        "charge_seed": [SEED, s, z, d, 1],
                        "members": members,
                    }
                )
                name = batches[-1]["name"]
                print(f"{name:16} {sum(n):6d} atoms {len(n):5d} structures {dict(pbc)}")

    doc = {"seed": SEED, "sizes": SIZES, "sets": SETS, "batches": batches}
    Path(args.out).write_text(json.dumps(doc, separators=(",", ":")) + "\n")


if __name__ == "__main__":
    main()
