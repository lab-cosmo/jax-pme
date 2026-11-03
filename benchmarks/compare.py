"""Compare benchmark results: one markdown table per op, arms as columns.

    python benchmarks/compare.py main.jsonl branch.jsonl [...]

The first arm is the reference: ratios are reference / arm (> 1 means faster),
`dev` is the largest relative deviation of the output fingerprints from it.
Results from different devices are never mixed in one table.
"""

import json
import sys
from collections import defaultdict


def load(paths):
    rows = []
    for path in paths:
        with open(path) as f:
            rows += [json.loads(line) for line in f if line.strip()]
    return rows


def deviation(a, b):
    # per output: fingerprint differences relative to its |x| sum (sums of forces are ~0)
    return max(
        max(abs(x - y) for x, y in zip(fa, fb)) / max(fa[1], fb[1], 1e-30)
        for fa, fb in zip(a, b)
    )


def table(rows, op):
    arms = list(dict.fromkeys(r["label"] for r in rows))
    by = defaultdict(dict)
    for r in rows:
        by[r["batch"]][r["label"]] = r
    ref = arms[0]
    head = (
        ["batch", "pairs"]
        + [f"{a} ms" for a in arms]
        + [f"{ref}/{a}" for a in arms[1:]]
        + ["dev"]
    )
    lines = ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    for batch, cells in by.items():
        ms = [
            "OOM"
            if cells.get(a, {}).get("oom")
            else f"{1e3 * cells[a]['median_s']:.3f}"
            if a in cells
            else "-"
            for a in arms
        ]
        ratios, devs = [], []
        r0 = cells.get(ref)
        for a in arms[1:]:
            r = cells.get(a)
            ok = r0 and r and "median_s" in r0 and "median_s" in r
            ratios.append(f"{r0['median_s'] / r['median_s']:.2f}" if ok else "-")
            if ok:
                devs.append(deviation(r0["fingerprint"], r["fingerprint"]))
        pairs = "/".join(
            str(cells[a]["pairs"] + cells[a]["pairs_nonpbc"]) for a in arms if a in cells
        )
        dev = f"{max(devs):.1e}" if devs else "-"
        lines.append("| " + " | ".join([batch, pairs] + ms + ratios + [dev]) + " |")
    return "\n".join(lines)


def main(paths):
    rows = load(paths)
    for device in dict.fromkeys(r["device"] for r in rows):
        sub = [r for r in rows if r["device"] == device]
        arms = {
            r["label"]: (r["rev"] or r["jaxpme_version"], r["jax"], r["dtype"]) for r in sub
        }
        print(f"## {device}\n")
        for label, (rev, jax, dtype) in arms.items():
            print(f"- `{label}`: jax-pme {rev}, jax {jax}, {dtype}")
        for op in dict.fromkeys(r["op"] for r in sub):
            print(f"\n### {op}\n")
            print(table([r for r in sub if r["op"] == op], op))
        print()


if __name__ == "__main__":
    main(sys.argv[1:])
