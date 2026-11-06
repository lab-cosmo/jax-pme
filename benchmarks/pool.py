"""Structure pool: random samples of public datasets, stored as ragged npz files.

The npz files are not in the repo; `pool.json` lists them with sha256s, and
`load` refuses files that don't match. Per file (S structures, N atoms):
`n_atoms` (S,), `numbers` (N,), `positions` (N, 3), `cell` (S, 3, 3), `pbc` (S, 3),
`index` (S,) row in the source split, and dataset-specific extras (`charge`).
"""

import numpy as np

import hashlib
import json
from pathlib import Path

HERE = Path(__file__).parent
MANIFEST = json.loads((HERE / "pool.json").read_text())


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class Dataset:
    def __init__(self, path):
        d = np.load(path)
        self.n_atoms = d["n_atoms"].astype(int)
        self.offsets = np.concatenate([[0], np.cumsum(self.n_atoms)])
        self.numbers, self.positions = d["numbers"], d["positions"]
        self.cell, self.pbc = d["cell"], d["pbc"]
        self.charge = (
            d["charge"].astype(int) if "charge" in d else np.zeros(len(self.n_atoms), int)
        )

    def __len__(self):
        return len(self.n_atoms)

    def atoms(self, i):
        from ase import Atoms

        sl = slice(self.offsets[i], self.offsets[i + 1])
        return Atoms(
            numbers=self.numbers[sl],
            positions=self.positions[sl],
            cell=self.cell[i],
            pbc=self.pbc[i],
        )


def load(pool_dir):
    out = {}
    for name, entry in MANIFEST["datasets"].items():
        path = Path(pool_dir) / entry["file"]
        if sha256(path) != entry["sha256"]:
            raise ValueError(f"{path} does not match pool.json (sha256)")
        out[name] = Dataset(path)
    return out


def charges(n_atoms, totals, seed):
    """Per-atom charges ~ N(0, 0.4), shifted so each structure sums to its total."""
    rng = np.random.default_rng(seed)
    q = []
    for n, total in zip(n_atoms, totals):
        x = rng.normal(0.0, 0.4, size=n)
        q.append(x - x.mean() + total / n)
    return q


def build(pool, batch):
    """ase.Atoms with initial charges for one entry of batches.json."""
    members = [(pool[name], i) for name, i in batch["members"]]
    q = charges(
        [ds.n_atoms[i] for ds, i in members],
        [ds.charge[i] for ds, i in members],
        batch["charge_seed"],
    )
    atoms = []
    for (ds, i), qi in zip(members, q):
        a = ds.atoms(i)
        a.set_initial_charges(qi)
        atoms.append(a)
    return atoms
