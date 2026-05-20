"""Adversarial tests targeting the tile-dispatch kernel and batching invariants.

Assumptions under test:

  [K1] atom_off[b] % BM == 0 for every b
       group_p2 = atom_off[b] // BM + mt must land in the right bin.

  [K2] n_kvec_tiles == 1: K_pad == BK (single k-tile per system)
       group_p1 = b * 1 + 0 = b — must still be correct.

  [K6] BM=1, BK=1 (degenerate tiling) gives the same energy as defaults.

  [B1] _build_dispatch_tables produces exhaustive, disjoint coverage.
       Every (b, atom-tile, k-tile) triple appears exactly once in pass2_flat.
"""

import numpy as np
import jax

import pytest
from ase import Atoms

jax.config.update("jax_enable_x64", True)

_CUTOFF = 4.0
_NUM_K = 100


def _make_random_pbc(n, seed=0, box=6.0):
    rng = np.random.default_rng(seed)
    pos = rng.uniform(0, box, (n, 3))
    q = rng.choice([-1.0, 1.0], size=n).astype(np.float64)
    q[-1] = -q[:-1].sum()
    atoms = Atoms(numbers=[1] * n, positions=pos, cell=np.diag([box, box, box]), pbc=True)
    atoms.set_initial_charges(q)
    return atoms


def _energy_of(atoms, BM=32, BK=128):
    from jaxpme.batched_tiled.calculators import Ewald

    calc = Ewald(prefactor=1.0)
    c, b, bnp, bp = calc.prepare([atoms], num_k=_NUM_K, cutoff=_CUTOFF, BM=BM, BK=BK)
    return float(np.array(calc.energy(c, b, bnp, bp))[b.structure_mask][0])


# ---------------------------------------------------------------------------
# [K6] BM=1, BK=1 (degenerate tiling) matches default tile sizes
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("n_atoms", [3, 5, 8])
def test_bm1_bk1_matches_default(n_atoms):
    """BM=BK=1 (every tile is one atom × one k-vec) must give the same energy."""
    a = _make_random_pbc(n_atoms, seed=77)
    np.testing.assert_allclose(
        _energy_of(a, BM=1, BK=1),
        _energy_of(a, BM=32, BK=128),
        rtol=1e-10,
        err_msg=f"BM=1,BK=1 gives different energy than default for n_atoms={n_atoms}",
    )


# ---------------------------------------------------------------------------
# [K6] BM larger than all atom counts → exactly one atom-tile per system
# ---------------------------------------------------------------------------

def test_large_bm_one_tile_per_system():
    """When BM > max(N_b), every system uses a single atom-tile; dispatch table is minimal."""
    from jaxpme.batched_tiled.calculators import Ewald

    systems = [_make_random_pbc(n, seed=i) for i, n in enumerate([3, 5, 7])]
    BM = 64  # larger than all atom counts

    calc = Ewald(prefactor=1.0)
    c, b, bnp, bp = calc.prepare(systems, num_k=_NUM_K, cutoff=_CUTOFF, BM=BM)
    e_batch = np.array(calc.energy(c, b, bnp, bp))[b.structure_mask]

    for i, atoms in enumerate(systems):
        np.testing.assert_allclose(
            e_batch[i],
            _energy_of(atoms, BM=BM),
            rtol=1e-10,
            err_msg=f"BM={BM} (one-tile-per-system): system {i} energy mismatch",
        )


# ---------------------------------------------------------------------------
# [K2] n_kvec_tiles=1: K_pad == BK (single k-tile per system)
# ---------------------------------------------------------------------------

def test_single_kvec_tile():
    """When BK covers all k-vecs in one tile, group_p1 = b; must still be correct."""
    from jaxpme.batched_tiled.calculators import Ewald

    a = _make_random_pbc(5, seed=13)
    BK = 256

    calc_ref = Ewald(prefactor=1.0)
    c_ref, b_ref, bnp_ref, bp_ref = calc_ref.prepare([a], num_k=_NUM_K, cutoff=_CUTOFF, BK=32)
    e_ref = float(np.array(calc_ref.energy(c_ref, b_ref, bnp_ref, bp_ref))[b_ref.structure_mask][0])

    calc = Ewald(prefactor=1.0)
    c, b, bnp, bp = calc.prepare([a], num_k=_NUM_K, cutoff=_CUTOFF, BK=BK)
    n_kvec_tiles = bp.k_grid.shape[1] // bp.BK.shape[0]
    assert n_kvec_tiles == 1, (
        f"BK={BK} still gives {n_kvec_tiles} k-tiles with num_k={_NUM_K}; increase BK"
    )
    e = float(np.array(calc.energy(c, b, bnp, bp))[b.structure_mask][0])
    np.testing.assert_allclose(
        e, e_ref, rtol=1e-10, err_msg="Single-k-tile (n_kvec_tiles=1) gives different energy"
    )


# ---------------------------------------------------------------------------
# [B1] Dispatch table coverage: exhaustive, disjoint (b, atom-tile, k-tile) triples
# ---------------------------------------------------------------------------

def test_dispatch_table_exhaustive_coverage():
    """pass2_flat must contain every (b, atom-tile, k-tile) triple exactly once."""
    from jaxpme.batched_tiled.calculators import Ewald

    systems = [_make_random_pbc(n, seed=i) for i, n in enumerate([3, 7, 5, 2])]
    BM, BK = 4, 4

    calc = Ewald(prefactor=1.0)
    c, b, bnp, bp = calc.prepare(systems, num_k=_NUM_K, cutoff=_CUTOFF, BM=BM, BK=BK)

    pbc_atom_off = np.array(bp.pbc_atom_off)
    n_kvec_tiles = bp.k_grid.shape[1] // BK
    B_pbc = len(pbc_atom_off) - 1

    expected = set()
    for b_idx in range(B_pbc):
        n_atoms_b = int(pbc_atom_off[b_idx + 1] - pbc_atom_off[b_idx])
        assert n_atoms_b % BM == 0, (
            f"System {b_idx}: sum-padded atom count {n_atoms_b} not divisible by BM={BM}"
        )
        n_atom_tiles = n_atoms_b // BM
        for mt in range(n_atom_tiles):
            for kt in range(n_kvec_tiles):
                expected.add((b_idx, mt, kt))

    pass2 = np.array(bp.pass2_flat)
    actual = set(map(tuple, pass2.tolist()))

    assert actual == expected, (
        f"pass2_flat coverage mismatch.\n"
        f"  Missing: {expected - actual}\n"
        f"  Extra:   {actual - expected}"
    )
