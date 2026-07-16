"""Adversarial tests targeting the tile-dispatch kernel and batching invariants.

Assumptions under test:

  [K1] atom_off[b] % BM == 0 for every b
       group_p2 = atom_off[b] // BM + mt must land in the right bin.

  [K2] n_kvec_tiles == 1: K_pad == BK (single k-tile per system)
       group_p1 = b * 1 + 0 = b — must still be correct.

  [K6] BM=1, BK=1 (degenerate tiling) gives the same energy as defaults.

  [B1] _build_dispatch_table produces exhaustive, disjoint coverage.
       Every (b, m_tile, k_tile) triple appears exactly once in dispatch_table.
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
    """When BM > max(N_b), every system uses a single atom-tile; dispatch table minimal."""
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
    c_ref, b_ref, bnp_ref, bp_ref = calc_ref.prepare(
        [a], num_k=_NUM_K, cutoff=_CUTOFF, BK=32
    )
    e_ref = float(
        np.array(calc_ref.energy(c_ref, b_ref, bnp_ref, bp_ref))[b_ref.structure_mask][0]
    )

    calc = Ewald(prefactor=1.0)
    c, b, bnp, bp = calc.prepare([a], num_k=_NUM_K, cutoff=_CUTOFF, BK=BK)
    n_kvec_tiles = bp.k_grid.shape[1] // bp.BK.shape[0]
    assert n_kvec_tiles == 1, (
        f"BK={BK} still gives {n_kvec_tiles} k-tiles with num_k={_NUM_K}; increase BK"
    )
    e = float(np.array(calc.energy(c, b, bnp, bp))[b.structure_mask][0])
    np.testing.assert_allclose(
        e,
        e_ref,
        rtol=1e-10,
        err_msg="Single-k-tile (n_kvec_tiles=1) gives different energy",
    )


# ---------------------------------------------------------------------------
# [B1] Dispatch table coverage: exhaustive, disjoint (b, atom-tile, k-tile) triples
# ---------------------------------------------------------------------------


def test_dispatch_table_exhaustive_coverage():
    """dispatch_table must contain every (b, m_tile, k_tile) triple exactly once."""
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

    table = np.array(bp.dispatch_table)
    actual = set(map(tuple, table.tolist()))

    assert actual == expected, (
        f"dispatch_table coverage mismatch.\n"
        f"  Missing: {expected - actual}\n"
        f"  Extra:   {actual - expected}"
    )


# ---------------------------------------------------------------------------
# get_batch: empty samples, dtype threading, size-reserve contract
# ---------------------------------------------------------------------------


def _make_random_nonpbc(n, seed=0, box=3.0):
    rng = np.random.default_rng(seed)
    atoms = Atoms(numbers=[1] * n, positions=rng.uniform(0, box, (n, 3)), pbc=False)
    atoms.set_initial_charges(np.zeros(n))
    return atoms


_TINY_SIZES = dict(
    num_structures=2,
    num_structures_pbc=1,
    num_atoms=8,
    num_atoms_pbc=32,
    num_pairs=16,
    num_pairs_nonpbc=4,
    num_k=128,
)


def test_get_batch_empty_samples():
    """`samples=[]` with explicit dtype yields a pure-padding batch: identity
    cells, all-False masks, every atom mapped to the padding structure, and
    padding pairs at atom row 0 (`padding_atom_idx = total_atoms = 0`) —
    masked and position-zero either way, pinned so the convention is
    deliberate. The batch must run through the kernel and yield exact zeros."""
    from jaxpme.batched_tiled.batching import get_batch
    from jaxpme.batched_tiled.calculators import Ewald

    charges, sr, bnp, bp = get_batch([], dtype=np.float64, **_TINY_SIZES)

    np.testing.assert_array_equal(sr.cell, np.broadcast_to(np.eye(3), sr.cell.shape))
    for mask in (
        sr.structure_mask,
        sr.atom_mask,
        sr.pair_mask,
        sr.pbc_mask,
        bnp.pair_mask,
        bp.structure_mask,
        bp.pbc_atom_mask,
    ):
        assert not mask.any()
    np.testing.assert_array_equal(sr.atom_to_structure, _TINY_SIZES["num_structures"] - 1)
    np.testing.assert_array_equal(sr.centers, 0)
    np.testing.assert_array_equal(sr.others, 0)
    np.testing.assert_array_equal(sr.smearing, 1.0)

    e = np.array(Ewald(prefactor=1.0).energy(charges, sr, bnp, bp))
    np.testing.assert_array_equal(e, 0.0)


def test_get_batch_empty_samples_requires_dtype():
    from jaxpme.batched_tiled.batching import get_batch

    with pytest.raises(ValueError, match="dtype"):
        get_batch([], **_TINY_SIZES)


def test_get_batch_int_dtype():
    """`int_dtype` sets the NL index arrays; the kernel-internal flat-layout
    arrays stay int32 regardless; default is int64."""
    from jaxpme.batched_tiled.batching import get_batch, prepare

    structure = prepare(_make_random_pbc(4, seed=3), num_k=_NUM_K, cutoff=_CUTOFF)
    sizes = {**_TINY_SIZES, "num_atoms_pbc": 64, "num_pairs": 512}

    _, sr, bnp, bp = get_batch([structure], int_dtype=np.int32, **sizes)
    for arr in (
        sr.centers,
        sr.others,
        sr.cell_shifts,
        sr.atom_to_structure,
        sr.pair_to_structure,
        bnp.centers,
        bnp.others,
        bp.structure_to_structure,
    ):
        assert arr.dtype == np.int32
    for arr in (bp.pbc_atom_off, bp.pbc_segment_atom, bp.pbc_to_flat, bp.dispatch_table):
        assert arr.dtype == np.int32

    _, sr64, _, _ = get_batch([structure], **sizes)
    assert sr64.centers.dtype == np.int64


def test_get_batch_minimal_sizes_contract():
    """The size-reserve contract (see `sample_shapes`): `get_batch` succeeds
    at exactly the minimal sizes and rejects anything below each axis's assert
    boundary. `next_size` clamps its minimum to 1, so reserve-0 axes bottom
    out at max(need, 1). `num_atoms_pbc` is BM-rounded *up* after its assert:
    its boundary is sum(padded) + 1, and anything in [sum+1, sum+BM] lands on
    the same final sum + BM."""
    from jaxpme.batched_tiled.batching import get_batch, prepare, sample_shapes

    BM, BK = 8, 16
    structures = [
        prepare(_make_random_pbc(5, seed=1), num_k=_NUM_K, cutoff=_CUTOFF),
        prepare(_make_random_nonpbc(3, seed=2), num_k=_NUM_K, cutoff=_CUTOFF),
    ]
    shapes = [sample_shapes(s, BM=BM) for s in structures]
    pbc_padded_sum = sum(s["n_atoms_pbc"] for s in shapes)

    # exact assert boundary per axis (the minimum that succeeds)
    boundary = dict(
        num_structures=len(structures) + 1,
        num_structures_pbc=max(sum(s["is_pbc"] for s in shapes), 1),
        num_atoms=sum(s["n_atoms"] for s in shapes) + 1,
        num_atoms_pbc=pbc_padded_sum + 1,
        num_pairs=sum(s["n_pairs"] for s in shapes) + 1,
        num_pairs_nonpbc=max(sum(s["n_pairs_nonpbc"] for s in shapes), 1),
        num_k=max(max(s["num_k"] for s in shapes), 1),
    )

    # minimal *final* sizes (what a size planner would request)
    minimal = {**boundary, "num_atoms_pbc": pbc_padded_sum + BM}
    get_batch(structures, BM=BM, BK=BK, **minimal)

    # the boundary itself succeeds too; num_atoms_pbc BM-rounds to sum + BM
    _, _, _, bp = get_batch(structures, BM=BM, BK=BK, **boundary)
    assert bp.pbc_segment_atom.shape[0] == pbc_padded_sum + BM

    for axis in boundary:
        below = {**boundary, axis: boundary[axis] - 1}
        with pytest.raises(AssertionError):
            get_batch(structures, BM=BM, BK=BK, **below)


def _make_slab(n=6, L=6.0, vacuum=40.0, seed=5):
    """2D slab with enough vacuum that `shrink_2d_cell` actually shrinks."""
    rng = np.random.default_rng(seed)
    pos = rng.uniform(0, L, (n, 3))
    pos[:, 2] = rng.uniform(0, 2.0, n)
    q = rng.choice([-1.0, 1.0], size=n).astype(np.float64)
    q[-1] = -q[:-1].sum()
    atoms = Atoms(
        numbers=[1] * n,
        positions=pos,
        cell=np.diag([L, L, vacuum]),
        pbc=[True, True, False],
    )
    atoms.set_initial_charges(q)
    return atoms


def test_prepare_2d_keeps_raw_cell_stores_effective():
    """For 2D pbc, `prepare` keeps `structure["cell"]` raw and stores the
    shrunk cell in `structure["effective_cell"]` instead of overwriting."""
    from jaxpme.batched_tiled.batching import prepare

    atoms = _make_slab()
    structure = prepare(atoms, num_k=_NUM_K, cutoff=_CUTOFF)

    np.testing.assert_array_equal(structure["cell"], atoms.get_cell().array)
    eff = structure["effective_cell"]
    np.testing.assert_array_equal(eff[:2], structure["cell"][:2])
    assert np.linalg.norm(eff[2]) < np.linalg.norm(structure["cell"][2])


def test_get_batch_cell_split_and_pbc_rows():
    """`Batch.cell` is raw, `Batch.effective_cell` is what the Ewald math
    consumes, `Batch.pbc` are the per-structure periodic rows; padding
    structures carry identity cells and all-False rows."""
    from jaxpme.batched_tiled.batching import get_batch, prepare

    structures = [
        prepare(_make_random_pbc(4, seed=1), num_k=_NUM_K, cutoff=_CUTOFF),
        prepare(_make_slab(), num_k=_NUM_K, cutoff=_CUTOFF),
        prepare(_make_random_nonpbc(3, seed=2), num_k=_NUM_K, cutoff=_CUTOFF),
    ]
    _, sr, _, _ = get_batch(
        structures,
        num_structures=4,
        num_structures_pbc=2,
        num_atoms=16,
        num_atoms_pbc=96,
        num_pairs=2048,
        num_pairs_nonpbc=8,
        num_k=1024,
    )

    for idx, s in enumerate(structures):
        np.testing.assert_array_equal(sr.cell[idx], s["cell"])
        np.testing.assert_array_equal(
            sr.effective_cell[idx], s.get("effective_cell", s["cell"])
        )
        np.testing.assert_array_equal(sr.pbc[idx], s["pbc"])
    assert (sr.effective_cell[1] != sr.cell[1]).any()  # the slab actually shrunk
    np.testing.assert_array_equal(sr.cell[3], np.eye(3))
    np.testing.assert_array_equal(sr.effective_cell[3], np.eye(3))
    assert not sr.pbc[3].any()


def test_2d_cell_gradient_drops_shrink_artifact():
    """Gradients to the raw `Batch.cell` flow through periodic rows only:
    the 2D shrink is a convergence trick whose cell-gradient is an artifact,
    dropped inside `compose_cell`. The vacuum row's gradient is exactly zero;
    the periodic rows' are not."""
    from jaxpme.batched_tiled.calculators import Ewald

    calc = Ewald(prefactor=1.0)
    charges, sr, bnp, bp = calc.prepare([_make_slab()], num_k=_NUM_K, cutoff=_CUTOFF)

    def total_energy(cell):
        return calc.energy(charges, sr._replace(cell=cell), bnp, bp).sum()

    g = np.array(jax.grad(total_energy)(sr.cell))[0]
    np.testing.assert_array_equal(g[2], 0.0)
    assert np.abs(g[:2]).max() > 0.0


def test_compose_cell_none_passthrough():
    """`effective_cell=None` (legacy / hand-built batches) means the cell is
    already effective — `compose_cell` returns it untouched."""
    from jaxpme.batched_tiled.batching import Batch
    from jaxpme.utils import compose_cell

    cell = np.arange(9.0).reshape(1, 3, 3)
    batch = Batch(
        positions=None,
        centers=None,
        others=None,
        cell_shifts=None,
        distances=None,
        cell=cell,
        smearing=None,
        atom_mask=None,
        pair_mask=None,
        structure_mask=None,
        pbc_mask=None,
        atom_to_structure=None,
        pair_to_structure=None,
    )
    assert compose_cell(batch) is cell


def test_prepare_nonpbc_keeps_identity_cell():
    """to_structure normalizes zero non-PBC cells to the identity; `prepare`
    must keep that (a raw zero cell is singular under inv() downstream, e.g.
    for cells indexed by padding pbc rows)."""
    from jaxpme.batched_tiled.batching import get_batch, prepare

    rng = np.random.default_rng(0)
    atoms = Atoms(numbers=[1] * 4, positions=rng.uniform(0, 3.0, (4, 3)), pbc=False)

    structure = prepare(atoms, num_k=_NUM_K, cutoff=_CUTOFF)
    np.testing.assert_array_equal(structure["cell"], np.eye(3))

    _, sr, _, _ = get_batch(
        [structure],
        num_structures=2,
        num_structures_pbc=1,
        num_atoms=8,
        num_atoms_pbc=32,
        num_pairs=32,
        num_pairs_nonpbc=8,
        num_k=128,
        BM=32,
        BK=128,
    )
    np.testing.assert_array_equal(sr.cell[0], np.eye(3))
