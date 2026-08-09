"""The kernel must stay tiled: no intermediate may scale with K_pad.

`per_triple` slices its work tile out of `[B, K_pad, ...]` arrays. Indexing by
`b` first (`kvec[b]`) turns into a `[T, K_pad, 3]` gather under vmap -- the
whole k-axis per tile, which is the matrix the tiling exists to avoid. XLA
fuses it away at small sizes and gives up as `T · K_pad` grows, so the cost
appears suddenly; these tests pin it at a size where it shows.

`memory_analysis()` reads XLA's own scratch accounting off the compiled
executable -- no execution, no timing flake.
"""

import numpy as np
import jax

import pytest
from ase.io import read
from conftest import REFERENCE_STRUCTURES_DIR

jax.config.update("jax_enable_x64", True)


def _tiled_case(cutoff=4.0, lr_div=16.0, BM=32, BK=128):
    from jaxpme.batched_tiled.calculators import Ewald

    atoms = read(REFERENCE_STRUCTURES_DIR / "coulomb_test_frames.xyz", index=3)
    L = np.linalg.norm(atoms.get_cell().array, axis=-1)
    num_k = int(np.prod(L) / (2 * (cutoff / lr_div) ** 3))

    calculator = Ewald(prefactor=1.0)
    args = calculator.prepare(
        [atoms], num_k=num_k, cutoff=cutoff, smearing=cutoff / 8, BM=BM, BK=BK
    )
    return calculator, args


def _scratch_bytes(fn, args):
    return jax.jit(fn).lower(*args).compile().memory_analysis().temp_size_in_bytes


@pytest.mark.parametrize("method", ["potentials", "energy_forces", "energy_forces_stress"])
def test_kernel_scratch_scales_with_tile_not_kpad(method):
    """Scratch must track the [T, BM, BK] trig blocks, not T·K_pad.

    A correctly tiled kernel sits at 3-5x the trig block across sizes; one that
    gathers `kvec[b]` per tile runs an order of magnitude above that.
    """
    calculator, args = _tiled_case()
    _, _, _, batch_pbc = args

    T = batch_pbc.dispatch_table.shape[0]
    BM = batch_pbc.BM.shape[0]
    BK = batch_pbc.BK.shape[0]
    K_pad = batch_pbc.k_grid.shape[1]
    itemsize = np.dtype(np.float64).itemsize

    trig_block = T * BM * BK * itemsize
    gathered_kaxis = T * K_pad * 3 * itemsize
    scratch = _scratch_bytes(getattr(calculator, method), args)

    assert scratch < 16 * trig_block, (
        f"{method}: {scratch:,} B scratch is {scratch / trig_block:.1f}x the "
        f"{trig_block:,} B of trig blocks -- an intermediate is scaling with "
        f"K_pad={K_pad} (one gathered k-axis would be {gathered_kaxis:,} B)"
    )


def test_scratch_grows_with_bk_not_kpad():
    """Refining the k-grid at fixed BK must not blow up scratch superlinearly.

    K_pad grows 8x between these two cases while T grows 8x with it, so a
    correctly tiled kernel grows ~8x (more tiles) and a gathering one grows
    ~64x (more tiles, each holding a longer k-axis).
    """
    coarse_calc, coarse = _tiled_case(lr_div=8.0)
    fine_calc, fine = _tiled_case(lr_div=16.0)

    k_ratio = fine[3].k_grid.shape[1] / coarse[3].k_grid.shape[1]
    t_ratio = fine[3].dispatch_table.shape[0] / coarse[3].dispatch_table.shape[0]

    coarse_bytes = _scratch_bytes(coarse_calc.energy_forces, coarse)
    fine_bytes = _scratch_bytes(fine_calc.energy_forces, fine)
    growth = fine_bytes / coarse_bytes

    assert growth < 4 * t_ratio, (
        f"scratch grew {growth:.1f}x for a {t_ratio:.0f}x tile-count increase "
        f"(K_pad grew {k_ratio:.0f}x) -- an intermediate is scaling with K_pad"
    )
