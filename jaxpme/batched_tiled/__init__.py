"""Batched Ewald via a tile-dispatched, pure-XLA reciprocal kernel.

Atoms are sum-padded per system into a flat array (variable per-system slot
count); k-vectors are rectangular `[B_pbc, K_pad, 3]` with K_pad shared across
systems. Tile sizes `(BM, BK)` are fixed at batch construction. `num_k` is
required by `prepare` — it sets the per-cell K target via
`lr_wavelength_for_num_k`, and the batcher max-pads to a common K_pad.

Differentiates cleanly through autograd: forces and stress via plain
`jax.value_and_grad`, no `custom_vjp`.
"""

from .calculators import Ewald

__all__ = ["Ewald"]
