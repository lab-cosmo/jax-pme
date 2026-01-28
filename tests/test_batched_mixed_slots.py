import numpy as np
import jax

import pytest
from ase.io import read
from conftest import REFERENCE_STRUCTURES_DIR

jax.config.update("jax_enable_x64", True)


@pytest.mark.parametrize("cutoff", [4.0, 5.0, 6.0])
def test_single_slot_vs_batched_mixed(cutoff):
    """Single slot should give identical results to batched_mixed."""
    from jaxpme.batched_mixed.calculators import Ewald as EwaldMixed
    from jaxpme.batched_mixed_slots.calculators import Ewald as EwaldSlots

    structures = read(REFERENCE_STRUCTURES_DIR / "coulomb_test_frames.xyz", index=":3")

    # batched_mixed
    calc_mixed = EwaldMixed(prefactor=1.0)
    charges_m, batch_m, nopbc_m, pbc_m = calc_mixed.prepare(structures, cutoff)
    energy_m = calc_mixed.energy(charges_m, batch_m, nopbc_m, pbc_m)
    pot_m = calc_mixed.potentials(charges_m, batch_m, nopbc_m, pbc_m)

    # batched_mixed_slots with single slot (no slot_num_k specified)
    calc_slots = EwaldSlots(prefactor=1.0)
    charges_s, batch_s, nopbc_s, pbc_s = calc_slots.prepare(structures, cutoff)
    energy_s = calc_slots.energy(charges_s, batch_s, nopbc_s, pbc_s)
    pot_s = calc_slots.potentials(charges_s, batch_s, nopbc_s, pbc_s)

    # pbc_s is a tuple with one element
    assert len(pbc_s) == 1

    np.testing.assert_allclose(energy_m, energy_s, rtol=1e-10)
    np.testing.assert_allclose(pot_m, pot_s, rtol=1e-10)


@pytest.mark.parametrize("cutoff", [4.0, 5.0, 6.0])
def test_two_slots_vs_single_slot(cutoff):
    """Two slots should give same results as single slot when all samples fit."""
    from jaxpme.batched_mixed_slots.calculators import Ewald

    structures = read(REFERENCE_STRUCTURES_DIR / "coulomb_test_frames.xyz", index=":3")

    calculator = Ewald(prefactor=1.0)

    # Single slot (auto-sized)
    charges_1, batch_1, nopbc_1, pbc_1 = calculator.prepare(structures, cutoff)
    energy_1 = calculator.energy(charges_1, batch_1, nopbc_1, pbc_1)
    pot_1 = calculator.potentials(charges_1, batch_1, nopbc_1, pbc_1)

    # Two slots with large capacities (all samples fit in slot 0)
    charges_2, batch_2, nopbc_2, pbc_2 = calculator.prepare(
        structures,
        cutoff,
        slot_num_k=[100000, 200000],
        slot_num_atoms_pbc=[1000, 2000],
    )
    energy_2 = calculator.energy(charges_2, batch_2, nopbc_2, pbc_2)
    pot_2 = calculator.potentials(charges_2, batch_2, nopbc_2, pbc_2)

    assert len(pbc_2) == 2

    np.testing.assert_allclose(energy_1, energy_2, rtol=1e-10)
    np.testing.assert_allclose(pot_1, pot_2, rtol=1e-10)


@pytest.mark.parametrize("cutoff", [5.0])
def test_slots_partition_samples(cutoff):
    """Test that samples are correctly partitioned into slots."""
    from jaxpme.batched_mixed.kspace import count_halfspace_kvectors
    from jaxpme.batched_mixed_slots.batching import prepare
    from jaxpme.batched_mixed_slots.calculators import Ewald

    structures = read(REFERENCE_STRUCTURES_DIR / "coulomb_test_frames.xyz", index=":6")

    # Prepare samples to see their k counts
    samples = [prepare(atoms, cutoff) for atoms in structures]
    k_counts = []
    for s in samples:
        if hasattr(s["lr"], "k_grid") and s["lr"].k_grid is not None:
            k_counts.append(count_halfspace_kvectors(s["lr"].k_grid.shape))

    k_counts = np.array(k_counts)

    # Choose a threshold that splits samples
    median_k = int(np.median(k_counts))
    threshold_k = median_k

    calculator = Ewald(prefactor=1.0)

    # Single slot for reference
    charges_ref, batch_ref, nopbc_ref, pbc_ref = calculator.prepare(structures, cutoff)
    energy_ref = calculator.energy(charges_ref, batch_ref, nopbc_ref, pbc_ref)

    # Two slots
    charges_2, batch_2, nopbc_2, pbc_2 = calculator.prepare(
        structures,
        cutoff,
        slot_num_k=[threshold_k, max(k_counts) + 100],
        slot_num_atoms_pbc=[
            max(len(s.positions) for s in structures),
            max(len(s.positions) for s in structures),
        ],
    )
    energy_2 = calculator.energy(charges_2, batch_2, nopbc_2, pbc_2)

    # Results should be numerically equivalent
    np.testing.assert_allclose(energy_ref, energy_2, rtol=1e-8)


@pytest.mark.parametrize("cutoff", [5.0])
def test_empty_slot(cutoff):
    """Test that empty slots are handled correctly."""
    from jaxpme.batched_mixed_slots.calculators import Ewald

    structures = read(REFERENCE_STRUCTURES_DIR / "coulomb_test_frames.xyz", index=":3")

    calculator = Ewald(prefactor=1.0)

    # Single slot for reference
    charges_ref, batch_ref, nopbc_ref, pbc_ref = calculator.prepare(structures, cutoff)
    energy_ref = calculator.energy(charges_ref, batch_ref, nopbc_ref, pbc_ref)

    # Two slots where slot 0 is too small for any sample (all go to slot 1)
    charges_2, batch_2, nopbc_2, pbc_2 = calculator.prepare(
        structures,
        cutoff,
        slot_num_k=[1, 100000],  # slot 0 has k=1, won't fit anything
        slot_num_atoms_pbc=[1, 1000],
    )
    energy_2 = calculator.energy(charges_2, batch_2, nopbc_2, pbc_2)

    # Slot 0 should be empty (only padding)
    assert not pbc_2[0].structure_mask.any()
    # Slot 1 should have all samples
    assert pbc_2[1].structure_mask.sum() == len(structures)

    np.testing.assert_allclose(energy_ref, energy_2, rtol=1e-8)


@pytest.mark.parametrize("cutoff", [5.0])
def test_forces_with_slots(cutoff):
    """Test that forces are correct with slots."""
    from jaxpme.batched_mixed.calculators import Ewald as EwaldMixed
    from jaxpme.batched_mixed_slots.calculators import Ewald as EwaldSlots

    structures = read(REFERENCE_STRUCTURES_DIR / "coulomb_test_frames.xyz", index=":3")

    # Reference
    calc_mixed = EwaldMixed(prefactor=1.0)
    charges_m, batch_m, nopbc_m, pbc_m = calc_mixed.prepare(structures, cutoff)
    energy_m, forces_m = calc_mixed.energy_forces(charges_m, batch_m, nopbc_m, pbc_m)

    # Slots
    calc_slots = EwaldSlots(prefactor=1.0)
    charges_s, batch_s, nopbc_s, pbc_s = calc_slots.prepare(
        structures,
        cutoff,
        slot_num_k=[5000, 100000],
        slot_num_atoms_pbc=[100, 500],
    )
    energy_s, forces_s = calc_slots.energy_forces(charges_s, batch_s, nopbc_s, pbc_s)

    np.testing.assert_allclose(energy_m, energy_s, rtol=1e-8)
    np.testing.assert_allclose(forces_m, forces_s, rtol=1e-8)


@pytest.mark.parametrize("cutoff", [5.0])
def test_stress_with_slots(cutoff):
    """Test that stress is correct with slots."""
    from jaxpme.batched_mixed.calculators import Ewald as EwaldMixed
    from jaxpme.batched_mixed_slots.calculators import Ewald as EwaldSlots

    structures = read(REFERENCE_STRUCTURES_DIR / "coulomb_test_frames.xyz", index=":3")

    # Reference
    calc_mixed = EwaldMixed(prefactor=1.0)
    charges_m, batch_m, nopbc_m, pbc_m = calc_mixed.prepare(structures, cutoff)
    energy_m, forces_m, stress_m = calc_mixed.energy_forces_stress(
        charges_m, batch_m, nopbc_m, pbc_m
    )

    # Slots
    calc_slots = EwaldSlots(prefactor=1.0)
    charges_s, batch_s, nopbc_s, pbc_s = calc_slots.prepare(
        structures,
        cutoff,
        slot_num_k=[5000, 100000],
        slot_num_atoms_pbc=[100, 500],
    )
    energy_s, forces_s, stress_s = calc_slots.energy_forces_stress(
        charges_s, batch_s, nopbc_s, pbc_s
    )

    np.testing.assert_allclose(energy_m, energy_s, rtol=1e-8)
    np.testing.assert_allclose(forces_m, forces_s, rtol=1e-8)
    np.testing.assert_allclose(stress_m, stress_s, rtol=1e-8)


@pytest.mark.parametrize("cutoff", [5.0])
def test_mixed_pbc_nopbc_with_slots(cutoff):
    """Test slots with mixed periodic and non-periodic systems."""
    from jaxpme.batched_mixed.calculators import Ewald as EwaldMixed
    from jaxpme.batched_mixed_slots.calculators import Ewald as EwaldSlots

    structures = read(REFERENCE_STRUCTURES_DIR / "coulomb_test_frames.xyz", index=":3")

    # Add a non-periodic copy
    atoms_no_pbc = structures[-1].copy()
    atoms_no_pbc.set_pbc(False)
    mixed_structures = list(structures) + [atoms_no_pbc]

    # Reference
    calc_mixed = EwaldMixed(prefactor=1.0)
    charges_m, batch_m, nopbc_m, pbc_m = calc_mixed.prepare(mixed_structures, cutoff)
    energy_m = calc_mixed.energy(charges_m, batch_m, nopbc_m, pbc_m)

    # Slots
    calc_slots = EwaldSlots(prefactor=1.0)
    charges_s, batch_s, nopbc_s, pbc_s = calc_slots.prepare(
        mixed_structures,
        cutoff,
        slot_num_k=[5000, 100000],
        slot_num_atoms_pbc=[100, 500],
    )
    energy_s = calc_slots.energy(charges_s, batch_s, nopbc_s, pbc_s)

    np.testing.assert_allclose(energy_m, energy_s, rtol=1e-8)


@pytest.mark.parametrize("cutoff", [5.0])
def test_three_slots(cutoff):
    """Test with three slots."""
    from jaxpme.batched_mixed_slots.calculators import Ewald

    structures = read(REFERENCE_STRUCTURES_DIR / "coulomb_test_frames.xyz", index=":6")

    calculator = Ewald(prefactor=1.0)

    # Single slot for reference
    charges_ref, batch_ref, nopbc_ref, pbc_ref = calculator.prepare(structures, cutoff)
    energy_ref = calculator.energy(charges_ref, batch_ref, nopbc_ref, pbc_ref)

    # Three slots
    charges_3, batch_3, nopbc_3, pbc_3 = calculator.prepare(
        structures,
        cutoff,
        slot_num_k=[1000, 10000, 100000],
        slot_num_atoms_pbc=[50, 200, 1000],
    )
    energy_3 = calculator.energy(charges_3, batch_3, nopbc_3, pbc_3)

    assert len(pbc_3) == 3
    np.testing.assert_allclose(energy_ref, energy_3, rtol=1e-8)


@pytest.mark.parametrize("cutoff", [5.0])
def test_jit_compilation(cutoff):
    """Test that the calculator can be JIT compiled."""
    from jaxpme.batched_mixed_slots.calculators import Ewald

    structures = read(REFERENCE_STRUCTURES_DIR / "coulomb_test_frames.xyz", index=":3")

    calculator = Ewald(prefactor=1.0)
    charges, batch, nopbc, pbc = calculator.prepare(
        structures,
        cutoff,
        slot_num_k=[5000, 100000],
        slot_num_atoms_pbc=[100, 500],
    )

    # JIT compile the energy function
    energy_jit = jax.jit(calculator.energy)
    energy = energy_jit(charges, batch, nopbc, pbc)

    # Should not raise and should give reasonable results
    assert not np.isnan(energy).any()
    assert energy[batch.structure_mask].sum() != 0


@pytest.mark.parametrize("cutoff", [5.0])
def test_gradient_flow(cutoff):
    """Test that gradients flow correctly through slots."""
    from jaxpme.batched_mixed_slots.calculators import Ewald

    structures = read(REFERENCE_STRUCTURES_DIR / "coulomb_test_frames.xyz", index=":3")

    calculator = Ewald(prefactor=1.0)
    charges, batch, nopbc, pbc = calculator.prepare(
        structures,
        cutoff,
        slot_num_k=[5000, 100000],
        slot_num_atoms_pbc=[100, 500],
    )

    def loss_fn(charges, batch, nopbc, pbc):
        return calculator.energy(charges, batch, nopbc, pbc).sum()

    # Compute gradient w.r.t. batch (which contains positions)
    grad_fn = jax.grad(loss_fn, argnums=1, allow_int=True)
    grads = grad_fn(charges, batch, nopbc, pbc)

    # Check gradients are not NaN and have reasonable magnitude
    assert not np.isnan(grads.positions).any()
    # Non-padding atoms should have non-zero gradients
    assert (grads.positions[batch.atom_mask] != 0).any()
