import numpy as np

def test_batching():
    """Test end-to-end batching with mixed periodic/non-periodic systems."""
    from ase.build import molecule, bulk

    from jaxpme.batched_mixed.batching import prepare, get_batch

    h2o = molecule("H2O")
    h2o.set_initial_charges([0.4, -0.8, 0.4])

    nacl = bulk("NaCl", "rocksalt", a=5.6)
    nacl.set_initial_charges(np.tile([1.0, -1.0], len(nacl) // 2))

    cutoff = 3.0

    systems = [h2o, nacl]
    samples = []
    for atoms in systems:
        charges, structure, lr = prepare(atoms, cutoff)
        samples.append((charges, structure, lr))

    charges, sr_batch, nonperiodic_batch, periodic_batch = get_batch(samples)

    total_atoms = len(h2o) + len(nacl)
    assert sr_batch.atom_mask.sum() == total_atoms
    assert sr_batch.system_mask.sum() == 2

    # Verify positions for H2O
    h2o_pos_batch = sr_batch.positions[: len(h2o)]
    assert np.allclose(h2o_pos_batch, h2o.get_positions())

    # Verify positions for NaCl
    nacl_pos_batch = sr_batch.positions[len(h2o) : total_atoms]
    assert np.allclose(nacl_pos_batch, nacl.get_positions())

    # Verify periodic and non-periodic batches
    assert periodic_batch.system_mask.sum() == 1
    assert nonperiodic_batch.pair_mask.sum() > 0

    # Verify periodic system indexing
    pbc_idx = 0
    atom_indices = periodic_batch.atom_to_atom[pbc_idx]
    atom_indices = atom_indices[periodic_batch.atom_mask[pbc_idx]]

    nacl_positions_from_batch = sr_batch.positions[atom_indices]
    assert np.allclose(nacl_positions_from_batch, nacl.get_positions())

    # Verify atom_to_system and pair_to_system
    assert sr_batch.atom_to_system[: len(h2o)].sum() == 0  # H2O is system 0
    assert sr_batch.atom_to_system[len(h2o) : total_atoms].sum() == len(
        nacl
    )  # NaCl is system 1

    h2o_pairs = sr_batch.pair_mask & (sr_batch.pair_to_system == 0)
    nacl_pairs = sr_batch.pair_mask & (sr_batch.pair_to_system == 1)
    assert h2o_pairs.sum() > 0
    assert nacl_pairs.sum() > 0


def test_two_systems_sanity():
    """Sanity check that batching two identical systems gives expected results."""
    from ase.build import molecule

    from jaxpme.batched_mixed.batching import prepare, get_batch

    h2o_1 = molecule("H2O")
    h2o_1.set_initial_charges([0.4, -0.8, 0.4])

    h2o_2 = molecule("H2O")
    h2o_2.translate([5.0, 0, 0])
    h2o_2.set_initial_charges([0.4, -0.8, 0.4])

    cutoff = 2.0
    systems = [h2o_1, h2o_2]
    samples = []
    for atoms in systems:
        charges, structure, lr = prepare(atoms, cutoff)
        samples.append((charges, structure, lr))

    charges, sr_batch, nonperiodic_batch, periodic_batch = get_batch(samples)

    assert sr_batch.atom_mask.sum() == 6
    assert sr_batch.system_mask.sum() == 2
    assert periodic_batch.system_mask.sum() == 0

    # Verify positions for the first H2O
    h2o_1_pos_batch = sr_batch.positions[:3]
    assert np.allclose(h2o_1_pos_batch, h2o_1.get_positions())

    # Verify positions for the second H2O
    h2o_2_pos_batch = sr_batch.positions[3:6]
    assert np.allclose(h2o_2_pos_batch, h2o_2.get_positions())

    # Verify total charge is neutral
    total_charge = 0
    for charges, _, _ in samples:
        total_charge += charges.sum()
    assert abs(total_charge) < 1e-10

    assert nonperiodic_batch.pair_mask.sum() > 0

    # Verify atom_to_system for both systems
    assert np.all(sr_batch.atom_to_system[:3] == 0)  # First H2O
    assert np.all(sr_batch.atom_to_system[3:6] == 1)  # Second H2O

    # Verify pair_to_system
    valid_pairs = sr_batch.pair_mask
    pair_systems = sr_batch.pair_to_system[valid_pairs]
    assert np.all((pair_systems == 0) | (pair_systems == 1))


def test_padding():
    """Test that padding works correctly."""
    from ase.build import molecule

    from jaxpme.batched_mixed.batching import prepare, get_batch

    h2 = molecule("H2")
    h2.set_initial_charges([0.5, -0.5])

    charges, structure, lr = prepare(h2, cutoff=2.0)
    samples = [(charges, structure, lr)]

    charges, sr_batch, nonperiodic_batch, periodic_batch = get_batch(
        samples, num_systems=8, num_atoms=16, num_pairs=32, num_pairs_nonpbc=8
    )

    assert len(sr_batch.system_mask) == 8
    assert len(sr_batch.atom_mask) == 16
    assert len(sr_batch.pair_mask) == 32
    assert len(nonperiodic_batch.pair_mask) == 8

    assert sr_batch.system_mask.sum() == 1
    assert sr_batch.atom_mask.sum() == 2

    # Verify atom_to_system and pair_to_system for padding
    padding_system = 7  # num_systems - 1
    assert np.all(sr_batch.atom_to_system[~sr_batch.atom_mask] == padding_system)
    assert np.all(sr_batch.pair_to_system[~sr_batch.pair_mask] == padding_system)
