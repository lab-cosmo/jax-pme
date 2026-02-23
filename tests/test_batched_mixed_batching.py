import numpy as np


def test_batching():
    """Test end-to-end batching with mixed periodic/non-periodic structures."""
    from ase.build import bulk, molecule

    from jaxpme.batched_mixed.batching import get_batch, prepare

    h2o = molecule("H2O")
    h2o.set_initial_charges([0.4, -0.8, 0.4])

    nacl = bulk("NaCl", "rocksalt", a=5.6)
    nacl.set_initial_charges(np.tile([1.0, -1.0], len(nacl) // 2))

    cutoff = 3.0

    structures = [h2o, nacl]
    samples = []
    samples = [prepare(atoms, cutoff) for atoms in structures]

    charges, sr_batch, nonperiodic_batch, periodic_batch = get_batch(samples)

    total_atoms = len(h2o) + len(nacl)
    assert sr_batch.atom_mask.sum() == total_atoms
    assert sr_batch.structure_mask.sum() == 2

    # Verify positions for H2O
    h2o_pos_batch = sr_batch.positions[: len(h2o)]
    assert np.allclose(h2o_pos_batch, h2o.get_positions())

    # Verify positions for NaCl
    nacl_pos_batch = sr_batch.positions[len(h2o) : total_atoms]
    assert np.allclose(nacl_pos_batch, nacl.get_positions())

    # Verify periodic and non-periodic batches
    assert periodic_batch.structure_mask.sum() == 1
    assert nonperiodic_batch.pair_mask.sum() > 0

    # Verify periodic structure indexing
    pbc_idx = 0
    atom_indices = periodic_batch.atom_to_atom[pbc_idx]
    atom_indices = atom_indices[periodic_batch.atom_mask[pbc_idx]]

    nacl_positions_from_batch = sr_batch.positions[atom_indices]
    assert np.allclose(nacl_positions_from_batch, nacl.get_positions())

    # Verify atom_to_structure and pair_to_structure
    assert sr_batch.atom_to_structure[: len(h2o)].sum() == 0  # H2O is structure 0
    assert sr_batch.atom_to_structure[len(h2o) : total_atoms].sum() == len(
        nacl
    )  # NaCl is structure 1

    h2o_pairs = sr_batch.pair_mask & (sr_batch.pair_to_structure == 0)
    nacl_pairs = sr_batch.pair_mask & (sr_batch.pair_to_structure == 1)
    assert h2o_pairs.sum() > 0
    assert nacl_pairs.sum() > 0


def test_two_structures_sanity():
    """Sanity check that batching two identical structures gives expected results."""
    from ase.build import molecule

    from jaxpme.batched_mixed.batching import get_batch, prepare

    h2o_1 = molecule("H2O")
    h2o_1.set_initial_charges([0.4, -0.8, 0.4])

    h2o_2 = molecule("H2O")
    h2o_2.translate([5.0, 0, 0])
    h2o_2.set_initial_charges([0.4, -0.8, 0.4])

    cutoff = 2.0
    structures = [h2o_1, h2o_2]
    samples = [prepare(atoms, cutoff) for atoms in structures]

    charges, sr_batch, nonperiodic_batch, periodic_batch = get_batch(samples)

    assert sr_batch.atom_mask.sum() == 6
    assert sr_batch.structure_mask.sum() == 2
    assert periodic_batch.structure_mask.sum() == 0

    # Verify positions for the first H2O
    h2o_1_pos_batch = sr_batch.positions[:3]
    assert np.allclose(h2o_1_pos_batch, h2o_1.get_positions())

    # Verify positions for the second H2O
    h2o_2_pos_batch = sr_batch.positions[3:6]
    assert np.allclose(h2o_2_pos_batch, h2o_2.get_positions())

    # Verify total charge is neutral
    total_charge = 0
    for structure in samples:
        total_charge += structure["charges"].sum()
    assert abs(total_charge) < 1e-10

    assert nonperiodic_batch.pair_mask.sum() > 0

    # Verify atom_to_structure for both structures
    assert np.all(sr_batch.atom_to_structure[:3] == 0)  # First H2O
    assert np.all(sr_batch.atom_to_structure[3:6] == 1)  # Second H2O

    # Verify pair_to_structure
    valid_pairs = sr_batch.pair_mask
    pair_structures = sr_batch.pair_to_structure[valid_pairs]
    assert np.all((pair_structures == 0) | (pair_structures == 1))


def test_padding():
    """Test that padding works correctly."""
    from ase.build import molecule

    from jaxpme.batched_mixed.batching import get_batch, prepare

    h2 = molecule("H2")
    h2.set_initial_charges([0.5, -0.5])
    diags = [7.0, 8.0, 9.0]
    h2.set_cell(np.diag(diags))
    h2.set_pbc([True, True, True])
    lr_wavelength = 2.0

    samples = [prepare(h2, cutoff=2.0, lr_wavelength=lr_wavelength)]

    charges, sr_batch, nonperiodic_batch, periodic_batch = get_batch(
        samples, num_structures=8, num_atoms=16, num_pairs=32, num_pairs_nonpbc=8
    )

    assert len(sr_batch.structure_mask) == 8
    assert len(sr_batch.atom_mask) == 16
    assert len(sr_batch.pair_mask) == 32
    assert len(nonperiodic_batch.pair_mask) == 8

    assert sr_batch.structure_mask.sum() == 1
    assert sr_batch.atom_mask.sum() == 2

    # Lazy check to verify that all generated, non-padded k-vectors are different
    # With halfspace=True (default), we have roughly half the k-vectors
    from jaxpme.batched_mixed.kspace import count_halfspace_kvectors

    k_shape = (
        int(np.ceil(diags[0] / lr_wavelength)),
        int(np.ceil(diags[1] / lr_wavelength)),
        int(np.ceil(diags[2] / lr_wavelength)),
    )
    expected_kpoints = count_halfspace_kvectors(k_shape)
    # Exclude zero-padding vectors when counting unique k-vectors
    # k_grid has shape (num_pbc, num_k, 3); take first structure
    k_grid = periodic_batch.k_grid[0]  # shape (num_k, 3)
    nonzero_mask = np.any(k_grid != 0, axis=1)  # check each k-vector
    actual_unique = np.unique(k_grid[nonzero_mask], axis=0).shape[0]
    assert actual_unique == expected_kpoints

    # Verify atom_to_structure and pair_to_structure for padding
    padding_structure = 7  # num_structures - 1
    assert np.all(sr_batch.atom_to_structure[~sr_batch.atom_mask] == padding_structure)
    assert np.all(sr_batch.pair_to_structure[~sr_batch.pair_mask] == padding_structure)


def test_mixed_pbc_smoke():
    """Smoke test for mixed PBC: 2D orthorhombic + non-periodic."""
    from ase import Atoms

    from jaxpme.batched_mixed.batching import get_batch, prepare

    # 2D orthorhombic structure
    cell_2d = np.diag([10.0, 10.0, 20.0])
    atoms_2d = Atoms(
        "H2",
        positions=[[5.0, 5.0, 5.0], [5.0, 5.0, 6.0]],
        cell=cell_2d,
        pbc=[True, True, False],
    )
    atoms_2d.set_initial_charges([1.0, -1.0])

    # Non-periodic structure
    atoms_0d = Atoms(
        "H2", positions=[[0.0, 0.0, 0.0], [0.0, 0.0, 1.0]], pbc=[False, False, False]
    )
    atoms_0d.set_initial_charges([1.0, -1.0])

    cutoff = 3.0
    samples = [prepare(atoms_2d, cutoff), prepare(atoms_0d, cutoff)]

    # This should not blow up
    charges, sr_batch, nonp_batch, pbc_batch = get_batch(samples)

    assert sr_batch.structure_mask.sum() == 2
    assert sr_batch.pbc_mask[0]
    assert not sr_batch.pbc_mask[1]
    assert pbc_batch.structure_mask.sum() == 1
    assert nonp_batch.pair_mask.sum() > 0


def test_k_grid_shape_homogeneous():
    """k_grid_shape forces all periodic systems to use the same k-grid shape."""
    from ase.build import bulk

    from jaxpme.batched_mixed.batching import get_batch, prepare

    # Two NaCl cells with different sizes -> without k_grid_shape they'd get different grids
    small = bulk("NaCl", "rocksalt", a=4.0)
    small.set_initial_charges(np.tile([1.0, -1.0], len(small) // 2))

    large = bulk("NaCl", "rocksalt", a=8.0)
    large.set_initial_charges(np.tile([1.0, -1.0], len(large) // 2))

    cutoff = 3.0
    k_grid_shape = (6, 6, 6)

    samples = [
        prepare(atoms, cutoff, k_grid_shape=k_grid_shape) for atoms in [small, large]
    ]

    # Both structures should produce k-grids with exactly shape (6, 6, 6)
    for sample in samples:
        assert sample["lr"].k_grid.shape == k_grid_shape

    # Batching should work; padding is minimal because k-vector count is the same
    charges, sr_batch, nonperiodic_batch, periodic_batch = get_batch(samples)
    assert periodic_batch.structure_mask.sum() == 2

    from jaxpme.batched_mixed.kspace import count_halfspace_kvectors

    expected_k = count_halfspace_kvectors(k_grid_shape)

    # Both structures should have the same number of non-zero (non-padded) k-vectors
    for pbc_idx in range(int(periodic_batch.structure_mask.sum())):
        nonzero = np.any(periodic_batch.k_grid[pbc_idx] != 0, axis=1).sum()
        assert nonzero == expected_k


def test_k_grid_shape_int_shorthand():
    """k_grid_shape accepts a single int as shorthand for (n, n, n)."""
    from ase.build import bulk

    from jaxpme.batched_mixed.batching import prepare

    nacl = bulk("NaCl", "rocksalt", a=5.6)
    nacl.set_initial_charges(np.tile([1.0, -1.0], len(nacl) // 2))

    sample_tuple = prepare(nacl, cutoff=3.0, k_grid_shape=(8, 8, 8))
    sample_int = prepare(nacl, cutoff=3.0, k_grid_shape=8)

    assert sample_tuple["lr"].k_grid.shape == sample_int["lr"].k_grid.shape == (8, 8, 8)


def test_k_grid_shape_system_specific_smearing():
    """When k_grid_shape is set, smearing is derived per system from cell size."""
    from ase.build import bulk

    from jaxpme.batched_mixed.batching import prepare
    from jaxpme.kspace import lr_wavelength_for_kgrid_shape

    small = bulk("NaCl", "rocksalt", a=4.0)
    small.set_initial_charges(np.tile([1.0, -1.0], len(small) // 2))

    large = bulk("NaCl", "rocksalt", a=8.0)
    large.set_initial_charges(np.tile([1.0, -1.0], len(large) // 2))

    k_grid_shape = (6, 6, 6)
    sample_small = prepare(small, cutoff=3.0, k_grid_shape=k_grid_shape)
    sample_large = prepare(large, cutoff=3.0, k_grid_shape=k_grid_shape)

    # smearing scales with cell size: larger cell -> larger lr_wavelength_eff -> larger smearing
    assert sample_large["smearing"] > sample_small["smearing"]

    # smearing should equal lr_wavelength_eff * 2.0
    lr_small = lr_wavelength_for_kgrid_shape(small.get_cell().array, k_grid_shape)
    lr_large = lr_wavelength_for_kgrid_shape(large.get_cell().array, k_grid_shape)
    assert np.isclose(sample_small["smearing"], lr_small * 2.0)
    assert np.isclose(sample_large["smearing"], lr_large * 2.0)


def test_k_grid_shape_accuracy():
    """k_grid_shape with explicit smearing matches default lr_wavelength result exactly."""
    import jax

    jax.config.update("jax_enable_x64", True)

    from ase.io import read
    from conftest import REFERENCE_STRUCTURES_DIR

    from jaxpme.batched_mixed.calculators import Ewald
    from jaxpme.kspace import get_kgrid_ewald_shape

    cutoff = 5.0
    atoms = read(REFERENCE_STRUCTURES_DIR / "coulomb_test_frames.xyz", index="0")

    # Compute what the default heuristic would give
    lr_wavelength = cutoff / 8.0
    smearing = cutoff / 4.0
    k_grid_shape = get_kgrid_ewald_shape(atoms.get_cell().array, lr_wavelength)

    calculator = Ewald(prefactor=1.0)

    # Default: lr_wavelength and smearing derived from cutoff
    charges1, sr1, nonp1, pbc1 = calculator.prepare([atoms], cutoff)
    E1 = calculator.energy(charges1, sr1, nonp1, pbc1)

    # k_grid_shape API: same k-grid shape + same explicit smearing -> identical result
    charges2, sr2, nonp2, pbc2 = calculator.prepare(
        [atoms], cutoff, k_grid_shape=k_grid_shape, smearing=smearing
    )
    E2 = calculator.energy(charges2, sr2, nonp2, pbc2)

    np.testing.assert_allclose(E1, E2, rtol=1e-10)


def test_orthorhombic_tolerance():
    """Test that nearly-orthorhombic cells (within tolerance) are accepted for 2D PBC."""
    import pytest
    from ase import Atoms

    from jaxpme.batched_mixed.batching import is_orthorhombic, prepare

    # Cell with tiny off-diagonal elements (within 1e-8 tolerance) should work
    cell_nearly_ortho = np.diag([10.0, 10.0, 20.0])
    cell_nearly_ortho[0, 1] = 1e-10  # tiny off-diagonal, within tolerance

    atoms_ok = Atoms(
        "H2",
        positions=[[5.0, 5.0, 5.0], [5.0, 5.0, 6.0]],
        cell=cell_nearly_ortho,
        pbc=[True, True, False],
    )
    atoms_ok.set_initial_charges([1.0, -1.0])

    # Should not raise
    prepare(atoms_ok, cutoff=3.0)

    # Cell with off-diagonal elements outside tolerance should fail
    cell_not_ortho = np.diag([10.0, 10.0, 20.0])
    cell_not_ortho[0, 1] = 1e-5  # outside 1e-8 tolerance

    atoms_bad = Atoms(
        "H2",
        positions=[[5.0, 5.0, 5.0], [5.0, 5.0, 6.0]],
        cell=cell_not_ortho,
        pbc=[True, True, False],
    )
    atoms_bad.set_initial_charges([1.0, -1.0])

    with pytest.raises(ValueError, match="orthorhombic"):
        prepare(atoms_bad, cutoff=3.0)

    # Unit tests for is_orthorhombic helper
    assert is_orthorhombic(np.eye(3) * 5.0)
    assert is_orthorhombic(np.diag([3.0, 4.0, 5.0]))
    assert not is_orthorhombic(
        np.array([[1.0, 0.1, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
    )
    assert is_orthorhombic(np.eye(3) + 1e-11)  # within default tolerance
    assert not is_orthorhombic(np.eye(3) + 1e-5)  # outside default tolerance
