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


def test_lr_wavelength_for_num_k_round_trip():
    """Analytical lr_wavelength_for_num_k gives k-grid count close to target."""
    from jaxpme.batched_mixed.kspace import count_halfspace_kvectors
    from jaxpme.kspace import get_kgrid_ewald_shape, lr_wavelength_for_num_k

    cells = [
        np.eye(3) * 10.0,  # cubic
        np.diag([5.0, 5.0, 20.0]),  # elongated
        np.array([[10.0, 0.0, 0.0], [2.0, 9.0, 0.0], [0.0, 0.0, 8.0]]),  # skewed
    ]

    for cell in cells:
        for num_k in [50, 200, 1000]:
            lw = lr_wavelength_for_num_k(cell, num_k)
            shape = get_kgrid_ewald_shape(cell, lw)
            actual = count_halfspace_kvectors(shape)

            # halfspace ≈ N_total/2 is approximate; allow some slack
            ratio = actual / num_k
            assert ratio > 0.75, f"ratio {ratio:.2f} too low for {cell}"
            assert ratio < 1.5, f"ratio {ratio:.2f} too high for {cell}"


def test_num_k_equivalence():
    """num_k is a pure shorthand — identical to manually deriving lr_wavelength."""
    import jax

    jax.config.update("jax_enable_x64", True)

    from ase.io import read
    from conftest import REFERENCE_STRUCTURES_DIR

    from jaxpme.batched_mixed.calculators import Ewald
    from jaxpme.kspace import lr_wavelength_for_num_k

    atoms = read(REFERENCE_STRUCTURES_DIR / "coulomb_test_frames.xyz", index="0")
    cell = atoms.get_cell().array

    num_k = 200
    lw = lr_wavelength_for_num_k(cell, num_k)
    cutoff = lw * 8.0
    smearing = lw * 2.0

    calculator = Ewald(prefactor=1.0)

    # Path A: num_k
    charges_a, sr_a, nonp_a, pbc_a = calculator.prepare([atoms], num_k=num_k)
    E_a, F_a, S_a = calculator.energy_forces_stress(charges_a, sr_a, nonp_a, pbc_a)

    # Path B: explicit equivalent parameters
    charges_b, sr_b, nonp_b, pbc_b = calculator.prepare(
        [atoms], cutoff=cutoff, lr_wavelength=lw, smearing=smearing
    )
    E_b, F_b, S_b = calculator.energy_forces_stress(charges_b, sr_b, nonp_b, pbc_b)

    np.testing.assert_allclose(E_a, E_b, rtol=1e-10)
    np.testing.assert_allclose(F_a, F_b, rtol=1e-10)
    np.testing.assert_allclose(S_a, S_b, rtol=1e-10)


def test_num_k_accuracy():
    """Energy/forces/stress with num_k match a well-converged reference."""
    import jax

    jax.config.update("jax_enable_x64", True)

    from ase.io import read
    from conftest import REFERENCE_STRUCTURES_DIR

    from jaxpme.batched_mixed.calculators import Ewald
    from jaxpme.batched_mixed.kspace import count_halfspace_kvectors
    from jaxpme.kspace import get_kgrid_ewald_shape

    cutoff_ref = 5.0
    atoms = read(REFERENCE_STRUCTURES_DIR / "coulomb_test_frames.xyz", index="0")

    calculator = Ewald(prefactor=1.0)

    # Reference: default heuristic
    c1, sr1, nonp1, pbc1 = calculator.prepare([atoms], cutoff_ref)
    E_ref, F_ref, S_ref = calculator.energy_forces_stress(c1, sr1, nonp1, pbc1)

    # Compute the num_k that the default heuristic produces
    lr_wavelength_ref = cutoff_ref / 8.0
    shape_ref = get_kgrid_ewald_shape(atoms.get_cell().array, lr_wavelength_ref)
    num_k_ref = count_halfspace_kvectors(shape_ref)

    # num_k path with same budget
    c2, sr2, nonp2, pbc2 = calculator.prepare([atoms], num_k=num_k_ref)
    E_nk, F_nk, S_nk = calculator.energy_forces_stress(c2, sr2, nonp2, pbc2)

    # Cutoff differs slightly (derived from analytical λ vs exact λ),
    # so results won't be bit-identical, but should be close.
    np.testing.assert_allclose(E_nk, E_ref, rtol=1e-3)
    np.testing.assert_allclose(F_nk, F_ref, rtol=1e-3, atol=1e-5)
    np.testing.assert_allclose(S_nk, S_ref, rtol=1e-3, atol=1e-5)


def test_num_k_batching():
    """Two structures with different cells, same num_k, batch correctly."""
    from ase.build import bulk

    from jaxpme.batched_mixed.batching import get_batch, prepare

    small = bulk("NaCl", "rocksalt", a=4.0)
    small.set_initial_charges(np.tile([1.0, -1.0], len(small) // 2))

    large = bulk("NaCl", "rocksalt", a=8.0)
    large.set_initial_charges(np.tile([1.0, -1.0], len(large) // 2))

    num_k = 300
    samples = [prepare(atoms, num_k=num_k) for atoms in [small, large]]

    charges, sr_batch, nonperiodic_batch, periodic_batch = get_batch(samples)
    assert periodic_batch.structure_mask.sum() == 2

    # Both k-grids should have count approximately num_k
    for pbc_idx in range(2):
        nonzero = np.any(periodic_batch.k_grid[pbc_idx] != 0, axis=1).sum()
        assert nonzero >= num_k, f"structure {pbc_idx}: {nonzero} < {num_k}"
        assert nonzero < num_k * 2, f"structure {pbc_idx}: {nonzero} >> {num_k}"
