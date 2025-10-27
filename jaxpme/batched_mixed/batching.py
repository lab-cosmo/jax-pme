import numpy as np

from collections import namedtuple

from .kspace import generate_ewald_k_grid

Batch = namedtuple(
    "Batch",
    (
        "positions",
        "centers",
        "others",
        "cell_shifts",
        "distances",
        "cell",
        "smearing",
        "atom_mask",
        "pair_mask",
        "structure_mask",
        "pbc_mask",
        "atom_to_structure",
        "pair_to_structure",
    ),
)
Periodic = namedtuple(
    "Periodic",
    ("k_grid", "atom_to_atom", "structure_to_structure", "atom_mask", "structure_mask"),
    # atom_to_atom: indices into SR batch positions [num_pbc, num_atoms_pbc]
    #               -> positions[atom_to_atom]
    # structure_to_structure: indices into SR batch structures [num_pbc]
    #                   -> cell[structure_to_structure]
    # structure_mask: mask for valid periodic structures [num_pbc]
    #              -> True for real structures, False for padding
)
NonPeriodic = namedtuple("NonPeriodic", ("centers", "others", "pair_mask"))


def get_batch(
    samples,
    num_structures=None,
    num_structures_pbc=None,
    num_atoms=None,
    num_atoms_pbc=None,
    num_pairs=None,
    num_pairs_nonpbc=None,
    num_k=None,
    strategy="powers_of_2",
):
    _num_structures = len(samples)
    _num_atoms = []
    _num_pairs = []
    _num_pairs_nonpbc = []
    _is_pbc = []
    _num_k = []

    for structure in samples:
        lr = structure["lr"]
        _num_atoms.append(len(structure["positions"]))
        _num_pairs.append(len(structure["centers"]))
        if hasattr(lr, "k_grid"):
            _is_pbc.append(True)
            _num_k.append(lr.k_grid.shape[0] * lr.k_grid.shape[1] * lr.k_grid.shape[2])
        else:
            _is_pbc.append(False)
            _num_pairs_nonpbc.append(len(lr.centers))

    _num_atoms = np.array(_num_atoms)
    _num_pairs = np.array(_num_pairs)
    _num_pairs_nonpbc = np.array(_num_pairs_nonpbc)
    _is_pbc = np.array(_is_pbc)
    _num_k = np.array(_num_k)

    _total_atoms = _num_atoms.sum()
    _total_pairs = _num_pairs.sum()
    _max_atoms_pbc = _num_atoms[_is_pbc].max() if _is_pbc.any() else 0
    _max_k = _num_k.max() if len(_num_k) > 0 else 0
    _total_pairs_nonpbc = _num_pairs_nonpbc.sum() if len(_num_pairs_nonpbc) > 0 else 0
    _total_pbc = _is_pbc.sum()

    num_structures = get_size(num_structures, _num_structures, strategy=strategy)
    num_atoms = get_size(num_atoms, _total_atoms, strategy=strategy)
    num_pairs = get_size(num_pairs, _total_pairs, strategy=strategy)
    num_atoms_pbc = get_size(num_atoms_pbc, _max_atoms_pbc, strategy=strategy)
    num_k = get_size(num_k, _max_k, strategy=strategy)
    num_pairs_nonpbc = get_size(num_pairs_nonpbc, _total_pairs_nonpbc, strategy=strategy)
    num_pbc = get_size(num_structures_pbc, _total_pbc, strategy=strategy)

    padding_atom_idx = _total_atoms
    padding_structure_idx = num_structures - 1

    charges = np.zeros(num_atoms, dtype=float)
    positions = np.zeros((num_atoms, 3), dtype=float)
    cell = np.zeros((num_structures, 3, 3), dtype=float)
    cell[:] = np.eye(3)
    smearing = np.zeros(num_structures, dtype=float)
    centers = np.ones(num_pairs, dtype=int) * padding_atom_idx
    others = np.ones(num_pairs, dtype=int) * padding_atom_idx
    cell_shifts = np.zeros((num_pairs, 3), dtype=int)
    atom_to_structure = np.ones(num_atoms, dtype=int) * padding_structure_idx
    pair_to_structure = np.ones(num_pairs, dtype=int) * padding_structure_idx
    structure_mask = np.zeros(num_structures, dtype=bool)
    pbc_mask = np.zeros(num_structures, dtype=bool)
    atom_mask = np.zeros(num_atoms, dtype=bool)
    pair_mask = np.zeros(num_pairs, dtype=bool)

    nonpbc_centers = np.ones(num_pairs_nonpbc, dtype=int) * padding_atom_idx
    nonpbc_others = np.ones(num_pairs_nonpbc, dtype=int) * padding_atom_idx
    nonpbc_pair_mask = np.zeros(num_pairs_nonpbc, dtype=bool)

    pbc_kgrid = np.zeros((num_pbc, num_k, 3), dtype=float)
    pbc_atom_to_atom = np.ones((num_pbc, num_atoms_pbc), dtype=int) * padding_atom_idx
    pbc_structure_to_structure = np.ones(num_pbc, dtype=int) * padding_structure_idx
    pbc_atom_mask = np.zeros((num_pbc, num_atoms_pbc), dtype=bool)
    pbc_structure_mask = np.zeros(num_pbc, dtype=bool)

    atom_offset = 0
    pair_offset = 0
    nonpbc_offset = 0
    pbc_idx = 0

    for idx, structure in enumerate(samples):
        lr = structure["lr"]
        is_periodic = hasattr(lr, "k_grid")

        num_n = len(structure["positions"])
        num_p = len(structure["centers"])

        atom_slice = slice(atom_offset, atom_offset + num_n)
        pair_slice = slice(pair_offset, pair_offset + num_p)

        charges[atom_slice] = structure["charges"]
        positions[atom_slice] = structure["positions"]
        cell[idx] = structure["cell"]
        centers[pair_slice] = structure["centers"] + atom_offset
        others[pair_slice] = structure["others"] + atom_offset
        cell_shifts[pair_slice] = structure["cell_shifts"]

        atom_to_structure[atom_slice] = idx
        pair_to_structure[pair_slice] = idx
        structure_mask[idx] = True
        atom_mask[atom_slice] = True
        pair_mask[pair_slice] = True

        if is_periodic:
            pbc_mask[idx] = True
            smearing[idx] = structure["smearing"]
            pbc_structure_to_structure[pbc_idx] = idx
            pbc_atom_to_atom[pbc_idx, :num_n] = np.arange(atom_offset, atom_offset + num_n)

            k_grid_shape = lr.k_grid.shape
            pbc_kgrid[pbc_idx] = generate_ewald_k_grid(k_grid_shape, size=num_k)

            pbc_atom_mask[pbc_idx, :num_n] = True
            pbc_structure_mask[pbc_idx] = True
            pbc_idx += 1
        else:
            num_nonpbc = len(lr.centers)
            nonpbc_slice = slice(nonpbc_offset, nonpbc_offset + num_nonpbc)

            smearing[idx] = 1.0
            nonpbc_centers[nonpbc_slice] = lr.centers + atom_offset
            nonpbc_others[nonpbc_slice] = lr.others + atom_offset
            nonpbc_pair_mask[nonpbc_slice] = True

            nonpbc_offset += num_nonpbc

        atom_offset += num_n
        pair_offset += num_p

    sr_batch = Batch(
        positions=positions,
        cell=cell,
        smearing=smearing,
        centers=centers,
        others=others,
        cell_shifts=cell_shifts,
        atom_mask=atom_mask,
        pair_mask=pair_mask,
        structure_mask=structure_mask,
        pbc_mask=pbc_mask,
        atom_to_structure=atom_to_structure,
        pair_to_structure=pair_to_structure,
        distances=None,
    )

    periodic_batch = Periodic(
        k_grid=pbc_kgrid,
        atom_to_atom=pbc_atom_to_atom,
        structure_to_structure=pbc_structure_to_structure,
        atom_mask=pbc_atom_mask,
        structure_mask=pbc_structure_mask,
    )

    nonperiodic_batch = NonPeriodic(
        centers=nonpbc_centers,
        others=nonpbc_others,
        pair_mask=nonpbc_pair_mask,
    )

    return charges, sr_batch, nonperiodic_batch, periodic_batch


def prepare(atoms, cutoff, lr_wavelength=None, smearing=None):
    structure = to_structure(atoms, cutoff)

    structure["charges"] = atoms.get_initial_charges()

    if lr_wavelength is None:
        lr_wavelength = cutoff / 8.0

    if smearing is None:
        smearing = cutoff / 4.0

    smearing, lr = to_lr(atoms, structure, lr_wavelength, smearing)

    structure["lr"] = lr
    if smearing is not None:
        structure["smearing"] = smearing

    return structure


def to_lr(atoms, structure, lr_wavelength, smearing):
    if atoms.pbc.all():
        cell = atoms.get_cell().array
        k_grid = get_kgrid_ewald(cell, lr_wavelength)
        return smearing, Periodic(
            k_grid=k_grid,
            atom_to_atom=None,
            structure_to_structure=None,
            atom_mask=None,
            structure_mask=None,
        )
    elif not atoms.pbc.all():
        N = len(atoms)
        full_i = np.arange(N).repeat(N)
        full_j = np.tile(np.arange(N), N)

        return None, NonPeriodic(
            centers=full_i,
            others=full_j,
            pair_mask=None,
        )

    else:
        raise ValueError("no mixed pbc yet")


def get_kgrid_ewald(cell, lr_wavelength):
    ns = np.ceil(np.linalg.norm(cell, axis=-1) / lr_wavelength)
    return np.ones((int(ns[0]), int(ns[1]), int(ns[2])))


def to_structure(atoms, cutoff):
    from vesin import ase_neighbor_list as neighbor_list

    structure = {}
    structure["cell"] = atoms.get_cell().array
    structure["positions"] = atoms.get_positions()
    structure["atomic_numbers"] = atoms.get_atomic_numbers().astype(int)
    structure["charges"] = atoms.get_initial_charges()

    if atoms.pbc.all():
        centers, others, D, S = neighbor_list("ijDS", atoms, cutoff)
    elif atoms.pbc.any():
        raise ValueError  # not supported here
    else:
        assert not atoms.pbc.any()

        centers, others, D = neighbor_list("ijD", atoms, cutoff)
        S = np.zeros((len(centers), 3), dtype=int)
        if (structure["cell"] == 0).all():
            structure["cell"] = np.eye(3)

    structure["centers"] = centers
    structure["others"] = others
    structure["cell_shifts"] = S
    structure["displacements"] = D
    structure["pbc"] = atoms.get_pbc()

    return structure


def get_size(proposed, actual, strategy="powers_of_2"):
    if proposed is not None:
        if isinstance(proposed, str):
            return _get_size(actual + 1, strategy=proposed)
        elif isinstance(proposed, int):
            assert proposed > actual
            return proposed
        else:
            msg = "could not determine size.\n"
            msg += f"proposed: {proposed} actual: {actual} strategy: {strategy}"
            raise ValueError(msg)
    else:
        return _get_size(actual + 1, strategy=strategy)


def _get_size(n, strategy="powers_of_2"):
    if strategy == "powers_of_2":
        # return next largest power of 2
        result = 2 ** np.ceil(np.log2(n))
    elif strategy == "powers_of_4":
        # return next largest power of 4
        result = 4 ** np.ceil(np.log(n) / np.log(4))
    elif strategy == "powers_of_8":
        # return next largest power of 8
        result = 8 ** np.ceil(np.log(n) / np.log(8))
    elif strategy == "multiples":
        if n <= 32:
            return next_multiple(n, 4)

        if n <= 64:
            return next_multiple(n, 16)

        if n <= 256:
            return next_multiple(n, 64)

        if n <= 1024:
            return next_multiple(n, 256)

        if n <= 4096:
            return next_multiple(n, 1024)

        if n <= 32768:
            return next_multiple(n, 4096)

        result = next_multiple(n, 16384)
    else:
        raise ValueError(f"sizing strategy {strategy} is not known")

    return int(result)


def next_multiple(val, n):
    return n * (1 + int(val // n))


## test ##


assert (
    _get_size(
        13,
        strategy="powers_of_2",
    )
    == 16
)

assert (
    _get_size(
        33,
        strategy="powers_of_4",
    )
    == 64
)

assert (
    _get_size(
        11,
        strategy="multiples",
    )
    == 12
)
