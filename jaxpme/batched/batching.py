import numpy as np


def to_batch(samples):
    num_systems = len(samples) + 1

    num_atoms = 0
    num_pairs = 0
    num_k = 0
    num_extra = 0

    for charges, cell, positions, i, _, _, lr in samples:
        num_atoms += positions.shape[0]
        num_pairs += i.shape[0]
        if hasattr(lr, "kgrid"):
            s = lr.kgrid.shape
            num_k += s[0] * s[1] * s[2]
        else:
            num_extra += lr.i.shape[0]

    num_atoms = get_size(num_atoms + 1)
    num_pairs = get_size(num_pairs + 1)
    num_trigs = get_size(num_k * num_atoms + 1)
    num_k = get_size(num_k + 1)
    num_extra = get_size(num_extra + 1)

    charges = np.zeros(num_atoms, dtype=float)
    cell_shifts = np.zeros((num_pairs, 3), dtype=int)
    i = np.zeros(num_pairs, dtype=int)
    j = np.zeros(num_pairs, dtype=int)
    atom_to_system = np.zeros(num_atoms, dtype=int)
    pair_to_system = np.zeros(num_pairs, dtype=int)
    system_mask = np.zeros(num_systems, dtype=bool)
    atom_mask = np.zeros(num_atoms, dtype=bool)
    pair_mask = np.zeros(num_pairs, dtype=bool)

    cell = np.zeros((num_systems, 3, 3), dtype=float)
    smearing = np.zeros(num_systems, dtype=float)
    positions = np.zeros((num_atoms, 3), dtype=float)
    periodic_atom_mask = np.zeros(num_atoms, dtype=bool)

    kgrid = np.zeros((num_k, 3), dtype=float)
    k_to_system = np.zeros(num_k, dtype=int)
    trig_to_system = np.zeros(num_trigs, dtype=int)
    trig_to_k = np.zeros(num_trigs, dtype=int)
    trig_to_atom = np.zeros(num_trigs, dtype=int)
    trig_mask = np.zeros(num_trigs, dtype=bool)
    k_mask = np.zeros(num_k, dtype=bool)

    extra_i = np.zeros(num_extra, dtype=int)
    extra_j = np.zeros(num_extra, dtype=int)
    extra_pair_mask = np.zeros(num_extra, dtype=bool)

    atom_offset = 0
    pair_offset = 0
    k_offset = 0
    ki_offset = 0
    extra_offset = 0
    for idx, (_charges, _cell, _positions, _i, _j, _cell_shifts, lr) in enumerate(samples):
        periodic = hasattr(lr, "kgrid")

        num_n = _positions.shape[0]
        num_p = _i.shape[0]

        atom_slice = slice(atom_offset, atom_offset + num_n)
        pair_slice = slice(pair_offset, pair_offset + num_p)

        if not periodic:
            num_ex = lr.i.shape[0]
            extra_slice = slice(extra_offset, extra_offset + num_ex)
        else:
            s = lr.kgrid.shape
            num_k = s[0] * s[1] * s[2]
            num_ki = num_k * num_n

            k_slice = slice(k_offset, k_offset + num_k)
            ki_slice = slice(ki_offset, ki_offset + num_ki)

        charges[atom_slice] = _charges
        cell_shifts[pair_slice] = _cell_shifts
        i[pair_slice] = _i + atom_offset
        j[pair_slice] = _j + atom_offset

        positions[atom_slice] = _positions
        if not periodic:
            extra_i[extra_slice] = lr.i + atom_offset
            extra_j[extra_slice] = lr.j + atom_offset
            extra_pair_mask[extra_slice] = True

            cell[idx] = np.eye(3)
            smearing[idx] = 1.0
        else:
            shape = lr.kgrid.shape
            # manually do an outer product of all the kvecs w/ each other
            kgrid[k_slice][:, 0] = shape[0] * np.fft.fftfreq(shape[0]).repeat(
                shape[1] * shape[2]
            )
            kgrid[k_slice][:, 1] = np.tile(
                shape[1] * np.fft.fftfreq(shape[1]).repeat(shape[0]), shape[2]
            )
            kgrid[k_slice][:, 2] = np.tile(
                shape[2] * np.fft.fftfreq(shape[2]), shape[0] * shape[1]
            )

            k_to_system[k_slice] = idx

            # convention: trigs is laid out such that we have blocks of
            # k = 0, i = 0 1 2 ...
            trig_to_system[ki_slice] = idx
            trig_to_k[ki_slice] = np.arange(num_k).repeat(num_n) + k_offset
            trig_to_atom[ki_slice] = np.tile(np.arange(num_n), num_k) + atom_offset

            trig_mask[ki_slice] = True
            k_mask[k_slice] = True

            cell[idx] = _cell
            smearing[idx] = lr.smearing
            periodic_atom_mask[atom_slice] = True

        atom_to_system[atom_slice] = idx
        pair_to_system[pair_slice] = idx

        system_mask[idx] = True
        atom_mask[atom_slice] = True
        pair_mask[pair_slice] = True

        atom_offset += num_n
        pair_offset += num_p

        if not periodic:
            extra_offset += num_ex
        else:
            k_offset += num_k
            ki_offset += num_ki

    # now we add the padding

    # skip cell_shift -- already zero
    i[pair_offset:] = atom_offset
    j[pair_offset:] = atom_offset

    atom_to_system[atom_offset:] = num_systems - 1
    pair_to_system[pair_offset:] = num_systems - 1

    # skip masks -- already False

    # padding for lr stuff

    cell[(idx + 1) :] = np.eye(3)
    smearing[(idx + 1) :] = 1.0
    # positions -- already 0
    # kgrid -- already 0
    k_to_system[k_offset:] = num_systems - 1
    trig_to_system[ki_offset:] = num_systems - 1
    trig_to_k[ki_offset:] = num_k - 1
    trig_to_atom[ki_offset:] = atom_offset
    # trig_mask -- already False
    # k_mask -- already False
    extra_i[extra_offset:] = atom_offset
    extra_j[extra_offset:] = atom_offset
    # extra_pair_mask -- already False

    return (
        # per system
        cell,
        smearing,
        charges,
        system_mask,
        # per atom
        positions,
        atom_to_system,
        atom_mask,
        periodic_atom_mask,  # True if periodic
        # per pair (within cutoff)
        None,
        cell_shifts,  # used for distances if those are missing
        i,
        j,
        pair_to_system,
        pair_mask,
        # per pair (outside cutoff, non-pbc)
        extra_i,
        extra_j,
        extra_pair_mask,
        # per k-vector
        kgrid,
        k_to_system,
        k_mask,
        # per k,i (trig)
        trig_to_atom,
        trig_to_k,
    )


def get_size(n):
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

    return next_multiple(n, 16384)


def next_multiple(val, n):
    return n * (1 + int(val // n))
