import numpy as np

from collections import namedtuple

from jaxpme.utils import atoms_to_graph

Periodic = namedtuple("Periodic", ("cell", "smearing", "kgrid"))
NonPeriodic = namedtuple("NonPeriodic", ("i", "j"))


def prepare(atoms, cutoff, charges=None, lr_wavelength=None, smearing=None):
    graph = atoms_to_graph(atoms, cutoff, full_list=True)

    if charges is None:
        c = atoms.get_initial_charges()
        if (c != 0.0).any():
            charges = c

    if charges is None:
        charges = np.array([-1.0, 1.0])
        charges = np.tile(charges, len(atoms) // 2)

    if lr_wavelength is None:
        lr_wavelength = cutoff / 8.0

    if smearing is None:
        smearing = cutoff / 4.0

    lr = to_lr(atoms, graph, lr_wavelength, smearing)

    return charges, *graph, lr


def to_lr(atoms, graph, lr_wavelength, smearing):
    i = graph[2]
    j = graph[3]

    if atoms.pbc.all():
        cell = atoms.get_cell().array
        kgrid = get_kgrid_ewald(cell, lr_wavelength)
        return Periodic(cell, smearing, kgrid)
    elif not atoms.pbc.all():
        N = len(atoms)
        full_i = np.arange(N).repeat(N)
        full_j = np.tile(np.arange(N), N)

        remaining_i, remaining_j = setdiff2d(full_i, full_j, i, j)

        return NonPeriodic(remaining_i, remaining_j)

    else:
        raise ValueError("no mixed pbc yet")


def get_kgrid_ewald(cell, lr_wavelength):
    ns = np.ceil(np.linalg.norm(cell, axis=-1) / lr_wavelength)
    return np.ones((int(ns[0]), int(ns[1]), int(ns[2])))


def setdiff2d(I, J, i, j):
    # I,J is a full neighborlist, i,j a truncated one,
    # this filters I,J to all the pairs that aren't in j,i
    max_value = np.max(I)
    factor = int(10 ** np.ceil(np.log10(max_value)))

    full = I * factor + J
    sub = i * factor + j

    diff = np.setdiff1d(full, sub)

    return np.divmod(diff, factor)
