import numpy as np
import jax
import jax.numpy as jnp

from collections import namedtuple
from time import monotonic

from ase import units
from ase.io import read
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution, Stationary

from jaxpme import PME, Ewald

Point = namedtuple("Point", ("R", "P"))

MDState = namedtuple("MDState", ("point", "F", "fixed"))

# -- confing
multiplier = 10
T = 10
nsteps = 2**14
mode = "PME"

# L40s:
# 1024 atoms
# PME   (245W): took: 3.207ms/step (16384steps @ 1024atoms)
# Ewald (275W): took: 0.352ms/step (16384steps @ 1024atoms)

# 8192 atoms
# Ewald (270W): took: 6.924ms/step (16384steps @ 8192atoms)
# PME   (260W): took: 7.635ms/step (16384steps @ 8192atoms)

# 16000 atoms

# PME   (211W):

# -- end config


def to_velocities(momenta, masses):
    return momenta / masses[:, None]


def to_momenta(velocities, masses):
    return velocities * masses[:, None]


primitive = read("../tests/reference_structures/benchmark_geometry.in")


atoms = primitive * [multiplier, multiplier, multiplier]

print(f"running {len(atoms)} atoms")

MaxwellBoltzmannDistribution(atoms, temperature_K=T)
Stationary(atoms)

cutoff = (np.linalg.norm(atoms.get_cell().array, axis=-1).min()) / 2 - 1e-6
smearing = cutoff / 5
if mode == "Ewald":
    lr_wavelength = smearing / 2
    print(
        f"{mode}: cutoff={cutoff:.2f}, smearing={smearing:.2f}, lr_wavelength={lr_wavelength:.2f}"
    )
    initial = Ewald.prepare(atoms, None, cutoff, lr_wavelength, smearing)
    _energy = Ewald.energy
elif mode == "PME":
    mesh_spacing = smearing / 8
    print(
        f"{mode}: cutoff={cutoff:.2f}, smearing={smearing:.2f}, mesh_spacing={mesh_spacing:.2f}"
    )
    initial = PME.prepare(atoms, None, cutoff, mesh_spacing, smearing)
    _energy = PME.energy


velocities = jnp.array(atoms.get_velocities())
masses = jnp.array(atoms.get_masses())


def prepare_md(initial, velocities, masses, dt):
    charges, *graph, k_grid, smearing = initial
    cell, positions, i, j, S = graph

    point = Point(positions, to_momenta(velocities, masses))

    fixed_inputs = charges, cell, i, j, S, k_grid, smearing

    def energy(point, fixed):
        charges, cell, i, j, S, k_grid, smearing = fixed

        return 14.399484341230986 * _energy(
            charges,
            cell,
            point.R,
            i,
            j,
            S,
            k_grid,
            smearing,
        )

    def forces(point, fixed):
        F = jax.grad(lambda p: energy(p, fixed), argnums=0)(point).R
        return -F

    state = MDState(point, forces(point, fixed_inputs), fixed_inputs)

    to_V = lambda P: to_velocities(P, masses)

    def step_fn(md_state, ignored):
        P_halfdt = md_state.point.P + 0.5 * dt * md_state.F
        R = md_state.point.R + dt * to_V(P_halfdt)

        point = md_state.point._replace(R=R)
        F = forces(point, md_state.fixed)

        P = P_halfdt + 0.5 * dt * F
        point = point._replace(P=P)

        md_state = MDState(point, F, md_state.fixed)

        return md_state, (point, F)

    return state, step_fn


state, step = prepare_md(initial, velocities, masses, 1.0 * units.fs)


print("warmup...")
start = monotonic()
step = jax.jit(step)
state, _ = step(state, None)
state, res = jax.lax.scan(step, state, None, length=1)
state.point.R.block_until_ready()
print(f"took: {1e3*(monotonic()-start):.3f}ms")

print("running...")
# with jax.profiler.trace("trace/"):
start = monotonic()
state, res = jax.lax.scan(step, state, None, length=nsteps)
state.point.R.block_until_ready()
print(
    f"took: {1e3*(monotonic()-start)/nsteps:.3f}ms/step ({nsteps}steps @ {len(atoms)}atoms)"
)
