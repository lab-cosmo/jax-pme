# `jax-pme`: Particle-mesh based calculations of long-range interactions in JAX

This is an experimental version of the [`torch-pme`](https://github.com/lab-cosmo/torch-pme) package, written using [`jax`](https://github.com/jax-ml/jax) instead of `torch`. It currently offers a *subset* of its features. Note that the API is not yet finalised -- we appreciate any feedback! **There will be breaking changes without announcement.**

To learn more about Ewald summation and its particle-mesh variants, which this package implements, please have a look at our [preprint](https://arxiv.org/abs/2412.03281):

```
Title: Fast and flexible range-separated models for atomistic machine learning
Authors: Philip Loche, Kevin K. Huguenin-Dumittan, Melika Honarmand, Qianjun Xu, Egor Rumiantsev, Wei Bin How, Marcel F. Langer, Michele Ceriotti
Preprint: arXiv:2412.03281 (2024)
```

You should also check the [`torch-pme` documentation](https://lab-cosmo.github.io/torch-pme/latest)!

## Installation

This package requires `jax`. Please make sure to install the appropriate version for your setup.

Once this has been done, you can clone this repository, and run:

```bash
pip install -e .
```

inside the folder. The `dev` group contains development dependencies, and can be installed by adding `.[dev]` to the command above.

## Usage

**The interface of this package is not yet fully designed. Please file an issue or get in touch via `marcel.langer@epfl.ch` if you have a particular use-case in mind and would like to chat about how to best support it.**

Currently, the high-level API of this package is designed to compute (a) potentials, (b) total energy, (c) forces, and (d) stress for pairwise potentials of the form `1/r**p`, with `p=1` defining the important case of Coulomb interactions, i.e., electrostatics. The API is designed for standalone computation of these quantities, and is not particularly optimised for integration into machine learning model architectures yet. For example, it expects to be given positions and the cell as inputs, as opposed to graph edges, and it expects a half neighborlist.

### Design

This package has to respect the constraints of `jax`, and is therefore designed differently than its `torch` counterpart. The relevant issue here is that `jax` is not designed to manage stateful classes, and it requires array shapes to be known ahead of time. For example, the shape of the reciprocal-space grid (the $k$-grid), which depends both on convergence parameters like `lr_wavelength` (the cutoff in reciprocal space, i.e., the minimum wavelength) and the periodic `cell` of the system, has to be known ahead of time. This is different to the `torch` version, where we can just compute it in `forward`. We also need to be careful to ensure that all operations relevant to differentiation happen inside the scope of the calculation function, since `jax` traces the computation, and doesn't track the arrays across compuations like `torch`.

`jax-pme` is designed accordingly: The actual compute functions are pure functions that accept the relevant arguments for differentiation (`positions`, `charges`, `cell`) as well as information like the shape of the `k-grid`. As a consequence, they can be traced and transformed by `jax`, for instance with `jax.grad` and `jax.jit`. Provided inputs are padded appropriately, even `vmap` and `scan` can be used.

For convenience, we also provide helper functions that transform more conventional inputs, together with the relevant convergence parameters, into the inputs required by `jax-pme`. We currently expect the user to take care of padding inputs to common shapes, but we support masking out padded inputs to enable this.

### API

The high-level API is provided by `Calculator` classes, which are simple `namedtuple`s of functions:

```python
Calculator = namedtuple(
    "Calculator",
    ("prepare", "potentials", "energy", "energy_forces", "energy_forces_stress"),
)
```

They are instantiated just like any other class, by calling

```python
from jaxpme import Ewald, PME

calculator = Ewald(
	exponent=1,  # corresponds to electrostatics
	exclusion_radius=None,  # if this is not None, purely long-range potentials are computed (see preprint)
	prefactor=1.0,  # default to Gauss units. jaxpme.prefactors.eV_A for standard ase units
	custom_potential=None,  # mostly for testing -- you can define custom potential functions
	)

calculator = PME(
    exponent=1,  # corresponds to electrostatics
    exclusion_radius=None,  # if this is not None, purely long-range potentials are computed (see preprint)
    prefactor=1.0,  # default to Gauss units. jaxpme.prefactors.eV_A for standard ase units
    interpolation_nodes=4,  # currently only 4 is supported
    custom_potential=None,  # mostly for testing -- you can define custom potential functions
	)

# -> calculator.prepare, .energy, etc ... can be called
```

The functions exposed by `Calculator` consist of a `prepare` function that arranges all the inputs required for calculations of some input structure, including determining the shape of the reciprocal-space grid, and a bundle of functions that then execute different calculations.

`prepare` expects the arguments `atoms` (`ase.Atoms` instance), `charges`, `cutoff` (for the real-space neighborlist), `mesh_spacing` (PME) or `lr_cutoff` (Ewald) (defining the resolution/cutoff in reciprocal space), `smearing` (range separation parameter, related to `cutoff`). The parameters can be tuned with `torch-pme` or set heuristically (see `torch-pme` docs). It returns a tuple of inputs `charges, *graph, k_grid, smearing`, where `*graph` collects `cell`, `positions`, neighbor indices `i` and `j`, and `cell_shifts`. `k_grid` is a dummy array that defines the *shape* of the reciprocal-space grid via its `shape`, its values are not used. `prepare` is not `jax.jit`-able as it returns variable-shape output.

The following calculation functions are implemented:

- `potentials`: Accepts the above inputs and returns the potential values at each position.
- `energy`: Computes the total energy obtained by multplying the charge at each position with the potential at each positions and summing up.
- `energy_forces`: Energy as above, and its derivative with respect to positions.
- `energy_forces_stress`: The above, with additionally the stress.

All these other functions can be `jit`-ed and support function transformations like `vmap` and `grad`. They optionally accept boolean mask arrays `atom_mask` and `pair_mask` to exclude irrelevant inputs from the output, typically introduced by padding. We currently do not support padding the $k$-grid, you should simply use the biggest grid consistently. Make sure that padding indices do not connect non-padded edges.

***

The *low-level* API is not yet ready for public consumption. We split the calculation task into sub-problems: `solvers.py` defines the actual implementations of the Ewald and PME method, `potentials.py` defines the actual potential functions, and the other files implement various helper functionality.

### Recommendations

We find in benchmarks that for moderately-sized systems up to a few thousand atoms, the asymptotically less efficient `Ewald` method works best. For large systems, `PME` is preferable, as it scales $O(N \log N)$. **Note that PME is not smooth in the forces -- be careful when using it for dynamics**. The P3M method, which fixes this and is also more accurate, will be implemented soon.

It is *highly* recommended to tune convergence parameters for your specific system. `torchpme.utils.tune_ewald` (and its `pme` version) exists for this purpose. Paramters can be used directly in `jax-pme`. You should typically tune the parameters for the larges system in a given dataset.

## Development

The package uses `ruff` for linting and formatting and `pytest` for testing. Please run `ruff format . && ruff check --fix .` before *every* commit or set up a commit hook to do it. Tests can be run in the `tests/` folder with `pytest`. Be aware that the test suite can take a few minutes to run.
