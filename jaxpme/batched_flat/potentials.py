import jax

from collections import namedtuple

from jaxpme.potentials import coulomb, inverse_power_law
from jaxpme.potentials import potential as _potential

# -- high-level interface --
Potential = namedtuple(
    "Potential", ("sr", "lr", "real", "correction", "batched_correction")
)


def potential(exponent=1, exclusion_radius=None, custom_potential=None):
    if custom_potential is None:
        if exponent == 1:
            raw_pot = coulomb()
        else:
            raw_pot = inverse_power_law(exponent)
    else:
        raw_pot = custom_potential

    pot = _potential(
        exponent=exponent,
        exclusion_radius=exclusion_radius,
        custom_potential=custom_potential,
    )

    def batched_correction(smearing, charges, volume, atom_to_system, system_mask):
        # charges: per atom
        # smearing, volume: per system
        # atom_to_system: map per system <> per atom
        # system_mask: True for non-padded system

        c = -charges * raw_pot.correction_self(smearing)[atom_to_system]

        charge_tot = jax.ops.segment_sum(
            charges, atom_to_system, num_segments=system_mask.shape[0]
        )

        prefac = raw_pot.correction_background(smearing)[atom_to_system]
        c -= 2 * prefac * (charge_tot / volume)[atom_to_system]

        return c

    return Potential(*pot, batched_correction)
