from collections import namedtuple

Prefactors = namedtuple("Prefactors", ("SI", "eV_A", "kcalmol_A", "kJmol"))

prefactors = Prefactors(
    SI=2.3070775523417355e-28,  # -> SI units
    eV_A=14.399645478425667,  # -> electron volt / Ångstrom
    kcalmol_A=332.0637132991921,  # -> kilocalories per mole / Ångstrom
    kJmol=1389.3545764438197,  # -> kilojoule per mole / Ångstrom
)
