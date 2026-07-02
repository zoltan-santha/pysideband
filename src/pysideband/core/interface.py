from __future__ import annotations

from enum import Enum

class AbInitioCalculator(Enum):
    VASP = "vasp"
    ABINIT = "abinit"
    QE = "qe"
    PWMAT = "pwmat"
    WIEN2K = "wien2k"
    ELK = "elk"
    SIESTA = "siesta"
    CP2K = "cp2k"
    CRYSTAL = "crystal"
    DFTBP = "dftbp"
    TURBOMOLE = "turbomole"
    AIMS = "aims"
    CASTEP = "castep"
    FLEUR = "fleur"
    ABACUS = "abacus"
    LAMMPS = "lammps"
    QLM = "qlm"
    OTHER = "other"
