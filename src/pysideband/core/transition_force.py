from __future__ import annotations

from typing import Sequence
import numpy as np

from pysideband.core.displacement import Displacement
from pysideband.core.structure import Structure
import pysideband.core.phonons as ph

class TransitionForce(Displacement):
    def __init__(self, initial: Structure, final: Structure, phonons: ph.Phonons):
        super().__init__(initial, final)
        self._reference_structure = Structure(
            file_path = "",
            calculator = "other",
            phonopyatoms = phonons.structure,
            optional_crystal_info = None
        )
        mass_weighted_displacement = super().mass_weighted()
        f, v = phonons.eigh()
        f[f < 0] = 0.0
        self._transition_force = (v @ (f**2 * (v.conj().T @ mass_weighted_displacement.flatten()))).reshape(mass_weighted_displacement.shape)
    
    def mass_weighted(self):
        return self._transition_force
    
    def mass_weighted_cell_resolved(self, reference: Structure | None = None):
        if reference is None:
            reference = self._reference_structure
        return super().mass_weighted_cell_resolved(reference)
    
    def mass_weighted_q_gauged(self, q: Sequence[float] | np.ndarray, reference: Structure | None = None):
        if reference is None:
            reference = self._reference_structure
        return super().mass_weighted_q_gauged(q, reference)