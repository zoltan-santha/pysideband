from __future__ import annotations

import numpy as np
from typing import Sequence

from pysideband.core.phonons import Phonons

class QMesh:
    def __init__(
        self,
        phonons: Phonons,
        mesh_size: Sequence[int] | np.ndarray,
        mesh_shift: Sequence[float] | np.ndarray = np.zeros(3),
    ) -> None:
        if len(mesh_size) != 3:
            raise ValueError(f"Invalid mesh size: {mesh_size}. Must be a sequence of 3 integers.")
        if len(mesh_shift) != 3:
            raise ValueError(f"Invalid mesh shift: {mesh_shift}. Must be a sequence of 3 floats.")
        phonons._phonon.init_mesh(
            mesh=mesh_size,
            shift=mesh_shift,
            is_time_reversal=True,
            is_mesh_symmetry=True,
            is_gamma_center=True,
        )
        self._qpoints = phonons._phonon._mesh.qpoints
        self._weights = phonons._phonon._mesh.weights
    
    @property
    def qpoints(self) -> np.ndarray:
        return self._qpoints
    
    @property
    def weights(self) -> np.ndarray:
        return self._weights
