from __future__ import annotations

import numpy as np
from typing import Sequence

from phonopy import Phonopy
from phonopy import load as Phonopy_load
from phonopy.physical_units import get_physical_units as Phonopy_get_physical_units

from pysideband.core.structure import Structure
from pysideband.core.interface import AbInitioCalculator
import pysideband.core.transition_force as tf


class Phonons:
    def __init__(self, phonopy_yaml_path: str, FORCE_SETS_path: str | None, FORCE_CONSTANTS_path: str | None = None):
        self._phonopy_yaml_path: str =              phonopy_yaml_path
        self._FORCE_SETS_path: str | None =         FORCE_SETS_path
        self._FORCE_CONSTANTS_path: str | None =    FORCE_CONSTANTS_path
        self._phonon: Phonopy | None =           None
        self._structure: Structure | None =         None
        self.__post_init__()
    
    def __post_init__(self):
        try:
            with open(self._phonopy_yaml_path, 'r') as f:
                pass
            if self._FORCE_SETS_path is not None:
                with open(self._FORCE_SETS_path, 'r') as f:
                    pass
            if self._FORCE_CONSTANTS_path is not None:
                with open(self._FORCE_CONSTANTS_path, 'r') as f:
                    pass
        except FileNotFoundError as exc:
            raise FileNotFoundError(f"Phonon data file not found: {exc.filename}") from exc
        self._phonon = Phonopy_load(
            phonopy_yaml =              self._phonopy_yaml_path,
            force_sets_filename =       self._FORCE_SETS_path,
            force_constants_filename =  self._FORCE_CONSTANTS_path
            )
        self._structure = Structure(
            file_path = self._phonopy_yaml_path,
            calculator = AbInitioCalculator.OTHER,
            phonopyatoms = self._phonon.unitcell,
            optional_crystal_info = None
        )
    
    @property
    def structure(self):
        if self._structure is None:
            raise ValueError(f"Structure not loaded for phonon data: {self._phonopy_yaml_path}")
        return self._structure.structure
    
    @property
    def degrees_of_freedom(self):
        if self._structure is None:
            raise ValueError(f"Phonon data not loaded for file: {self._phonopy_yaml_path}")
        return np.prod(self.structure.scaled_positions.shape)
    
    def eigh(self, q: Sequence[float] | np.ndarray = np.zeros(3)):
        if len(q) != 3:
            raise ValueError(f"Invalid q-point: {q}. Must be a sequence of 3 floats.")
        if self._phonon is None:
            raise ValueError(f"Phonon data not loaded for file: {self._phonopy_yaml_path}")
        frequencies, eigenvectors = self._phonon.get_frequencies_with_eigenvectors(q)
        return (frequencies * Phonopy_get_physical_units().THzToEv), eigenvectors
    
    def run_interpolation(self, transition_force: tf.TransitionForce, q_points: np.ndarray, q_weights: np.ndarray):
        frequencies_set = []
        interpolated_quantity_set = []
        for i, (q, w) in enumerate(zip(q_points, q_weights)):
            #print(f"Interpolating for q-point {i+1}/{len(q_points)}: {q} with weight {w}.", flush=True)
            gauged_mwd = transition_force.mass_weighted_q_gauged(q).flatten()
            frequencies, eigenvectors = self.eigh(q)
            frequencies_set.append(frequencies)
            _factor = 1.0 / (2 * Phonopy_get_physical_units().EV * Phonopy_get_physical_units().Hbar**2)
            _factor *= Phonopy_get_physical_units().AMU * Phonopy_get_physical_units().Angstrom**2
            interpolated_quantity = w * _factor * np.abs(eigenvectors.conj().T @ gauged_mwd)**2 / (frequencies ** 3)
            interpolated_quantity_set.append(interpolated_quantity)
        if len(frequencies_set) > 0:
            frequencies_set = np.concatenate(frequencies_set, axis=None)
            interpolated_quantity_set = np.concatenate(interpolated_quantity_set, axis=None)
        return frequencies_set, interpolated_quantity_set