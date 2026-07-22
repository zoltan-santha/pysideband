from __future__ import annotations

import numpy as np
from scipy.optimize import linear_sum_assignment as SciPy_linear_sum_assignment

from phonopy.structure.atoms import PhonopyAtoms
try:
    from phonopy.interface.calculator import StructureInfo as PhonopyStructureInfo
except ImportError: # older implementation of phonopy does not have StructureInfo type, defines it as a tuple instead
    PhonopyStructureInfo = tuple
from phonopy.interface.calculator import read_crystal_structure as Phonopy_read_crystal_structure

from pysideband.core.interface import AbInitioCalculator

class Structure:
    def __init__(self, file_path: str, calculator: AbInitioCalculator, phonopyatoms: PhonopyAtoms, optional_crystal_info: PhonopyStructureInfo | None): # type: ignore
        self._file_path: str = file_path
        self._calculator: AbInitioCalculator = calculator
        self._PhonopyAtoms: PhonopyAtoms = phonopyatoms
        self._optional_crystal_info: PhonopyStructureInfo | None = optional_crystal_info # type: ignore
    
    @classmethod
    def from_file(cls, file_path: str, calculator: AbInitioCalculator):
        if calculator == AbInitioCalculator.OTHER:
            raise ValueError(f"Unsupported calculator invoked for structure file: {file_path}")
        try:
            with open(file_path, 'r') as f:
                pass
        except FileNotFoundError as exc:
            raise FileNotFoundError(f"Structure file not found: {exc.filename}") from exc
        phonopyAtoms, optional_crystal_info = Phonopy_read_crystal_structure(
            filename =          file_path,
            interface_mode =    calculator.value
            )
        return cls(file_path, calculator, phonopyAtoms, optional_crystal_info)
    
    @property
    def structure(self):
        if self._PhonopyAtoms is None:
            raise ValueError(f"Structure not loaded for file: {self._file_path}")
        return self._PhonopyAtoms
    
    def align_to(self, other: PhonopyAtoms):
        lattice =           self._PhonopyAtoms.cell
        self_f_positions =  self._PhonopyAtoms.scaled_positions
        self_symbols =      self._PhonopyAtoms.symbols
        self_masses =       self._PhonopyAtoms.masses
        other_f_positions = other.scaled_positions
        other_symbols =     other.symbols
        
        if len(self._PhonopyAtoms) != len(other):
            raise ValueError(f"Cannot align structures with different number of atoms: {len(self._PhonopyAtoms)} vs {len(other)}")
        if sorted(self_symbols) != sorted(other_symbols):
            raise ValueError(f"Cannot align structures with different atomic species: {set(self_symbols)} vs {set(other_symbols)}")
        
        self_new_positions = np.empty_like(self_f_positions)
        self_new_symbols = np.empty_like(self_symbols, dtype=np.dtypes.StringDType)
        self_new_masses = np.empty_like(self_masses)
        
        for symbol in set(self_symbols):
            self_atom_indices = np.array([i for i, s in enumerate(self_symbols) if s == symbol])
            self_f_positions_subset = self_f_positions[self_atom_indices]
            other_atom_indices = np.array([i for i, s in enumerate(other_symbols) if s == symbol])
            other_f_positions_subset = other_f_positions[other_atom_indices]
            neighbor_cells = np.array([
                np.array([i,j,k])
                for i in range(-1, 2)
                for j in range(-1, 2)
                for k in range(-1, 2)
            ]).transpose()[None, :, :]
            self_f_image_positions_subset = self_f_positions_subset[:, :, None] + neighbor_cells
            distance_vectors = (other_f_positions_subset @ lattice)[:, None, :, None] - np.einsum("ibk,ba->iak", self_f_image_positions_subset, lattice)[None, :, :, :]
            distances = np.linalg.norm(distance_vectors, axis=2)
            min_distance_indices = np.argmin(distances, axis=2)
            min_distances = np.take_along_axis(
                arr =       distances,
                indices =   min_distance_indices[:, :, None],
                axis=2
            ).squeeze(axis=2)
            row_indices, col_indices = SciPy_linear_sum_assignment(min_distances)
            self_new_atom_indices = self_atom_indices[row_indices]
            self_old_atom_indices = other_atom_indices[col_indices]
            self_new_positions[self_new_atom_indices] = self_f_positions[self_old_atom_indices]
            self_new_symbols[self_new_atom_indices] = np.array(self_symbols, dtype=np.dtypes.StringDType)[self_old_atom_indices]
            self_new_masses[self_new_atom_indices] = self_masses[self_old_atom_indices]
        self._PhonopyAtoms.scaled_positions = self_new_positions
        self._PhonopyAtoms._symbols = self_new_symbols.tolist()
        self._PhonopyAtoms._masses = self_new_masses
