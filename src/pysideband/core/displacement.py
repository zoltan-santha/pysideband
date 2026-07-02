from __future__ import annotations

import numpy as np
from typing import Sequence

from pysideband.core.structure import Structure

class Displacement:
    def __init__(self, initial: Structure, final: Structure):
        self._initial_structure: Structure = initial
        self._final_structure: Structure = final
        self._masses: np.ndarray = self._initial_structure.structure.masses
        self._displacement: np.ndarray | None = None
        self.__post_init__()
    
    def __post_init__(self):
        if len(self._initial_structure.structure) != len(self._final_structure.structure):
            raise ValueError(f"Cannot create displacement with different number of atoms: {len(self._initial_structure.structure)} vs {len(self._final_structure.structure)}")
        if sorted(self._initial_structure.structure.symbols) != sorted(self._final_structure.structure.symbols):
            raise ValueError(f"Cannot create displacement with different atomic species: {set(self._initial_structure.structure.symbols)} vs {set(self._final_structure.structure.symbols)}")
        
        lattice = self._initial_structure.structure.cell
        initial_f_positions = self._initial_structure.structure.scaled_positions
        final_f_positions = self._final_structure.structure.scaled_positions
        neighbor_cells = np.array([
            np.array([i,j,k])
            for i in range(-1, 2)
            for j in range(-1, 2)
            for k in range(-1, 2)
        ]).transpose()[None, :, :]
        distance_vectors = np.einsum(
            "ab,ibk->iak",
            lattice,
            (final_f_positions[:, :, None] + neighbor_cells) - initial_f_positions[:, :, None]
        )
        distances = np.linalg.norm(distance_vectors, axis=1)
        min_distance_indices = np.argmin(distances, axis=1)
        self._displacement = np.take_along_axis(
            distance_vectors,
            np.broadcast_to(min_distance_indices[:, None, None], (distance_vectors.shape[0], 3, 1)),
            axis=2
        ).squeeze(axis=2)
    
    def mass_weighted(self):
        if self._displacement is None:
            raise ValueError(f"Displacement data not available for initial and final structures.")
        dimension_factor = 1.0
        return self._displacement * np.sqrt(np.outer(self._masses, np.ones(3))) * dimension_factor
    
    def mass_weighted_cell_resolved(self, reference: Structure):
        cell_resolved_mwd: dict[tuple[int,int,int], np.ndarray] = {}
        mass_weighted_displacement = self.mass_weighted()
        mwd_measure = np.linalg.norm(mass_weighted_displacement, axis=1)
        cyclic_fractional_atom_positions = np.exp(2j * np.pi * reference.structure.scaled_positions)
        cyclic_center_of_mwd = (np.angle(
            np.sum(cyclic_fractional_atom_positions * mwd_measure[:, None], axis=0)
        )/(2 * np.pi)) % 1.0
        neighbor_cells = np.array([
            np.array([i,j,k])
            for i in range(-1, 2)
            for j in range(-1, 2)
            for k in range(-1, 2)
        ]).transpose()[None, :, :]
        lattice = reference.structure.cell
        distance_vectors_from_cyclic_center = np.einsum(
            "ab,ibk->iak",
            lattice,
            (reference.structure.scaled_positions[:, :, None] + neighbor_cells) - cyclic_center_of_mwd[None, :, None]
        )
        distances_from_cyclic_center = np.linalg.norm(distance_vectors_from_cyclic_center, axis=1)
        min_distance_indices = np.argmin(distances_from_cyclic_center, axis=1)
        for i in range(len(reference.structure)):
            cell = tuple(neighbor_cells[0, :, min_distance_indices[i]])
            if cell not in cell_resolved_mwd:
                cell_resolved_mwd[cell] = np.zeros_like(mass_weighted_displacement)
            cell_resolved_mwd[cell][i] = mass_weighted_displacement[i]
        return cell_resolved_mwd
    
    def mass_weighted_q_gauged(self, q: Sequence[float] | np.ndarray, reference: Structure):
        if len(q) != 3:
            raise ValueError(f"Invalid q-point: {q}. Must be a sequence of 3 floats.")
        mwcr = self.mass_weighted_cell_resolved(reference)
        gauged_mwd = np.zeros_like(self._displacement, dtype=np.complex128)
        for cell, mwd in mwcr.items():
            phase_factor = np.exp(-2j * np.pi * np.dot(q, cell))
            gauged_mwd += mwd * phase_factor
        structure_gauge = np.exp(-2j * np.pi * np.dot(q, reference.structure.scaled_positions.T))
        gauged_mwd *= structure_gauge[:, None]
        return gauged_mwd
