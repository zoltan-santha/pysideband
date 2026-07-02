from __future__ import annotations

from dataclasses import dataclass
import time
import numpy as np
from pathlib import Path
from typing import Any

from pysideband.mpi import MPIContext
from pysideband.core import AbInitioCalculator, Phonons, Structure, TransitionForce, QMesh
from pysideband.methods.method import Method, MethodResult
from pysideband.methods.parameters import Field, Parameter, MethodParameters, get_parameters, InternalError, UserInputError


params: dict[str, Any] = {
    "grid": {
        "size": Field(
            syntax="\[$size\]",
            description="Grid size for interpolation (e.g., 'size: [1, 1, 1]').",
            params={
                "size": Parameter(
                    attr="grid_size",
                    type=np.fromstring, type_kwargs={"sep": ",", "dtype": int}, length=3,
                    default=np.array([1, 1, 1], dtype=int)
                ),
            },
        ),
        "shift": Field(
            syntax="\[$shift\]",
            description="Grid shift for interpolation (e.g., 'shift: [0.0, 0.0, 0.0]').",
            params={
                "shift": Parameter(
                    attr="grid_shift",
                    type=np.fromstring, type_kwargs={"sep": ",", "dtype": float}, length=3,
                    default=np.array([0.0, 0.0, 0.0], dtype=float)
                ),
            },
        ),
    },
    "initial state": {
        "structure": Field(
            syntax="$file [| $abinitio_calculator]",
            description="Initial state structure file (e.g., 'path/to/CONTCAR | vasp').",
            params={
                "file": Parameter(
                    attr="initial_structure_file",
                    type=Path
                ),
                "abinitio_calculator": Parameter(
                    attr="initial_structure_format",
                    type=AbInitioCalculator,
                    default=AbInitioCalculator.VASP
                ),
            },
        ),
    },
    "final state": {
        "structure": Field(
            syntax="$file [| $abinitio_calculator]",
            description="Final state structure file (e.g., 'path/to/CONTCAR | vasp').",
            params={
                "file": Parameter(
                    attr="final_structure_file",
                    type=Path
                ),
                "abinitio_calculator": Parameter(
                    attr="final_structure_format",
                    type=AbInitioCalculator,
                    default=AbInitioCalculator.VASP
                ),
            },
        ),
        "phonons": {
            "force constants": Field(
                syntax="$file",
                description="Force constants file for final state phonons.",
                params={
                    "file": Parameter(
                        attr="final_structure_force_constants_file",
                        type=Path
                    ),
                },
            ),
            "phonopy displacements": Field(
                syntax="$file",
                description="Phonopy displacements YAML file for final state phonons.",
                params={
                    "file": Parameter(
                        attr="final_structure_phonopy_disp_yaml_file",
                        type=Path
                    ),
                },
            ),
        },
    },
}


@dataclass
class InterpolationParameters(MethodParameters):
    grid_size: np.ndarray | None = None
    grid_shift: np.ndarray | None = None
    initial_structure_file: Path | None = None
    initial_structure_format: AbInitioCalculator | None = None
    final_structure_file: Path | None = None
    final_structure_format: AbInitioCalculator | None = None
    final_structure_force_constants_file: Path | None = None
    final_structure_phonopy_disp_yaml_file: Path | None = None


class Interpolation(Method):
    def __init__(self, parameters: dict[str, Any]) -> None:
        super().__init__()
        try:
            self.parameters: InterpolationParameters = get_parameters(InterpolationParameters, params, parameters)
        except UserInputError as e:
            print(f"User input error: {e}")
        except InternalError as e:
            print(f"Internal error: {e}")

    def apply_input(self, inputs: MethodResult | None) -> None:
        if inputs is not None:
            raise ValueError(
                "Interpolation method does not accept input from previous steps."
            )

    def run(self, mpi: MPIContext) -> MethodResult:
        if mpi.is_root: print(
            f"  parameters:" "\n"
            f"    grid size: {self.parameters.grid_size} (shift: {self.parameters.grid_shift})" "\n"
            f"    initial state:" "\n"
            f"      structure: {self.parameters.initial_structure_file} ({self.parameters.initial_structure_format.value})" "\n"
            f"    final state:" "\n"
            f"      structure: {self.parameters.final_structure_file} ({self.parameters.final_structure_format.value})" "\n"
            f"      phonons:" "\n"
            f"        force constants: {self.parameters.final_structure_force_constants_file}" "\n"
            f"        displacements: {self.parameters.final_structure_phonopy_disp_yaml_file}",
            flush=True
        )
        
        mpi.barrier()
        
        _start = time.time()
        if mpi.is_root: print(
            f"  Loading force constants and reference structure for phonons...",
            end="", flush=True
        )
        phonons = Phonons(
            phonopy_yaml_path=self.parameters.final_structure_phonopy_disp_yaml_file,
            FORCE_SETS_path=None,
            FORCE_CONSTANTS_path=self.parameters.final_structure_force_constants_file,
        )
        if mpi.is_root: print(
            f"  done ({time.time() - _start:.2f} seconds)",
            flush=True
        )
        
        _start = time.time()
        if mpi.is_root: print(
            f"  Generating q-mesh for interpolation...",
            end="", flush=True
        )
        qmesh = QMesh(
            phonons=phonons,
            mesh_size=self.parameters.grid_size,
            mesh_shift=self.parameters.grid_shift
        )
        qpoints = qmesh.qpoints
        weights = qmesh.weights
        if mpi.is_root: print(
            f"  done ({time.time() - _start:.2f} seconds)",
            flush=True,
        )
        
        _start = time.time()
        if mpi.is_root: print(
            f"  Loading initial state structure...",
            end="", flush=True
        )
        initial_structure = Structure.from_file(
            file_path=self.parameters.initial_structure_file,
            calculator=self.parameters.initial_structure_format,
        )
        if mpi.is_root: print(
            f"  done ({time.time() - _start:.2f} seconds)",
            flush=True,
        )
        
        _start = time.time()
        if mpi.is_root: print(
            f"  Aligning initial structure to phonon structure...",
            end="", flush=True
        )
        initial_structure.align_to(phonons.structure)
        if mpi.is_root: print(
            f"  done ({time.time() - _start:.2f} seconds)",
            flush=True,
        )
        
        _start = time.time()
        if mpi.is_root: print(
            f"  Loading final state structure...",
            end="", flush=True
        )
        final_structure = Structure.from_file(
            file_path=self.parameters.final_structure_file,
            calculator=self.parameters.final_structure_format,
        )
        if mpi.is_root: print(
            f"  done ({time.time() - _start:.2f} seconds)",
            flush=True,
        )
        
        _start = time.time()
        if mpi.is_root: print(
            f"  Aligning final structure to phonon structure...",
            end="", flush=True
        )
        final_structure.align_to(phonons.structure)
        if mpi.is_root: print(
            f"  done ({time.time() - _start:.2f} seconds)",
            flush=True,
        )
        
        _start = time.time()
        if mpi.is_root: print(
            f"  Calculating transition force...",
            end="", flush=True
        )
        transition_force = TransitionForce(
            initial=initial_structure,
            final=final_structure,
            phonons=phonons,
        )
        if mpi.is_root: print(
            f"  done ({time.time() - _start:.2f} seconds)",
            flush=True,
        )
        
        qpoint_chunks = np.array_split(qpoints, mpi.size)
        weight_chunks = np.array_split(weights, mpi.size)
        
        local_qpoints = qpoint_chunks[mpi.rank]
        local_weights = weight_chunks[mpi.rank]
        
        mpi.comm.barrier()
        
        _start = time.time()
        if mpi.is_root: print(
            f"  Running phonon interpolation..." "\n"
            f"    q-points: {len(qpoints)}" "\n"
            f"    phonon modes: {phonons.degrees_of_freedom} ({len(phonons.structure)} atoms)",
            flush=True
        )
        
        local_frequencies, local_phrf = phonons.run_interpolation(
            transition_force=transition_force,
            q_points=local_qpoints,
            q_weights=local_weights,
        )
        
        gathered = mpi.gather((local_frequencies, local_phrf), root=0)
        
        savefile_name = f"interpolated_{self.parameters.grid_size[0]}x{self.parameters.grid_size[1]}x{self.parameters.grid_size[2]}"
        if mpi.is_root:
            frequencies = np.concatenate([item[0] for item in gathered], axis=None)
            partial_hrf = np.concatenate([item[1] for item in gathered], axis=None)
            
            partial_hrf /= np.sum(weights)
            
            total_hr_factor = np.sum(partial_hrf)
            print(
                f"  ({time.time() - _start:.2f} seconds)" "\n"
                f"  Total HR factor: {total_hr_factor:.3f}",
                flush=True
            )
            
            with open(f"{savefile_name}.frequencies", "wb") as file:
                np.save(file, frequencies)
            with open(f"{savefile_name}.pHRf", "wb") as file:
                np.save(file, partial_hrf)
            print(
                f"Saved:" "\n"
                f"  phonon frequencies: {savefile_name}.frequencies" "\n"
                f"  partial HR factors: {savefile_name}.pHRf",
                flush=True
            )
        
        mpi.comm.barrier()
        
        if mpi.is_root: print(
            "",
            flush=True
        )
        
        return MethodResult(
            output_files={
                "frequencies": Path(
                    f"{savefile_name}.frequencies"
                ),
                "pHRf": Path(
                    f"{savefile_name}.pHRf"
                ),
            }
        )
