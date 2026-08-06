from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import time
from typing import Any
import numpy as np

from pysideband.methods.method import Method, MethodResult
from pysideband.methods.interpolation import Interpolation
from pysideband.mpi import MPIContext
from pysideband.methods.parameters import Field, Parameter, MethodParameters, get_parameters, InternalError, UserInputError
from pysideband.core.units import energy, energy_inverse_to_units, energy_to_units


params: dict[str, Any] = {
    "input": {
        "frequencies": Field(
            syntax="$file",
            description="Path to the frequencies file (e.g., 'path/to/frequencies').",
            params={
                "file": Parameter(
                    attr="frequencies_file",
                    type=Path,
                    default=Path()
                ),
            },
        ),
        "pHRf": Field(
            syntax="$file",
            description="Path to the file containing the partial Huang-Rhys factors (e.g., 'path/to/pHRf').",
            params={
                "file": Parameter(
                    attr="pHRf_file",
                    type=Path,
                    default=Path()
                ),
            },
        ),
    },
    "output": Field(
        syntax="$directory",
        description="Path to the output directory where the results will be saved (e.g., 'path/to/output').",
        params={
            "directory": Parameter(
                attr="output_directory",
                type=Path,
                default=Path("internal-pysideband-parent-folder")
            ),
        },
    ),
    "energy window": {
        "range": Field(
            syntax="\[$value\] [$units]",
            description="Energy window range for the single phonon method (list of two floats and a string for units, e.g., '[0.0, 1.0] eV').",
            params={
                "value": Parameter(
                    attr="energy_window_range_value",
                    type=np.fromstring, type_kwargs={"sep": ",", "dtype": float}, length=2,
                    default=np.array([0.0, 1.0], dtype=float)
                ),
                "units": Parameter(
                    attr="energy_window_range_units",
                    type=str,
                    default="eV"
                ),
            },
        ),
        "step": Field(
            syntax="$value [$units]",
            description="Energy window step size for the single phonon method (float and a string for units, e.g., '0.1 meV').",
            params={
                "value": Parameter(
                    attr="energy_window_step_value",
                    type=float,
                    default=0.1
                ),
                "units": Parameter(
                    attr="energy_window_step_units",
                    type=str,
                    default="meV"
                ),
            },
        ),
    },
    "smearing": {
        "gaussian": Field(
            syntax="$value [$units]",
            description="Gaussian smearing sigma for the single phonon method (float and a string for units, e.g., '1 meV').",
            params={
                "value": Parameter(
                    attr="smearing_gaussian_sigma_value",
                    type=float,
                    default=1
                ),
                "units": Parameter(
                    attr="smearing_gaussian_sigma_units",
                    type=str,
                    default="meV"
                ),
            },
        ),
    },
}


@dataclass
class SinglePhononParameters(MethodParameters):
    frequencies_file: Path | None = None
    pHRf_file: Path | None = None
    output_directory: Path | None = None
    energy_window_range_value: np.ndarray | None = None
    energy_window_range_units: str | None = None
    energy_window_step_value: float | None = None
    energy_window_step_units: str | None = None
    smearing_gaussian_sigma_value: float | None = None
    smearing_gaussian_sigma_units: str | None = None


class GaussianSmearer:
    def __init__(self, sigma: float):
        self._sigma = sigma
    
    def smear(self, base_grid: np.ndarray, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        smeared = np.zeros_like(base_grid)
        for z, f in zip(x, y):
            smeared += f * np.exp(-0.5 * ((base_grid - z) / self._sigma)**2) / (self._sigma * np.sqrt(2 * np.pi))
        return smeared


class SinglePhonon(Method):
    def __init__(self, parameters: dict[str, Any]) -> None:
        super().__init__()
        try:
            self.parameters: SinglePhononParameters = get_parameters(SinglePhononParameters, params, parameters)
        except UserInputError as e:
            print(f"User input error: {e}")
        except InternalError as e:
            print(f"Internal error: {e}")

    def apply_input(self, inputs: MethodResult | None) -> None:
        if inputs is not None:
            if inputs.method_type is not Interpolation:
                raise UserInputError("Single phonon method only accepts an Interpolation method as input.")
            if "frequencies" in inputs.output_files:
                self.parameters.frequencies_file = Path(
                    inputs.output_files["frequencies"]
                )
            if "pHRf" in inputs.output_files:
                self.parameters.pHRf_file = Path(inputs.output_files["pHRf"])

    def run(self, mpi: MPIContext) -> MethodResult:
        if self.parameters.energy_window_range_units.lower() in ["nm"]:
            raise UserInputError("Single phonon method does not support 'nm' units for energy window range.")
        if self.parameters.energy_window_step_units.lower() in ["nm"]:
            raise UserInputError("Single phonon method does not support 'nm' units for energy window step size.")
        if self.parameters.smearing_gaussian_sigma_units.lower() in ["nm"]:
            raise UserInputError("Single phonon method does not support 'nm' units for Gaussian smearing sigma.")
        
        if mpi.is_root: print(
            f"  parameters:" "\n"
            f"    inputs:" "\n"
            f"      phonon frequencies: {self.parameters.frequencies_file}" "\n"
            f"      partial HR factors: {self.parameters.pHRf_file}" "\n"
            f"    energy window:" "\n"
            f"      range: {self.parameters.energy_window_range_value} {self.parameters.energy_window_range_units}" "\n"
            f"      step: {self.parameters.energy_window_step_value} {self.parameters.energy_window_step_units}" "\n"
            f"    smearing:" "\n"
            f"      Gaussian sigma: {self.parameters.smearing_gaussian_sigma_value} {self.parameters.smearing_gaussian_sigma_units}" "\n"
            f"  output directory: {self.parameters.output_directory}",
            flush=True
        )
        
        mpi.barrier()
        
        _start = time.time()
        if mpi.is_root: print(
            f"  Loading phonon frequencies...",
            end="", flush=True
        )
        frequencies = np.load(self.parameters.frequencies_file)
        if mpi.is_root: print(
            f"  done ({time.time() - _start:.2f} seconds)",
            flush=True
        )
        
        _start = time.time()
        if mpi.is_root: print(
            f"  Loading partial HR factors...",
            end="", flush=True
        )
        partial_hrf = np.load(self.parameters.pHRf_file)
        if mpi.is_root: print(
            f"  done ({time.time() - _start:.2f} seconds)",
            flush=True
        )
        
        mpi.barrier()
        
        energy_window_range = energy(self.parameters.energy_window_range_value, self.parameters.energy_window_range_units)
        if self.parameters.energy_window_step_units.lower() in ["point", "points", "p", "pts", "steps", "step"]:
            energy_window_step = (np.max(energy_window_range) - np.min(energy_window_range)) / self.parameters.energy_window_step_value
        else:
            energy_window_step = energy(self.parameters.energy_window_step_value, self.parameters.energy_window_step_units)
        
        smearing_gaussian_sigma = energy(self.parameters.smearing_gaussian_sigma_value, self.parameters.smearing_gaussian_sigma_units)
        
        output_units = self.parameters.energy_window_range_units
        
        mpi.barrier()
        
        _start = time.time()
        if mpi.is_root: print(
            f"  Running singlephonon calculation..." "\n"
            f"    energy window: {energy_window_range} eV (dE = {energy_window_step} eV)" "\n"
            f"    Gaussian smearing sigma: {smearing_gaussian_sigma} eV",
            flush=True
        )
        
        smearer = GaussianSmearer(smearing_gaussian_sigma)
        energy_grid = np.arange(energy_window_range[0], energy_window_range[1] + energy_window_step, energy_window_step)
        local_frequencies = np.array_split(frequencies, mpi.size)[mpi.rank]
        local_partial_hrf = np.array_split(partial_hrf, mpi.size)[mpi.rank]
        
        local_spectrum = smearer.smear(energy_grid, local_frequencies, local_partial_hrf)
        gathered_spectrum = mpi.gather(local_spectrum, root=0)
        
        output_dir = self.parameters.output_directory
        if output_dir == Path("internal-pysideband-parent-folder"):
            output_dir = self.parameters.frequencies_file.parent
        output_dir.mkdir(parents=True, exist_ok=True)
        
        savefile_name = "sph_spectrum"
        if mpi.is_root:
            spectrum = np.sum(gathered_spectrum, axis=0)
            
            print(
                f"  ({time.time() - _start:.2f} seconds)" "\n"
                f"  Area under the spectrum curve: {np.trapezoid(spectrum, energy_grid):.3f}",
            )
            
            with open(f"{output_dir}/{savefile_name}.energy", "wb") as file:
                np.save(file, energy_to_units(energy_grid, output_units))
            with open(f"{output_dir}/{savefile_name}.spectrum", "wb") as file:
                np.save(file, energy_inverse_to_units(spectrum, output_units))
            print(
                f"Saved:" "\n"
                f"  energy grid: {output_dir}/{savefile_name}.energy (units: {output_units})" "\n"
                f"  spectrum: {output_dir}/{savefile_name}.spectrum (units: 1/{output_units})",
                flush=True
            )
        
        mpi.barrier()
        
        if mpi.is_root: print(
            "",
            flush=True
        )
        
        return MethodResult(
            method_type=type(self),
            output_files={
                "energy": Path(f"{output_dir}/{savefile_name}.energy"),
                "spectrum": Path(f"{output_dir}/{savefile_name}.spectrum"),
            },
            details={
                "output_units": output_units,
                "output_dir": output_dir,
            }
        )
