from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from pathlib import Path
import time
from typing import Any
import numpy as np

from pysideband.methods.method import Method, MethodResult
from pysideband.mpi import MPIContext
from pysideband.methods.parameters import Field, Parameter, MethodParameters, get_parameters, InternalError, UserInputError
from pysideband.core.units import energy, energy_to_units, temperature, temperature_to_units


class Process(Enum):
    ABSORPTION = "absorption"
    EMISSION = "emission"
    
    @classmethod
    def from_string(cls, value: str) -> Process:
        aliases = {
            "absorption": cls.ABSORPTION,
            "abs": cls.ABSORPTION,
            "a": cls.ABSORPTION,
            "emission": cls.EMISSION,
            "emi": cls.EMISSION,
            "e": cls.EMISSION,
        }
        if value.lower() in aliases:
            return aliases[value.lower()]
        raise ValueError(f"Unknown process type: {value}")


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
    "process": Field(
        syntax="$process",
        description="Process type for the multi phonon method (string, either 'absorption' or 'emission').",
        params={
            "process": Parameter(
                attr="process",
                type=Process.from_string,
                default=Process.ABSORPTION
            ),
        },
    ),
    "ZPL": Field(
        syntax="$energy [$units]",
        description="Zero-phonon line energy for the multi phonon method (float and a string for units, e.g., '1.945 eV').",
        params={
            "energy": Parameter(
                attr="zpl_energy_value",
                type=float
            ),
            "units": Parameter(
                attr="zpl_energy_units",
                type=str,
                default="eV"
            ),
        },
    ),
    "energy window": {
        "range": Field(
            syntax="\[$range\] [$units]",
            description="Energy window range for the multi phonon method (two floats and a string for units, e.g., '[1.2, 2.1] eV').",
            params={
                "range": Parameter(
                    attr="energy_window_range_value",
                    type=np.fromstring, type_kwargs={"sep": ",", "dtype": float}, length=2
                ),
                "units": Parameter(
                    attr="energy_window_range_units",
                    type=str,
                    default="eV"
                ),
            },
        ),
        "step": Field(
            syntax="$step [$units]",
            description="Energy window step size for the multi phonon method (float and a string for units, e.g., '0.1 meV').",
            params={
                "step": Parameter(
                    attr="energy_window_step_value",
                    type=float
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
            syntax="$sigma [$units]",
            description="Gaussian smearing for the multi phonon method (float and a string for units, e.g., '0.5 meV').",
            params={
                "sigma": Parameter(
                    attr="smearing_gaussian_sigma_value",
                    type=float
                ),
                "units": Parameter(
                    attr="smearing_gaussian_sigma_units",
                    type=str,
                    default="meV"
                ),
            },
        ),
        "lorentzian": Field(
            syntax="$gamma [$units]",
            description="Lorentzian smearing for the multi phonon method (float and a string for units, e.g., '0.5 meV').",
            params={
                "gamma": Parameter(
                    attr="smearing_lorentzian_gamma_value",
                    type=float
                ),
                "units": Parameter(
                    attr="smearing_lorentzian_gamma_units",
                    type=str,
                    default="meV"
                ),
            },
        ),
    },
    "temperature": Field(
        syntax="\[$temperature\] [$units]",
        description="Temperature for the multi phonon method (float or list of floats and a string for units, e.g., '0 K' or '[0, 4, 77, 300] K').",
        params={
            "temperature": Parameter(
                attr="temperature_value",
                type=np.fromstring, type_kwargs={"sep": ",", "dtype": float},
                default=np.array([0.0], dtype=float)
            ),
            "units": Parameter(
                attr="temperature_units",
                type=str,
                default="K"
            ),
        },
    ),
}


@dataclass
class MultiPhononParameters(MethodParameters):
    frequencies_file: Path | None = None
    pHRf_file: Path | None = None
    process: Process | None = None
    zpl_energy_value: float | None = None
    zpl_energy_units: str | None = None
    energy_window_range_value: np.ndarray | None = None
    energy_window_range_units: str | None = None
    energy_window_step_value: float | None = None
    energy_window_step_units: str | None = None
    smearing_gaussian_sigma_value: float | None = None
    smearing_gaussian_sigma_units: str | None = None
    smearing_lorentzian_gamma_value: float | None = None
    smearing_lorentzian_gamma_units: str | None = None
    temperature_value: np.ndarray | None = None
    temperature_units: str | None = None


def _bose_occupation(frequencies: np.ndarray, T: float) -> np.ndarray:
    k_B = 8.617333262145e-5  # Boltzmann constant in eV/K
    if T <= 0:
        return np.zeros_like(frequencies)
    
    x = frequencies / (k_B * T)
    nbar = np.zeros_like(x)
    
    under_overflow_mask = x < 700  # Avoid overflow in exp for large x
    mask = under_overflow_mask
    nbar[mask] = 1.0 / np.expm1(x[mask])
    
    return nbar


def _estimate_nyquist_eV(
    frequencies: np.ndarray,
    partial_hrf: np.ndarray,
    T_max: float,
    kappa_min: float,
    kappa_max: float,
    alias_sigma: float = 8,
    safety_eV: float = 0.1
) -> float:
    abs_value_needed = max(abs(kappa_min), abs(kappa_max))
    
    effective_abs_frequency = abs(np.sum(frequencies * partial_hrf))
    nbar = _bose_occupation(frequencies, T_max)
    frequency_variance = np.sum(partial_hrf * (2.0 * nbar + 1.0) * frequencies**2)
    frequency_std_dev = np.sqrt(max(float(frequency_variance), 0.0))
    frequency_max = float(np.max(frequencies))
    
    estimated_support = effective_abs_frequency + alias_sigma * frequency_std_dev + 5.0 * frequency_max
    
    return max(abs_value_needed*1.1, estimated_support, safety_eV)


def _next_power_of_two(n: int) -> int:
    if n <= 0:
        return 1
    return 1 << (int(n) - 1).bit_length()


class MultiPhonon(Method):
    def __init__(self, parameters: dict[str, Any]) -> None:
        super().__init__()
        try:
            self.parameters: MultiPhononParameters = get_parameters(MultiPhononParameters, params, parameters)
        except UserInputError as e:
            print(f"User input error: {e}")
        except InternalError as e:
            print(f"Internal error: {e}")

    def apply_input(self, inputs: MethodResult | None) -> None:
        if inputs is not None:
            if "frequencies" in inputs.output_files:
                self.parameters.frequencies_file = Path(
                    inputs.output_files["frequencies"]
                )
            if "pHRf" in inputs.output_files:
                self.parameters.pHRf_file = Path(inputs.output_files["pHRf"])

    def run(self, mpi: MPIContext) -> MethodResult:
        if mpi.is_root: print(
            f"  parameters:" "\n"
            f"    inputs:" "\n"
            f"      phonon frequencies: {self.parameters.frequencies_file}" "\n"
            f"      partial HR factors: {self.parameters.pHRf_file}" "\n"
            f"    process: {self.parameters.process.value}" "\n"
            f"    ZPL energy: {self.parameters.zpl_energy_value} {self.parameters.zpl_energy_units}" "\n"
            f"    energy window:" "\n"
            f"      range: {self.parameters.energy_window_range_value} {self.parameters.energy_window_range_units}" "\n"
            f"      step: {self.parameters.energy_window_step_value} {self.parameters.energy_window_step_units}" "\n"
            f"    smearing:" "\n"
            f"      Gaussian sigma: {self.parameters.smearing_gaussian_sigma_value} {self.parameters.smearing_gaussian_sigma_units}" "\n"
            f"      Lorentzian gamma: {self.parameters.smearing_lorentzian_gamma_value} {self.parameters.smearing_lorentzian_gamma_units}" "\n"
            f"    temperature: {self.parameters.temperature_value} {self.parameters.temperature_units}",
            flush=True
        )
        
        output_files = {
            "energy": None,
            "spectrum": {},
            "lineshape": {},
        }
        
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
        
        zpl_energy_eV = np.abs(energy(self.parameters.zpl_energy_value, self.parameters.zpl_energy_units))
        
        energy_window_range_eV = energy(self.parameters.energy_window_range_value, self.parameters.energy_window_range_units)
        if self.parameters.energy_window_step_units.lower() in ["point", "points", "p", "pts", "steps", "step"]:
            energy_window_step_eV = (np.max(energy_window_range_eV) - np.min(energy_window_range_eV)) / self.parameters.energy_window_step_value
        elif self.parameters.energy_window_step_units.lower() in ["nm"]:
            _energy_window_range_nm = energy_to_units(energy_window_range_eV, "nm")
            energy_window_step_eV = self.parameters.energy_window_step_value * energy(np.max(_energy_window_range_nm), "nm") / np.max(_energy_window_range_nm)
        else:
            energy_window_step_eV = energy(self.parameters.energy_window_step_value, self.parameters.energy_window_step_units)
        
        if self.parameters.smearing_gaussian_sigma_units.lower() in ["nm"]:
            _energy_window_range_nm = energy_to_units(energy_window_range_eV, "nm")
            smearing_gaussian_sigma_eV = self.parameters.smearing_gaussian_sigma_value * energy(np.max(_energy_window_range_nm), "nm") / np.max(_energy_window_range_nm)
        else:
            smearing_gaussian_sigma_eV = energy(self.parameters.smearing_gaussian_sigma_value, self.parameters.smearing_gaussian_sigma_units)
        
        if self.parameters.smearing_lorentzian_gamma_units.lower() in ["nm"]:
            _energy_window_range_nm = energy_to_units(energy_window_range_eV, "nm")
            smearing_lorentzian_gamma_eV = self.parameters.smearing_lorentzian_gamma_value * energy(np.max(_energy_window_range_nm), "nm") / np.max(_energy_window_range_nm)
        else:
            smearing_lorentzian_gamma_eV = energy(self.parameters.smearing_lorentzian_gamma_value, self.parameters.smearing_lorentzian_gamma_units)
        
        temperatures_K = temperature(self.parameters.temperature_value, self.parameters.temperature_units)
        
        output_units = self.parameters.energy_window_range_units
        
        mpi.barrier()
        
        if mpi.is_root: print(
            f"  Running multiphonon calculation...",
            flush=True
        )
        
        E_min = np.min(energy_window_range_eV)
        E_max = np.max(energy_window_range_eV)
        dE_max_eV = energy_window_step_eV
        if mpi.is_root: print(
            f"    energy window: {np.array(energy_window_range_eV)} eV (dE = {dE_max_eV} eV)" "\n"
            f"    Gaussian smearing sigma: {smearing_gaussian_sigma_eV} eV" "\n"
            f"    Lorentzian smearing gamma: {smearing_lorentzian_gamma_eV} eV" "\n"
            f"    temperatures: {temperatures_K} K",
            flush=True
        )
        
        kappa_min_needed = E_min - zpl_energy_eV
        kappa_max_needed = E_max - zpl_energy_eV
        
        nyquist_eV = _estimate_nyquist_eV(
            frequencies=frequencies,
            partial_hrf=partial_hrf,
            T_max=np.max(temperatures_K),
            kappa_min=kappa_min_needed,
            kappa_max=kappa_max_needed,
        )
        if mpi.is_root: print(
            f"    estimated Nyquist energy: {nyquist_eV} eV",
            flush=True
        )
        
        dtau = np.pi / nyquist_eV
        n_min = np.ceil(2.0 * nyquist_eV / dE_max_eV)
        N = _next_power_of_two(max(n_min, 16))
        
        dE_fft = 2.0 * nyquist_eV / N
        
        n_out = int(np.floor((E_max - E_min) / dE_fft)) + 1
        E_out = E_min + np.arange(n_out, dtype=float) * dE_fft
        kappa_out = E_out - zpl_energy_eV
        
        if mpi.is_root: print(
            f"    FFT grid size: {N} points",
            flush=True
        )
        
        _start = time.time()
        if mpi.is_root: print(
            f"    Saving energy grid...",
            end="", flush=True
        )
        
        output_files["energy"] = Path(f"mph_spectrum.energy")
        if mpi.is_root:
            with open(f"{output_files['energy']}", "wb") as file:
                np.save(file, energy_to_units(E_out, output_units))
        
        if mpi.is_root: print(
            f"  done ({time.time() - _start:.2f} seconds)",
            flush=True
        )
        
        tau = (np.arange(N, dtype=float) - N // 2) * dtau
        
        local_frequencies = np.array_split(frequencies, mpi.size)[mpi.rank]
        local_partial_hrf = np.array_split(partial_hrf, mpi.size)[mpi.rank]
        
        if self.parameters.process == Process.ABSORPTION:
            sign = +1.0
        elif self.parameters.process == Process.EMISSION:
            sign = -1.0
        else:
            raise InternalError(f"Unreachable code reached: Unknown process type: {self.parameters.process}")
        
        _start = time.time()
        if mpi.is_root: print(
            f"    Calculating temperature independent contribution...",
            end="", flush=True
        )
        
        local_S_tau_0 = np.zeros_like(tau, dtype=np.complex128)
        
        local_single_tau_contribution = np.exp(sign * 1j * local_frequencies * tau[0])
        local_phrf_sum = np.sum(local_partial_hrf)
        dphase = np.exp(sign * 1j * local_frequencies * dtau)
        
        for i in range(len(tau)):
            local_S_tau_0[i] = np.sum(local_partial_hrf * local_single_tau_contribution) - local_phrf_sum
            local_single_tau_contribution *= dphase
        
        if mpi.is_root: print(
            f"  done ({time.time() - _start:.2f} seconds)",
            flush=True
        )
        
        mpi.barrier()
        
        gathered_S_tau_0 = mpi.gather(local_S_tau_0, root=0)
        S_tau_0 = np.sum(gathered_S_tau_0, axis=0)
        S_tau_0 = mpi.bcast(S_tau_0, root=0)
        
        raw_kappa = np.fft.fftshift(
            2.0 * np.pi * np.fft.fftfreq(
                N, d=dtau
            )
        )
        
        if mpi.is_root: print(
            f"    Calculating temperature dependent contributions...",
            flush=True
        )
        
        for T in temperatures_K:
            _start = time.time()
            if mpi.is_root: print(
                f"      T = {T} K...",
                end="", flush=True
            )
            
            local_nbar = _bose_occupation(local_frequencies, T)
            local_S_tau_T = np.zeros_like(tau, dtype=np.complex128)
            
            local_single_tau_contribution = np.exp(1j * local_frequencies * tau[0])
            local_weighted_phrf_sum = np.sum(local_partial_hrf * local_nbar)
            dphase = np.exp(1j * local_frequencies * dtau)
            
            for i in range(len(tau)):
                factor = local_single_tau_contribution + np.conj(local_single_tau_contribution)
                local_S_tau_T[i] = np.sum(local_nbar * local_partial_hrf * factor) - 2*local_weighted_phrf_sum
                local_single_tau_contribution *= dphase
            
            if mpi.is_root: print(
                f"  done ({time.time() - _start:.2f} seconds)",
                flush=True
            )
            
            gathered_S_tau_T = mpi.gather(local_S_tau_T, root=0)
            S_tau_T = np.sum(gathered_S_tau_T, axis=0)
            S_tau_T = mpi.bcast(S_tau_T, root=0)
            
            G_tau = np.exp(S_tau_0 + S_tau_T)
            
            if smearing_gaussian_sigma_eV > 0.0:
                sigma_E = smearing_gaussian_sigma_eV / (2.0 * np.sqrt(2.0 * np.log(2.0)))
                G_tau *= np.exp(-0.5 * (sigma_E * tau)**2)
            
            if smearing_lorentzian_gamma_eV > 0.0:
                gamma_E = smearing_lorentzian_gamma_eV / 2.0
                G_tau *= np.exp(-gamma_E * np.abs(tau))
            
            _start = time.time()
            if mpi.is_root: print(
                f"        Calculating spectral function...",
                end="", flush=True
            )
            
            raw_A = np.fft.fftshift(
                np.fft.fft(
                    np.fft.ifftshift(
                        G_tau
                    )
                )
            ) * dtau / (2.0 * np.pi)
            
            A_real = np.interp(kappa_out, raw_kappa, raw_A.real)
            A_imag = np.interp(kappa_out, raw_kappa, raw_A.imag)
            
            max_real = np.max(np.abs(A_real))
            max_imag = np.max(np.abs(A_imag))
            
            if max_real > 0.0 and max_imag/max_real > 1e-6:
                print(f"          Warning: Imaginary part of the spectrum is significant (max ratio: {max_imag/max_real:.2e})")
            
            A = A_real
            
            if mpi.is_root: print(
                f"  done ({time.time() - _start:.2f} seconds)",
                flush=True
            )
            
            _start = time.time()
            if mpi.is_root: print(
                f"        Calculating lineshape function...",
                end="", flush=True
            )
            
            if self.parameters.process == Process.ABSORPTION:
                prefactor = np.where(E_out > 0.0, E_out, 0.0)
            elif self.parameters.process == Process.EMISSION:
                prefactor = np.where(E_out > 0.0, E_out**3, 0.0)
            else:
                raise InternalError(f"Unreachable code reached: Unknown process type: {self.parameters.process}")
            
            L = prefactor * A
            L /= np.trapezoid(L, E_out)
            
            if mpi.is_root: print(
                f"  done ({time.time() - _start:.2f} seconds)",
                flush=True
            )
            
            _start = time.time()
            if mpi.is_root: print(
                f"        Saving...",
                end="", flush=True
            )
            
            savefile_name = f"mph_spectrum_T{int(T)}K"
            output_files["spectrum"][f"T{int(T)}K"] = Path(f"{savefile_name}.spectrum")
            output_files["lineshape"][f"T{int(T)}K"] = Path(f"{savefile_name}.lineshape")
            if mpi.is_root:
                with open(f"{savefile_name}.spectrum", "wb") as file:
                    np.save(file, A)
                with open(f"{savefile_name}.lineshape", "wb") as file:
                    np.save(file, L)
            
            if mpi.is_root: print(
                f"  done ({time.time() - _start:.2f} seconds)",
                flush=True
            )
        
        mpi.barrier()
        
        if mpi.is_root: print(
            f"Saved:" "\n"
            f"  energy grid: {output_files['energy']} (units: {output_units})",
            flush=True
        )
        max_header_length = max(len(f"T = {int(T)} K:") for T in temperatures_K)
        for T in temperatures_K:
            padding = " " * (max_header_length - len(f"T = {int(T)} K:"))
            header = f"  T = {int(T)} K:"
            blank = " " * len(header)
            if mpi.is_root: print(
                f"{header}{padding} spectrum:  {output_files['spectrum'][f'T{int(T)}K']}  (units: 1/eV)" "\n"
                f"{blank}{padding} lineshape: {output_files['lineshape'][f'T{int(T)}K']} (units: a.u.)",
                flush=True
            )
        
        return MethodResult(
            output_files=output_files
        )
