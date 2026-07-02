from __future__ import annotations

from typing import Sequence

import numpy as np

def energy(value: float | np.ndarray, units: str) -> Sequence[float]:
    """Convert units of energy to eV."""
    if units.lower() in ["ev"]:
        return value
    elif units.lower() in ["mev"]:
        return value / 1000.0
    elif units.lower() in ["nm"]:
        return 1239.84193 / value
    elif units.lower() in ["cm-1", "cm^-1"]:
        return value * 0.000123984193
    else:
        raise ValueError(f"Unsupported energy units provided: '{units}', supported units are: eV, meV, nm, cm-1, cm^-1")

def energy_to_units(value: Sequence[float], units: str) -> Sequence[float]:
    """Convert energy in eV to specified units."""
    if units.lower() in ["ev"]:
        return value
    elif units.lower() in ["mev"]:
        return value * 1000.0
    elif units.lower() in ["nm"]:
        return 1239.84193 / value
    elif units.lower() in ["cm-1", "cm^-1"]:
        return value / 0.000123984193
    else:
        raise ValueError(f"Unsupported energy units provided: '{units}', supported units are: eV, meV, nm, cm-1, cm^-1")

def energy_inverse_to_units(value: Sequence[float], units: str) -> Sequence[float]:
    """Convert quantity from 1/eV to specified 1/units."""
    if units.lower() in ["ev"]:
        return value
    elif units.lower() in ["mev"]:
        return value / 1000.0
    elif units.lower() in ["nm"]:
        return value / 1239.84193
    elif units.lower() in ["cm-1", "cm^-1"]:
        return value * 0.000123984193
    else:
        raise ValueError(f"Unsupported energy units provided: '{units}', supported units are: eV, meV, nm, cm-1, cm^-1")

def temperature(value: Sequence[float], units: str) -> Sequence[float]:
    """Convert units of temperature to K."""
    if units.lower() in ["k"]:
        return value
    elif units.lower() in ["c", "celsius"]:
        return value + 273.15
    elif units.lower() in ["f", "fahrenheit"]:
        return (value - 32) * 5 / 9 + 273.15
    else:
        raise ValueError(f"Unsupported temperature units provided: '{units}', supported units are: K, °C, °F")

def temperature_to_units(value: Sequence[float], units: str) -> Sequence[float]:
    """Convert temperature in K to specified units."""
    if units.lower() in ["k"]:
        return value
    elif units.lower() in ["c", "celsius", "°c"]:
        return value - 273.15
    elif units.lower() in ["f", "fahrenheit", "°f"]:
        return (value - 273.15) * 9 / 5 + 32
    else:
        raise ValueError(f"Unsupported temperature units provided: '{units}', supported units are: K, °C, °F")
