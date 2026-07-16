from __future__ import annotations

import math
from typing import Optional

from tspi.torchtem.physics import energy2wavelength

_unit_categories = {
    "real_space": ("Å", "Angstrom", "nm", "um", "mm", "m"),
    "reciprocal_space": ("1/Å", "1/Angstrom", "1/nm", "1/um", "1/mm", "1/m"),
    "angular": ("rad", "mrad", "deg"),
    "energy": ("eV", "keV"),
}

units_type = {
    unit: category for category, units in _unit_categories.items() for unit in units
}

_conversion_factors = {
    "Å": 1.0,
    "nm": 1e-1,
    "um": 1e-4,
    "mm": 1e-7,
    "m": 1e-10,
    "1/Å": 1.0,
    "1/nm": 10.0,
    "1/um": 1e4,
    "1/mm": 1e7,
    "1/m": 1e10,
    "mrad": 1.0,
    "rad": 1e3,
    "deg": 1e3 / math.pi * 180.0,
}

_tex_units = {
    "Å": r"\mathrm{\AA}",
    "nm": r"\mathrm{nm}",
    "um": r"\mathrm{\mu m}",
    "mm": r"\mathrm{mm}",
    "m": r"\mathrm{m}",
    "1/Å": r"\mathrm{\AA}^{-1}",
    "1/nm": r"\mathrm{nm}^{-1}",
    "1/um": r"\mathrm{\mu m}^{-1}",
    "1/mm": r"\mathrm{mm}^{-1}",
    "1/m": r"\mathrm{m}^{-1}",
    "mrad": r"\mathrm{mrad}",
    "deg": r"\mathrm{deg}",
    "e/Å^2": r"\mathrm{e}^-/\mathrm{\AA}^2",
}


def format_units(units: Optional[str], use_tex: bool = False) -> str:
    if units is None:
        return ""
    if not use_tex:
        return units
    try:
        formatted = _tex_units[units]
    except KeyError:
        if units == "%":
            formatted = r"\mathrm{\%}"
        else:
            formatted = r"\mathrm{" + f"{units}" + r"}"
    return f"${formatted}$"


def validate_units(
    units: Optional[str] = None, old_units: Optional[str] = None
) -> Optional[str]:
    if old_units is None and units is None:
        return None
    if units is None:
        units = old_units
    elif old_units is not None:
        if units_type[units] != units_type[old_units]:
            raise RuntimeError(f"cannot convert units {old_units} to {units}")

    if units not in units_type:
        return units
    if units_type[units] == "real_space":
        return "Å" if units == "Angstrom" else units
    if units_type[units] == "reciprocal_space":
        return "1/Å" if units == "1/Angstrom" else units
    if units_type[units] in ("angular", "energy"):
        return units
    raise ValueError(f"Invalid units: {units}")


def get_conversion_factor(
    units: Optional[str] = None,
    old_units: Optional[str] = None,
    energy: Optional[float] = None,
) -> float:
    if units is None:
        return 1.0
    if old_units is None:
        raise RuntimeError("old_units must be provided if units is provided")

    if units_type[old_units] == "reciprocal_space" and units_type[units] == "angular":
        if energy is None:
            raise RuntimeError(
                "energy must be provided to convert from reciprocal space to angular units"
            )
        wavelength = float(energy2wavelength(energy).item())
        validated = validate_units(units, "mrad")
        assert validated is not None
        return wavelength * 1e3 * _conversion_factors[validated]

    validated = validate_units(units, old_units)
    assert validated is not None
    return _conversion_factors[validated]
