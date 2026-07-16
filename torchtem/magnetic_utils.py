from __future__ import annotations

import torch

bohr_magneton = 9.2740100657e-24 * 1e20  # A * Å^2
vacuum_permeability = 1.25663706127e-6 * 1e10  # T * Å / A


def saturation_magnetization(
    magnetic_moments: torch.Tensor | list[float] | list[list[float]], volume: float
) -> torch.Tensor:
    moments = torch.as_tensor(magnetic_moments, dtype=torch.float64)
    return bohr_magneton * vacuum_permeability * moments.sum() / float(volume)


def set_magnetic_moments(
    positions: torch.Tensor | list[list[float]],
    magnetic_moments: torch.Tensor | list[float] | list[list[float]],
) -> dict[str, torch.Tensor]:
    positions_tensor = torch.as_tensor(positions, dtype=torch.float64)
    moments = torch.as_tensor(magnetic_moments, dtype=torch.float64)
    if moments.ndim == 1:
        moments = moments.unsqueeze(0).repeat(positions_tensor.shape[0], 1)
    if moments.shape != (positions_tensor.shape[0], 3):
        raise ValueError("magnetic_moments must have shape (n_atoms, 3)")
    return {"positions": positions_tensor, "magnetic_moments": moments}
