from __future__ import annotations

from typing import Sequence

import torch
from torch import nn

from torchtem.atomic_parametrizations import (
    kirkland_scaled_projected_parameters,
    lobato_scaled_projected_parameters,
    peng_scaled_parameters,
    render_kirkland_projected_potential,
    render_lobato_projected_potential,
    render_peng_finite_projected_potential,
)
from torchtem.core_loss import electron_configurations
from torchtem.integrals import (
    GaussianProjectionIntegrals,
    QuadratureProjectionIntegrals,
    ScatteringFactorProjectionIntegrals,
)
from torchtem.slicing import slice_limits, validate_slice_thickness


def _atomic_symbol_to_number() -> dict[str, int]:
    return {symbol: i + 1 for i, symbol in enumerate(electron_configurations().keys())}


class IAMPotentialBuilder(nn.Module):
    """Differentiable IAM-style potential builder that returns multislice-ready slices."""

    def __init__(
        self,
        *,
        gpts: tuple[int, int],
        sampling: tuple[float, float],
        positions_A: torch.Tensor,
        symbols: Sequence[str],
        cell: torch.Tensor,
        slice_thickness: float | tuple[float, ...],
        parametrization: str = "lobato",
        projection: str = "infinite",
        amplitudes: torch.Tensor | None = None,
        peng_table: str = "peng_high.json",
        dtype: torch.dtype = torch.float64,
        device=None,
    ) -> None:
        super().__init__()
        self.gpts = gpts
        self.sampling = sampling
        self.symbols = tuple(symbols)
        self.parametrization = parametrization
        self.projection = projection
        self.peng_table = peng_table
        self.dtype = dtype
        self.device_hint = device
        self.register_buffer("cell", cell.to(device=device, dtype=dtype))
        self.positions_A = nn.Parameter(positions_A.to(device=device, dtype=dtype))
        if amplitudes is None:
            amplitudes = torch.ones((positions_A.shape[0],), dtype=dtype)
        self.amplitudes = nn.Parameter(amplitudes.to(device=device, dtype=dtype))
        self.slice_thickness = validate_slice_thickness(
            slice_thickness, thickness=float(self.cell[2, 2].item())
        )

        if len(self.symbols) != int(positions_A.shape[0]):
            raise ValueError("Length of symbols must match number of atoms")
        symbol_to_number = _atomic_symbol_to_number()
        self.register_buffer(
            "atomic_numbers",
            torch.tensor([symbol_to_number[s] for s in self.symbols], dtype=torch.int64, device=device),
        )

        if parametrization == "lobato":
            scaled = torch.stack(
                [lobato_scaled_projected_parameters(symbol) for symbol in self.symbols], dim=0
            )
            self.register_buffer("scaled_parameters", scaled.to(device=device, dtype=dtype))
        elif parametrization == "kirkland":
            scaled = torch.stack(
                [kirkland_scaled_projected_parameters(symbol) for symbol in self.symbols], dim=0
            )
            self.register_buffer("scaled_parameters", scaled.to(device=device, dtype=dtype))
        elif parametrization == "peng":
            scaled = torch.stack(
                [peng_scaled_parameters(symbol, peng_table)["finite_projected_potential"] for symbol in self.symbols],
                dim=0,
            )
            self.register_buffer("scaled_parameters", scaled.to(device=device, dtype=dtype))
        else:
            raise ValueError(f"Unsupported parametrization {parametrization}")

        self._infinite_integrator = ScatteringFactorProjectionIntegrals(self.parametrization)
        self._finite_integrator = None
        if self.parametrization in ("lobato", "kirkland"):
            self._finite_integrator = QuadratureProjectionIntegrals(parametrization=self.parametrization)

    @property
    def num_slices(self) -> int:
        return len(self.slice_thickness)

    @property
    def limits(self) -> list[tuple[float, float]]:
        return slice_limits(self.slice_thickness)

    def _slice_mask(self, a: float, b: float) -> torch.Tensor:
        z = self.positions_A[:, 2]
        return (z >= a) & (z < b)

    def _render_infinite_slice(self, mask: torch.Tensor) -> torch.Tensor:
        if mask.sum() == 0:
            return torch.zeros(self.gpts, dtype=self.dtype, device=self.positions_A.device)
        if self._infinite_integrator is None:
            raise RuntimeError("No infinite-projection integrator available for this parametrization.")
        return self._integrate_with_amplitudes(
            self._infinite_integrator,
            mask=mask,
            a=0.0,
            b=float(self.cell[2, 2].item()),
        )

    def _render_finite_slice(self, a: float, b: float) -> torch.Tensor:
        if self.parametrization == "peng":
            return render_peng_finite_projected_potential(
                gpts=self.gpts,
                sampling=self.sampling,
                positions_A=self.positions_A[:, :2],
                scaled_parameters=self.scaled_parameters,
                slice_limits_A=(a, b),
                z_positions_A=self.positions_A[:, 2],
                amplitudes=self.amplitudes,
                dtype=self.dtype,
                device=self.positions_A.device,
            )
        if self._finite_integrator is None:
            raise NotImplementedError("No finite-projection integrator available for this parametrization.")
        return self._integrate_with_amplitudes(
            self._finite_integrator,
            mask=torch.ones((self.positions_A.shape[0],), dtype=torch.bool, device=self.positions_A.device),
            a=a,
            b=b,
        )

    def _integrate_with_amplitudes(
        self,
        integrator: ScatteringFactorProjectionIntegrals | GaussianProjectionIntegrals,
        *,
        mask: torch.Tensor,
        a: float,
        b: float,
    ) -> torch.Tensor:
        if mask.sum() == 0:
            return torch.zeros(self.gpts, dtype=self.dtype, device=self.positions_A.device)
        device = self.positions_A.device
        dtype = self.dtype
        positions = self.positions_A[mask]
        symbols = [self.symbols[i] for i, keep in enumerate(mask.tolist()) if keep]
        amplitudes = self.amplitudes[mask]

        if integrator.finite and not integrator.periodic:
            positions, symbols, amplitudes = self._pad_periodic_images(
                positions=positions,
                symbols=symbols,
                amplitudes=amplitudes,
                integrator=integrator,
            )

        out = torch.zeros(self.gpts, dtype=dtype, device=device)
        for i, symbol in enumerate(symbols):
            contribution = integrator.integrate_on_grid(
                positions_A=positions[i : i + 1],
                symbols=[symbol],
                a=a,
                b=b,
                gpts=self.gpts,
                sampling=self.sampling,
                device=device,
                dtype=dtype,
            )
            out = out + amplitudes[i] * contribution
        return out

    def _pad_periodic_images(
        self,
        *,
        positions: torch.Tensor,
        symbols: list[str],
        amplitudes: torch.Tensor,
        integrator: GaussianProjectionIntegrals | QuadratureProjectionIntegrals,
    ) -> tuple[torch.Tensor, list[str], torch.Tensor]:
        if positions.numel() == 0:
            return positions, symbols, amplitudes

        cell_x = float(self.cell[0, 0].item())
        cell_y = float(self.cell[1, 1].item())
        cell_z = float(self.cell[2, 2].item())
        if cell_x <= 0.0 or cell_y <= 0.0 or cell_z <= 0.0:
            return positions, symbols, amplitudes

        margin = max(float(integrator.cutoff(symbol)) for symbol in symbols)
        nx = int(torch.ceil(torch.tensor(margin / cell_x, dtype=self.dtype)).item())
        ny = int(torch.ceil(torch.tensor(margin / cell_y, dtype=self.dtype)).item())
        nz = int(torch.ceil(torch.tensor(margin / cell_z, dtype=self.dtype)).item())

        shifts = []
        for ix in range(-nx, nx + 1):
            for iy in range(-ny, ny + 1):
                for iz in range(-nz, nz + 1):
                    shifts.append((ix * cell_x, iy * cell_y, iz * cell_z))

        tiled_positions = []
        tiled_symbols: list[str] = []
        tiled_amplitudes = []
        for position, symbol, amplitude in zip(positions, symbols, amplitudes):
            for shift_x, shift_y, shift_z in shifts:
                shifted = position.clone()
                shifted[0] = shifted[0] + shift_x
                shifted[1] = shifted[1] + shift_y
                shifted[2] = shifted[2] + shift_z
                tiled_positions.append(shifted)
                tiled_symbols.append(symbol)
                tiled_amplitudes.append(amplitude)

        return (
            torch.stack(tiled_positions, dim=0),
            tiled_symbols,
            torch.stack(tiled_amplitudes, dim=0),
        )

    def forward(self) -> torch.Tensor:
        slices = []
        for a, b in self.limits:
            if self.projection == "infinite":
                slices.append(self._render_infinite_slice(self._slice_mask(a, b)))
            elif self.projection == "finite":
                slices.append(self._render_finite_slice(a, b))
            else:
                raise ValueError(f"Unknown projection mode {self.projection}")
        return torch.stack(slices, dim=0)
