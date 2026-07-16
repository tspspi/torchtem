from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Sequence

import numpy as np
import torch

from torchtem.parametrizations import Parametrization, validate_parametrization
from torchtem.physics import ensure_tuple2, real_space_mesh, reciprocal_mesh, spatial_frequencies


class FieldIntegrator(ABC):
    """Base class for tensor-native projection integrators."""

    def __init__(self, *, periodic: bool, finite: bool) -> None:
        self.periodic = bool(periodic)
        self.finite = bool(finite)

    @abstractmethod
    def integrate_on_grid(
        self,
        *,
        positions_A: torch.Tensor,
        symbols: Sequence[str],
        a: float | torch.Tensor,
        b: float | torch.Tensor,
        gpts: tuple[int, int],
        sampling: tuple[float, float] | float,
        device=None,
        dtype: torch.dtype = torch.float64,
    ) -> torch.Tensor:
        raise NotImplementedError

    @abstractmethod
    def cutoff(self, symbol: str) -> float:
        raise NotImplementedError


def sinc(
    gpts: tuple[int, int],
    sampling: tuple[float, float] | float,
    *,
    device=None,
    dtype: torch.dtype = torch.float64,
) -> torch.Tensor:
    sampling = ensure_tuple2(sampling)
    ky, kx = spatial_frequencies(gpts, sampling, device=device, dtype=dtype)
    k = torch.sqrt((ky[:, None] * sampling[0]).square() + (kx[None] * sampling[1]).square())
    dk2 = sampling[0] * sampling[1]
    out = torch.empty(gpts, device=device, dtype=dtype)
    mask = k == 0
    out[mask] = dk2
    out[~mask] = torch.sin(k[~mask]) / k[~mask] * dk2
    return out


def optimize_cutoff(
    func,
    tolerance: float,
    *,
    a: float,
    b: float,
    max_iter: int = 80,
) -> float:
    """Monotone bisection cutoff estimate for radial functions."""
    lo = float(a)
    hi = float(b)
    for _ in range(max_iter):
        mid = 0.5 * (lo + hi)
        value = func(torch.tensor(mid, dtype=torch.float64))
        scalar = abs(float(torch.as_tensor(value).item()))
        if scalar > tolerance:
            lo = mid
        else:
            hi = mid
    return hi


def gaussian_projected_scattering_factors(
    symbol: str,
    gpts: tuple[int, int],
    sampling: tuple[float, float] | float,
    *,
    parametrization: str | Parametrization = "peng",
    device=None,
    dtype: torch.dtype = torch.float64,
) -> torch.Tensor:
    parametrization = validate_parametrization(parametrization)
    sampling = ensure_tuple2(sampling)
    ky, kx = reciprocal_mesh(gpts, sampling, device=device, dtype=dtype)
    k2 = ky.square() + kx.square()
    parameters = parametrization.scaled_parameters(symbol, "projected_scattering_factor").to(
        device=device, dtype=dtype
    )
    return parameters[0, :, None, None] * torch.exp(-parameters[1, :, None, None] * k2[None])


def gaussian_projection_weights(
    symbol: str,
    a: torch.Tensor,
    b: torch.Tensor,
    *,
    parametrization: str | Parametrization = "peng",
    device=None,
    dtype: torch.dtype = torch.float64,
) -> torch.Tensor:
    parametrization = validate_parametrization(parametrization)
    parameters = parametrization.scaled_parameters(symbol, "projected_scattering_factor").to(
        device=device, dtype=dtype
    )
    scales = torch.pi / torch.sqrt(parameters[1])[:, None]
    return torch.abs(torch.erf(scales * b[None]) - torch.erf(scales * a[None])) / 2.0


def correction_projected_scattering_factors(
    symbol: str,
    gpts: tuple[int, int],
    sampling: tuple[float, float] | float,
    *,
    short_range: str | Parametrization = "lobato",
    long_range: str | Parametrization = "peng",
    device=None,
    dtype: torch.dtype = torch.float64,
) -> torch.Tensor:
    short_range = validate_parametrization(short_range)
    long_range = validate_parametrization(long_range)
    ky, kx = reciprocal_mesh(gpts, ensure_tuple2(sampling), device=device, dtype=dtype)
    k2 = ky.square() + kx.square()
    short = short_range.get_function("projected_scattering_factor", symbol)(k2)
    long = long_range.get_function("projected_scattering_factor", symbol)(k2)
    return short - long


def superpose_deltas(
    positions: torch.Tensor,
    gpts: tuple[int, int],
    *,
    weights: torch.Tensor | None = None,
    sampling: tuple[float, float] | float = (1.0, 1.0),
    device=None,
    dtype: torch.dtype = torch.float64,
) -> torch.Tensor:
    """Fourier-native periodic delta superposition, differentiable in positions."""
    sampling = ensure_tuple2(sampling)
    device = device or positions.device
    if positions.numel() == 0:
        return torch.zeros(gpts, dtype=torch.complex128, device=device)
    ky, kx = reciprocal_mesh(gpts, sampling, device=device, dtype=dtype)
    phase = -2.0j * torch.pi * (
        ky.unsqueeze(0) * positions[:, 0].unsqueeze(-1).unsqueeze(-1)
        + kx.unsqueeze(0) * positions[:, 1].unsqueeze(-1).unsqueeze(-1)
    )
    factors = torch.exp(phase.to(torch.complex128))
    if weights is not None:
        factors = factors * weights.to(torch.complex128).unsqueeze(-1).unsqueeze(-1)
    return factors.sum(dim=0)


class ScatteringFactorProjectionIntegrals(FieldIntegrator):
    """Infinite projected-potential integrator using projected scattering factors."""

    def __init__(self, parametrization: str | Parametrization = "lobato") -> None:
        super().__init__(periodic=True, finite=False)
        self.parametrization = validate_parametrization(parametrization)

    def cutoff(self, symbol: str) -> float:
        return float("inf")

    def integrate_on_grid(
        self,
        *,
        positions_A: torch.Tensor,
        symbols: Sequence[str],
        a: float | torch.Tensor,
        b: float | torch.Tensor,
        gpts: tuple[int, int],
        sampling: tuple[float, float] | float,
        device=None,
        dtype: torch.dtype = torch.float64,
    ) -> torch.Tensor:
        del a, b
        sampling = ensure_tuple2(sampling)
        device = device or positions_A.device
        sinc_term = sinc(gpts, sampling, device=device, dtype=dtype).to(torch.complex128)
        out = torch.zeros(gpts, dtype=torch.complex128, device=device)
        unique_symbols = sorted(set(symbols))
        for symbol in unique_symbols:
            mask = torch.tensor([s == symbol for s in symbols], device=positions_A.device)
            positions_xy = positions_A[mask][:, :2]
            if positions_xy.numel() == 0:
                continue
            ky, kx = reciprocal_mesh(gpts, sampling, device=device, dtype=dtype)
            k2 = ky.square() + kx.square()
            scattering = self.parametrization.get_function("projected_scattering_factor", symbol)(k2)
            deltas = superpose_deltas(
                positions_xy,
                gpts,
                sampling=sampling,
                device=device,
                dtype=dtype,
            )
            out = out + deltas * scattering.to(torch.complex128) / sinc_term
        return torch.fft.ifft2(out).real


class GaussianProjectionIntegrals(FieldIntegrator):
    """Finite projected-potential integrator using Gaussian long range plus correction."""

    def __init__(
        self,
        *,
        parametrization: str | Parametrization = "lobato",
        gaussian_parametrization: str | Parametrization = "peng",
        cutoff_tolerance: float = 1e-3,
    ) -> None:
        super().__init__(periodic=True, finite=True)
        self.parametrization = validate_parametrization(parametrization)
        self.gaussian_parametrization = validate_parametrization(gaussian_parametrization)
        self.cutoff_tolerance = float(cutoff_tolerance)

    def cutoff(self, symbol: str) -> float:
        potential = self.gaussian_parametrization.get_function("projected_scattering_factor", symbol)
        return optimize_cutoff(potential, self.cutoff_tolerance, a=1e-3, b=1e3)

    def integrate_on_grid(
        self,
        *,
        positions_A: torch.Tensor,
        symbols: Sequence[str],
        a: float | torch.Tensor,
        b: float | torch.Tensor,
        gpts: tuple[int, int],
        sampling: tuple[float, float] | float,
        device=None,
        dtype: torch.dtype = torch.float64,
    ) -> torch.Tensor:
        sampling = ensure_tuple2(sampling)
        device = device or positions_A.device
        a_tensor = torch.as_tensor(a, device=device, dtype=dtype)
        b_tensor = torch.as_tensor(b, device=device, dtype=dtype)
        sinc_term = sinc(gpts, sampling, device=device, dtype=dtype).to(torch.complex128)
        out = torch.zeros(gpts, dtype=torch.complex128, device=device)
        unique_symbols = sorted(set(symbols))

        for symbol in unique_symbols:
            mask = torch.tensor([s == symbol for s in symbols], device=positions_A.device)
            positions = positions_A[mask]
            if positions.numel() == 0:
                continue

            shifted_a = a_tensor - positions[:, 2]
            shifted_b = b_tensor - positions[:, 2]
            weights = gaussian_projection_weights(
                symbol,
                shifted_a,
                shifted_b,
                parametrization=self.gaussian_parametrization,
                device=device,
                dtype=dtype,
            )
            gaussians = gaussian_projected_scattering_factors(
                symbol,
                gpts,
                sampling,
                parametrization=self.gaussian_parametrization,
                device=device,
                dtype=dtype,
            )
            deltas = superpose_deltas(
                positions[:, :2],
                gpts,
                sampling=sampling,
                device=device,
                dtype=dtype,
            )
            weighted_scattering = (weights[:, :, None, None] * gaussians[:, None]).sum(dim=0).sum(dim=0)
            gaussian_term = deltas * weighted_scattering.to(torch.complex128)

            in_slice = (positions[:, 2] >= a_tensor) & (positions[:, 2] < b_tensor)
            correction_positions = positions[in_slice][:, :2]
            if correction_positions.numel() == 0:
                correction_term = torch.zeros_like(out)
            else:
                correction = correction_projected_scattering_factors(
                    symbol,
                    gpts,
                    sampling,
                    short_range=self.parametrization,
                    long_range=self.gaussian_parametrization,
                    device=device,
                    dtype=dtype,
                )
                correction_term = superpose_deltas(
                    correction_positions,
                    gpts,
                    sampling=sampling,
                    device=device,
                    dtype=dtype,
                ) * correction.to(torch.complex128)

            out = out + (gaussian_term + correction_term) / sinc_term
        return torch.fft.ifft2(out).real


class ProjectionIntegralTable:
    """Tabulated projection antiderivatives on a radial grid."""

    def __init__(
        self,
        radial_gpts: torch.Tensor,
        limits: torch.Tensor,
        values: torch.Tensor,
    ) -> None:
        if values.shape != (limits.shape[0], radial_gpts.shape[0]):
            raise ValueError("values shape must be (n_limits, n_radial)")
        self.radial_gpts = radial_gpts
        self.limits = limits
        self.values = values

    def _interpolate(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.as_tensor(x, device=self.limits.device, dtype=self.limits.dtype)
        idx = torch.searchsorted(self.limits, x, right=True) - 1
        idx = idx.clamp(0, self.limits.shape[0] - 2)
        x0 = self.limits[idx]
        x1 = self.limits[idx + 1]
        w = (x - x0) / (x1 - x0)
        return self.values[idx] + w.unsqueeze(-1) * (self.values[idx + 1] - self.values[idx])

    def integrate(self, a: torch.Tensor | float, b: torch.Tensor | float) -> torch.Tensor:
        a_tensor = torch.atleast_1d(torch.as_tensor(a, device=self.limits.device, dtype=self.limits.dtype))
        b_tensor = torch.atleast_1d(torch.as_tensor(b, device=self.limits.device, dtype=self.limits.dtype))
        return self._interpolate(b_tensor) - self._interpolate(a_tensor)


class QuadratureProjectionIntegrals(FieldIntegrator):
    """Finite projected-potential integrator using quadrature-tabulated radial slices."""

    def __init__(
        self,
        parametrization: str | Parametrization = "lobato",
        *,
        cutoff_tolerance: float = 1e-4,
        inner_cutoff_factor: float = 2.0,
        taper: float = 0.85,
        integration_step: float = 0.02,
        quad_order: int = 8,
    ) -> None:
        super().__init__(periodic=False, finite=True)
        self.parametrization = validate_parametrization(parametrization)
        self.cutoff_tolerance = float(cutoff_tolerance)
        self.inner_cutoff_factor = float(inner_cutoff_factor)
        self.taper = float(taper)
        self.integration_step = float(integration_step)
        self.quad_order = int(quad_order)
        self._tables: dict[tuple[str, tuple[float, float], torch.dtype, str], ProjectionIntegralTable] = {}
        nodes, weights = np.polynomial.legendre.leggauss(self.quad_order)
        self._quad_nodes_np = nodes
        self._quad_weights_np = weights

    def cutoff(self, symbol: str) -> float:
        potential = self.parametrization.get_function("potential", symbol)
        return optimize_cutoff(potential, self.cutoff_tolerance, a=1e-3, b=1e3)

    @staticmethod
    def _radial_gpts(
        inner_cutoff: float,
        cutoff: float,
        *,
        device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        num_points = int(torch.ceil(torch.tensor(cutoff / inner_cutoff, dtype=dtype)).item())
        start = torch.log(torch.tensor(inner_cutoff, device=device, dtype=dtype))
        stop = torch.log(torch.tensor(cutoff, device=device, dtype=dtype))
        return torch.exp(torch.linspace(start, stop, num_points, device=device, dtype=dtype))

    def _integral_limits(self, cutoff: float, *, device, dtype: torch.dtype) -> torch.Tensor:
        num = int(torch.ceil(torch.tensor(cutoff / self.integration_step, dtype=dtype)).item())
        limits = torch.linspace(-cutoff, 0.0, num, device=device, dtype=dtype)
        return torch.cat((limits, -limits.flip(0)[1:]), dim=0)

    def _taper_values(self, radial_gpts: torch.Tensor, cutoff: float) -> torch.Tensor:
        taper_start = self.taper * cutoff
        taper_values = torch.ones_like(radial_gpts)
        taper_mask = radial_gpts > taper_start
        taper_values = torch.where(
            taper_mask,
            (torch.cos(torch.pi * (radial_gpts - taper_start) / (cutoff - taper_start)) + 1.0) / 2.0,
            taper_values,
        )
        return torch.where(radial_gpts <= cutoff, taper_values, torch.zeros_like(taper_values))

    def _fixed_quad(
        self,
        func,
        a: float,
        b: float,
        *,
        device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        nodes = torch.as_tensor(self._quad_nodes_np, device=device, dtype=dtype)
        weights = torch.as_tensor(self._quad_weights_np, device=device, dtype=dtype)
        x = 0.5 * (b - a) * nodes + 0.5 * (b + a)
        fx = func(x)
        return 0.5 * (b - a) * (fx * weights[None]).sum(dim=1)

    def _calculate_integral_table(
        self,
        symbol: str,
        sampling: tuple[float, float],
        *,
        device,
        dtype: torch.dtype,
    ) -> ProjectionIntegralTable:
        potential = self.parametrization.get_function("potential", symbol)
        cutoff = self.cutoff(symbol)
        inner_limit = min(sampling) / self.inner_cutoff_factor
        radial_gpts = self._radial_gpts(inner_limit, cutoff, device=device, dtype=dtype)
        limits = self._integral_limits(cutoff, device=device, dtype=dtype)

        def project_along_z(z: torch.Tensor) -> torch.Tensor:
            return potential(torch.sqrt(radial_gpts[:, None].square() + z[None].square()))

        table = torch.zeros((limits.shape[0] - 1, radial_gpts.shape[0]), device=device, dtype=dtype)
        table[0] = self._fixed_quad(
            project_along_z,
            float((-limits[0] * 2.0).item()),
            float(limits[0].item()),
            device=device,
            dtype=dtype,
        )

        for j, (a_value, b_value) in enumerate(zip(limits[1:-1], limits[2:])):
            table[j + 1] = table[j] + self._fixed_quad(
                project_along_z,
                float(a_value.item()),
                float(b_value.item()),
                device=device,
                dtype=dtype,
            )

        table = table * self._taper_values(radial_gpts, cutoff).unsqueeze(0)
        return ProjectionIntegralTable(radial_gpts, limits[1:], table)

    def get_integral_table(
        self,
        symbol: str,
        sampling: tuple[float, float],
        *,
        device,
        dtype: torch.dtype,
    ) -> ProjectionIntegralTable:
        key = (symbol, sampling, dtype, str(device))
        if key not in self._tables:
            self._tables[key] = self._calculate_integral_table(
                symbol,
                sampling,
                device=device,
                dtype=dtype,
            )
        return self._tables[key]

    @staticmethod
    def _interpolate_radial_table(
        radial_values: torch.Tensor,
        radial_gpts: torch.Tensor,
        r: torch.Tensor,
    ) -> torch.Tensor:
        cutoff = radial_gpts[-1]
        inner = radial_gpts[0]
        idx = torch.searchsorted(radial_gpts, r, right=True) - 1
        idx = idx.clamp(0, radial_gpts.shape[0] - 2)
        r0 = radial_gpts[idx]
        r1 = radial_gpts[idx + 1]
        flat_idx = idx.reshape(radial_values.shape[0], -1)
        v0 = radial_values.gather(1, flat_idx).reshape(idx.shape)
        v1 = radial_values.gather(1, (flat_idx + 1)).reshape(idx.shape)
        w = (r - r0) / (r1 - r0)
        interpolated = v0 + w * (v1 - v0)
        interpolated = torch.where(
            r < inner,
            radial_values[:, 0].view(-1, 1, 1).expand_as(interpolated),
            interpolated,
        )
        return torch.where(r <= cutoff, interpolated, torch.zeros_like(interpolated))

    def integrate_on_grid(
        self,
        *,
        positions_A: torch.Tensor,
        symbols: Sequence[str],
        a: float | torch.Tensor,
        b: float | torch.Tensor,
        gpts: tuple[int, int],
        sampling: tuple[float, float] | float,
        device=None,
        dtype: torch.dtype = torch.float64,
    ) -> torch.Tensor:
        sampling = ensure_tuple2(sampling)
        device = device or positions_A.device
        out = torch.zeros(gpts, dtype=dtype, device=device)
        yy, xx = real_space_mesh(gpts, sampling, device=device, dtype=dtype)
        a_tensor = torch.as_tensor(a, device=device, dtype=dtype)
        b_tensor = torch.as_tensor(b, device=device, dtype=dtype)

        unique_symbols = sorted(set(symbols))
        for symbol in unique_symbols:
            mask = torch.tensor([s == symbol for s in symbols], device=positions_A.device)
            positions = positions_A[mask].to(device=device, dtype=dtype)
            if positions.numel() == 0:
                continue
            table = self.get_integral_table(symbol, sampling, device=device, dtype=dtype)
            shifted_a = a_tensor - positions[:, 2]
            shifted_b = b_tensor - positions[:, 2]
            radial_values = table.integrate(shifted_a, shifted_b)
            dy = yy.unsqueeze(0) - positions[:, 0].unsqueeze(-1).unsqueeze(-1)
            dx = xx.unsqueeze(0) - positions[:, 1].unsqueeze(-1).unsqueeze(-1)
            r = torch.sqrt(dy.square() + dx.square() + 1e-12)
            out = out + self._interpolate_radial_table(radial_values, table.radial_gpts, r).sum(dim=0)

        return out
