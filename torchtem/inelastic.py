from __future__ import annotations

from dataclasses import dataclass

import math
import numpy as np
import torch
from torch import nn

from torchtem.complex_support import abs2
from torchtem.fft_support import fft2, fft2_convolve
from torchtem.physics import ensure_tuple2, real_space_mesh
from torchtem.scan import fft_shift_wave


def draw_scattering_depths(
    num_depths: int,
    num_samples: int,
    mean_free_path_A: float,
    max_depth_A: float,
    *,
    max_batch: int = 10_000,
    max_attempts: int = 50_000_000,
    rng=None,
    device=None,
    dtype: torch.dtype = torch.float64,
) -> torch.Tensor:
    if rng is None:
        rng = np.random.default_rng()

    if num_depths == 0:
        return torch.zeros((num_samples, 0), device=device, dtype=dtype)

    max_num_batches = max_attempts // max_batch
    depths = np.zeros((num_samples, num_depths), dtype=np.float64)
    k = 0
    for _ in range(max_num_batches):
        new_depths = np.cumsum(
            -mean_free_path_A * np.log(rng.random((max_batch, num_depths + 1))), axis=-1
        )
        new_depths = new_depths[
            (new_depths[:, -1] > max_depth_A) * (new_depths[:, -2] < max_depth_A)
        ]
        new_k = min(num_samples, k + len(new_depths))
        depths[k:new_k] = new_depths[: new_k - k, :num_depths]
        k = new_k
        if k == num_samples:
            break

    if k != num_samples:
        raise ValueError(
            f"requested scattering events did not occur in {max_attempts} attempts"
        )

    return torch.as_tensor(depths, device=device, dtype=dtype)


def draw_radial_scattering_angle(
    critical_angle_mrad: float,
    characteristic_angle_mrad: float,
    num_samples: int,
    num_depths: int,
    *,
    rng=None,
    device=None,
    dtype: torch.dtype = torch.float64,
) -> torch.Tensor:
    if rng is None:
        rng = np.random.default_rng()

    u = rng.random((num_samples, num_depths))
    radial_scattering_angles = np.sqrt(
        characteristic_angle_mrad**2
        * (
            (critical_angle_mrad**2 + characteristic_angle_mrad**2)
            / characteristic_angle_mrad**2
        )
        ** u
        - characteristic_angle_mrad**2
    )
    return torch.as_tensor(radial_scattering_angles, device=device, dtype=dtype)


def draw_azimuthal_angle(
    num_samples: int,
    num_depths: int,
    *,
    rng=None,
    device=None,
    dtype: torch.dtype = torch.float64,
) -> torch.Tensor:
    if rng is None:
        rng = np.random.default_rng()
    azimuthal_angles = 2.0 * math.pi * rng.random((num_samples, num_depths))
    return torch.as_tensor(azimuthal_angles, device=device, dtype=dtype)


def excitations_weights(
    n: int,
    thickness_A: float | torch.Tensor,
    mean_free_path_A: float | torch.Tensor,
) -> torch.Tensor:
    thickness = torch.as_tensor(thickness_A, dtype=torch.float64)
    mean_free_path = torch.as_tensor(mean_free_path_A, dtype=torch.float64, device=thickness.device)
    ratio = thickness / mean_free_path
    return ratio.pow(int(n)) * torch.exp(-ratio) / float(math.factorial(int(n)))


class TransitionPotential(nn.Module):
    """Differentiable localized transition-potential layer."""

    def __init__(
        self,
        *,
        gpts: tuple[int, int],
        sampling: tuple[float, float] | float,
        positions_A: torch.Tensor,
        amplitudes: torch.Tensor,
        widths_A: torch.Tensor,
        phase_shifts_rad: torch.Tensor | None = None,
        dtype: torch.dtype = torch.float64,
        device=None,
    ) -> None:
        super().__init__()
        self.gpts = gpts
        self.sampling = ensure_tuple2(sampling)
        self.dtype = dtype
        self.device_hint = device
        self.positions_A = nn.Parameter(positions_A.to(device=device, dtype=dtype))
        self.amplitudes = nn.Parameter(amplitudes.to(device=device, dtype=dtype))
        self.log_widths_A = nn.Parameter(widths_A.to(device=device, dtype=dtype).log())
        if phase_shifts_rad is None:
            phase_shifts_rad = torch.zeros_like(amplitudes, dtype=dtype)
        self.phase_shifts_rad = nn.Parameter(phase_shifts_rad.to(device=device, dtype=dtype))

    @property
    def widths_A(self) -> torch.Tensor:
        return self.log_widths_A.exp()

    def kernel(self) -> torch.Tensor:
        yy, xx = real_space_mesh(
            self.gpts, self.sampling, device=self.positions_A.device, dtype=self.dtype
        )
        dy = yy.unsqueeze(0) - self.positions_A[:, 0].unsqueeze(-1).unsqueeze(-1)
        dx = xx.unsqueeze(0) - self.positions_A[:, 1].unsqueeze(-1).unsqueeze(-1)
        sigma2 = self.widths_A.square().unsqueeze(-1).unsqueeze(-1)
        envelope = torch.exp(-(dx.square() + dy.square()) / (2.0 * sigma2))
        phase = torch.exp(1.0j * self.phase_shifts_rad.to(torch.complex128)).unsqueeze(-1).unsqueeze(-1)
        amp = self.amplitudes.to(torch.complex128).unsqueeze(-1).unsqueeze(-1)
        return amp * phase * envelope

    def local_potential(self, space: str = "real") -> torch.Tensor:
        kernel = self.kernel()
        if space == "real":
            return abs2(kernel)
        if space == "reciprocal":
            return abs2(fft2(kernel))
        raise ValueError("space must be 'real' or 'reciprocal'")

    def integrated_intensities(self) -> torch.Tensor:
        return self.local_potential(space="real").sum(dim=(-2, -1)) * (
            self.sampling[0] * self.sampling[1]
        )

    def filter_by_intensity(self, threshold: float) -> "TransitionPotential":
        intensities = self.integrated_intensities()
        order = torch.argsort(intensities, descending=True)
        ordered_intensities = intensities[order]
        cumulative = torch.cumsum(ordered_intensities / ordered_intensities.sum(), dim=0)
        n = (
            int(
                torch.searchsorted(
                    cumulative,
                    torch.as_tensor(threshold, dtype=cumulative.dtype, device=cumulative.device),
                ).item()
            )
            + 1
        )
        included = order[:n]
        if included.numel() == 0:
            raise RuntimeError("No transition channels remain after filtering")

        return self.__class__(
            gpts=self.gpts,
            sampling=self.sampling,
            positions_A=self.positions_A.detach().clone(),
            amplitudes=self.amplitudes.detach()[included].clone(),
            widths_A=self.widths_A.detach()[included].clone(),
            phase_shifts_rad=self.phase_shifts_rad.detach()[included].clone(),
            dtype=self.dtype,
            device=self.positions_A.device,
        )

    def absolute_threshold(self, wave: torch.Tensor, threshold: float = 1.0) -> torch.Tensor:
        if threshold >= 1.0:
            return torch.as_tensor(0.0, dtype=self.dtype, device=self.positions_A.device)

        local_potential = self.local_potential(space="real").sum(dim=0)
        array = abs2(wave.to(torch.complex128))
        overlap = fft2_convolve(
            local_potential.to(torch.complex128),
            fft2(array.to(torch.complex128)),
        ).real
        overlap = torch.sort(overlap.reshape(-1), descending=True).values
        cumulative = torch.cumsum(overlap, dim=0) / overlap.sum()
        index = torch.searchsorted(
            cumulative,
            torch.as_tensor(threshold, dtype=cumulative.dtype, device=cumulative.device),
            right=False,
        ) - 1
        index = torch.clamp(index, min=0)
        return overlap[index]

    def forward(self, wave: torch.Tensor) -> torch.Tensor:
        kernels = self.kernel()
        return kernels * wave.to(torch.complex128)


@dataclass
class PlasmonEventSpec:
    radial_angles_mrad: torch.Tensor
    azimuthal_angles_rad: torch.Tensor
    weights: torch.Tensor
    depths_A: torch.Tensor | None = None
    excitation_counts: torch.Tensor | None = None
    ensemble_mean: bool = False


class PlasmonScatteringEvents(nn.Module):
    """Weighted ensemble of plasmon scattering events via beam shifts."""

    def __init__(
        self,
        spec: PlasmonEventSpec,
        *,
        sampling: tuple[float, float],
        device=None,
        dtype: torch.dtype = torch.float64,
    ) -> None:
        super().__init__()
        self.sampling = sampling
        self.dtype = dtype
        self.device_hint = device
        self._ensemble_mean = bool(spec.ensemble_mean)
        if spec.depths_A is None:
            depths = torch.zeros((spec.weights.shape[0], 0), device=device, dtype=dtype)
            counts = torch.zeros((spec.weights.shape[0],), device=device, dtype=torch.int64)
        else:
            depths = spec.depths_A.to(device=device, dtype=dtype)
            if spec.excitation_counts is None:
                raise ValueError("excitation_counts must be provided when depths_A is provided")
            counts = spec.excitation_counts.to(device=device, dtype=torch.int64)
        self.register_buffer("depths_A", depths)
        self.register_buffer("excitation_counts", counts)
        self.register_buffer("radial_angles_mrad", spec.radial_angles_mrad.to(device=device, dtype=dtype))
        self.register_buffer("azimuthal_angles_rad", spec.azimuthal_angles_rad.to(device=device, dtype=dtype))
        weights = spec.weights.to(device=device, dtype=dtype)
        self.register_buffer("weights", weights)

    @property
    def ensemble_mean(self) -> bool:
        return self._ensemble_mean

    @property
    def normalized_weights(self) -> torch.Tensor:
        return self.weights / self.weights.sum()

    @property
    def num_events(self) -> int:
        return int(self.weights.shape[0])

    @property
    def num_excitations(self) -> tuple[int, ...]:
        return tuple(int(n) for n in self.excitation_counts.tolist())

    @property
    def max_excitations(self) -> int:
        return max(self.num_excitations, default=0)

    def shifts_A(self) -> torch.Tensor:
        radial = self.radial_angles_mrad / 1e3
        dy = radial * torch.sin(self.azimuthal_angles_rad)
        dx = radial * torch.cos(self.azimuthal_angles_rad)
        return torch.stack((dy, dx), dim=-1)

    def get_scattering_event_depths(self, num_excitations: int = 1) -> dict[int, torch.Tensor]:
        event_depths: dict[int, list[torch.Tensor]] = {}
        if self.depths_A.numel() == 0:
            return {}
        for depths, n in zip(self.depths_A, self.excitation_counts.tolist()):
            n = int(n)
            if n >= num_excitations:
                event_depths.setdefault(n, []).append(depths[num_excitations - 1])
        return {k: torch.stack(v) for k, v in event_depths.items()}

    def forward(self, wave: torch.Tensor) -> torch.Tensor:
        return fft_shift_wave(wave, self.shifts_A(), self.sampling)

    def weighted_intensity(self, wave: torch.Tensor) -> torch.Tensor:
        shifted = self.forward(wave)
        intensity = shifted.abs().square()
        weights = self.normalized_weights.reshape((self.weights.shape[0],) + (1,) * (intensity.ndim - 1))
        return torch.sum(weights * intensity, dim=0)


class MonteCarloPlasmons:
    """Torch-native analogue of the source-side plasmon event generator."""

    def __init__(
        self,
        *,
        mean_free_path_A: float,
        excitation_energy_eV: float,
        critical_angle_mrad: float,
        num_excitations: int | tuple[int, ...],
        num_samples: int,
        ensemble_mean: bool = False,
        seed: int | None = None,
        device=None,
        dtype: torch.dtype = torch.float64,
    ) -> None:
        if isinstance(num_excitations, int):
            num_excitations = tuple(range(int(num_excitations) + 1))
        self.mean_free_path_A = float(mean_free_path_A)
        self.excitation_energy_eV = float(excitation_energy_eV)
        self.critical_angle_mrad = float(critical_angle_mrad)
        self.num_excitations = tuple(int(n) for n in num_excitations)
        self.num_samples = int(num_samples)
        self.ensemble_mean = bool(ensemble_mean)
        self.seed = seed
        self.device = device
        self.dtype = dtype

    def __len__(self) -> int:
        return self.num_samples

    def characteristic_angle(self, energy_eV: float) -> float:
        return self.excitation_energy_eV / (2.0 * float(energy_eV)) * 1.0e3

    def draw_spec(
        self,
        *,
        energy_eV: float,
        thickness_A: float,
    ) -> PlasmonEventSpec:
        rng = np.random.default_rng(self.seed)

        depths = []
        radial_angles = []
        azimuthal_angles = []
        weights = []

        for n in self.num_excitations:
            num_samples = 1 if n == 0 else self.num_samples

            depths.append(
                draw_scattering_depths(
                    n,
                    num_samples,
                    self.mean_free_path_A,
                    thickness_A,
                    rng=rng,
                    device=self.device,
                    dtype=self.dtype,
                )
            )
            radial_angles.append(
                draw_radial_scattering_angle(
                    self.critical_angle_mrad,
                    self.characteristic_angle(energy_eV),
                    num_samples,
                    n,
                    rng=rng,
                    device=self.device,
                    dtype=self.dtype,
                )
            )
            azimuthal_angles.append(
                draw_azimuthal_angle(
                    num_samples,
                    n,
                    rng=rng,
                    device=self.device,
                    dtype=self.dtype,
                )
            )
            weights.append(
                torch.full(
                    (num_samples,),
                    float(excitations_weights(n, thickness_A, self.mean_free_path_A).item()),
                    device=self.device,
                    dtype=self.dtype,
                )
            )

        flat_radial = []
        flat_azimuthal = []
        flat_depths = []
        excitation_counts = []
        max_n = max(self.num_excitations, default=0)
        for depth, radial, azimuthal in zip(depths, radial_angles, azimuthal_angles):
            padded_depth = torch.zeros((depth.shape[0], max_n), device=depth.device, dtype=depth.dtype)
            if depth.shape[1] > 0:
                padded_depth[:, : depth.shape[1]] = depth
            flat_depths.append(padded_depth)
            excitation_counts.append(
                torch.full((depth.shape[0],), depth.shape[1], device=depth.device, dtype=torch.int64)
            )
            if radial.shape[1] == 0:
                flat_radial.append(torch.zeros((radial.shape[0],), device=radial.device, dtype=radial.dtype))
                flat_azimuthal.append(
                    torch.zeros((azimuthal.shape[0],), device=azimuthal.device, dtype=azimuthal.dtype)
                )
            else:
                flat_radial.append(radial.sum(dim=1))
                flat_azimuthal.append(azimuthal.sum(dim=1))

        return PlasmonEventSpec(
            depths_A=torch.cat(flat_depths, dim=0),
            excitation_counts=torch.cat(excitation_counts, dim=0),
            radial_angles_mrad=torch.cat(flat_radial, dim=0),
            azimuthal_angles_rad=torch.cat(flat_azimuthal, dim=0),
            weights=torch.cat(weights, dim=0),
            ensemble_mean=self.ensemble_mean,
        )

    def draw_events(
        self,
        *,
        energy_eV: float,
        thickness_A: float,
        sampling: tuple[float, float],
    ) -> PlasmonScatteringEvents:
        return PlasmonScatteringEvents(
            self.draw_spec(energy_eV=energy_eV, thickness_A=thickness_A),
            sampling=sampling,
            device=self.device,
            dtype=self.dtype,
        )
