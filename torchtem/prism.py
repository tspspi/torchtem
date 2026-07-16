from __future__ import annotations

import math

import torch
from torch import nn

from torchtem.natural_neighbors import pairwise_natural_neighbor_weights
from torchtem.optics import AberrationCoefficients, CTF
from torchtem.physics import spatial_frequencies
from torchtem.physics import energy2wavelength


def prism_wave_vectors(
    cutoff_mrad: float,
    extent_A: tuple[float, float],
    energy_eV: float,
    interpolation: int | tuple[int, int] = 1,
    device=None,
    dtype: torch.dtype = torch.float64,
) -> torch.Tensor:
    """Simple reciprocal-space wave-vector selection for a PRISM basis."""
    wavelength = energy2wavelength(energy_eV, device=device, dtype=dtype)
    if isinstance(interpolation, int):
        interpolation = (interpolation, interpolation)
    interpolation_y = max(1, int(interpolation[0]))
    interpolation_x = max(1, int(interpolation[1]))

    n_term = cutoff_mrad / 1.0e3 / (wavelength / extent_A[0] * interpolation_y)
    m_term = cutoff_mrad / 1.0e3 / (wavelength / extent_A[1] * interpolation_x)
    n_max = int(torch.ceil(torch.as_tensor(n_term, device=device, dtype=dtype)).item())
    m_max = int(torch.ceil(torch.as_tensor(m_term, device=device, dtype=dtype)).item())

    n = torch.arange(-n_max, n_max + 1, dtype=dtype, device=device)
    m = torch.arange(-m_max, m_max + 1, dtype=dtype, device=device)
    ky = n / extent_A[0] * interpolation_y
    kx = m / extent_A[1] * interpolation_x
    KY, KX = torch.meshgrid(ky, kx, indexing="ij")
    mask = KY.square() + KX.square() < (cutoff_mrad / 1.0e3 / wavelength) ** 2
    vectors = torch.stack((KY[mask], KX[mask]), dim=-1)
    if vectors.numel() == 0:
        return torch.zeros((1, 2), device=device, dtype=dtype)
    return vectors


def smatrix_wave_vectors(
    cutoff_mrad: float,
    extent_A: tuple[float, float],
    gpts: tuple[int, int],
    energy_eV: float,
    interpolation: int | tuple[int, int] = 1,
    device=None,
    dtype: torch.dtype = torch.float64,
) -> torch.Tensor:
    """Match abTEM SMatrix parent-wave selection from the discrete soft aperture support."""
    if isinstance(interpolation, int):
        interpolation = (interpolation, interpolation)

    interpolation_y = max(1, int(interpolation[0]))
    interpolation_x = max(1, int(interpolation[1]))
    window_gpts = (
        math.ceil(gpts[0] / interpolation_y),
        math.ceil(gpts[1] / interpolation_x),
    )
    window_extent_A = (
        window_gpts[0] * (extent_A[0] / gpts[0]),
        window_gpts[1] * (extent_A[1] / gpts[1]),
    )
    sampling_A = (
        window_extent_A[0] / window_gpts[0],
        window_extent_A[1] / window_gpts[1],
    )

    ky = torch.fft.fftfreq(
        window_gpts[0],
        d=sampling_A[0],
        device=device,
        dtype=dtype,
    )
    kx = torch.fft.fftfreq(
        window_gpts[1],
        d=sampling_A[1],
        device=device,
        dtype=dtype,
    )
    ky_grid, kx_grid = torch.meshgrid(ky, kx, indexing="ij")

    wavelength = energy2wavelength(energy_eV, device=device, dtype=dtype)
    alpha = torch.sqrt(ky_grid.square() + kx_grid.square()) * wavelength
    phi = torch.atan2(ky_grid, kx_grid)

    angular_sampling_mrad = (
        wavelength / (window_gpts[0] * sampling_A[0]) * 1.0e3,
        wavelength / (window_gpts[1] * sampling_A[1]) * 1.0e3,
    )
    denominator = torch.sqrt(
        (torch.cos(phi) * angular_sampling_mrad[0] * 1.0e-3).square()
        + (torch.sin(phi) * angular_sampling_mrad[1] * 1.0e-3).square()
    )
    denominator[0, 0] = 1.0

    soft_aperture = torch.clamp(
        (cutoff_mrad * 1.0e-3 - alpha) / denominator + 0.5,
        min=0.0,
        max=1.0,
    )
    soft_aperture[0, 0] = 1.0

    indices = torch.nonzero(soft_aperture > 0.0, as_tuple=False)
    n = torch.fft.fftfreq(
        window_gpts[0],
        d=1.0 / window_gpts[0],
        device=device,
        dtype=dtype,
    )[indices[:, 0]]
    m = torch.fft.fftfreq(
        window_gpts[1],
        d=1.0 / window_gpts[1],
        device=device,
        dtype=dtype,
    )[indices[:, 1]]

    return torch.stack(
        (
            n / window_extent_A[0],
            m / window_extent_A[1],
        ),
        dim=-1,
    )


def plane_waves(
    wave_vectors: torch.Tensor,
    extent_A: tuple[float, float],
    gpts: tuple[int, int],
    *,
    reverse: bool = False,
    device=None,
    dtype: torch.dtype = torch.float64,
) -> torch.Tensor:
    y = torch.arange(gpts[0], device=device, dtype=dtype) * (extent_A[0] / gpts[0])
    x = torch.arange(gpts[1], device=device, dtype=dtype) * (extent_A[1] / gpts[1])
    yy, xx = torch.meshgrid(y, x, indexing="ij")
    sign = -1.0 if reverse else 1.0
    phase = 2.0j * torch.pi * sign * (
        wave_vectors[:, 0, None, None] * yy[None] + wave_vectors[:, 1, None, None] * xx[None]
    )
    return torch.exp(phase)


def pairwise_inverse_distance_weights(
    parent_wave_vectors: torch.Tensor,
    wave_vectors: torch.Tensor,
    eps: float = 1e-12,
) -> torch.Tensor:
    diff = parent_wave_vectors[:, None, :] - wave_vectors[None, :, :]
    dist2 = diff.square().sum(dim=-1)
    exact = dist2 <= eps
    weights = 1.0 / torch.clamp(dist2, min=eps)
    if exact.any():
        weights = weights * (~exact)
        weights = weights + exact.to(weights.dtype) * 1e12
    return weights / weights.sum(dim=0, keepdim=True)


def partitioned_prism_wave_vectors(
    cutoff_mrad: float,
    extent_A: tuple[float, float],
    energy_eV: float,
    *,
    num_rings: int,
    num_points_per_ring: int = 6,
    device=None,
    dtype: torch.dtype = torch.float64,
) -> torch.Tensor:
    wavelength = energy2wavelength(energy_eV, device=device, dtype=dtype)
    rings = [torch.tensor([[0.0, 0.0]], device=device, dtype=dtype)]
    if num_rings < 1:
        raise ValueError("num_rings must be at least 1")
    if num_rings == 1:
        return rings[0]

    n = int(num_points_per_ring)
    for r in torch.linspace(cutoff_mrad / (num_rings - 1), cutoff_mrad, num_rings - 1, device=device, dtype=dtype):
        angles = torch.arange(n, device=device, dtype=dtype) * 2.0 * math.pi / n + math.pi / 2.0
        ky = torch.round(r * torch.sin(angles) / 1000.0 / wavelength * extent_A[0]) / extent_A[0]
        kx = torch.round(r * torch.cos(-angles) / 1000.0 / wavelength * extent_A[1]) / extent_A[1]
        n += int(num_points_per_ring)
        rings.append(torch.stack((ky, kx), dim=-1))
    return torch.vstack(rings).to(dtype)


def pairwise_natural_weights(
    parent_wave_vectors: torch.Tensor,
    wave_vectors: torch.Tensor,
) -> torch.Tensor:
    return pairwise_natural_neighbor_weights(parent_wave_vectors, wave_vectors)


def calculate_positions_coefficients(
    positions_A: torch.Tensor,
    wave_vectors: torch.Tensor,
) -> torch.Tensor:
    positions = positions_A.to(torch.float64)
    wave_vectors = wave_vectors.to(torch.float64)
    return torch.exp(
        -2.0j
        * torch.pi
        * (
            positions[..., 0, None] * wave_vectors[:, 0][None]
            + positions[..., 1, None] * wave_vectors[:, 1][None]
        )
    )


def calculate_ctf_coefficients(
    ctf: CTF,
    wave_vectors: torch.Tensor,
    *,
    normalize: bool = True,
) -> torch.Tensor:
    wave_vectors = wave_vectors.to(torch.float64)
    alpha = (
        torch.sqrt(wave_vectors[:, 0].square() + wave_vectors[:, 1].square())
        * ctf.wavelength.to(wave_vectors.device)
    )
    phi = torch.arctan2(wave_vectors[:, 1], wave_vectors[:, 0])
    coefficients = ctf.evaluate(alpha, phi)
    if normalize:
        denominator = torch.sqrt((coefficients.square()).sum(dim=-1, keepdim=True))
        safe_denominator = torch.where(
            denominator.abs()
            < torch.as_tensor(1e-30, dtype=denominator.real.dtype, device=denominator.device),
            torch.ones_like(denominator),
            denominator,
        )
        coefficients = coefficients / safe_denominator
    return coefficients


def beamlet_weights(
    parent_wave_vectors: torch.Tensor,
    wave_vectors: torch.Tensor,
    gpts: tuple[int, int],
    sampling: tuple[float, float],
) -> torch.Tensor:
    ky, kx = spatial_frequencies(
        gpts,
        sampling,
        device=parent_wave_vectors.device,
        dtype=parent_wave_vectors.dtype,
    )
    ky_grid, kx_grid = torch.meshgrid(ky, kx, indexing="ij")
    k = torch.stack((ky_grid.reshape(-1), kx_grid.reshape(-1)), dim=-1)

    indices = []
    for wave_vector in wave_vectors.to(k.dtype):
        matches = torch.all(torch.isclose(wave_vector[None], k, atol=1e-8, rtol=1e-5), dim=1)
        if not torch.any(matches):
            raise ValueError("wave_vectors must lie on the reciprocal-space grid defined by gpts and sampling")
        indices.append(int(torch.argmax(matches.to(torch.int64)).item()))

    weights = torch.zeros(
        (parent_wave_vectors.shape[0],) + tuple(ky_grid.shape),
        dtype=parent_wave_vectors.dtype,
        device=parent_wave_vectors.device,
    )
    point_weights = pairwise_natural_weights(parent_wave_vectors, wave_vectors)
    for i, flat_index in enumerate(indices):
        row, col = divmod(flat_index, ky_grid.shape[1])
        weights[:, row, col] = point_weights[:, i]

    return weights


def beamlet_basis(
    ctf_or_coefficients: torch.Tensor | CTF,
    parent_wave_vectors: torch.Tensor,
    wave_vectors: torch.Tensor,
    extent_or_sampling: tuple[float, float],
    gpts: tuple[int, int],
) -> torch.Tensor:
    if isinstance(ctf_or_coefficients, CTF):
        basis = ctf_or_coefficients.evaluate_on_grid(gpts=gpts, sampling=extent_or_sampling)
        basis = (
            beamlet_weights(
                parent_wave_vectors,
                wave_vectors,
                gpts,
                extent_or_sampling,
            ).to(torch.complex128)
            * basis
            / math.sqrt(len(wave_vectors))
        )
        basis = torch.fft.ifft2(basis, norm="ortho")
        return torch.fft.fftshift(basis, dim=(-2, -1))

    ctf_coefficients = ctf_or_coefficients
    weights = pairwise_inverse_distance_weights(parent_wave_vectors, wave_vectors)
    parent_plane_waves = plane_waves(
        parent_wave_vectors, extent_or_sampling, gpts, device=ctf_coefficients.device
    )
    weighted_ctf = weights * ctf_coefficients[:, None]
    basis = torch.einsum("pq,pyx->qyx", weighted_ctf.to(parent_plane_waves.dtype), parent_plane_waves)
    basis = torch.fft.ifft2(basis, norm="ortho")
    return basis / math.sqrt(wave_vectors.shape[0])


def partitioned_beamlet_basis(
    ctf_coefficients: torch.Tensor,
    parent_wave_vectors: torch.Tensor,
    wave_vectors: torch.Tensor,
    extent_A: tuple[float, float],
    gpts: tuple[int, int],
    *,
    weight_mode: str = "natural",
) -> torch.Tensor:
    if weight_mode == "natural":
        weights = pairwise_natural_weights(parent_wave_vectors, wave_vectors)
    elif weight_mode == "inverse_distance":
        weights = pairwise_inverse_distance_weights(parent_wave_vectors, wave_vectors)
    else:
        raise ValueError(f"Unknown weight_mode {weight_mode}")
    parent_plane_waves = plane_waves(
        parent_wave_vectors, extent_A, gpts, device=ctf_coefficients.device
    )
    weighted_ctf = weights * ctf_coefficients[:, None]
    basis = torch.einsum("pq,pyx->qyx", weighted_ctf.to(parent_plane_waves.dtype), parent_plane_waves)
    basis = torch.fft.ifft2(basis, norm="ortho")
    return torch.fft.fftshift(basis, dim=(-2, -1)) / math.sqrt(wave_vectors.shape[0])


def interpolate_full(
    array: torch.Tensor,
    parent_wave_vectors: torch.Tensor,
    wave_vectors: torch.Tensor,
    extent_A: tuple[float, float],
    gpts: tuple[int, int],
    energy_eV: float,
    defocus_A: float = 0.0,
) -> torch.Tensor:
    interpolated_array = plane_waves(
        wave_vectors,
        extent_A,
        gpts,
        device=array.device,
    )

    weights = pairwise_natural_weights(parent_wave_vectors, wave_vectors)
    alpha = (
        torch.sqrt(wave_vectors[:, 0].square() + wave_vectors[:, 1].square())
        * energy2wavelength(energy_eV, device=wave_vectors.device, dtype=wave_vectors.dtype)
    )
    phi = torch.arctan2(wave_vectors[:, 0], wave_vectors[:, 1])

    ctf = CTF(
        energy_eV=energy_eV,
        gpts=gpts,
        sampling=(extent_A[0] / gpts[0], extent_A[1] / gpts[1]),
        semiangle_cutoff_mrad=float("inf"),
        aberrations=AberrationCoefficients(C10=-defocus_A),
        dtype=wave_vectors.dtype,
        device=array.device,
    )
    interpolated_array = interpolated_array * ctf.evaluate(alpha, phi)[:, None, None]
    weighted_parent = torch.einsum(
        "pq,pyx->qyx",
        weights.to(array.dtype),
        array.to(torch.complex128),
    )
    return interpolated_array * weighted_parent


def remove_tilt(
    array: torch.Tensor,
    planewave_cutoff_mrad: float,
    extent_A: tuple[float, float],
    gpts: tuple[int, int],
    energy_eV: float,
    interpolation: int | tuple[int, int],
    partitions: int | None,
    accumulated_defocus_A: float,
    block_info=None,
) -> torch.Tensor:
    if block_info is None:
        start, end = 0, array.shape[-3]
    else:
        start, end = block_info[0]["array-location"][-3]

    if partitions is None:
        wave_vectors = prism_wave_vectors(
            planewave_cutoff_mrad,
            extent_A,
            energy_eV,
            interpolation,
            device=array.device,
            dtype=torch.float64,
        )
    else:
        wave_vectors = partitioned_prism_wave_vectors(
            planewave_cutoff_mrad,
            extent_A,
            energy_eV,
            num_rings=partitions,
            device=array.device,
            dtype=torch.float64,
        )

    wave_vectors = wave_vectors[start:end]
    wavelength = energy2wavelength(energy_eV, device=array.device, dtype=torch.float64)
    alpha = torch.sqrt(wave_vectors[:, 0].square() + wave_vectors[:, 1].square()) * wavelength
    phi = torch.arctan2(wave_vectors[:, 0], wave_vectors[:, 1])

    ctf = CTF(
        energy_eV=energy_eV,
        gpts=gpts,
        sampling=(extent_A[0] / gpts[0], extent_A[1] / gpts[1]),
        semiangle_cutoff_mrad=float("inf"),
        aberrations=AberrationCoefficients(C10=accumulated_defocus_A),
        dtype=torch.float64,
        device=array.device,
    )
    ctf_coefficients = ctf.evaluate(alpha, phi)
    expand_dims = tuple(range(len(array.shape) - 3)) + (-2, -1)
    ctf_coefficients = ctf_coefficients.view(
        *((1,) * (len(array.shape) - 3)),
        ctf_coefficients.shape[0],
        1,
        1,
    )

    tilt_removed = (
        array.to(torch.complex128)
        * plane_waves(
            wave_vectors,
            extent_A,
            gpts,
            reverse=True,
            device=array.device,
            dtype=torch.float64,
        )
        * ctf_coefficients
    )
    return tilt_removed


def reduce_beamlets_nearest_no_interpolation(
    waves: torch.Tensor,
    basis: torch.Tensor,
    parent_s_matrix: torch.Tensor,
    shifts: torch.Tensor,
) -> torch.Tensor:
    if waves.shape[0] != shifts.shape[0]:
        raise ValueError("Number of output waves must match number of shifts")
    if shifts.ndim != 2 or shifts.shape[1] != 2:
        raise ValueError("shifts must have shape (n_waves, 2)")

    basis = basis.to(parent_s_matrix.dtype)
    parent_s_matrix = parent_s_matrix.to(parent_s_matrix.dtype)
    out = waves.clone().to(parent_s_matrix.dtype)

    for i in range(out.shape[0]):
        shift_y = int(shifts[i, 0].item())
        shift_x = int(shifts[i, 1].item())
        for j in range(out.shape[1]):
            for k in range(out.shape[2]):
                out[i, j, k] = torch.dot(
                    basis[:, j, k],
                    parent_s_matrix[
                        :,
                        (j + shift_y) % parent_s_matrix.shape[1],
                        (k + shift_x) % parent_s_matrix.shape[2],
                    ],
                )

    return out


def array_row_intersection(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    a = a.to(torch.float64)
    b = b.to(torch.float64)
    tmp = torch.all(torch.isclose(a[:, None, :], b[None, :, :], atol=1e-8, rtol=1e-5), dim=2)
    tmp_int = tmp.to(torch.int64)
    return torch.sum((torch.cumsum(tmp_int, dim=0) * tmp_int) == 1, dim=1).to(torch.bool)


class SMatrixArray(nn.Module):
    """Differentiable S-matrix container with PRISM-style reduction."""

    def __init__(
        self,
        array: torch.Tensor,
        *,
        parent_wave_vectors: torch.Tensor,
        extent_A: tuple[float, float],
        energy_eV: float,
    ) -> None:
        super().__init__()
        if array.ndim != 3:
            raise ValueError("Expected S-matrix array with shape (n_beamlets, ny, nx)")
        self.register_buffer("array", array.to(torch.complex128))
        self.register_buffer("parent_wave_vectors", parent_wave_vectors.to(torch.float64))
        self.extent_A = extent_A
        self.energy_eV = float(energy_eV)

    def reduce_to_waves(self, coefficients: torch.Tensor) -> torch.Tensor:
        coefficients = coefficients.to(self.array.dtype)
        if coefficients.shape[-1] != self.array.shape[0]:
            raise ValueError("Last dimension of coefficients must match the number of beamlets")
        return torch.einsum("...p,pyx->...yx", coefficients, self.array)

    def scan(self, scan_positions_A: torch.Tensor, ctf_coefficients: torch.Tensor) -> torch.Tensor:
        phase = calculate_positions_coefficients(scan_positions_A, self.parent_wave_vectors)
        coeffs = ctf_coefficients.to(torch.complex128) * phase
        return self.reduce_to_waves(coeffs)


class SMatrixBuilder(nn.Module):
    """Build a simple S-matrix by multislicing parent plane waves."""

    def __init__(
        self,
        multislice_system: nn.Module,
        parent_wave_vectors: torch.Tensor,
        extent_A: tuple[float, float],
        gpts: tuple[int, int],
        energy_eV: float,
        interpolation: int | tuple[int, int] = 1,
    ) -> None:
        super().__init__()
        self.multislice_system = multislice_system
        self.register_buffer("parent_wave_vectors", parent_wave_vectors.to(torch.float64))
        self.extent_A = extent_A
        self.gpts = gpts
        self.energy_eV = float(energy_eV)
        if isinstance(interpolation, int):
            interpolation = (interpolation, interpolation)
        self.interpolation = tuple(int(value) for value in interpolation)

    def forward(self, potential_slices: torch.Tensor) -> SMatrixArray:
        incident = plane_waves(
            self.parent_wave_vectors,
            self.extent_A,
            self.gpts,
            device=potential_slices.device,
        )
        incident = incident * (
            (self.interpolation[0] * self.interpolation[1])
            / float(self.gpts[0] * self.gpts[1])
        )
        propagated = torch.stack(
            [self.multislice_system(incident[i], potential_slices) for i in range(incident.shape[0])],
            dim=0,
        )
        return SMatrixArray(
            propagated,
            parent_wave_vectors=self.parent_wave_vectors,
            extent_A=self.extent_A,
            energy_eV=self.energy_eV,
        )


class PartitionedSMatrixInterpolator(nn.Module):
    """Reduce parent S-matrix beamlets with partitioned PRISM interpolation weights."""

    def __init__(
        self,
        *,
        extent_A: tuple[float, float],
        gpts: tuple[int, int],
        weight_mode: str = "natural",
    ) -> None:
        super().__init__()
        self.extent_A = extent_A
        self.gpts = gpts
        self.weight_mode = weight_mode

    def forward(
        self,
        smatrix: SMatrixArray,
        ctf_coefficients: torch.Tensor,
        wave_vectors: torch.Tensor,
    ) -> torch.Tensor:
        wave_vectors = wave_vectors.to(smatrix.parent_wave_vectors.device, dtype=torch.float64)
        if self.weight_mode == "natural":
            weights = pairwise_natural_weights(smatrix.parent_wave_vectors, wave_vectors)
        elif self.weight_mode == "inverse_distance":
            weights = pairwise_inverse_distance_weights(smatrix.parent_wave_vectors, wave_vectors)
        else:
            raise ValueError(f"Unknown weight_mode {self.weight_mode}")
        coefficients = weights * ctf_coefficients.to(torch.complex128)[:, None]
        return torch.einsum("pq,pyx->qyx", coefficients, smatrix.array)
