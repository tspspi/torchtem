from __future__ import annotations

import math

import torch
from torch import nn

from torchtem.optics import CTF
from torchtem.physics import spatial_frequencies


class DiagonalMCF(nn.Module):
    """Dense diagonal mixed-coherence decomposition on a CTF grid."""

    def __init__(
        self,
        ctf: CTF,
        *,
        eigenvectors: int | tuple[int, ...],
        focal_spread_A: float = 0.0,
        source_size_A: float = 0.0,
        rectangular_offset_A: tuple[float, float] = (0.0, 0.0),
    ) -> None:
        super().__init__()
        self.ctf = ctf
        if isinstance(eigenvectors, int):
            self.eigenvectors = tuple(range(int(eigenvectors)))
        else:
            self.eigenvectors = tuple(int(i) for i in eigenvectors)
        self.focal_spread_A = float(focal_spread_A)
        self.source_size_A = float(source_size_A)
        self.rectangular_offset_A = rectangular_offset_A

    def reciprocal_coordinates(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        alpha, _ = self.ctf.angular_coordinates()
        ky, kx = spatial_frequencies(
            self.ctf.gpts,
            self.ctf.sampling,
            device=alpha.device,
            dtype=self.ctf.dtype,
        )
        kxky_y, kxky_x = torch.meshgrid(ky, kx, indexing="ij")
        k2 = kxky_y.square() + kxky_x.square()
        aperture = alpha <= self.ctf.semiangle_cutoff_mrad
        return kxky_y, kxky_x, k2 * aperture.to(k2.dtype)

    def mutual_coherence_matrix(self) -> tuple[torch.Tensor, torch.Tensor]:
        wavelength = self.ctf.wavelength.to(dtype=self.ctf.dtype)
        ky, kx = spatial_frequencies(
            self.ctf.gpts,
            self.ctf.sampling,
            device=self.ctf.device_hint,
            dtype=self.ctf.dtype,
        )
        ky, kx = torch.meshgrid(ky, kx, indexing="ij")
        alpha, _ = self.ctf.angular_coordinates()
        mask = (alpha <= self.ctf.semiangle_cutoff_mrad).reshape(-1)

        kx_flat = kx.reshape(-1)[mask]
        ky_flat = ky.reshape(-1)[mask]
        k2_flat = (kx.square() + ky.square()).reshape(-1)[mask]

        E = torch.ones(
            (kx_flat.shape[0], kx_flat.shape[0]),
            dtype=self.ctf.dtype,
            device=kx_flat.device,
        )

        if self.focal_spread_A > 0.0:
            dk2 = k2_flat[:, None] - k2_flat[None, :]
            E = E * torch.exp(
                -((0.5 * math.pi * wavelength * self.focal_spread_A) ** 2) * dk2.square()
            )

        if self.source_size_A > 0.0:
            dkx = kx_flat[:, None] - kx_flat[None, :]
            dky = ky_flat[:, None] - ky_flat[None, :]
            E = E * torch.exp(-((math.pi * self.source_size_A) ** 2) * (dkx.square() + dky.square()))

        if self.rectangular_offset_A != (0.0, 0.0):
            dkx = kx_flat[:, None] - kx_flat[None, :]
            dky = ky_flat[:, None] - ky_flat[None, :]
            E = E * torch.sinc(dkx * self.rectangular_offset_A[0]) * torch.sinc(
                dky * self.rectangular_offset_A[1]
            )

        return E, mask

    def reciprocal_modes(self) -> torch.Tensor:
        E, mask = self.mutual_coherence_matrix()
        values, vectors = torch.linalg.eigh(E)
        order = torch.argsort(values, descending=True)
        selected = order[torch.as_tensor(self.eigenvectors, device=order.device)]
        selected_values = torch.clamp(values[selected], min=0.0)
        selected_vectors = vectors[:, selected].T

        modes = torch.zeros(
            (len(self.eigenvectors), self.ctf.gpts[0] * self.ctf.gpts[1]),
            dtype=torch.complex128,
            device=E.device,
        )
        modes[:, mask] = torch.sqrt(selected_values).to(torch.complex128)[:, None] * selected_vectors.to(
            torch.complex128
        )
        modes = modes.reshape(len(self.eigenvectors), *self.ctf.gpts)
        transfer = self.ctf()
        aperture = self.ctf.aperture()
        safe_aperture = torch.where(
            aperture > 0,
            torch.clamp(aperture, min=1e-30),
            torch.ones_like(aperture),
        ).to(torch.complex128)
        phase_only = torch.where(
            aperture > 0,
            transfer / safe_aperture,
            torch.zeros_like(transfer),
        )
        reciprocal_modes = modes * phase_only.unsqueeze(0)
        norm = torch.sqrt(torch.sum(torch.abs(transfer).square()))
        return reciprocal_modes / norm.clamp_min(torch.finfo(self.ctf.dtype).eps)

    def real_space_modes(self) -> torch.Tensor:
        return torch.fft.ifft2(self.reciprocal_modes())

    def average_intensity(self) -> torch.Tensor:
        modes = self.real_space_modes()
        return modes.abs().square().sum(dim=0)

    def forward(self, real_space: bool = True) -> torch.Tensor:
        if real_space:
            return self.real_space_modes()
        return self.reciprocal_modes()
