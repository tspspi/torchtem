from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from torchtem.potentials import render_gaussian_projection


@dataclass
class FrozenPhononSpec:
    displacement_sigmas_A: torch.Tensor
    num_configs: int
    seed: int = 0
    directions: str = "xy"


class FrozenPhononGaussianEnsemble(nn.Module):
    """Frozen-phonon ensemble over differentiable Gaussian projected atoms."""

    def __init__(
        self,
        *,
        gpts: tuple[int, int],
        sampling: tuple[float, float] | float,
        positions_A: torch.Tensor,
        amplitudes: torch.Tensor,
        sigmas_A: torch.Tensor,
        frozen_phonons: FrozenPhononSpec,
        dtype: torch.dtype = torch.float64,
        device=None,
    ) -> None:
        super().__init__()
        self.gpts = gpts
        self.sampling = sampling
        self.dtype = dtype
        self.device_hint = device
        self.positions_A = nn.Parameter(positions_A.to(device=device, dtype=dtype))
        self.amplitudes = nn.Parameter(amplitudes.to(device=device, dtype=dtype))
        self.log_sigmas_A = nn.Parameter(sigmas_A.to(device=device, dtype=dtype).log())
        self.register_buffer(
            "displacement_sigmas_A",
            frozen_phonons.displacement_sigmas_A.to(device=device, dtype=dtype),
        )
        self.num_configs = int(frozen_phonons.num_configs)
        self.seed = int(frozen_phonons.seed)
        self.directions = frozen_phonons.directions
        self.register_buffer("displacements_A", self._sample_displacements())

    @property
    def sigmas_A(self) -> torch.Tensor:
        return self.log_sigmas_A.exp()

    def _axis_mask(self) -> torch.Tensor:
        mask = torch.zeros(2, device=self.positions_A.device, dtype=self.dtype)
        directions = set(self.directions.lower())
        if "y" in directions:
            mask[0] = 1.0
        if "x" in directions:
            mask[1] = 1.0
        return mask

    def _sample_displacements(self) -> torch.Tensor:
        generator = torch.Generator(device="cpu")
        generator.manual_seed(self.seed)
        noise = torch.randn(
            (self.num_configs, self.positions_A.shape[0], 2),
            generator=generator,
            dtype=self.dtype,
        ).to(self.positions_A.device)
        sigmas = self.displacement_sigmas_A
        if sigmas.ndim == 1:
            sigmas = sigmas[:, None].expand(-1, 2)
        return noise * sigmas.unsqueeze(0) * self._axis_mask()

    def configuration_positions(self) -> torch.Tensor:
        return self.positions_A.unsqueeze(0) + self.displacements_A

    def forward(self) -> torch.Tensor:
        return torch.stack(
            [
                render_gaussian_projection(
                    gpts=self.gpts,
                    sampling=self.sampling,
                    positions_A=positions,
                    amplitudes=self.amplitudes,
                    sigmas_A=self.sigmas_A,
                    dtype=self.dtype,
                    device=self.positions_A.device,
                )
                for positions in self.configuration_positions()
            ],
            dim=0,
        )

    def ensemble_mean(self) -> torch.Tensor:
        return self.forward().mean(dim=0)
