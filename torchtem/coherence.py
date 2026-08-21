from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from torchtem.complex_support import abs2
from torchtem.scan import fft_shift_wave


@dataclass
class CoherenceEnsemble:
    defocus_offsets_A: torch.Tensor | None = None
    source_offsets_A: torch.Tensor | None = None
    weights: torch.Tensor | None = None


class CoherenceAverager(nn.Module):
    """Approximate partial coherence by averaging over shifted/defocused probes."""

    def __init__(self, ensemble: CoherenceEnsemble) -> None:
        super().__init__()
        defocus_count = 1
        source_count = 1

        if ensemble.defocus_offsets_A is not None:
            if ensemble.defocus_offsets_A.ndim != 1:
                raise ValueError("defocus_offsets_A must be a 1-D tensor of Angstrom offsets")
            self.register_buffer(
                "defocus_offsets_A", ensemble.defocus_offsets_A.to(torch.float64)
            )
            defocus_count = int(ensemble.defocus_offsets_A.shape[0])
        else:
            self.defocus_offsets_A = None

        if ensemble.source_offsets_A is not None:
            if ensemble.source_offsets_A.ndim != 2:
                raise ValueError("source_offsets_A must have shape (count, 2)")
            if ensemble.source_offsets_A.shape[1] != 2:
                raise ValueError("source_offsets_A must provide x/y offsets in Angstrom")
            self.register_buffer(
                "source_offsets_A", ensemble.source_offsets_A.to(torch.float64)
            )
            source_count = int(ensemble.source_offsets_A.shape[0])
        else:
            self.source_offsets_A = None

        num_members = max(defocus_count, source_count)
        if ensemble.weights is not None:
            weights = ensemble.weights.to(torch.float64)
            if weights.ndim != 1:
                raise ValueError("weights must be a 1-D tensor of coherence weights")
            if weights.shape[0] != num_members:
                raise ValueError(
                    "weights length must match the number of coherence members"
                )
        else:
            weights = torch.full((num_members,), 1.0 / num_members, dtype=torch.float64)

        weight_sum = weights.sum()
        if float(weight_sum.detach().cpu()) == 0.0:
            raise ValueError("weights must sum to a non-zero value")
        self.register_buffer("weights", weights / weight_sum)

    def num_members(self) -> int:
        return int(self.weights.shape[0])

    def apply_source_offsets(
        self, wave: torch.Tensor, sampling: tuple[float, float]
    ) -> torch.Tensor:
        members = self.num_members()
        if self.source_offsets_A is None:
            return wave.unsqueeze(0).expand(members, *wave.shape)
        return fft_shift_wave(wave, self.source_offsets_A, sampling)

    def average_intensity(
        self, waves: torch.Tensor, detector: nn.Module | None = None
    ) -> torch.Tensor:
        if waves.shape[0] != self.num_members():
            raise ValueError("waves leading dimension must match the number of coherence members")

        if detector is None:
            intensity = abs2(waves)
            return torch.sum(self.weights[:, None, None] * intensity, dim=0)

        outputs = []
        for index in range(waves.shape[0]):
            outputs.append(detector(waves[index]))
        outputs = torch.stack(outputs, dim = 0)
        weights = self.weights.to(outputs.dtype).reshape(
            (self.weights.shape[0],) + (1,) * (outputs.ndim - 1)
        )
        return torch.sum(weights * outputs, dim = 0)
