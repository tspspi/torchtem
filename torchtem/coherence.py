from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from tspi.torchtem.complex_support import abs2
from tspi.torchtem.scan import fft_shift_wave


@dataclass
class CoherenceEnsemble:
    defocus_offsets_A: torch.Tensor | None = None
    source_offsets_A: torch.Tensor | None = None
    weights: torch.Tensor | None = None


class CoherenceAverager(nn.Module):
    """Approximate partial coherence by averaging over shifted/defocused probes."""

    def __init__(self, ensemble: CoherenceEnsemble) -> None:
        super().__init__()
        if ensemble.defocus_offsets_A is not None:
            self.register_buffer(
                "defocus_offsets_A", ensemble.defocus_offsets_A.to(torch.float64)
            )
        else:
            self.defocus_offsets_A = None
        if ensemble.source_offsets_A is not None:
            self.register_buffer(
                "source_offsets_A", ensemble.source_offsets_A.to(torch.float64)
            )
        else:
            self.source_offsets_A = None
        if ensemble.weights is not None:
            weights = ensemble.weights.to(torch.float64)
        else:
            n = 1
            if ensemble.defocus_offsets_A is not None:
                n = max(n, int(ensemble.defocus_offsets_A.shape[0]))
            if ensemble.source_offsets_A is not None:
                n = max(n, int(ensemble.source_offsets_A.shape[0]))
            weights = torch.full((n,), 1.0 / n, dtype=torch.float64)
        self.register_buffer("weights", weights / weights.sum())

    def num_members(self) -> int:
        return int(self.weights.shape[0])

    def apply_source_offsets(
        self, wave: torch.Tensor, sampling: tuple[float, float]
    ) -> torch.Tensor:
        if self.source_offsets_A is None:
            return wave.unsqueeze(0).expand(self.num_members(), *wave.shape)
        return fft_shift_wave(wave, self.source_offsets_A, sampling)

    def average_intensity(
        self, waves: torch.Tensor, detector: nn.Module | None = None
    ) -> torch.Tensor:
        if detector is None:
            intensity = abs2(waves)
            return torch.sum(self.weights[:, None, None] * intensity, dim=0)
        outputs = torch.stack([detector(waves[i]) for i in range(waves.shape[0])], dim=0)
        weights = self.weights.to(outputs.dtype).reshape(
            (self.weights.shape[0],) + (1,) * (outputs.ndim - 1)
        )
        return torch.sum(weights * outputs, dim=0)
