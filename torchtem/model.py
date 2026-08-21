from __future__ import annotations

import torch
from torch import nn

from torchtem.coherence import CoherenceAverager
from torchtem.detectors import AnnularDetector
from torchtem.multislice import MultisliceSystem
from torchtem.optics import Probe
from torchtem.scan import CustomScan, GridScan, fft_shift_wave


def _validate_num_slices(num_slices: int) -> int:
    value = int(num_slices)
    if value < 1:
        raise ValueError("num_slices must be at least 1")
    return value


class TEMModel(nn.Module):
    """Minimal differentiable TEM/STEM composition built from torchtem operators."""

    def __init__(
        self,
        *,
        probe: Probe,
        multislice: MultisliceSystem,
        potential: nn.Module,
        detector: nn.Module | None = None,
        scan: GridScan | CustomScan | None = None,
        coherence: CoherenceAverager | None = None,
        num_slices: int = 1,
    ) -> None:
        super().__init__()

        # Validate public configuration before storing it on the model.
        self.probe = probe
        self.multislice = multislice
        self.potential = potential
        self.detector = detector
        self.scan = scan
        self.coherence = coherence
        self.num_slices = _validate_num_slices(num_slices)

    def potential_slices(self) -> torch.Tensor:
        potential = self.potential()
        if potential.ndim == 3:
            return potential
        return potential.unsqueeze(0).repeat(self.num_slices, 1, 1) / self.num_slices

    def exit_wave(self) -> torch.Tensor:
        incident = self.probe()
        return self.multislice(incident, self.potential_slices())

    def scanned_exit_waves(self) -> torch.Tensor:
        # Build the shifted incident waves for each scan position.
        incident = self.probe()
        if self.scan is None:
            positions = torch.zeros((1, 2), device=incident.device, dtype=torch.float64)
        else:
            positions = self.scan().to(device=incident.device, dtype=torch.float64)

        shifted = fft_shift_wave(incident, positions, self.multislice.sampling)
        slices = self.potential_slices()

        # Propagate each scan position independently through the specimen.
        exit_waves = []
        for index in range(shifted.shape[0]):
            exit_waves.append(self.multislice(shifted[index], slices))
        return torch.stack(exit_waves, dim = 0)

    def forward(self) -> torch.Tensor:
        if self.scan is None:
            wave = self.exit_wave()
            if self.coherence is not None:
                waves = self.coherence.apply_source_offsets(wave, self.multislice.sampling)
                return self.coherence.average_intensity(waves, self.detector)
            if self.detector is None:
                return wave
            return self.detector(wave)

        waves = self.scanned_exit_waves()
        if self.coherence is not None:
            outputs = []
            for index in range(waves.shape[0]):
                coherent_members = self.coherence.apply_source_offsets(
                    waves[index], self.multislice.sampling
                )
                outputs.append(self.coherence.average_intensity(coherent_members, self.detector))
            return torch.stack(outputs, dim = 0)
        if self.detector is None:
            return waves

        outputs = []
        for index in range(waves.shape[0]):
            outputs.append(self.detector(waves[index]))
        return torch.stack(outputs, dim = 0)
