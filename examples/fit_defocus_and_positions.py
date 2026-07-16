from __future__ import annotations

import pathlib
import sys

import torch

if __package__ in (None, ""):
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

from tspi.torchtem import (
    AberrationCoefficients,
    AnnularDetector,
    CTF,
    GaussianAtomProjection,
    MultisliceSystem,
    Probe,
    TEMModel,
)


def build_model(
    *,
    c10: float,
    positions: torch.Tensor,
    amplitudes: torch.Tensor,
    sigmas: torch.Tensor,
) -> TEMModel:
    ctf = CTF(
        energy_eV=200e3,
        gpts=(64, 64),
        sampling=(0.2, 0.2),
        semiangle_cutoff_mrad=30.0,
        aberrations=AberrationCoefficients(C10=c10, C30=5.0e4),
    )
    probe = Probe(ctf)
    potential = GaussianAtomProjection(
        gpts=(64, 64),
        sampling=(0.2, 0.2),
        positions_A=positions,
        amplitudes=amplitudes,
        sigmas_A=sigmas,
    )
    multislice = MultisliceSystem(
        energy_eV=200e3,
        gpts=(64, 64),
        sampling=(0.2, 0.2),
        slice_thickness_A=1.0,
    )
    detector = AnnularDetector(
        energy_eV=200e3,
        gpts=(64, 64),
        sampling=(0.2, 0.2),
        inner_mrad=10.0,
        outer_mrad=80.0,
    )
    return TEMModel(
        probe=probe,
        multislice=multislice,
        potential=potential,
        detector=detector,
        num_slices=3,
    )


def main() -> None:
    target = build_model(
        c10=-25.0,
        positions=torch.tensor([[4.0, 4.5], [6.5, 7.0]], dtype=torch.float64),
        amplitudes=torch.tensor([15.0, 12.0], dtype=torch.float64),
        sigmas=torch.tensor([0.5, 0.8], dtype=torch.float64),
    )
    target_signal = target().detach()

    fit = build_model(
        c10=-5.0,
        positions=torch.tensor([[3.0, 5.0], [7.0, 6.0]], dtype=torch.float64),
        amplitudes=torch.tensor([15.0, 12.0], dtype=torch.float64),
        sigmas=torch.tensor([0.5, 0.8], dtype=torch.float64),
    )

    optimizer = torch.optim.Adam(fit.parameters(), lr=5e-2)

    for step in range(200):
        optimizer.zero_grad()
        prediction = fit()
        loss = (prediction - target_signal).abs().square()
        loss.backward()
        optimizer.step()

        if step % 25 == 0:
            print(
                f"step={step:03d} loss={loss.item():.6e} "
                f"C10={fit.probe.ctf.C10.detach().item():.3f} "
                f"positions={fit.potential.positions_A.detach().cpu().tolist()}"
            )


if __name__ == "__main__":
    main()
