from __future__ import annotations

import pathlib
import sys

import matplotlib
import torch

if __package__ in (None, ""):
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from torchtem import (
    AnnularDetectorConfig,
    ExperimentBuilder,
    GaussianAtomProjection,
    GridScanConfig,
    MultisliceConfig,
    ProbeSourceConfig,
    STEMConfig,
)


OUTPUT_DIR = pathlib.Path(__file__).resolve().with_name("output")


def _robust_limits(image: torch.Tensor, *, lower: float = 0.01, upper: float = 0.99) -> tuple[float, float]:
    flat = image.reshape(-1).to(torch.float64)
    q = torch.quantile(flat, torch.tensor([lower, upper], dtype=flat.dtype))
    vmin = float(q[0])
    vmax = float(q[1])
    if vmax <= vmin:
        vmax = vmin + 1e-12
    return vmin, vmax


def build_experiment(
    *,
    c10: float,
    positions_A: torch.Tensor,
    amplitudes: torch.Tensor,
    sigmas_A: torch.Tensor,
) -> ExperimentBuilder:
    min_corner = positions_A.amin(dim=0) - 2.5
    max_corner = positions_A.amax(dim=0) + 2.5
    return ExperimentBuilder.from_stem(
        STEMConfig(
            source=ProbeSourceConfig(
                energy_eV=200e3,
                gpts=(96, 96),
                sampling=(0.2, 0.2),
                semiangle_cutoff_mrad=25.0,
            ),
            multislice=MultisliceConfig(
                energy_eV=200e3,
                gpts=(96, 96),
                sampling=(0.2, 0.2),
                slice_thickness_A=1.0,
                num_slices=4,
                backend="fourier",
            ),
            potential=GaussianAtomProjection(
                gpts=(96, 96),
                sampling=(0.2, 0.2),
                positions_A=positions_A,
                amplitudes=amplitudes,
                sigmas_A=sigmas_A,
            ),
            detector=AnnularDetectorConfig(
                inner_mrad=20.0,
                outer_mrad=80.0,
            ),
            scan=GridScanConfig(
                start_A=tuple(float(value) for value in min_corner),
                end_A=tuple(float(value) for value in max_corner),
                shape=(8, 8),
            ),
        )
    )


def main() -> None:
    OUTPUT_DIR.mkdir(exist_ok=True)

    target = build_experiment(
        c10=-22.0,
        positions_A=torch.tensor([[5.5, 5.0], [9.0, 10.5]], dtype=torch.float64),
        amplitudes=torch.tensor([10.0, 7.5], dtype=torch.float64),
        sigmas_A=torch.tensor([0.7, 0.9], dtype=torch.float64),
    )
    target.set_parameter("source.ctf.C10", -22.0)
    target_signal = target().detach()

    fit = build_experiment(
        c10=-5.0,
        positions_A=torch.tensor([[4.5, 6.0], [10.5, 9.0]], dtype=torch.float64),
        amplitudes=torch.tensor([10.0, 7.5], dtype=torch.float64),
        sigmas_A=torch.tensor([0.7, 0.9], dtype=torch.float64),
    )
    fit.set_parameter("source.ctf.C10", -5.0)

    optimizer = torch.optim.Adam(fit.parameters(), lr=5e-2)
    history: list[float] = []

    for step in range(80):
        optimizer.zero_grad()
        prediction = fit()
        loss = (prediction - target_signal).square().mean()
        loss.backward()
        optimizer.step()
        history.append(float(loss.detach()))

        if step % 20 == 0 or step == 79:
            print(
                f"step={step:03d} loss={loss.item():.6e} "
                f"C10={fit.source.ctf.C10.detach().item():.3f} "
                f"positions={fit.potential.positions_A.detach().cpu().tolist()}"
            )

    final_prediction = fit().detach().cpu().reshape(8, 8)
    target_image = target_signal.detach().cpu().reshape(8, 8)
    display_vmin, display_vmax = _robust_limits(
        torch.stack([target_image, final_prediction]),
        lower=0.02,
        upper=0.98,
    )

    fig, axes = plt.subplots(1, 3, figsize=(12.5, 4.4), dpi=180)
    loss_ax, target_ax, fit_ax = axes

    loss_ax.plot(history, color="black", linewidth=1.5)
    loss_ax.set_title("Optimization Loss")
    loss_ax.set_xlabel("step")
    loss_ax.set_ylabel("MSE")

    target_im = target_ax.imshow(
        target_image.numpy(),
        cmap="inferno",
        aspect="equal",
        vmin=display_vmin,
        vmax=display_vmax,
    )
    target_ax.set_title("Target STEM Signal")
    fig.colorbar(target_im, ax=target_ax, fraction=0.046, pad=0.04)

    fit_im = fit_ax.imshow(
        final_prediction.numpy(),
        cmap="inferno",
        aspect="equal",
        vmin=display_vmin,
        vmax=display_vmax,
    )
    fit_ax.set_title("Fitted STEM Signal")
    fig.colorbar(fit_im, ax=fit_ax, fraction=0.046, pad=0.04)

    fig.tight_layout()
    image_path = OUTPUT_DIR / "fit_stem_experiment_parameters.png"
    fig.savefig(image_path, dpi=180)
    plt.close(fig)

    print("saved_image:", image_path)


if __name__ == "__main__":
    main()
