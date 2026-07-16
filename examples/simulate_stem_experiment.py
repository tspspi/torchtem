from __future__ import annotations

import pathlib
import sys

import matplotlib
import torch

if __package__ in (None, ""):
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from tspi.torchtem import (
    AnnularDetectorConfig,
    ExperimentBuilder,
    GaussianAtomProjection,
    GridScanConfig,
    MultisliceConfig,
    PixelatedDetectorConfig,
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


def build_experiment() -> ExperimentBuilder:
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
                positions_A=torch.tensor([[5.5, 5.0], [9.0, 10.5]], dtype=torch.float64),
                amplitudes=torch.tensor([10.0, 7.5], dtype=torch.float64),
                sigmas_A=torch.tensor([0.7, 0.9], dtype=torch.float64),
            ),
            detector={
                "haadf": AnnularDetectorConfig(inner_mrad=20.0, outer_mrad=80.0),
                "pixelated": PixelatedDetectorConfig(),
            },
            scan=GridScanConfig(
                start_A=(0.0, 0.0),
                end_A=(3.0, 3.0),
                shape=(8, 8),
            ),
        )
    )


def main() -> None:
    OUTPUT_DIR.mkdir(exist_ok=True)
    experiment = build_experiment()
    result = experiment.run(include_exit_wave=True)
    haadf = result.outputs["haadf"].detach().cpu().reshape(8, 8)
    pixelated = result.outputs["pixelated"].detach().cpu()
    exit_wave = result.exit_wave.detach().cpu()
    mean_diffraction = pixelated.mean(dim=0)
    log_diffraction = torch.log1p(mean_diffraction)
    haadf_vmin, haadf_vmax = _robust_limits(haadf, lower=0.02, upper=0.98)
    diff_vmin, diff_vmax = _robust_limits(log_diffraction, lower=0.02, upper=0.995)

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.8), dpi=180)
    haadf_im = axes[0].imshow(
        haadf.numpy(),
        cmap="inferno",
        aspect="equal",
        vmin=haadf_vmin,
        vmax=haadf_vmax,
    )
    axes[0].set_title("Simulated STEM HAADF")
    axes[0].set_xlabel("scan x")
    axes[0].set_ylabel("scan y")
    fig.colorbar(haadf_im, ax=axes[0], fraction=0.046, pad=0.04)

    diff_im = axes[1].imshow(
        log_diffraction.numpy(),
        cmap="magma",
        aspect="equal",
        vmin=diff_vmin,
        vmax=diff_vmax,
    )
    axes[1].set_title("Mean Diffraction Pattern (log contrast)")
    axes[1].set_xlabel("kx pixels")
    axes[1].set_ylabel("ky pixels")
    fig.colorbar(diff_im, ax=axes[1], fraction=0.046, pad=0.04)

    fig.tight_layout()
    image_path = OUTPUT_DIR / "simulate_stem_experiment.png"
    exit_wave_path = OUTPUT_DIR / "simulate_stem_experiment_exit_wave.pt"
    fig.savefig(image_path, dpi=180)
    plt.close(fig)
    torch.save(exit_wave, exit_wave_path)

    print("mode:", result.metadata["mode"])
    print("propagation_backend:", result.metadata["propagation_backend"])
    print("detector_names:", result.detector_names)
    print("output_shapes:", result.output_shapes())
    print("exit_wave_shape:", tuple(exit_wave.shape))
    print("haadf_mean:", float(result.outputs["haadf"].detach().mean()))
    print("haadf_minmax:", (float(haadf.min()), float(haadf.max())))
    print("pixelated_shape:", tuple(result.outputs["pixelated"].shape))
    print("display_limits:", {"haadf": (haadf_vmin, haadf_vmax), "log_diffraction": (diff_vmin, diff_vmax)})
    print("saved_image:", image_path)
    print("saved_exit_wave:", exit_wave_path)


if __name__ == "__main__":
    main()
