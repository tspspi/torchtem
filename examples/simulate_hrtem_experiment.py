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
    AberrationCoefficients,
    ExperimentBuilder,
    GaussianAtomProjection,
    HRTEMConfig,
    ImageDetectorConfig,
    MultisliceConfig,
    PixelatedDetectorConfig,
    PlaneWaveSourceConfig,
)


OUTPUT_DIR = pathlib.Path(__file__).resolve().with_name("output")


def _robust_limits(image: torch.Tensor, *, lower: float = 0.01, upper: float = 0.995) -> tuple[float, float]:
    flat = image.reshape(-1).to(torch.float64)
    q = torch.quantile(flat, torch.tensor([lower, upper], dtype=flat.dtype))
    vmin = float(q[0])
    vmax = float(q[1])
    if vmax <= vmin:
        vmax = vmin + 1e-12
    return vmin, vmax


def build_experiment() -> ExperimentBuilder:
    return ExperimentBuilder.from_hrtem(
        HRTEMConfig(
            source=PlaneWaveSourceConfig(
                gpts=(96, 96),
                sampling=(0.2, 0.2),
                energy_eV=200e3,
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
                "image": ImageDetectorConfig(
                    output="intensity",
                    semiangle_cutoff_mrad=30.0,
                    aberrations=AberrationCoefficients(C10=-120.0, C30=1.0e4),
                ),
                "diffraction": PixelatedDetectorConfig(),
            },
        )
    )


def main() -> None:
    OUTPUT_DIR.mkdir(exist_ok=True)
    experiment = build_experiment()
    result = experiment.run(return_intermediates=True, include_exit_wave=True)
    image_intensity = result.outputs["image"].detach().cpu()
    diffraction = result.outputs["diffraction"].detach().cpu()
    exit_wave = result.exit_wave.detach().cpu()
    display_diffraction = torch.log1p(diffraction)
    image_vmin, image_vmax = _robust_limits(image_intensity, lower=0.01, upper=0.99)
    diff_vmin, diff_vmax = _robust_limits(display_diffraction, lower=0.01, upper=0.999)

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.8), dpi=180)
    image_im = axes[0].imshow(
        image_intensity.numpy(),
        cmap="gray",
        aspect="equal",
        vmin=image_vmin,
        vmax=image_vmax,
    )
    axes[0].set_title("Simulated TEM Image")
    axes[0].set_xlabel("x pixels")
    axes[0].set_ylabel("y pixels")
    fig.colorbar(image_im, ax=axes[0], fraction=0.046, pad=0.04)

    diff_im = axes[1].imshow(
        display_diffraction.numpy(),
        cmap="magma",
        aspect="equal",
        vmin=diff_vmin,
        vmax=diff_vmax,
    )
    axes[1].set_title("Diffraction Pattern (log contrast)")
    axes[1].set_xlabel("kx pixels")
    axes[1].set_ylabel("ky pixels")
    fig.colorbar(diff_im, ax=axes[1], fraction=0.046, pad=0.04)
    fig.tight_layout()
    image_path = OUTPUT_DIR / "simulate_hrtem_experiment.png"
    exit_wave_path = OUTPUT_DIR / "simulate_hrtem_experiment_exit_wave.pt"
    fig.savefig(image_path, dpi=180)
    plt.close(fig)
    torch.save(exit_wave, exit_wave_path)

    print("mode:", result.metadata["mode"])
    print("propagation_backend:", result.metadata["propagation_backend"])
    print("detector_names:", result.detector_names)
    print("output_shapes:", result.output_shapes())
    print("exit_wave_shape:", tuple(exit_wave.shape))
    print("intermediate_keys:", sorted(result.intermediates))
    print("image_mean:", float(image_intensity.mean()))
    print("image_minmax:", (float(image_intensity.min()), float(image_intensity.max())))
    print("diffraction_max:", float(diffraction.max()))
    print("display_limits:", {"image": (image_vmin, image_vmax), "log_diffraction": (diff_vmin, diff_vmax)})
    print("saved_image:", image_path)
    print("saved_exit_wave:", exit_wave_path)


if __name__ == "__main__":
    main()
