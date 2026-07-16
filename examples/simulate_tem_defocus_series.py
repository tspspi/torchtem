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
    AberrationCoefficients,
    ExperimentBuilder,
    HRTEMConfig,
    IAMPotentialBuilder,
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


def build_experiment(*, image_defocus_A: float) -> ExperimentBuilder:
    gpts = (96, 96)
    sampling = (0.2, 0.2)
    cell = torch.diag(torch.tensor([gpts[0] * sampling[0], gpts[1] * sampling[1], 4.0], dtype=torch.float64))
    positions_A = torch.tensor(
        [
            [2.4, 2.4, 0.8],
            [2.4, 7.2, 2.2],
            [2.4, 12.0, 1.1],
            [7.2, 2.4, 2.0],
            [7.2, 7.2, 0.9],
            [7.2, 12.0, 2.4],
            [12.0, 2.4, 1.3],
            [12.0, 7.2, 2.7],
            [12.0, 12.0, 1.6],
            [4.8, 4.8, 3.1],
            [9.6, 4.8, 3.3],
            [4.8, 9.6, 3.0],
            [9.6, 9.6, 3.2],
        ],
        dtype=torch.float64,
    )
    symbols = (
        ["Si"] * 9
        + ["O"] * 4
    )
    return ExperimentBuilder.from_hrtem(
        HRTEMConfig(
            source=PlaneWaveSourceConfig(
                gpts=gpts,
                sampling=sampling,
                energy_eV=200e3,
            ),
            multislice=MultisliceConfig(
                energy_eV=200e3,
                gpts=gpts,
                sampling=sampling,
                slice_thickness_A=1.0,
                num_slices=4,
                backend="fourier",
            ),
            potential=IAMPotentialBuilder(
                gpts=gpts,
                sampling=sampling,
                positions_A=positions_A,
                symbols=symbols,
                cell=cell,
                slice_thickness=1.0,
                parametrization="lobato",
                projection="infinite",
            ),
            detector={
                "image": ImageDetectorConfig(
                    output="intensity",
                    semiangle_cutoff_mrad=35.0,
                    aberrations=AberrationCoefficients(C10=image_defocus_A, C30=1.0e4),
                ),
                "diffraction": PixelatedDetectorConfig(),
            },
        )
    )


def main() -> None:
    OUTPUT_DIR.mkdir(exist_ok=True)

    defocus_values_A = torch.tensor([-180.0, -60.0, 60.0, 180.0], dtype=torch.float64)
    experiment = build_experiment(image_defocus_A=float(defocus_values_A[0].item()))
    series = experiment.run_parameter_series("image.ctf.C10", defocus_values_A)
    image_panels = [panel.detach().cpu() for panel in series.outputs["image"]]
    image_contrast_panels = [(panel / panel.mean()) - 1.0 for panel in image_panels]
    diffraction_panels = [
        torch.fft.fftshift(torch.log1p(panel.detach().cpu()), dim=(-2, -1))
        for panel in series.outputs["diffraction"]
    ]

    image_stack = torch.stack(image_contrast_panels, dim=0)
    diffraction_stack = torch.stack(diffraction_panels, dim=0)
    image_vmin, image_vmax = _robust_limits(image_stack, lower=0.01, upper=0.995)
    image_abs = max(abs(image_vmin), abs(image_vmax))
    image_vmin, image_vmax = -image_abs, image_abs
    diff_vmin, diff_vmax = _robust_limits(diffraction_stack, lower=0.01, upper=0.999)

    fig, axes = plt.subplots(len(defocus_values_A), 2, figsize=(9.0, 15.5), dpi=180)
    for row, defocus_A in enumerate(defocus_values_A.tolist()):
        image = image_contrast_panels[row]
        diffraction = diffraction_panels[row]

        image_im = axes[row, 0].imshow(
            image.numpy(),
            cmap="RdBu_r",
            aspect="equal",
            vmin=image_vmin,
            vmax=image_vmax,
        )
        axes[row, 0].set_title(f"Image Plane Contrast, C10 = {defocus_A:.0f} Å")
        axes[row, 0].set_xlabel("x pixels")
        axes[row, 0].set_ylabel("y pixels")

        diff_im = axes[row, 1].imshow(
            diffraction.numpy(),
            cmap="magma",
            aspect="equal",
            vmin=diff_vmin,
            vmax=diff_vmax,
        )
        axes[row, 1].set_title(f"Diffraction Plane, C10 = {defocus_A:.0f} Å")
        axes[row, 1].set_xlabel("kx pixels")
        axes[row, 1].set_ylabel("ky pixels")

    fig.colorbar(image_im, ax=axes[:, 0], fraction=0.02, pad=0.02)
    fig.colorbar(diff_im, ax=axes[:, 1], fraction=0.02, pad=0.02)
    fig.tight_layout()

    image_path = OUTPUT_DIR / "simulate_tem_defocus_series.png"
    fig.savefig(image_path, dpi=180)
    plt.close(fig)

    print("defocus_values_A:", defocus_values_A)
    print("series_parameter_name:", series.parameter_name)
    print("image_relative_stds:", [float(panel.std()) for panel in image_contrast_panels])
    print("image_limits:", (image_vmin, image_vmax))
    print("log_diffraction_limits:", (diff_vmin, diff_vmax))
    print("saved_image:", image_path)


if __name__ == "__main__":
    main()
