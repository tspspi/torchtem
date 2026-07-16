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
    GridScanConfig,
    IAMPotentialBuilder,
    MultisliceConfig,
    PixelatedDetectorConfig,
    ProbeSourceConfig,
    STEMConfig,
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
    symbols = ["Si"] * 9 + ["O"] * 4
    return ExperimentBuilder.from_stem(
        STEMConfig(
            source=ProbeSourceConfig(
                energy_eV=200e3,
                gpts=gpts,
                sampling=sampling,
                semiangle_cutoff_mrad=25.0,
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
                projection="finite",
            ),
            detector={
                "haadf": AnnularDetectorConfig(inner_mrad=25.0, outer_mrad=85.0),
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

    defocus_values_A = torch.tensor([-120.0, -40.0, 40.0, 120.0], dtype=torch.float64)
    experiment = build_experiment()
    series = experiment.run_parameter_series("source.ctf.C10", defocus_values_A)

    haadf_panels = [panel.detach().cpu().reshape(8, 8) for panel in series.outputs["haadf"]]
    diffraction_panels = [
        torch.fft.fftshift(torch.log1p(panel.detach().cpu().mean(dim=0)), dim=(-2, -1))
        for panel in series.outputs["pixelated"]
    ]

    haadf_stack = torch.stack(haadf_panels, dim=0)
    diffraction_stack = torch.stack(diffraction_panels, dim=0)
    haadf_vmin, haadf_vmax = _robust_limits(haadf_stack, lower=0.02, upper=0.99)
    diff_vmin, diff_vmax = _robust_limits(diffraction_stack, lower=0.01, upper=0.999)

    fig, axes = plt.subplots(len(defocus_values_A), 2, figsize=(9.5, 15.5), dpi=180)
    for row, defocus_A in enumerate(defocus_values_A.tolist()):
        haadf = haadf_panels[row]
        diffraction = diffraction_panels[row]

        haadf_im = axes[row, 0].imshow(
            haadf.numpy(),
            cmap="inferno",
            aspect="equal",
            vmin=haadf_vmin,
            vmax=haadf_vmax,
        )
        axes[row, 0].set_title(f"HAADF Scan, C10 = {defocus_A:.0f} Å")
        axes[row, 0].set_xlabel("scan x")
        axes[row, 0].set_ylabel("scan y")

        diff_im = axes[row, 1].imshow(
            diffraction.numpy(),
            cmap="magma",
            aspect="equal",
            vmin=diff_vmin,
            vmax=diff_vmax,
        )
        axes[row, 1].set_title(f"Mean Diffraction, C10 = {defocus_A:.0f} Å")
        axes[row, 1].set_xlabel("kx pixels")
        axes[row, 1].set_ylabel("ky pixels")

    fig.colorbar(haadf_im, ax=axes[:, 0], fraction=0.02, pad=0.02)
    fig.colorbar(diff_im, ax=axes[:, 1], fraction=0.02, pad=0.02)
    fig.tight_layout()

    image_path = OUTPUT_DIR / "simulate_stem_defocus_series.png"
    fig.savefig(image_path, dpi=180)
    plt.close(fig)

    print("defocus_values_A:", defocus_values_A)
    print("series_parameter_name:", series.parameter_name)
    print("haadf_limits:", (haadf_vmin, haadf_vmax))
    print("log_diffraction_limits:", (diff_vmin, diff_vmax))
    print("saved_image:", image_path)


if __name__ == "__main__":
    main()
