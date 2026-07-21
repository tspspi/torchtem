from __future__ import annotations

import pathlib
import sys

import matplotlib
import torch

if __package__ in (None, ""):
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from torchtem import FixedStackTEMControls, FixedStackTEMSimulator, IAMPotentialBuilder


OUTPUT_DIR = pathlib.Path(__file__).resolve().with_name("output")


def build_fixed_sample():
    gpts = (96, 96)
    sampling = (0.2, 0.2)
    cell = torch.diag(torch.tensor([gpts[0] * sampling[0], gpts[1] * sampling[1], 4.0], dtype=torch.float64))
    positions_A = torch.tensor(
        [
            [2.4, 2.4, 0.7],
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
    potential = IAMPotentialBuilder(
        gpts=gpts,
        sampling=sampling,
        positions_A=positions_A,
        symbols=symbols,
        cell=cell,
        slice_thickness=1.0,
        parametrization="lobato",
        projection="infinite",
    )
    return potential, gpts, sampling


def build_simulator() -> FixedStackTEMSimulator:
    potential, gpts, sampling = build_fixed_sample()
    controls = FixedStackTEMControls(
        initial_vector=torch.tensor(
            [
                0.75,   # c1 current
                -0.35,  # c2 current
                0.12,   # condenser stig dx
                -0.08,  # condenser stig dy
                0.55,   # objective focus
                0.07,   # objective stig dx
                -0.05,  # objective stig dy
                0.65,   # corrector c30
                -0.18,  # corrector c32 x
                0.09,   # corrector c32 y
                0.06,   # corrector c34 x
                -0.04,  # corrector c34 y
            ],
            dtype=torch.float64,
        )
    )
    return FixedStackTEMSimulator(
        controls=controls,
        potential=potential,
        energy_eV=200e3,
        gpts=gpts,
        sampling=sampling,
        slice_thickness_A=1.0,
        num_slices=4,
        source_semiangle_cutoff_mrad=5.0,
        image_semiangle_cutoff_mrad=35.0,
    )


def main() -> None:
    OUTPUT_DIR.mkdir(exist_ok=True)

    simulator = build_simulator()
    outputs = simulator()
    image = outputs["image"].detach().cpu()
    image_contrast = (image / image.mean()) - 1.0
    diffraction = torch.fft.fftshift(torch.log1p(outputs["diffraction"].detach().cpu()), dim=(-2, -1))
    exit_wave = outputs["exit_wave"].detach().cpu()

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.8), dpi=180)
    image_ax, diffraction_ax = axes

    image_im = image_ax.imshow(image_contrast.numpy(), cmap="RdBu_r", aspect="equal")
    image_ax.set_title("Image Plane Contrast")
    image_ax.set_xlabel("x pixels")
    image_ax.set_ylabel("y pixels")
    fig.colorbar(image_im, ax=image_ax, fraction=0.046, pad=0.04)

    diff_im = diffraction_ax.imshow(diffraction.numpy(), cmap="magma", aspect="equal")
    diffraction_ax.set_title("Momentum Plane")
    diffraction_ax.set_xlabel("kx pixels")
    diffraction_ax.set_ylabel("ky pixels")
    fig.colorbar(diff_im, ax=diffraction_ax, fraction=0.046, pad=0.04)

    fig.tight_layout()
    image_path = OUTPUT_DIR / "simulate_stacked_tem_stackup.png"
    fig.savefig(image_path, dpi=180)
    plt.close(fig)
    exit_wave_path = OUTPUT_DIR / "simulate_stacked_tem_stackup_exit_wave.pt"
    torch.save(exit_wave, exit_wave_path)

    decoded = simulator.controls.decode()
    print("control_layout:", [entry.name for entry in simulator.control_layout()])
    print("control_vector:", simulator.control_vector().detach().cpu().tolist())
    print(
        "decoded_source_C10_C12:",
        float(decoded["source"]["C10"].detach()),
        float(decoded["source"]["C12"].detach()),
    )
    print(
        "decoded_image_C10_C30_C32:",
        float(decoded["image"]["C10"].detach()),
        float(decoded["image"]["C30"].detach()),
        float(decoded["image"]["C32"].detach()),
    )
    print("saved_image:", image_path)
    print("saved_exit_wave:", exit_wave_path)


if __name__ == "__main__":
    main()
