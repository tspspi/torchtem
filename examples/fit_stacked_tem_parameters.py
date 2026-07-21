from __future__ import annotations

import pathlib
import sys

import matplotlib
import torch

if __package__ in (None, ""):
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from examples.simulate_stacked_tem_stackup import build_fixed_sample
from torchtem import ActiveFixedStackTEMControls, FixedStackTEMControls, FixedStackTEMSimulator


OUTPUT_DIR = pathlib.Path(__file__).resolve().with_name("output")
FRAME_DIR = OUTPUT_DIR / "fit_stacked_tem_parameters_frames"
DIFFRACTION_WEIGHT = 0.25
EXIT_WAVE_WEIGHT = 1.0
SOURCE_WAVE_WEIGHT = 1.0


def build_simulator(
    initial_vector: torch.Tensor,
    *,
    active_names: tuple[str, ...] = (),
    bounds: dict[str, tuple[float, float]] | None = None,
) -> FixedStackTEMSimulator:
    potential, gpts, sampling = build_fixed_sample()
    if active_names:
        controls = ActiveFixedStackTEMControls(
            base_vector=initial_vector,
            active_names=active_names,
            bounds=bounds,
        )
    else:
        controls = FixedStackTEMControls(initial_vector=initial_vector)
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


def image_contrast(
    image: torch.Tensor,
    *,
    reference_mean: torch.Tensor | None = None,
) -> torch.Tensor:
    if reference_mean is None:
        reference_mean = image.mean(dim=(-2, -1), keepdim=True)
    return image / reference_mean - 1.0


def save_progress_frame(
    step: int,
    *,
    target_image: torch.Tensor,
    prediction: torch.Tensor,
    frame_dir: pathlib.Path,
) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(12.5, 4.3), dpi=160)
    target_ax, pred_ax, diff_ax = axes

    target_im = target_ax.imshow(target_image.numpy(), cmap="RdBu_r", aspect="equal")
    target_ax.set_title("Target HRTEM Contrast")
    fig.colorbar(target_im, ax=target_ax, fraction=0.046, pad=0.04)

    pred_im = pred_ax.imshow(prediction.numpy(), cmap="RdBu_r", aspect="equal")
    pred_ax.set_title(f"Prediction Step {step:03d}")
    fig.colorbar(pred_im, ax=pred_ax, fraction=0.046, pad=0.04)

    diff_im = diff_ax.imshow((prediction - target_image).numpy(), cmap="coolwarm", aspect="equal")
    diff_ax.set_title("Prediction - Target")
    fig.colorbar(diff_im, ax=diff_ax, fraction=0.046, pad=0.04)

    fig.tight_layout()
    fig.savefig(frame_dir / f"frame_{step:04d}.png", dpi=160)
    plt.close(fig)


def save_candidate_screening(
    *,
    target_image: torch.Tensor,
    target_mean: torch.Tensor,
    selected: dict[str, torch.Tensor],
    path: pathlib.Path,
) -> None:
    num_candidates = int(selected["top_scores"].shape[0])
    fig, axes = plt.subplots(1, num_candidates + 1, figsize=(4.0 * (num_candidates + 1), 4.3), dpi=160)
    target_ax = axes[0]
    target_im = target_ax.imshow(target_image.numpy(), cmap="RdBu_r", aspect="equal")
    target_ax.set_title("Target HRTEM Contrast")
    fig.colorbar(target_im, ax=target_ax, fraction=0.046, pad=0.04)

    for i in range(num_candidates):
        ax = axes[i + 1]
        prediction = selected["top_outputs"]["image"][i].detach().cpu()
        prediction_contrast = image_contrast(prediction, reference_mean=target_mean)
        im = ax.imshow(prediction_contrast.numpy(), cmap="RdBu_r", aspect="equal")
        ax.set_title(f"Candidate {i+1}\nscore={float(selected['top_scores'][i].detach()):.3e}")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def save_grid_screening_map(
    *,
    x_values: torch.Tensor,
    y_values: torch.Tensor,
    x_label: str,
    y_label: str,
    scores: torch.Tensor,
    path: pathlib.Path,
) -> None:
    fig, ax = plt.subplots(1, 1, figsize=(5.6, 4.8), dpi=170)
    im = ax.imshow(
        scores.numpy(),
        cmap="viridis",
        origin="lower",
        aspect="auto",
        extent=[
            float(x_values[0]),
            float(x_values[-1]),
            float(y_values[0]),
            float(y_values[-1]),
        ],
    )
    ax.set_title("Coarse Screening Scores")
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="MSE")
    fig.tight_layout()
    fig.savefig(path, dpi=170)
    plt.close(fig)


def save_exit_wave_fit_summary(
    *,
    history: list[float],
    target_exit_wave: torch.Tensor,
    initial_exit_wave: torch.Tensor,
    fitted_exit_wave: torch.Tensor,
    path: pathlib.Path,
) -> None:
    fig, axes = plt.subplots(1, 4, figsize=(16.0, 4.6), dpi=180)
    loss_ax, target_ax, init_ax, fit_ax = axes

    loss_ax.plot(history, color="black", linewidth=1.5)
    loss_ax.set_title("Exit-Wave Loss")
    loss_ax.set_xlabel("step")
    loss_ax.set_ylabel("MSE")

    target_amp = target_exit_wave.abs().numpy()
    initial_amp = initial_exit_wave.abs().numpy()
    fitted_amp = fitted_exit_wave.abs().numpy()

    target_im = target_ax.imshow(target_amp, cmap="magma", aspect="equal")
    target_ax.set_title("Target Exit-Wave |psi|")
    fig.colorbar(target_im, ax=target_ax, fraction=0.046, pad=0.04)

    init_im = init_ax.imshow(initial_amp, cmap="magma", aspect="equal")
    init_ax.set_title("Initial Exit-Wave |psi|")
    fig.colorbar(init_im, ax=init_ax, fraction=0.046, pad=0.04)

    fit_im = fit_ax.imshow(fitted_amp, cmap="magma", aspect="equal")
    fit_ax.set_title("Fitted Exit-Wave |psi|")
    fig.colorbar(fit_im, ax=fit_ax, fraction=0.046, pad=0.04)

    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def save_source_wave_fit_summary(
    *,
    history: list[float],
    target_source_wave: torch.Tensor,
    initial_source_wave: torch.Tensor,
    fitted_source_wave: torch.Tensor,
    path: pathlib.Path,
) -> None:
    fig, axes = plt.subplots(1, 4, figsize=(16.0, 4.6), dpi=180)
    loss_ax, target_ax, init_ax, fit_ax = axes

    loss_ax.plot(history, color="black", linewidth=1.5)
    loss_ax.set_title("Source-Wave Loss")
    loss_ax.set_xlabel("step")
    loss_ax.set_ylabel("MSE")

    target_amp = target_source_wave.abs().numpy()
    initial_amp = initial_source_wave.abs().numpy()
    fitted_amp = fitted_source_wave.abs().numpy()

    target_im = target_ax.imshow(target_amp, cmap="magma", aspect="equal")
    target_ax.set_title("Target Source |psi|")
    fig.colorbar(target_im, ax=target_ax, fraction=0.046, pad=0.04)

    init_im = init_ax.imshow(initial_amp, cmap="magma", aspect="equal")
    init_ax.set_title("Initial Source |psi|")
    fig.colorbar(init_im, ax=init_ax, fraction=0.046, pad=0.04)

    fit_im = fit_ax.imshow(fitted_amp, cmap="magma", aspect="equal")
    fit_ax.set_title("Fitted Source |psi|")
    fig.colorbar(fit_im, ax=fit_ax, fraction=0.046, pad=0.04)

    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def main() -> None:
    OUTPUT_DIR.mkdir(exist_ok=True)
    FRAME_DIR.mkdir(exist_ok=True)

    active_names = ("c1_current", "c2_current", "objective_focus", "corrector_c30")
    active_bounds = {
        "c1_current": (-1.0, 1.0),
        "c2_current": (-1.0, 1.0),
        "objective_focus": (-0.5, 0.8),
        "corrector_c30": (-0.2, 1.0),
    }
    target_vector = torch.tensor(
        [0.72, -0.28, 0.11, -0.06, 0.58, 0.05, -0.03, 0.62, -0.16, 0.08, 0.04, -0.02],
        dtype=torch.float64,
    )
    initial_vector = target_vector.clone()
    initial_vector[0] = -0.35
    initial_vector[1] = 0.44
    initial_vector[4] = -0.22
    initial_vector[7] = 0.18

    target_simulator = build_simulator(target_vector)
    target_outputs = target_simulator()
    target_raw_image = target_outputs["image"].detach()
    target_raw_diffraction = target_outputs["diffraction"].detach()
    target_image = image_contrast(target_raw_image)
    target_mean = target_raw_image.mean(dim=(-2, -1), keepdim=True).detach().cpu()
    target_exit_wave = target_outputs["exit_wave"].detach().cpu()
    target_source_wave = target_outputs["source_wave"].detach().cpu()

    coarse_c1_current = torch.linspace(-0.5, 0.9, steps=8, dtype=torch.float64)
    coarse_objective_focus = torch.linspace(-0.3, 0.8, steps=7, dtype=torch.float64)
    coarse_c2_current = torch.linspace(-0.8, 0.1, steps=8, dtype=torch.float64)
    screening_simulator = build_simulator(
        initial_vector,
        active_names=active_names,
        bounds=active_bounds,
    )
    def on_step(step: int, state: dict[str, torch.Tensor | dict[str, torch.Tensor]]) -> None:
        predicted_image = state["predicted_image"] if "predicted_image" in state else None
        if predicted_image is None:
            predicted_image = screening_simulator.image_contrast(state["outputs"]["image"])
        if step % 10 == 0 or step == 119:
            save_progress_frame(
                step,
                target_image=target_image.detach().cpu(),
                prediction=predicted_image.detach().cpu(),
                frame_dir=FRAME_DIR,
            )
        if step % 30 == 0 or step == 119:
            current = state["control_vector"].detach().cpu().tolist()
            print(
                f"step={step:03d} loss={float(state['loss'].detach()):.6e} "
                f"image_loss={float(state['image_loss'].detach()):.6e} "
                f"diffraction_loss={float(state['diffraction_loss'].detach()):.6e} "
                f"vector={current}"
            )

    coarse_to_fine = screening_simulator.coarse_to_fine_fit_from_named_parameter_grid(
        {
            "c1_current": coarse_c1_current,
            "objective_focus": coarse_objective_focus,
        },
        target_image=target_raw_image,
        target_diffraction=target_raw_diffraction,
        k=3,
        steps=120,
        lr=5e-2,
        diffraction_weight=DIFFRACTION_WEIGHT,
        callback=on_step,
    )
    screened = coarse_to_fine["screened"]
    candidate_screening_path = OUTPUT_DIR / "fit_stacked_tem_candidate_screening.png"
    save_candidate_screening(
        target_image=target_image.detach().cpu(),
        target_mean=target_mean,
        selected=screened,
        path=candidate_screening_path,
    )
    coarse_score_map = screening_simulator.score_named_parameter_grid(
        {
            "c1_current": coarse_c1_current,
            "objective_focus": coarse_objective_focus,
        },
        target_image=target_raw_image,
        target_diffraction=target_raw_diffraction,
        diffraction_weight=DIFFRACTION_WEIGHT,
    )["scores"].detach().cpu()
    candidate_grid_path = OUTPUT_DIR / "fit_stacked_tem_candidate_grid.png"
    save_grid_screening_map(
        x_values=coarse_objective_focus.detach().cpu(),
        y_values=coarse_c1_current.detach().cpu(),
        x_label="Objective focus",
        y_label="C1 current",
        scores=coarse_score_map,
        path=candidate_grid_path,
    )
    exit_wave_score_map = screening_simulator.score_named_parameter_grid(
        {
            "c1_current": coarse_c1_current,
            "c2_current": coarse_c2_current,
        },
        target_image=target_raw_image,
        target_exit_wave=target_outputs["exit_wave"].detach(),
        image_weight=0.0,
        exit_wave_weight=EXIT_WAVE_WEIGHT,
    )["scores"].detach().cpu()
    exit_wave_grid_best = screening_simulator.select_best_from_named_parameter_grid(
        {
            "c1_current": coarse_c1_current,
            "c2_current": coarse_c2_current,
        },
        target_image=target_raw_image,
        target_exit_wave=target_outputs["exit_wave"].detach(),
        image_weight=0.0,
        exit_wave_weight=EXIT_WAVE_WEIGHT,
    )
    exit_wave_candidate_grid_path = OUTPUT_DIR / "fit_stacked_tem_exit_wave_candidate_grid.png"
    save_grid_screening_map(
        x_values=coarse_c2_current.detach().cpu(),
        y_values=coarse_c1_current.detach().cpu(),
        x_label="C2 current",
        y_label="C1 current",
        scores=exit_wave_score_map,
        path=exit_wave_candidate_grid_path,
    )
    screened_initial_vector = coarse_to_fine["screened_initial_vector"].detach().cpu()
    fit_simulator = coarse_to_fine["fit_simulator"]
    optimization = coarse_to_fine
    initial_outputs = optimization["initial_outputs"]
    fitted_outputs = optimization["final_outputs"]
    history = optimization["history"]
    fitted_image = image_contrast(fitted_outputs["image"], reference_mean=target_mean.to(fitted_outputs["image"].device)).detach().cpu()
    target_image_cpu = target_image.detach().cpu()
    initial_prediction = image_contrast(initial_outputs["image"], reference_mean=target_mean.to(initial_outputs["image"].device)).detach().cpu()
    initial_exit_wave = initial_outputs["exit_wave"].detach().cpu()
    fitted_exit_wave = fitted_outputs["exit_wave"].detach().cpu()

    exit_wave_initial_vector = target_vector.clone()
    exit_wave_initial_vector[0] = -0.45
    exit_wave_initial_vector[1] = 0.35
    exit_wave_active_names = ("c1_current", "c2_current")
    exit_wave_bounds = {
        "c1_current": (-1.0, 1.0),
        "c2_current": (-1.0, 1.0),
    }
    exit_wave_simulator = build_simulator(
        exit_wave_initial_vector,
        active_names=exit_wave_active_names,
        bounds=exit_wave_bounds,
    )
    exit_wave_fit = exit_wave_simulator.optimize_to_match(
        target_image=target_raw_image,
        target_exit_wave=target_outputs["exit_wave"].detach(),
        steps=80,
        lr=5e-2,
        image_weight=0.0,
        exit_wave_weight=EXIT_WAVE_WEIGHT,
    )
    exit_wave_fit_path = OUTPUT_DIR / "fit_stacked_tem_exit_wave_fit.png"
    save_exit_wave_fit_summary(
        history=exit_wave_fit["exit_wave_history"],
        target_exit_wave=target_exit_wave,
        initial_exit_wave=exit_wave_fit["initial_outputs"]["exit_wave"].detach().cpu(),
        fitted_exit_wave=exit_wave_fit["final_outputs"]["exit_wave"].detach().cpu(),
        path=exit_wave_fit_path,
    )

    source_wave_initial_vector = target_vector.clone()
    source_wave_initial_vector[0] = -0.45
    source_wave_initial_vector[1] = 0.35
    source_wave_active_names = ("c1_current", "c2_current")
    source_wave_bounds = {
        "c1_current": (-1.0, 1.0),
        "c2_current": (-1.0, 1.0),
    }
    source_wave_simulator = build_simulator(
        source_wave_initial_vector,
        active_names=source_wave_active_names,
        bounds=source_wave_bounds,
    )
    source_wave_fit = source_wave_simulator.optimize_to_match(
        target_image=target_raw_image,
        target_source_wave=target_outputs["source_wave"].detach(),
        steps=80,
        lr=5e-2,
        image_weight=0.0,
        source_wave_weight=SOURCE_WAVE_WEIGHT,
    )
    source_wave_fit_path = OUTPUT_DIR / "fit_stacked_tem_source_wave_fit.png"
    save_source_wave_fit_summary(
        history=source_wave_fit["source_wave_history"],
        target_source_wave=target_source_wave,
        initial_source_wave=source_wave_fit["initial_outputs"]["source_wave"].detach().cpu(),
        fitted_source_wave=source_wave_fit["final_outputs"]["source_wave"].detach().cpu(),
        path=source_wave_fit_path,
    )

    fig, axes = plt.subplots(1, 4, figsize=(16.0, 4.6), dpi=180)
    loss_ax, target_ax, init_ax, fit_ax = axes

    loss_ax.plot(history, color="black", linewidth=1.5)
    loss_ax.set_title("Optimization Loss")
    loss_ax.set_xlabel("step")
    loss_ax.set_ylabel("MSE")

    target_im = target_ax.imshow(target_image_cpu.numpy(), cmap="inferno", aspect="equal")
    target_ax.set_title("Target HRTEM Contrast")
    fig.colorbar(target_im, ax=target_ax, fraction=0.046, pad=0.04)

    init_im = init_ax.imshow(initial_prediction.numpy(), cmap="inferno", aspect="equal")
    init_ax.set_title("Initial HRTEM Contrast")
    fig.colorbar(init_im, ax=init_ax, fraction=0.046, pad=0.04)

    fit_im = fit_ax.imshow(fitted_image.numpy(), cmap="inferno", aspect="equal")
    fit_ax.set_title("Fitted HRTEM Contrast")
    fig.colorbar(fit_im, ax=fit_ax, fraction=0.046, pad=0.04)

    fig.tight_layout()
    image_path = OUTPUT_DIR / "fit_stacked_tem_parameters.png"
    fig.savefig(image_path, dpi=180)
    plt.close(fig)
    target_exit_wave_path = OUTPUT_DIR / "fit_stacked_tem_parameters_target_exit_wave.pt"
    initial_exit_wave_path = OUTPUT_DIR / "fit_stacked_tem_parameters_initial_exit_wave.pt"
    fitted_exit_wave_path = OUTPUT_DIR / "fit_stacked_tem_parameters_fitted_exit_wave.pt"
    torch.save(target_exit_wave, target_exit_wave_path)
    torch.save(initial_exit_wave, initial_exit_wave_path)
    torch.save(fitted_exit_wave, fitted_exit_wave_path)

    print("target_vector:", target_vector.tolist())
    print("initial_vector:", initial_vector.tolist())
    print("active_names:", list(active_names))
    print("diffraction_weight:", DIFFRACTION_WEIGHT)
    print("screened_initial_vector:", screened_initial_vector.tolist())
    print("screened_top_scores:", screened["top_scores"].detach().cpu().tolist())
    print("saved_candidate_screening:", candidate_screening_path)
    print("saved_candidate_grid:", candidate_grid_path)
    print("saved_exit_wave_candidate_grid:", exit_wave_candidate_grid_path)
    print("exit_wave_grid_best_vector:", exit_wave_grid_best["best_vector"].detach().cpu().tolist())
    print("exit_wave_grid_best_score:", float(exit_wave_grid_best["best_score"].detach().cpu()))
    print("exit_wave_fit_initial_vector:", exit_wave_initial_vector.tolist())
    print("exit_wave_fit_final_vector:", exit_wave_fit["final_vector"].detach().cpu().tolist())
    print("exit_wave_fit_final_loss:", exit_wave_fit["exit_wave_history"][-1])
    print("saved_exit_wave_fit:", exit_wave_fit_path)
    print("source_wave_fit_initial_vector:", source_wave_initial_vector.tolist())
    print("source_wave_fit_final_vector:", source_wave_fit["final_vector"].detach().cpu().tolist())
    print("source_wave_fit_final_loss:", source_wave_fit["source_wave_history"][-1])
    print("saved_source_wave_fit:", source_wave_fit_path)
    print("fitted_vector:", fit_simulator.control_vector().detach().cpu().tolist())
    print("saved_image:", image_path)
    print("saved_frames_dir:", FRAME_DIR)
    print("saved_target_exit_wave:", target_exit_wave_path)
    print("saved_initial_exit_wave:", initial_exit_wave_path)
    print("saved_fitted_exit_wave:", fitted_exit_wave_path)


if __name__ == "__main__":
    main()
