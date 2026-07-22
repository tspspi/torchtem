# TorchTEM

This repository contains `torchtem`, a fully differentiable simulator for transmission electron microscopes (TEM). It has been inspired by [abTEM](https://github.com/abTEM/abTEM) __and is based entirely on the ideas from `abTEM`__, an ab inito transmission electron microscope simulator built by a [group at University of Vienna](https://www.mostlyphysics.net/). Consider `torchtem` being derivative work by the rules of GPL, but being in _no way associated to the authors of abTEM_.

The main goal was to keep the whole implementation fully differentiable to allow backpropagation through the models and pass in all parameters of the TEM as `torch` vector too.

* [What is covered](#what-is-covered)
* [Usage](#usage)
* [Theory](#theory)
* [CoQ proofs](#coq-proofs)
* [Examples](#examples)

![](https://raw.githubusercontent.com/tspspi/torchtem/refs/heads/master/_doc/titleimage.png)

## What is covered

At the moment what is covered:

* The summary of the [used models](latex/torchtem_layers.pdf) is written
* The basis for some proofs has been defined in `proofs` but not everything has been proofen to be correct till now
* Most algorithms have been implemented in torch tensors and validated against [abTEM](https://github.com/abTEM/abTEM)
* Most of the typical models that have also been supported in [abTEM](https://github.com/abTEM/abTEM) have been implemented
* The current Bloch branch is now validated against abTEM both for the low-level structure/scattering/dynamical operators and for the current experiment-level reciprocal-intensity, reciprocal-wave, normalized annular-detector, flexible-annular, and segmented-detector outputs with separate structure-factor support and selected-beam sets

## Usage

For the currently validated simple HRTEM/TEM surface, the most direct entry points are:

* `python -m examples.simulate_stacked_tem_stackup`
* `python -m examples.fit_stacked_tem_parameters`

The fixed-stack HRTEM control example keeps the specimen fixed and optimizes a microscope control vector through a differentiable image-formation path. The current example stack contains:

* condenser-1 excitation current
* condenser-2 excitation current
* condenser stigmator `dx`, `dy`
* objective focus
* objective stigmator `dx`, `dy`
* simple corrector controls for `C30`, `C32x`, `C32y`, `C34x`, `C34y`

A minimal Python setup looks like:

```python
import torch

from torchtem import FixedStackTEMControls, FixedStackTEMSimulator, IAMPotentialBuilder

gpts = (96, 96)
sampling = (0.2, 0.2)
cell = torch.diag(torch.tensor([gpts[0] * sampling[0], gpts[1] * sampling[1], 4.0], dtype=torch.float64))

controls = FixedStackTEMControls(
    initial_vector=torch.tensor(
        [0.75, -0.35, 0.12, -0.08, 0.55, 0.07, -0.05, 0.65, -0.18, 0.09, 0.06, -0.04],
        dtype=torch.float64,
    )
)

potential = IAMPotentialBuilder(
    gpts=gpts,
    sampling=sampling,
    positions_A=torch.tensor([[2.4, 2.4, 0.7], [7.2, 7.2, 2.0]], dtype=torch.float64),
    symbols=["Si", "O"],
    cell=cell,
    slice_thickness=1.0,
    parametrization="lobato",
    projection="infinite",
)

simulator = FixedStackTEMSimulator(
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

outputs = simulator()
image = outputs["image"]
diffraction = outputs["diffraction"]
source_wave = outputs["source_wave"]
exit_wave = outputs["exit_wave"]
```

This example uses an IAM atom-array specimen, not a Gaussian surrogate. The built-in stackup sample is a small hand-built `Si/O` cluster with `13` atoms (`9 x Si`, `4 x O`) in a `19.2 Å x 19.2 Å x 4.0 Å` cell. The current fixed-stack source aperture uses `5.0 mrad` so condenser-side controls change a genuinely multi-pixel pre-specimen wave rather than collapsing to a trivial single-pixel source.

## Theory

The used theoretical models, that are based on what is implemented in [abTEM](https://github.com/abTEM/abTEM), is described and summarized in the `latex` documentation.

## CoQ proofs

The proof directory contains CoQ modules of the simulation layers that aid making sure that the model is correct. __This is mainly work in progress__.

## Examples

The currently implemented subset of examples is very narrow and simple:

* `simulate_tem_defocus_series.py` generates a sweep through different focal settings of a simple sample (see `simulate_tem_defocus_series.png`) using an independent atom model potential of a small hand-built Si/O structure
* `simulate_stem_experiment.py` generates a simple STEM scan with annular and pixelated detector outputs for a toy sample (see `simulate_stem_experiment.png`) using a Gaussian atom projection model with two synthetic scatterers
* `simulate_stem_defocus_series.py` generates a sweep through different focal settings of a simple STEM sample (see `simulate_stem_defocus_series.png`) using an independent atom model potential of a small hand-built Si/O structure
* `simulate_hrtem_experiment.py` generates a simple HRTEM image of a toy sample together with the corresponding exit wave (see `simulate_hrtem_experiment.png`) using a Gaussian atom projection model with two synthetic scatterers
* `fit_defocus_and_positions.py` fits defocus and atomic positions of a simple sample by optimizing simulated detector responses against synthetic measurements using a Gaussian atom projection model with two synthetic scatterers
* `fit_stem_experiment_parameters.py` fits parameters of a simple STEM experiment by optimizing the simulated image against synthetic reference data (see `fit_stem_experiment_parameters.png`) using a Gaussian atom projection model with two synthetic scatterers
* `simulate_stacked_tem_stackup.py` generates a fixed-stack HRTEM image-plane and momentum-plane simulation for an IAM atom-array specimen using a vector of condenser/objective/corrector controls
* `fit_stacked_tem_parameters.py` fits the fixed-stack HRTEM control vector against a target HRTEM contrast image, also exposes source-side exit-wave and source-wave fitting branches, and writes progress frames that can later be assembled into a movie

The fixed-stack inverse example currently writes:

* `examples/output/fit_stacked_tem_parameters.png`
* `examples/output/fit_stacked_tem_candidate_screening.png`
* `examples/output/fit_stacked_tem_candidate_grid.png`
* `examples/output/fit_stacked_tem_exit_wave_candidate_grid.png`
* `examples/output/fit_stacked_tem_exit_wave_fit.png`
* `examples/output/fit_stacked_tem_source_wave_fit.png`
* `examples/output/fit_stacked_tem_parameters_frames/`

The reusable fixed-stack inverse API also now accepts `target_exit_wave`, `align_exit_wave_phase`, `exit_wave_weight`, `target_source_wave`, `align_source_wave_phase`, and `source_wave_weight` so condenser-side calibration can be driven directly from complex wave targets when detector-only fitting is too ambiguous.
