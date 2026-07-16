# TorchTEM

This repository contains `torchtem`, a fully differentiable simulator for transmission electron microscopes (TEM). It has been inspired by [abTEM](https://github.com/abTEM/abTEM) __and is based entirely on the ideas from `abTEM`__, an ab inito transmission electron microscope simulator built by a [group at University of Vienna](https://www.mostlyphysics.net/). Consider `torchtem` being derivative work by the rules of GPL.

The main goal was to keep the whole implementation fully differentiable to allow backpropagation through the models and pass in all parameters of the TEM as `torch` vector too.

## What is covered

At the moment what is covered:

* The summary of the [used models](latex/torchtem_layers.pdf) is written
* The basis for some proofs has been defined in `proofs` but not everything has been proofen to be correct till now
* Most algorithms have been implemented in torch tensors and validated against [abTEM](https://github.com/abTEM/abTEM)
* Most of the typical models that have also been supported in [abTEM](https://github.com/abTEM/abTEM) have been implemented

## Usage

__ToDo__

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
