# TorchTEM

This repository contains `torchtem`, a fully differentiable simulator for transmission electron microscopes (TEM). It has been inspired by [abTEM](https://github.com/abTEM/abTEM) __and is based entirely on the ideas from `abTEM`__, an ab inito transmission electron microscope simulator built by a [group at University of Vienna](https://www.mostlyphysics.net/).

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

The proof directory contains CoQ modules of the simulation layers that aid making sure that the model is correct.
