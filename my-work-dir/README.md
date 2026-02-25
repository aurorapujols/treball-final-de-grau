# Self‑Supervised Learning Pipeline for Meteor Imagery

This repository contains a modular, research‑grade implementation of a **Self‑Supervised Learning (SSL)** pipeline designed for meteor imagery.
The project is organized into clean, purpose‑specific modules to support training, evaluation, reproducibility, and experimentation on HPC clusters using **SLURM + Apptainer/Docker**.

## 📁 Project Structure

```
my-work-dir/
│
├── config/
│   ├── ssl_v1x.yaml
│   └── ssl_v2x.yaml
│
├── data/
│   ├── collate.py
│   ├── dataloaders.py
│   └── datasets.py
│
├── evaluation/
│   └── linear_probe.py
│
├── experiments/
│   └── run_ssl.py
│
├── logs/
│
├── losses/
│   └── contrastive_loss.py
│
├── models/
│   ├── modules.py
│   └── ssl_model.py
│
├── training/
│   └── ssl_training.py
│
├── transformations/
│   ├── augment.py
│   └── transform.py
│
├── utils/
│   ├── checkpoint.py
│   ├── plotting.py
│   └── seed.py
│
├── main.py
├── README.md
├── run_experiment.slurm
└── vgg-faiss-gpu.sif
```

---

## 📦 Directory Overview

``config/``

Experiment configuration files in YAML format.
Each file defines hyperparameters, dataset paths, model dimensions, augmentation settings, and output directories.

+ ``ssl_v1x.yaml`` — first‑generation SSL experiments
+ ``ssl_v2x.yaml`` — updated experiments with improved augmentations or model settings
---
``data/``

Everything related to dataset loading and preprocessing.

+ ``datasets.py`` — dataset classes for labeled and unlabeled meteor images
+ ``collate.py`` — custom collate functions (e.g., padding variable‑sized images)
+ ``dataloaders.py`` — functions that build PyTorch DataLoaders
---
``evaluation/``

Downstream evaluation modules.

+ ``linear_probe.py`` — trains and evaluates a logistic regression classifier on frozen backbone features
---
``experiments/``

High‑level experiment runners.

+ ``run_ssl.py`` — loads config, builds model + dataloaders, and launches SSL training
---
``logs/``

Automatically created directory for:

+ training logs
+ debug images
+ metrics
+ checkpoints (if configured)
---
``losses/``

Loss functions used in SSL.

+ ``contrastive_loss.py`` — NT‑Xent / SimCLR‑style contrastive loss
---
``models/``

Model architectures and building blocks.

+ ``modules.py`` — backbone, projection head, and reusable components
+ ``ssl_model.py`` — full SSL model combining backbone + projection head
---
``training/``

Training logic.

+ ``ssl_training.py`` — main SSL training loop, LR scheduling, early stopping, feature extraction, and evaluation hooks
---
``transformations/``

Image preprocessing and augmentations.

+ ``augment.py`` — custom augmentations (e.g., RandomAffineMeanFill, ControlledAugment)
+ ``transform.py`` — deterministic transforms for inference
---
``utils/``

General utilities.

+ ``checkpoint.py`` — save/load model checkpoints
+ ``plotting.py`` — plotting utilities (e.g., triplet debug visualization)
+ ``seed.py`` — reproducibility helpers
---
``main.py``

Entry point for running experiments locally or inside a container.
Loads a config file and dispatches to run_ssl.py.
---
``run_experiment.slurm``

SLURM job script for running SSL experiments on an HPC cluster using Apptainer/Docker.
---
``vgg-faiss-gpu.sif``

Apptainer container image containing the full GPU‑enabled environment (PyTorch, CUDA, FAISS, etc.).
---

## 🚀 Running an Experiment

### Local or container execution

```bash
python main.py --config config/ssl_v2x.yaml
```
### SLURM submission

```bash
sbatch run_experiment.slurm
```
The SLURM script automatically:

+ loads Apptainer
+ binds the working directory
+ runs the experiment inside the container
+ logs outputs to ``logs/``