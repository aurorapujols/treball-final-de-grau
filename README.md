# Contrastive Self-Supervised Learning for Astronomical Phenomena Identification (Bachelor Thesis)
**Aurora Pujols · Universitat Pompeu Fabra · 2026**

🌐 [Project Website](https://aurorapujols.github.io/tfg-website/) &nbsp;|&nbsp; 📦 [Dataset on Kaggle](https://doi.org/10.34740/kaggle/ds/10517445) &nbsp;|&nbsp; 📄 [Thesis Report](#)

---

## Overview

This repository contains the code developed for my Bachelor's Thesis (*Treball Final de Grau*) at UPF. The goal is to automate the classification of meteor camera detections — distinguishing real meteors from non-meteor events (insects, cosmic rays, satellites, artifacts, etc.) — using **contrastive self-supervised learning** followed by lightweight downstream classifiers, as well as non-meteor subclass discovery through clustering analysis.

The project was motivated by the need to reduce the manual review workload of meteor monitoring stations, which generate large volumes of detections that require human inspection.

---

## Pipeline

The full pipeline consists of three stages:

1. **Data labeling** — binary labeling of `meteor` and `non-meteor` samples was performed by the data provider; a custom labeling tool was used to manually annotate raw non-meteor samples into subclasses.
2. **Self-supervised pre-training** — a ResNet encoder is trained with contrastive loss on the unlabeled image pool to learn visual representations without requiring labels.
3. **Downstream classification** — the frozen encoder embeddings are used to train and evaluate three classifiers: Logistic Regression, SVM, and MLP.
4. **Clustering analysis** — K-Means clustering is applied to the learned embeddings (especially on the non-meteor class) to explore the visual substructure of confounding events.

---

## Repository Structure

```
treball-final-de-grau/
│
├── labeling_tool/              # Tool used to manually label raw camera frames, and visualize classifications
│
├── notebooks/                  # Exploratory analysis and result visualisation notebooks
│
├── src/                        # Main data preprocessing code
│   ├── dataset/                
│   ├── image_preprocessing/        # Functions for video and image processing
│   ├── pipelines/                  # Files used to execute full processing pipelines and similar
│   ├── utils/                      # Utils for general tasks
│   ├── xml_processing/             # Metadata processing
│   ├── *.slurm                     # Files for job execution on the cluster
│   ├── config.*                    # Files for easier configuration of executions
│   └── main.py
│
├── my-work-dir/                # Model training, and all related code with GPU needs
│   ├── config/                     # Files for easier configuration of executions
│   ├── data/                       # Dataloaders and Datasets
│   ├── evaluation/                 # Linear probe and metrics
│   ├── experiments/                # Main files of execution for different tasks
│   ├── losses/                     # Loss functions for model training
│   ├── models/                     # Modules and model structures
│   ├── others/                     # Mainly code for state-of-the-art comparison and additional output for small tasks
│   ├── training/                   # Main training and hyperparameter tunning files
│   ├── transformations/            # Basic PyTorch transforms and Augmentations for contrastive learning
│   ├── utils/                      # Utils for plotting and others
│   ├── *.slurm                     # Files for job execution on the cluster
│   └── main.py
│
└── Dockerfile                  # Dockerfile that was used to create the container
```

---

## Running on the CSUC Cluster

All heavy training (code in the `my-work-dir` folder) was executed on the **CSUC (Consorci de Serveis Universitaris de Catalunya) HPC cluster** using a containerized environment for `python 3.10` and `cuda 12.1.0`.

The environment was set up using **Apptainer**, which allows running Docker containers on HPC systems without root access.

---

## Dataset

The dataset was collected from home meteor monitoring cameras and manually annotated using the labeling tool included in this repository.

📦 **Dataset available on Kaggle:** [SPMN Night Sky Event Dataset](https://doi.org/10.34740/kaggle/ds/10517445)

The dataset contains images labeled as:
- `meteor` — confirmed meteor trails
- `non_meteor` — false positives (insects, satellites, cosmic rays, artifacts, etc.)

Further descriptions are in the link.

---

## Project Website

A public-facing website summarising the project, methodology, and key results is available at:

👉 [https://aurorapujols.github.io/tfg-website/](https://aurorapujols.github.io/tfg-website/)

---

## License

This repository is for academic purposes. Please contact the author before reusing the code or dataset.