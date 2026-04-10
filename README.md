# Robust 3D Vessel Aneurysm Classification: Voxel vs Point Representations

This repository contains the final code and generated outputs for the ECE1508 project on binary classification of 3D intracranial vessel segments as **normal vessel** or **aneurysm** using **VesselMNIST3D**.

## Repository Contents

- `ECE1508_vesselmnist3d_voxel_vs_pointnet_project_fixed.ipynb` — final notebook with data loading, model training, robustness benchmarking, multi-seed evaluation, plotting, and export utilities.
- `requirements.txt` — Python packages required to run the notebook.
- `output/` — extracted experiment artifacts from the current result bundle, including:
  - `checkpoints/` trained model weights
  - `figures/` generated plots
  - `tables/` CSV summaries and benchmark outputs
  - `run_artifact_summary.json` run metadata

## Project Summary

The project compares two representations of the same 3D medical input:

1. **Voxel pipeline**: a compact 3D CNN operating directly on the dense 3D volume.
2. **Point pipeline**: voxel-to-point conversion followed by a PointNet-style classifier.

The experiments evaluate both **clean performance** and **robustness under distribution shift**, using controlled:
- 3D rotations: **0°, 15°, 30°, 45°**
- Additive Gaussian noise: **0.00, 0.03, 0.06, 0.10** standard deviation

The notebook also includes:
- threshold calibration on the validation set
- richer evaluation metrics (AUROC, accuracy, balanced accuracy, precision, recall, F1)
- test-time augmentation (TTA)
- multi-seed experiments and seed-averaged summaries
- report-ready plots and zipped export of outputs

## Setup

### Option 1: Google Colab

1. Open `ECE1508_vesselmnist3d_voxel_vs_pointnet_project_fixed.ipynb` in Google Colab.
2. Run the installation/setup cells at the top of the notebook.
3. Run the notebook from top to bottom.

### Option 2: Local Python Environment

1. Create and activate a Python 3.10+ environment.
2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Launch Jupyter:

```bash
jupyter notebook
```

4. Open `ECE1508_vesselmnist3d_voxel_vs_pointnet_project_fixed.ipynb` and run all cells.

## Usage Notes

- The notebook is the **main demo and sample input-output artifact** for the project.
- By default, it creates a working directory under `/content/vesselmnist3d_project/` in Colab.
- Generated checkpoints, CSV tables, and figures are saved under the notebook's output directory.
- The current repository already includes an extracted snapshot of the latest generated outputs in `output/`.

## Key Output Files

Useful artifacts inside `output/` include:

- `tables/clean_summary.csv`
- `tables/robust_vs_tta_comparison.csv`
- `tables/seed_benchmark_mean_std.csv`
- `figures/main_comparison_auroc_curves.png`
- `figures/main_comparison_accuracy_curves.png`
- `figures/clean_main_metrics_bars.png`
- `figures/seed_mean_std_auroc.png`
- `figures/seed_mean_std_accuracy.png`

## Reproducibility

- The notebook sets global random seeds.
- Multi-seed evaluation is included to reduce dependence on a single run.
- Results may still vary slightly depending on hardware, CUDA version, and runtime environment.

