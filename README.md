# PushT ACT Training and Evaluation

A compact, reproducible notebook that adapts the official ACT implementation to the PushT replay dataset. It covers data loading, normalization, training, rollout-based selection, and video export.

## What this repo contains
- A single notebook: `pusht_act_official_training.ipynb`
- Lightweight setup files for running locally

## Data expectations
The notebook reads the PushT Zarr replay dataset and expects these folders at the repo root:
- `ACT_pusht_task/data/` (dataset location)
- `official_act_repo_probe/` (official ACT code)

The dataset is opened in read-only mode and mapped into:
- images (pixels)
- state vectors
- action vectors
- episode boundaries

## Training flow (high-level)
1. Resolve dataset path and load arrays
2. Normalize state/action fields
3. Build a dataset wrapper and split into train/val/test
4. Configure the official ACT policy
5. Train and log metrics per epoch
6. Select the best checkpoint using rollout statistics
7. Export a sample simulator video

<!-- ## Outputs
By default, the notebook writes to:
- `outputs/official_fresh/checkpoints/`
- `outputs/official_fresh/metrics/`
- `outputs/official_fresh/videos/` -->

<!-- ## Run
Open the notebook in Jupyter or VS Code and run the cells top to bottom.

```bash
jupyter notebook
```

## Environment
Install the core dependencies with:

```bash
pip install -r requirements.txt
``` -->
