# De-NATRM Core Code Map

This public release repository is intentionally centered on two layers:

1. The reusable De-NATRM training and inference pipeline.
2. The frozen final model export package.

Broader private experiment history, exploratory scripts, and internal figure workspaces are not part of this release repository.

## 1. Core Pipeline

The main execution chain is:

```text
config.py -> data.py -> model.py -> train.py -> run.py
```

### `config.py`

Responsibilities:

- Central experiment configuration
- Raw data path and fixed split path selection
- Feature-family switches
- ROI and image-risk branch controls
- Model hyperparameters
- Loss weights and threshold strategy

Why it matters:

This file defines the exact runtime regime. For manuscript-linked reproducibility review, it is the first place reviewers and collaborators will inspect.

### `data.py`

Responsibilities:

- Read authorized Excel tables
- Detect outcome and treatment columns
- Build fixed train/test splits
- Assemble tabular, image-risk, and ROI features
- Fit and save the sklearn preprocessor
- Export `data_meta.json`, treatment maps, and feature order metadata

Why it matters:

This is the most important non-model file in the repository. It determines what the model actually sees.

### `model.py`

Responsibilities:

- Define `CausalNet`
- Implement single-branch and multi-branch encoders
- Support clinical-only, clinical + image, clinical + ROI, and tri-branch settings
- Provide treatment-specific prediction heads
- Implement `mmd_rbf_multi`

Why it matters:

This is the architectural core of De-NATRM.

### `train.py`

Responsibilities:

- Wrap numpy arrays as `TorchDataset`
- Define the `Trainer`
- Compute factual loss
- Compute MMD loss
- Compute counterfactual consistency loss through nearest-neighbor matching
- Run optimization and training loops

Why it matters:

This file contains the real learning objective, not just the optimizer boilerplate.

### `run.py`

Responsibilities:

- Set seeds
- Call `load_and_prepare()`
- Build the model and trainer
- Handle validation split and optional sampler logic
- Train the model
- Evaluate train/test performance
- Export plots, metrics, thresholds, and prediction tables

Why it matters:

This is the default end-to-end entrypoint for De-NATRM.

## 2. Supporting Core Files

### `utils.py`

- Evaluation metrics
- ROC/PR/calibration plotting
- Optional bootstrap AUROC interval estimation

### `predictor.py`

- Detailed per-treatment prediction tables
- Recommendation reports
- Heatmap matrix export

## 3. Final Export Package

Directory:

`export/model_package/`

Purpose:

- Freeze a release-ready inference package for the final De-NATRM model
- Store metadata, schemas, examples, and validation reports
- Provide standalone prediction and package-validation scripts

Most relevant files:

- `export/model_package/scripts/predict_package.py`
- `export/model_package/scripts/validate_package.py`
- `export/model_package/manifest.json`
- `export/model_package/FINAL_MODEL_ID.json`
- `export/model_package/README.md`

## 4. Public Release Reading Order

Recommended reading order:

1. `README.md`
2. `docs/core_code_map.md`
3. `config.py`
4. `data.py`
5. `model.py`
6. `train.py`
7. `run.py`
8. `export/model_package/`
9. `tests/test_model_package.py`

## 5. Release Boundary Review

- Raw `.xlsx` clinical tables
- Generated prediction CSVs
- User-supplied derived assets under local data directories
- Local output folders
- Any report that may contain patient-identifiable metadata

Also review:

- `docs/github-release/data_availability.md`
- `docs/github-release/release_checklist.md`

## 6. Naming Guidance

For external release, use:

- Model name: `De-NATRM`
- Architecture implementation name: `CausalNet`

That split is useful because `De-NATRM` is the paper/repository identity, while `CausalNet` remains the concrete module/class name inside the code.
