# De-NATRM

De-NATRM is the official code repository accompanying the manuscript on deep causal treatment recommendation for HER2-positive breast cancer neoadjuvant therapy. The repository provides the main training pipeline, the frozen final model package, and the release-facing validation materials intended for manuscript-linked review and reuse.

## Release Snapshot

- Status: official manuscript-linked code release
- Version: `v1.0.0`
- License: `MIT`
- Primary entry: `run.py`
- Frozen package: `export/model_package/`

This public repository is intentionally limited to the De-NATRM main model release surface. It does not distribute the broader private research workspace, including exploratory experiment history, protected clinical spreadsheets, figure-production workspaces, or auxiliary analysis utilities.

## Repository Scope

This public repository is intended to support:

- code inspection
- manuscript-linked reproducibility review
- packaged inference validation
- repository-level release validation

Protected clinical source tables are not distributed in this release. Full end-to-end training reproduction therefore requires authorized user-supplied data assets.

## Main Model Path

The main De-NATRM training path is:

```text
config.py -> data.py -> model.py -> train.py -> run.py
```

- `config.py` defines runtime configuration and public-facing path overrides.
- `data.py` loads authorized tabular inputs, preprocesses features, and exports metadata.
- `model.py` implements `CausalNet`, the core De-NATRM architecture.
- `train.py` implements factual, MMD, and counterfactual-consistency training.
- `run.py` is the main end-to-end training and evaluation entrypoint.
- `export/model_package/` contains the frozen final model package and package validation scripts.

## Installation

Install the primary release dependencies with:

```bash
pip install -r requirements.txt
```

## Data Availability

The repository does not distribute protected clinical source tables, fixed split spreadsheets, or patient-level derived exports needed for full training reproduction.

Read [docs/github-release/data_availability.md](docs/github-release/data_availability.md) before attempting training.

If you have authorized local data, the main training entry can be configured through environment variables:

```bash
DE_NATRM_USE_FIXED_SPLIT_FILES=0 \
DE_NATRM_DATA_PATH=/path/to/authorized_dataset.xlsx \
DE_NATRM_OUTPUT_DIR=/path/to/output_dir \
python run.py
```

For fixed-split mode, the public release also supports:

```bash
DE_NATRM_USE_FIXED_SPLIT_FILES=1 \
DE_NATRM_TRAIN_DATA_PATHS=/path/train_batch1.xlsx:/path/train_batch2.xlsx:/path/train_batch3.xlsx \
DE_NATRM_TEST_DATA_PATH=/path/test.xlsx \
DE_NATRM_OUTPUT_DIR=/path/to/output_dir \
python run.py
```

## Validation

Run the release-facing checks:

```bash
python -m pytest tests/test_model_package.py tests/test_release_docs.py
python export/model_package/scripts/validate_package.py --package-dir export/model_package
```

Run the packaged inference example:

```bash
python export/model_package/scripts/predict_package.py \
  --input export/model_package/examples/example_patient_input.json
```

## Release Documents

- [Core code map](docs/core_code_map.md)
- [Data availability note](docs/github-release/data_availability.md)
- [Maintainer release checklist](docs/github-release/release_checklist.md)
- [Model asset audit](docs/model_asset_audit.md)
- [Final model package README](export/model_package/README.md)
- [Chinese README](README.zh-CN.md)

## Citation

If you use this repository in research, please cite the associated manuscript and the repository metadata in [CITATION.cff](CITATION.cff).

## License

This repository is released under the [MIT License](LICENSE).
