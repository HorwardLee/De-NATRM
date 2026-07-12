# De-NATRM Data Availability

## Summary

The public De-NATRM repository does not distribute protected clinical source tables, patient-level split spreadsheets, or patient-level derived exports.

This repository is therefore a manuscript-linked code release, not a complete public data release.

## Not Included in the Public Repository

The following asset categories are not publicly distributed here:

- raw clinical spreadsheets
- fixed train/test split spreadsheets
- patient-level derived prediction tables
- protected imaging assets or equivalent patient-linked imaging derivatives

## Why These Assets Are Not Included

These materials may contain protected or sensitive patient-linked information and therefore are not appropriate for unrestricted public distribution in the repository snapshot.

## What Is Included

The public release does include:

- the main De-NATRM training and evaluation code
- the frozen final model package under `export/model_package/`
- release-facing tests and validation commands
- documentation describing the code path and release boundaries

## What Is Required for Full Training Reproduction

To run `python run.py` for end-to-end training, you must provide your own authorized local input files.

The release-facing runtime supports the following environment variable overrides:

- `DE_NATRM_USE_FIXED_SPLIT_FILES`
- `DE_NATRM_DATA_PATH`
- `DE_NATRM_TRAIN_DATA_PATH`
- `DE_NATRM_TRAIN_DATA_PATHS`
- `DE_NATRM_TEST_DATA_PATH`
- `DE_NATRM_OUTPUT_DIR`

Recommended usage patterns:

Single-table mode:

```bash
DE_NATRM_USE_FIXED_SPLIT_FILES=0 \
DE_NATRM_DATA_PATH=/path/to/authorized_dataset.xlsx \
DE_NATRM_OUTPUT_DIR=/path/to/output_dir \
python run.py
```

Fixed-split mode:

```bash
DE_NATRM_USE_FIXED_SPLIT_FILES=1 \
DE_NATRM_TRAIN_DATA_PATHS=/path/train_batch1.xlsx:/path/train_batch2.xlsx:/path/train_batch3.xlsx \
DE_NATRM_TEST_DATA_PATH=/path/test.xlsx \
DE_NATRM_OUTPUT_DIR=/path/to/output_dir \
python run.py
```

On platforms where the path-list separator differs, use the operating system's path separator for `DE_NATRM_TRAIN_DATA_PATHS`.

## What Can Be Validated Without Protected Data

You can still validate the public release surface without retraining the full model:

```bash
python -m pytest tests/test_model_package.py tests/test_release_docs.py
python export/model_package/scripts/validate_package.py --package-dir export/model_package
python export/model_package/scripts/predict_package.py \
  --input export/model_package/examples/example_patient_input.json
```

## Maintainer Reminder

Before publishing the repository, manually review the working tree and the staged snapshot to ensure that no protected tables, derived patient-level exports, or local-only artifacts are included.
