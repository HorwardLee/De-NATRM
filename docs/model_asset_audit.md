# De-NATRM Model Asset Audit

## Final Export Target

The final frozen model package corresponds to the final release candidate recorded in the package metadata.

This is the final public-release model configuration:

| Component | Status |
| --- | --- |
| Causal model | with |
| 18 primary ROI | with |
| 7 composite ROI | with |
| 6 PET shape ROI | with |
| AUROC | 0.8693 |
| AUPRC | 0.9076 |
| Brier | 0.1946 |
| Precision | 0.8649 |
| Recall | 0.9412 |

The final De-NATRM package is under:

`export/model_package/`

Within the public De-NATRM release, this package is the main lightweight validation surface that can be checked without retraining the full model.

## Why Precision/Recall Are 0.8649/0.9412

The final release uses `threshold_strategy=precision_at_recall`, `target_recall=0.90`, and `selected_threshold=0.4630236625671386`.

An alternative fixed-threshold export shares the same model weights, preprocessor, and probability outputs, but reports Precision and Recall at threshold `0.5`. The public release adopts the precision-recall-targeted threshold configuration, so the package metadata records that release lineage as the authoritative export target.

Evidence copied into the package:

- `artifacts/test_metrics.txt`: AUROC 0.8693, AUPRC 0.9076, Brier 0.1946
- `artifacts/test_per_treatment_precision_recall.csv`: Overall Precision 0.8649, Recall 0.9412
- `artifacts/experiment_results_summary.csv`: selected threshold and hyperparameters

## Model Architecture

- Model class: `CausalNet`
- Weight file: `artifacts/causal_net.pt`
- Weight format: PyTorch `state_dict`
- Input dimension: 62
- Treatments: 2
- Treatment heads: `Standard`, `De-escalation`
- Representation dimension: 32
- Encoder hidden layers: `[64, 32]`
- Head hidden layers: `[8]`
- Dropout: 0.43
- BatchNorm: enabled
- Branch mode: clinical/table branch + ROI branch
- ROI branch dimension: 31
- Non-ROI numeric dimension: 31

## Preprocessing Assets

- Preprocessor: `artifacts/preprocessor.joblib`
- Type: sklearn `ColumnTransformer`
- Numeric pipeline: median imputation, then standard scaling
- Categorical pipeline: none in the final encoded schema
- Numeric features: 62
- Categorical features: 0

Special numeric strings are cleaned conservatively before inference:

- `<1.2` becomes `1.2`
- `>100` becomes `100`
- `*`, `＊`, `未测`, empty strings become missing

Missing values are left as missing for the saved median imputer.

## Feature Groups

Clinical/table features: 31 fields, listed in `artifacts/feature_order.json`.

ROI features: 31 fields:

- 18 primary ROI features
- 7 primary composite ROI features
- 6 selected PET shape radiomics features

Selected PET shape features:

- `pet_rad_original_shape_surfacevolumeratio`
- `pet_rad_original_shape_leastaxislength`
- `pet_rad_original_shape_minoraxislength`
- `pet_rad_original_shape_maximum2ddiameterslice`
- `pet_rad_original_shape_elongation`
- `pet_rad_original_shape_maximum2ddiameterrow`

ROI branch gating:

- valid ROI requires `mr_roi_available == 1`, `pet_roi_available == 1`, `roi_ambiguous_series == 0`
- if invalid, the ROI branch encoded features are zeroed before inference

## Treatment and Recommendation Contract

The package predicts pCR probabilities for:

- `Standard`
- `De-escalation`

The recommendation is the treatment head with the higher predicted pCR probability. The package does not directly support concrete regimen labels such as `TCbHP`, `THP`, or `HP`; those labels need an external, clinically reviewed mapping before migration.

## Exported Package Files

- `README.md`
- `FINAL_MODEL_ID.json`
- `manifest.json`
- `artifacts/causal_net.pt`
- `artifacts/preprocessor.joblib`
- `artifacts/data_meta.json`
- `artifacts/treatment_map.json`
- `artifacts/treatments_keep.json`
- `artifacts/feature_order.json`
- `artifacts/label_mapping.json`
- `artifacts/regimen_mapping.json`
- `schemas/input_schema.json`
- `schemas/output_schema.json`
- `examples/example_patient_input.json`
- `examples/example_prediction_output.json`
- `scripts/predict_package.py`
- `scripts/validate_package.py`
- `reports/parity_check.md`

## Non-Goals

This export does not retrain the model, retune thresholds, alter the training code, add external agent logic, or define clinical regimen mappings.
