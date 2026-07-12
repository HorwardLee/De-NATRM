# De-NATRM GitHub Release Checklist

Use this checklist before publishing the repository as the manuscript-linked GitHub release.

## Legal and Citation

- [ ] `LICENSE` is present and matches the intended public distribution terms.
- [ ] `CITATION.cff` is present and uses the De-NATRM release identity.
- [ ] Root README citation guidance matches `CITATION.cff`.

## Documentation

- [ ] `README.md` clearly explains what De-NATRM is, what the repository contains, and how to validate the public release.
- [ ] `README.zh-CN.md` is consistent with the English release-facing README.
- [ ] `docs/core_code_map.md` clearly points readers to the main model path.
- [ ] `docs/github-release/data_availability.md` clearly explains which assets are not distributed.

## Repository Hygiene and Privacy

- [ ] The intended public snapshot does not include raw clinical spreadsheets.
- [ ] The intended public snapshot does not include patient-level derived exports.
- [ ] The intended public snapshot does not include private local cache or temporary files.
- [ ] `.gitignore` reflects the intended public snapshot boundary.
- [ ] `git status --short` has been manually reviewed before publication.

## Setup and Runtime Clarity

- [ ] `requirements.txt` exists and is referenced by the root READMEs.
- [ ] The training entrypoint documents how authorized local data should be provided.
- [ ] Missing protected input files produce explicit guidance rather than silent assumptions.

## Validation

- [ ] `python -m pytest tests/test_model_package.py tests/test_release_docs.py` passes.
- [ ] `python export/model_package/scripts/validate_package.py --package-dir export/model_package` passes.
- [ ] `python export/model_package/scripts/predict_package.py --input export/model_package/examples/example_patient_input.json` runs successfully.

## Participation Metadata

- [ ] `CONTRIBUTING.md` is present.
- [ ] `CODE_OF_CONDUCT.md` is present.
- [ ] GitHub issue templates are present.
- [ ] GitHub pull request template is present.

## Final Publication Review

- [ ] The public-facing De-NATRM naming is consistent across repository docs and the frozen package docs.
- [ ] The main repository view emphasizes the core code path and release materials over internal research clutter.
- [ ] The repository is acceptable to link directly from the manuscript in its current public form.
