# Contributing to De-NATRM

Thank you for your interest in improving De-NATRM.

## Scope

This repository is a paper-aligned research code release. Contributions are most helpful when they improve:

- documentation clarity
- reproducibility and setup
- testing and validation
- packaging and release hygiene
- bug fixes that do not change the scientific intent without discussion

Large scientific changes, training redesigns, or changes that alter reported paper behavior should be discussed in an issue before implementation.

## Before Opening a Pull Request

1. Read `README.md` and `docs/github-release/data_availability.md`.
2. Confirm that your change does not add protected clinical data, derived patient tables, or local-only artifacts.
3. Run the relevant validation commands for the files you changed.

Recommended validation:

```bash
python -m pytest tests/test_model_package.py tests/test_release_docs.py
python export/model_package/scripts/validate_package.py --package-dir export/model_package
```

## Pull Request Expectations

- Keep changes focused and reviewable.
- Update documentation when behavior or usage changes.
- Preserve the public De-NATRM naming consistently.
- Explain any reproducibility impact.
- Do not commit raw clinical spreadsheets, patient-level exports, or local cache files.

## Issues

Please use GitHub Issues for:

- reproducibility problems
- documentation gaps
- packaging bugs
- release-surface inconsistencies

When reporting a bug, include the command you ran, the expected behavior, and the observed behavior.

## Data and Privacy

This public repository does not distribute protected clinical source data. By contributing, you agree not to upload:

- raw patient tables
- identifiable imaging data
- generated files containing protected patient-level content
- private local paths or credentials

## Review Standard

Changes should make the repository easier to understand, safer to release, or easier to validate. When in doubt, prefer smaller, well-documented pull requests.
