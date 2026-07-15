import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def load_config_module():
    spec = importlib.util.spec_from_file_location("release_config", ROOT / "config.py")
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_required_release_files_exist():
    required = [
        "LICENSE",
        "CITATION.cff",
        "CONTRIBUTING.md",
        "CODE_OF_CONDUCT.md",
        "requirements.txt",
        "README.md",
        "README.zh-CN.md",
        ".github/ISSUE_TEMPLATE/bug_report.md",
        ".github/ISSUE_TEMPLATE/feature_request.md",
        ".github/PULL_REQUEST_TEMPLATE.md",
        "docs/core_code_map.md",
        "docs/github-release/data_availability.md",
        "docs/github-release/release_checklist.md",
    ]
    for rel_path in required:
        assert (ROOT / rel_path).exists(), rel_path


def test_readmes_are_release_facing_and_portable():
    for rel_path in ["README.md", "README.zh-CN.md"]:
        text = (ROOT / rel_path).read_text(encoding="utf-8")
        assert "requirements.txt" in text
        assert "data_availability.md" in text
        assert "CITATION.cff" in text
        assert "/Users/" not in text


def test_requirements_cover_core_release_stack():
    text = (ROOT / "requirements.txt").read_text(encoding="utf-8")
    for package in ["numpy", "pandas", "scikit-learn", "torch", "openpyxl", "joblib", "pytest"]:
        assert package in text


def test_citation_file_uses_de_natrm_identity():
    text = (ROOT / "CITATION.cff").read_text(encoding="utf-8")
    assert "De-NATRM" in text
    assert "type: software" in text
    assert "license: MIT" in text
    assert "Official manuscript-linked code release" in text
    assert "<your-org-or-user>" not in text


def test_public_release_metadata_uses_professional_lineage_labels():
    final_id = (ROOT / "export" / "model_package" / "FINAL_MODEL_ID.json").read_text(encoding="utf-8")
    manifest = (ROOT / "export" / "model_package" / "manifest.json").read_text(encoding="utf-8")
    package_summary = (
        ROOT / "export" / "model_package" / "artifacts" / "experiment_results_summary.csv"
    ).read_text(encoding="utf-8-sig")

    assert "source_release_lineage" in final_id
    assert "source_code_snapshot" in final_id
    assert "source_experiment_dir" not in final_id
    assert "rollback_to_" not in final_id
    assert "source_release_lineage" in manifest
    assert "source_experiment_dir" not in manifest
    assert "rollback_to_" not in package_summary
    assert "HER2阳性-V3.0(2)" not in package_summary
    assert "authorized_dataset.xlsx" in package_summary


def test_config_respects_public_env_overrides(monkeypatch):
    monkeypatch.setenv("DE_NATRM_USE_FIXED_SPLIT_FILES", "0")
    monkeypatch.setenv("DE_NATRM_DATA_PATH", "/tmp/de_natrm_authorized_dataset.xlsx")
    monkeypatch.setenv("DE_NATRM_OUTPUT_DIR", "/tmp/de_natrm_output")

    config_module = load_config_module()

    assert config_module.USE_FIXED_SPLIT_FILES is False
    assert config_module.DATA_PATH == "/tmp/de_natrm_authorized_dataset.xlsx"
    assert config_module.OUTPUT_DIR == "/tmp/de_natrm_output"


def test_run_py_mentions_public_data_guidance():
    text = (ROOT / "run.py").read_text(encoding="utf-8")
    assert "data_availability.md" in text
    assert "DE_NATRM_DATA_PATH" in text
    assert "DE_NATRM_TRAIN_DATA_PATHS" in text


def test_config_uses_public_repo_placeholder_paths():
    text = (ROOT / "config.py").read_text(encoding="utf-8")
    assert "authorized_dataset.xlsx" in text
    assert "data_splits" in text
    assert "derived_assets" in text
    assert "de_natrm_run" in text
    assert "rollback_to_" not in text
    assert "de_natrm_public_release_snapshot_20260316" in text


def test_public_repo_omits_confidence_branch_variants():
    config_text = (ROOT / "config.py").read_text(encoding="utf-8")
    data_text = (ROOT / "data.py").read_text(encoding="utf-8")
    docs_text = (ROOT / "docs" / "core_code_map.md").read_text(encoding="utf-8")

    assert 'IMAGE_RISK_BRANCH_FEATURE_MODE = "raw"' in config_text
    assert "仅保留原始 image risk score 分支模式" in config_text
    assert "_img_risk" in data_text
    assert "interval estimation" in docs_text
