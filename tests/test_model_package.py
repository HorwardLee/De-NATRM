import importlib.util
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PACKAGE_DIR = ROOT / "export" / "model_package"
SCRIPT_PATH = PACKAGE_DIR / "scripts" / "predict_package.py"


def load_predict_module():
    spec = importlib.util.spec_from_file_location("predict_package", SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_final_model_identity_is_release_ready():
    final_id = json.loads((PACKAGE_DIR / "FINAL_MODEL_ID.json").read_text(encoding="utf-8"))

    assert final_id["source_release_lineage"] == "de_natrm_release_candidate_pr90_export"
    assert final_id["source_code_snapshot"] == "research_snapshot_20260316"
    assert "source_experiment_dir" not in final_id
    assert "source_code_version" not in final_id
    assert final_id["roi_summary"] == {
        "causal": True,
        "primary_roi_count": 18,
        "composite_roi_count": 7,
        "pet_shape_roi_count": 6,
    }
    assert final_id["metrics"] == {
        "auroc": 0.8693,
        "auprc": 0.9076,
        "brier": 0.1946,
        "precision": 0.8649,
        "recall": 0.9412,
    }
    assert final_id["threshold_strategy"] == "precision_at_recall"
    assert abs(final_id["selected_threshold"] - 0.4630236625671386) < 1e-12


def test_package_predictor_loads_and_predicts_minimal_patient():
    module = load_predict_module()
    predictor = module.PackagePredictor(PACKAGE_DIR)
    patient = {
        "Age": 50,
        "Height": 160,
        "Weight": 60,
        "BMI": 23.4,
        "mr_roi_available": 0,
        "pet_roi_available": 0,
        "roi_ambiguous_series": 1,
    }

    result = predictor.predict_one(patient)

    assert result["model_id"] == "de_natrm_final_release_pr90_20260423"
    assert set(result["probabilities"].keys()) == {"Standard", "De-escalation"}
    assert all(0.0 <= value <= 1.0 for value in result["probabilities"].values())
    assert result["recommended_treatment"] in {"Standard", "De-escalation"}
    assert abs(result["selected_threshold"] - 0.4630236625671386) < 1e-12
    assert result["threshold_strategy"] == "precision_at_recall"


def test_predict_script_is_self_contained_after_copy():
    source = SCRIPT_PATH.read_text(encoding="utf-8")

    assert "from model import CausalNet" not in source
    assert "class CausalNet" in source
