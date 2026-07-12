#!/usr/bin/env python3
"""Validate the frozen De-NATRM model package."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from predict_package import PackagePredictor


PACKAGE_DIR = Path(__file__).resolve().parents[1]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def validate(package_dir: Path) -> dict:
    manifest = load_json(package_dir / "manifest.json")
    final_id = load_json(package_dir / "FINAL_MODEL_ID.json")
    artifacts_dir = package_dir / "artifacts"

    hash_results = {}
    for rel_path, expected_hash in manifest["artifact_hashes_sha256"].items():
        actual_hash = sha256(package_dir / rel_path)
        hash_results[rel_path] = {
            "expected": expected_hash,
            "actual": actual_hash,
            "ok": actual_hash == expected_hash,
        }

    predictor = PackagePredictor(package_dir)
    smoke_patient = {
        "Age": 50,
        "Height": 160,
        "Weight": 60,
        "BMI": 23.4,
        "mr_roi_available": 0,
        "pet_roi_available": 0,
        "roi_ambiguous_series": 1,
    }
    prediction = predictor.predict_one(smoke_patient)

    meta = load_json(artifacts_dir / "data_meta.json")
    checks = {
        "model_id_ok": final_id["model_id"] == prediction["model_id"],
        "threshold_ok": abs(float(final_id["selected_threshold"]) - 0.4630236625671386) < 1e-12,
        "num_feature_count_ok": len(meta["num_features"]) == 62,
        "roi_branch_dim_ok": int(meta["roi_branch_feature_dim"]) == 31,
        "treatments_ok": sorted(prediction["probabilities"]) == ["De-escalation", "Standard"],
        "hashes_ok": all(item["ok"] for item in hash_results.values()),
    }
    checks["all_ok"] = all(checks.values())

    return {
        "package_dir": str(package_dir),
        "checks": checks,
        "hash_results": hash_results,
        "smoke_prediction": prediction,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate the frozen pCR/NAT model package.")
    parser.add_argument("--package-dir", default=str(PACKAGE_DIR), help="Path to export/model_package.")
    parser.add_argument("--output", help="Optional JSON report path.")
    args = parser.parse_args()

    report = validate(Path(args.package_dir))
    text = json.dumps(report, ensure_ascii=False, indent=2)
    if args.output:
        Path(args.output).write_text(text + "\n", encoding="utf-8")
    print(text)
    if not report["checks"]["all_ok"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
