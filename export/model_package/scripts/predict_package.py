#!/usr/bin/env python3
"""Self-contained inference entrypoint for the frozen De-NATRM model package."""

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn


PACKAGE_DIR = Path(__file__).resolve().parents[1]


def mlp(in_dim: int, hidden: List[int], out_dim: int, dropout: float = 0.0, bn: bool = False) -> nn.Sequential:
    layers: List[nn.Module] = []
    last = in_dim
    for h in hidden:
        layers.append(nn.Linear(last, h))
        if bn:
            layers.append(nn.BatchNorm1d(h))
        layers.append(nn.ReLU(inplace=True))
        if dropout > 0:
            layers.append(nn.Dropout(dropout))
        last = h
    layers.append(nn.Linear(last, out_dim))
    return nn.Sequential(*layers)


class CausalNet(nn.Module):
    """Inference-only copy of the De-NATRM CausalNet architecture."""

    def __init__(
        self,
        in_dim: int,
        rep_dim: int = 128,
        enc_hidden: List[int] = [256, 128],
        head_hidden: List[int] = [64, 32],
        num_treatments: int = 3,
        dropout: float = 0.1,
        bn: bool = True,
        num_features: Optional[int] = None,
        image_feature_dim: int = 0,
        secondary_feature_dim: int = 0,
        pet_feature_dim: int = 0,
        roi_feature_dim: int = 0,
        use_roi_branch: bool = False,
    ) -> None:
        super().__init__()

        self.num_treatments = num_treatments
        self.num_features = num_features
        self.image_feature_dim = int(image_feature_dim) if image_feature_dim is not None else 0
        self.secondary_feature_dim = int(secondary_feature_dim) if secondary_feature_dim is not None else 0
        self.pet_feature_dim = int(pet_feature_dim) if pet_feature_dim is not None else 0
        self.roi_feature_dim = int(roi_feature_dim) if roi_feature_dim is not None else 0
        self.use_roi_branch = bool(use_roi_branch)

        self.branch_mode = "single"
        explicit_image_dim = 0
        if self.num_features is not None and 0 < self.image_feature_dim < self.num_features:
            explicit_image_dim = self.image_feature_dim

        explicit_roi_dim = 0
        if self.use_roi_branch and self.num_features is not None and 0 < self.roi_feature_dim < self.num_features:
            explicit_roi_dim = self.roi_feature_dim

        if (
            self.num_features is not None
            and explicit_image_dim > 0
            and explicit_roi_dim > 0
            and explicit_image_dim + explicit_roi_dim < self.num_features
        ):
            self.branch_mode = "tri"
            self.image_feature_dim = explicit_image_dim
            self.roi_feature_dim = explicit_roi_dim
        elif explicit_roi_dim > 0:
            self.branch_mode = "roi"
            self.secondary_feature_dim = explicit_roi_dim
            self.roi_feature_dim = explicit_roi_dim
            self.image_feature_dim = 0
        elif explicit_image_dim > 0:
            self.branch_mode = "image"
            self.secondary_feature_dim = explicit_image_dim
            self.image_feature_dim = explicit_image_dim
            self.roi_feature_dim = 0
        elif self.num_features is not None and 0 < self.secondary_feature_dim < self.num_features:
            self.branch_mode = "secondary"
            self.image_feature_dim = self.secondary_feature_dim
            self.roi_feature_dim = 0
        elif self.num_features is not None and 0 < self.pet_feature_dim < self.num_features:
            self.branch_mode = "pet"
            self.secondary_feature_dim = self.pet_feature_dim
            self.image_feature_dim = self.pet_feature_dim
            self.roi_feature_dim = 0

        if self.branch_mode == "tri":
            self.secondary_feature_dim = int(self.image_feature_dim + self.roi_feature_dim)
        elif self.branch_mode != "single":
            self.secondary_feature_dim = int(max(self.secondary_feature_dim, self.image_feature_dim, self.roi_feature_dim))
        self.pet_feature_dim = int(max(self.pet_feature_dim, self.secondary_feature_dim))
        self.use_multi_branch = self.branch_mode != "single"

        if self.use_multi_branch:
            self.num_num_features = int(self.num_features)
            self.cat_dim = max(0, in_dim - self.num_num_features)

            if self.branch_mode == "tri":
                self.tab_num_dim = self.num_num_features - self.image_feature_dim - self.roi_feature_dim
                self.in_tab = self.tab_num_dim + self.cat_dim
                self.in_img = self.image_feature_dim
                self.in_roi = self.roi_feature_dim

                self.rep_tab = max(4, rep_dim // 2)
                rep_aux = max(2, rep_dim - self.rep_tab)
                self.rep_img = max(1, rep_aux // 2)
                self.rep_roi = max(1, rep_aux - self.rep_img)
                self.rep_dim = self.rep_tab + self.rep_img + self.rep_roi

                self.encoder_tab = mlp(self.in_tab, enc_hidden, self.rep_tab, dropout=dropout, bn=bn)
                self.encoder_img = mlp(self.in_img, enc_hidden, self.rep_img, dropout=dropout, bn=bn)
                self.encoder_roi = mlp(self.in_roi, enc_hidden, self.rep_roi, dropout=dropout, bn=bn)
            else:
                self.tab_num_dim = self.num_num_features - self.secondary_feature_dim
                self.in_tab = self.tab_num_dim + self.cat_dim
                self.in_img = self.secondary_feature_dim

                self.rep_tab = rep_dim // 2
                self.rep_img = rep_dim - self.rep_tab
                self.rep_dim = self.rep_tab + self.rep_img

                self.encoder_tab = mlp(self.in_tab, enc_hidden, self.rep_tab, dropout=dropout, bn=bn)
                self.encoder_img = mlp(self.in_img, enc_hidden, self.rep_img, dropout=dropout, bn=bn)
            head_in_dim = self.rep_dim
        else:
            self.encoder = mlp(in_dim, enc_hidden, rep_dim, dropout=dropout, bn=bn)
            self.rep_dim = rep_dim
            head_in_dim = rep_dim

        self.heads = nn.ModuleList(
            [mlp(head_in_dim, head_hidden, 1, dropout=dropout, bn=bn) for _ in range(num_treatments)]
        )

    def forward(self, x: torch.Tensor):
        if self.use_multi_branch:
            num_total = self.num_features
            x_num = x[:, :num_total]
            x_cat = x[:, num_total:] if self.cat_dim > 0 else None

            if self.branch_mode == "tri":
                x_tab_num = x_num[:, : self.tab_num_dim]
                x_img = x_num[:, self.tab_num_dim : self.tab_num_dim + self.image_feature_dim]
                x_roi = x_num[
                    :,
                    self.tab_num_dim
                    + self.image_feature_dim : self.tab_num_dim
                    + self.image_feature_dim
                    + self.roi_feature_dim,
                ]
                x_tab_in = torch.cat([x_tab_num, x_cat], dim=1) if x_cat is not None and x_cat.numel() > 0 else x_tab_num
                phi = torch.cat([self.encoder_tab(x_tab_in), self.encoder_img(x_img), self.encoder_roi(x_roi)], dim=1)
            else:
                x_tab_num = x_num[:, : self.tab_num_dim]
                x_img = x_num[:, self.tab_num_dim : self.tab_num_dim + self.secondary_feature_dim]
                x_tab_in = torch.cat([x_tab_num, x_cat], dim=1) if x_cat is not None and x_cat.numel() > 0 else x_tab_num
                phi = torch.cat([self.encoder_tab(x_tab_in), self.encoder_img(x_img)], dim=1)
        else:
            phi = self.encoder(x)

        logits_all = [head(phi) for head in self.heads]
        return phi, logits_all


MODEL_ID = "de_natrm_final_release_pr90_20260423"
DEFAULT_THRESHOLD = 0.4630236625671386


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _clean_numeric_value(value: Any) -> Any:
    """Mirror the frozen package's conservative numeric string cleanup."""
    if value is None:
        return np.nan
    if isinstance(value, (int, float, np.integer, np.floating)):
        if isinstance(value, float) and math.isnan(value):
            return np.nan
        return value
    text = str(value).strip()
    if text in {"", "nan", "NaN", "*", "＊", "未测"}:
        return np.nan
    text = re.sub(r"^[<>]\s*", "", text)
    text = text.rstrip("%")
    return text


def _as_dataframe(
    patient: Mapping[str, Any],
    num_features: Iterable[str],
    cat_features: Iterable[str],
) -> tuple[pd.DataFrame, list[str]]:
    columns = list(num_features) + list(cat_features)
    row: Dict[str, Any] = {}
    missing: list[str] = []
    for column in columns:
        if column in patient:
            row[column] = _clean_numeric_value(patient[column])
        else:
            row[column] = np.nan
            missing.append(column)
    frame = pd.DataFrame([row], columns=columns)
    for column in num_features:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    return frame, missing


def _roi_is_valid(patient: Mapping[str, Any]) -> bool:
    def number(name: str, default: float) -> float:
        value = _clean_numeric_value(patient.get(name, default))
        try:
            return float(value)
        except (TypeError, ValueError):
            return default

    return (
        number("mr_roi_available", 0.0) == 1.0
        and number("pet_roi_available", 0.0) == 1.0
        and number("roi_ambiguous_series", 1.0) == 0.0
    )


class PackagePredictor:
    """Load and run the final frozen De-NATRM CausalNet package."""

    def __init__(self, package_dir: Optional[Path | str] = None, device: str = "cpu") -> None:
        self.package_dir = Path(package_dir) if package_dir is not None else PACKAGE_DIR
        self.artifacts_dir = self.package_dir / "artifacts"
        self.device = torch.device(device)

        self.manifest = _read_json(self.package_dir / "manifest.json")
        self.final_id = _read_json(self.package_dir / "FINAL_MODEL_ID.json")
        self.meta = _read_json(self.artifacts_dir / "data_meta.json")
        self.treatment_map = _read_json(self.artifacts_dir / "treatment_map.json")
        self.preprocessor = joblib.load(self.artifacts_dir / "preprocessor.joblib")

        self.keep_order = list(self.treatment_map["keep_order"])
        self.num_features = list(self.meta["num_features"])
        self.cat_features = list(self.meta.get("cat_features") or [])
        self.selected_threshold = float(self.final_id.get("selected_threshold", DEFAULT_THRESHOLD))
        self.threshold_strategy = str(self.final_id.get("threshold_strategy", "precision_at_recall"))

        self.model = self._build_model()
        state_dict = torch.load(self.artifacts_dir / "causal_net.pt", map_location=self.device)
        self.model.load_state_dict(state_dict)
        self.model.to(self.device)
        self.model.eval()

    def _build_model(self) -> CausalNet:
        architecture = self.manifest["model"]["architecture"]
        return CausalNet(
            in_dim=int(architecture["input_dim"]),
            rep_dim=int(architecture["rep_dim"]),
            enc_hidden=list(architecture["enc_hidden"]),
            head_hidden=list(architecture["head_hidden"]),
            num_treatments=len(self.keep_order),
            dropout=float(architecture["dropout"]),
            bn=bool(architecture["batch_norm"]),
            num_features=len(self.num_features),
            image_feature_dim=int(self.meta.get("image_feature_dim", 0) or 0),
            secondary_feature_dim=int(self.meta.get("secondary_feature_dim", 0) or 0),
            pet_feature_dim=int(self.meta.get("pet_feature_dim", 0) or 0),
            roi_feature_dim=int(self.meta.get("roi_branch_feature_dim", 0) or 0),
            use_roi_branch=bool(self.meta.get("roi_branch_enabled", False)),
        )

    def _transform(self, patient: Mapping[str, Any]) -> tuple[np.ndarray, list[str], list[str]]:
        frame, missing = _as_dataframe(patient, self.num_features, self.cat_features)
        transformed = self.preprocessor.transform(frame)
        if hasattr(transformed, "toarray"):
            transformed = transformed.toarray()
        transformed = np.asarray(transformed, dtype=np.float32)

        warnings: list[str] = []
        roi_dim = int(self.meta.get("roi_branch_feature_dim", 0) or 0)
        if bool(self.meta.get("roi_branch_enabled", False)) and roi_dim > 0 and not _roi_is_valid(patient):
            transformed[:, -roi_dim:] = 0.0
            warnings.append(
                "ROI branch gated to zero because mr_roi_available/pet_roi_available/roi_ambiguous_series do not indicate a valid paired ROI."
            )
        return transformed, missing, warnings

    def predict_one(self, patient: Mapping[str, Any]) -> Dict[str, Any]:
        transformed, missing, warnings = self._transform(patient)
        tensor = torch.as_tensor(transformed, dtype=torch.float32, device=self.device)
        with torch.no_grad():
            _, logits_all = self.model(tensor)
            probs = torch.sigmoid(torch.stack([logit.squeeze(1) for logit in logits_all], dim=1))
        prob_values = probs.cpu().numpy()[0].astype(float)
        probabilities = {name: float(prob_values[idx]) for idx, name in enumerate(self.keep_order)}
        best_idx = int(np.argmax(prob_values))
        positive_at_threshold = {
            name: bool(probabilities[name] >= self.selected_threshold) for name in self.keep_order
        }

        if "Treatment" in patient and str(patient["Treatment"]).strip() not in self.keep_order:
            warnings.append(
                "Input Treatment is not a direct model label. This package supports only Standard and De-escalation."
            )

        return {
            "model_id": MODEL_ID,
            "probabilities": probabilities,
            "recommended_treatment": self.keep_order[best_idx],
            "recommended_probability": float(prob_values[best_idx]),
            "selected_threshold": self.selected_threshold,
            "threshold_strategy": self.threshold_strategy,
            "positive_at_threshold": positive_at_threshold,
            "missing_input_features": missing,
            "warnings": warnings,
        }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run one-patient prediction with the frozen pCR/NAT package.")
    parser.add_argument("--input", required=True, help="Path to a patient JSON object.")
    parser.add_argument("--package-dir", default=str(PACKAGE_DIR), help="Path to export/model_package.")
    parser.add_argument("--output", help="Optional path for prediction JSON output.")
    args = parser.parse_args()

    patient = _read_json(Path(args.input))
    predictor = PackagePredictor(args.package_dir)
    result = predictor.predict_one(patient)
    text = json.dumps(result, ensure_ascii=False, indent=2)
    if args.output:
        Path(args.output).write_text(text + "\n", encoding="utf-8")
    else:
        print(text)


if __name__ == "__main__":
    main()
