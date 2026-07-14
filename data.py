import os, re, json
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.decomposition import PCA
from sklearn.feature_selection import f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import log_loss
from sklearn.model_selection import StratifiedKFold, cross_val_predict
import joblib

from config import (
    DATA_PATH,
    USE_FIXED_SPLIT_FILES,
    TRAIN_DATA_PATH,
    TRAIN_DATA_PATHS,
    TEST_DATA_PATH,
    VARPOOL_PATH,
    OUTPUT_DIR,
    RANDOM_STATE,
    TOP_K_TREATMENTS,
    EXPLICIT_OUTCOME_COL,
    EXPLICIT_TREATMENT_COL,
    MP_COL_CANDIDATES,
    MP_AS_PCR_RULE,
    USE_C2_FEATURES,
    TABULAR_FEATURE_WHITELIST,
    USE_PET_FEATURES,
    USE_IMAGE_EMBEDDINGS,
    IMAGE_EMBEDDING_DIR,
    IMAGE_EMBED_MODALITIES,
    IMAGE_EMBED_POOLING,
    IMAGE_EMBED_MR_DIM,
    IMAGE_EMBED_PET_DIM,
    IMAGE_EMBED_ADD_MISSING_FLAGS,
    USE_IMAGE_RISK_SCORES,
    IMAGE_RISK_SCORE_MODE,
    IMAGE_RISK_MODEL_TYPE,
    IMAGE_RISK_MLP_HIDDEN,
    IMAGE_RISK_MLP_ALPHA,
    IMAGE_RISK_MLP_MAX_ITER,
    IMAGE_RISK_USE_SEPARATE_BRANCH,
    IMAGE_RISK_BRANCH_FEATURE_MODE,
    USE_ROI_FEATURES,
    ROI_FEATURE_TABLE_PATH,
    ROI_STRICT_SENSITIVITY,
    ROI_FEATURE_SELECTION_MODE,
    ROI_RAD_MAX_MISSING,
    ROI_RAD_CORR_THRESHOLD,
    ROI_RAD_CORR_WITHIN_MODALITY_ONLY,
    ROI_RAD_MAX_FEATURES,
    ROI_RAD_MAX_PET_FEATURES,
    ROI_RAD_MODALITIES,
    ROI_RAD_CLASSES,
    ROI_RAD_BALANCE_MODALITIES,
    ROI_RAD_TEXTURE_CLASSES,
    ROI_RAD_TEXTURE_BUDGET,
    ROI_RAD_HARMONIZE,
    ROI_RAD_HARMONIZE_MIN_BATCH_N,
    ROI_RAD_MAX_BATCH_SMD,
    ROI_RAD_MAX_BATCH_SMD_MR,
    ROI_RAD_MAX_BATCH_SMD_PET,
    ROI_RAD_STABILITY_MIN_BATCH_N,
    ROI_ADD_INTERACTIONS,
    ROI_ADD_PRIMARY_COMPOSITES,
    ROI_USE_SEPARATE_BRANCH,
    ROI_BRANCH_MISSING_POLICY,
)

ROI_FALLBACK_FEATURE_COLUMNS = [
    "mr_roi_available",
    "pet_roi_available",
    "roi_ambiguous_series",
    "roi_multifocal_any",
    "mr_lesion_count",
    "pet_lesion_count",
    "mr_total_roi_volume_ml",
    "mr_bbox_max_dim_mm",
    "mr_z_span_mm",
    "mr_max_slice_area_mm2",
    "pet_total_roi_volume_ml",
    "pet_bbox_max_dim_mm",
    "pet_z_span_mm",
    "pet_max_slice_area_mm2",
    "pet_suvbw_mean",
    "pet_suvbw_max",
    "pet_tlg",
    "log_pet_mr_volume_ratio",
]

ROI_NONFEATURE_COLUMNS = {
    "patient_id",
    "_patient_id_norm",
    "mr_valid_roi_names",
    "pet_valid_roi_names",
    "mr_excluded_roi_names",
    "pet_excluded_roi_names",
    "missing_flags",
    "in_all_data",
    "in_train",
    "in_test",
}

ROI_INTERACTION_COLUMNS = [
    "roi_cross_log_bbox_ratio",
    "roi_cross_log_zspan_ratio",
    "roi_cross_log_maxarea_ratio",
    "roi_cross_log_pet_suvmax_to_mr_vol",
    "roi_cross_pet_mr_max_diff",
    "roi_cross_pet_mr_p90_diff",
    "roi_cross_pet_mr_flatness_diff",
    "roi_cross_pet_mr_elongation_diff",
]

ROI_PRIMARY_COMPOSITE_COLUMNS = [
    "roi_comp_pet_metabolic_density",
    "roi_comp_pet_suvmax_to_mr_dim",
    "roi_comp_pet_mr_volume_ratio_raw",
    "roi_comp_pet_mr_zspan_ratio",
    "roi_comp_pet_mr_maxarea_ratio",
    "roi_comp_pet_mr_lesion_delta",
    "roi_comp_pet_suvmean_to_mr_vol",
]

ROI_SELECTION_PRIMARY_SET = set(ROI_FALLBACK_FEATURE_COLUMNS + ROI_INTERACTION_COLUMNS + ROI_PRIMARY_COMPOSITE_COLUMNS)


def _normalize_roi_rad_modalities(value: object) -> set[str]:
    text = str(value or "all").strip().lower()
    if text in {"", "all"}:
        return {"mr", "pet"}
    items = {part.strip().lower() for part in text.split(",") if part.strip()}
    valid = {"mr", "pet"}
    items = items & valid
    return items or {"mr", "pet"}


def _normalize_roi_rad_classes(value: object) -> Optional[set[str]]:
    text = str(value or "all").strip().lower()
    if text in {"", "all"}:
        return None
    items = {part.strip().lower() for part in text.split(",") if part.strip()}
    valid = {"shape", "firstorder", "glcm", "glrlm", "glszm", "gldm", "ngtdm"}
    items = items & valid
    return items or None


def _match_roi_radiomics_subset(col: str, allowed_modalities: set[str], allowed_classes: Optional[set[str]]) -> bool:
    if col in ROI_SELECTION_PRIMARY_SET:
        return True
    parts = col.split("_")
    if len(parts) < 4 or not parts[0] or not parts[1]:
        return False
    modality = parts[0].strip().lower()
    if modality not in allowed_modalities:
        return False
    if parts[1] != "rad":
        return False
    feature_class = parts[3].strip().lower()
    if allowed_classes is None:
        return True
    return feature_class in allowed_classes


def _roi_radiomics_modality(col: str) -> Optional[str]:
    parts = col.split("_")
    if not parts:
        return None
    modality = parts[0].strip().lower()
    if modality in {"mr", "pet"}:
        return modality
    return None


def _roi_radiomics_class(col: str) -> Optional[str]:
    parts = col.split("_")
    if len(parts) < 4 or parts[1] != "rad":
        return None
    return parts[3].strip().lower()


def _roi_radiomics_stability_threshold(col: str, default_threshold: float) -> float:
    modality = _roi_radiomics_modality(col)
    if modality == "mr":
        threshold = float(ROI_RAD_MAX_BATCH_SMD_MR or 0.0)
        if threshold > 0:
            return threshold
    if modality == "pet":
        threshold = float(ROI_RAD_MAX_BATCH_SMD_PET or 0.0)
        if threshold > 0:
            return threshold
    return float(default_threshold)


def _balanced_select_score_pairs(
    score_pairs: List[Tuple[str, float]],
    max_features: int,
    allowed_modalities: set[str],
) -> List[str]:
    if max_features <= 0:
        return [col for col, _ in score_pairs]

    grouped: Dict[str, List[Tuple[str, float]]] = {m: [] for m in sorted(allowed_modalities)}
    for col, score in score_pairs:
        modality = _roi_radiomics_modality(col)
        if modality in grouped:
            grouped[modality].append((col, score))

    active_modalities = [m for m in sorted(grouped) if grouped[m]]
    if len(active_modalities) <= 1:
        return [col for col, _ in score_pairs[:max_features]]

    base_quota = max_features // len(active_modalities)
    remainder = max_features % len(active_modalities)

    selected: List[str] = []
    leftovers: List[Tuple[str, float]] = []
    for rank, modality in enumerate(active_modalities):
        quota = base_quota + (1 if rank < remainder else 0)
        picks = grouped[modality][:quota]
        selected.extend([col for col, _ in picks])
        leftovers.extend(grouped[modality][quota:])

    if len(selected) < max_features:
        selected_set = set(selected)
        leftovers.sort(key=lambda item: item[1], reverse=True)
        for col, _ in leftovers:
            if col in selected_set:
                continue
            selected.append(col)
            selected_set.add(col)
            if len(selected) >= max_features:
                break

    return selected[:max_features]


def _select_score_pairs_with_pet_cap(
    score_pairs: List[Tuple[str, float]],
    max_features: int,
    pet_cap: int,
) -> List[str]:
    if max_features <= 0:
        max_features = len(score_pairs)
    if pet_cap <= 0:
        return [col for col, _ in score_pairs[:max_features]]

    selected: List[str] = []
    pet_count = 0
    for col, _ in score_pairs:
        modality = _roi_radiomics_modality(col)
        if modality == "pet" and pet_count >= pet_cap:
            continue
        selected.append(col)
        if modality == "pet":
            pet_count += 1
        if len(selected) >= max_features:
            break
    return selected


def _select_roi_score_pairs(
    score_pairs: List[Tuple[str, float]],
    max_features: int,
    allowed_modalities: set[str],
    pet_cap: int,
) -> List[str]:
    ordered_pairs = score_pairs
    if max_features > 0 and bool(ROI_RAD_BALANCE_MODALITIES) and allowed_modalities == {"mr", "pet"}:
        balanced_cols = _balanced_select_score_pairs(score_pairs, max_features, allowed_modalities)
        balanced_set = set(balanced_cols)
        ordered_pairs = (
            [(col, score) for col, score in score_pairs if col in balanced_set]
            + [(col, score) for col, score in score_pairs if col not in balanced_set]
        )
    select_n = max_features if max_features > 0 else len(ordered_pairs)
    return _select_score_pairs_with_pet_cap(ordered_pairs, select_n, pet_cap)

def _normalize(s: str) -> str:
    return re.sub(r"\s+", "", str(s)).lower()


def _normalize_patient_id(value: object) -> str:
    text = str(value).strip()
    if not text or text.lower() == "nan":
        return ""
    exam_match = re.fullmatch(r"(?:exam[_-])?(\d+)", text, flags=re.IGNORECASE)
    if exam_match:
        return exam_match.group(1)
    if re.fullmatch(r"\d+\.0+", text):
        return text.split(".", 1)[0]
    try:
        numeric = float(text)
    except ValueError:
        return text
    if numeric.is_integer():
        return str(int(numeric))
    return text

def find_col(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    norm = {_normalize(c): c for c in df.columns}
    # 精确
    for c in candidates:
        key = _normalize(c)
        if key in norm: return norm[key]
    # 模糊
    for c in candidates:
        key = _normalize(c)
        for col in df.columns:
            if key in _normalize(col): return col
    return None

def auto_detect_outcome_col(df: pd.DataFrame) -> Tuple[str, pd.Series]:
    if EXPLICIT_OUTCOME_COL and EXPLICIT_OUTCOME_COL in df.columns:
        return EXPLICIT_OUTCOME_COL, df[EXPLICIT_OUTCOME_COL]

    pcr_cands = ["pcr", "是否pcr", "pcr_label", "是否达到pcr", "ypT0N0", "ypT0/is"]
    col = find_col(df, pcr_cands)
    if col is not None:
        return col, df[col]

    mp_col = find_col(df, MP_COL_CANDIDATES)
    if mp_col is not None:
        print(f"[INFO] 未找到pCR列，使用 {mp_col} 构造近似pCR（MP==1）")
        return mp_col + "->pCR", MP_AS_PCR_RULE(df[mp_col]).astype(int)
    raise ValueError("未找到 pCR/MP 结局列；请在 config.EXPLICIT_OUTCOME_COL 指定。")

def auto_detect_treatment_col(df: pd.DataFrame) -> str:
    if EXPLICIT_TREATMENT_COL and EXPLICIT_TREATMENT_COL in df.columns:
        return EXPLICIT_TREATMENT_COL
    # 兼容新老列名
    t_cands = ["NAT方案性质", "NAT方案类型", "NAT方案模糊", "NAT方案（模糊）", "NAT_scheme", "治疗方案", "方案", "Regimen", "NAT", "Tx", "Treatment"]
    col = find_col(df, t_cands)
    if col is None:
        raise ValueError("未找到治疗/方案列；请在 config.EXPLICIT_TREATMENT_COL 指定。")
    return col

def load_varpool_names(path: Optional[str]) -> Optional[List[str]]:
    if not path or not os.path.exists(path): return None
    ext = os.path.splitext(path)[-1].lower()
    if ext in (".xlsx", ".xls"):
        try:
            vp = pd.read_excel(path, engine="openpyxl")
        except Exception:
            vp = pd.read_excel(path)
    elif ext == ".csv":
        vp = pd.read_csv(path)
    else:
        print("[WARN] 变量池不是 .xlsx/.csv，已忽略。")
        return None

    for c in vp.columns:
        if _normalize(c) in ["变量名","变量","var","vars","field","col","name","names"]:
            return vp[c].dropna().astype(str).tolist()
    return vp.iloc[:,0].dropna().astype(str).tolist()


def _resolve_roi_id_column(df: pd.DataFrame) -> Optional[str]:
    return find_col(df, ["影像ID", "影像id", "患者ID", "患者id", "patient_id", "ID", "id"])


def _resolve_roi_feature_columns(roi_df: pd.DataFrame, roi_path: str) -> List[str]:
    sidecar_path = Path(roi_path).with_suffix(".json")
    if sidecar_path.exists():
        try:
            sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
            sidecar_cols = sidecar.get("feature_columns") or sidecar.get("primary_feature_columns")
            if isinstance(sidecar_cols, list):
                cols = [str(col) for col in sidecar_cols if str(col) in roi_df.columns]
                if cols:
                    return cols
        except Exception as exc:
            print(f"[WARN] 读取 ROI 特征侧车文件失败，回退自动识别: {sidecar_path} | {exc}")

    fallback_cols = [col for col in ROI_FALLBACK_FEATURE_COLUMNS if col in roi_df.columns]
    if fallback_cols:
        return fallback_cols

    inferred_cols: List[str] = []
    for col in roi_df.columns:
        if col in ROI_NONFEATURE_COLUMNS:
            continue
        numeric = pd.to_numeric(roi_df[col], errors="coerce")
        if numeric.notna().any():
            inferred_cols.append(col)
    return inferred_cols


def _load_roi_feature_frame() -> Tuple[pd.DataFrame, List[str]]:
    roi_path = ROI_FEATURE_TABLE_PATH
    if not roi_path or not os.path.exists(roi_path):
        raise FileNotFoundError(f"未找到 ROI 特征表: {roi_path}")

    roi_df = pd.read_csv(roi_path)
    roi_feature_cols = _resolve_roi_feature_columns(roi_df, roi_path)
    required_cols = ["patient_id"] + roi_feature_cols
    missing = [col for col in required_cols if col not in roi_df.columns]
    if missing:
        raise ValueError(f"ROI 特征表缺少必要列: {missing}")

    roi_df = roi_df.copy()
    roi_df["patient_id"] = roi_df["patient_id"].astype(str)
    roi_df["_patient_id_norm"] = roi_df["patient_id"].map(_normalize_patient_id)
    roi_df = roi_df.loc[roi_df["_patient_id_norm"] != ""].copy()
    roi_df = roi_df.drop_duplicates("_patient_id_norm", keep="last")

    for col in roi_feature_cols:
        roi_df[col] = pd.to_numeric(roi_df[col], errors="coerce")

    return roi_df, roi_feature_cols


def _apply_roi_strict_filter(
    df: pd.DataFrame,
    split_labels: Optional[np.ndarray],
    source_labels: Optional[np.ndarray] = None,
    *,
    roi_df: pd.DataFrame,
    id_col: str,
) -> Tuple[pd.DataFrame, Optional[np.ndarray], Optional[np.ndarray]]:
    roi_lookup = roi_df.set_index("_patient_id_norm")
    match_ids = df[id_col].map(_normalize_patient_id)
    matched = roi_lookup.reindex(match_ids.tolist())
    strict_mask = (
        matched["mr_roi_available"].fillna(0).eq(1)
        & matched["pet_roi_available"].fillna(0).eq(1)
        & matched["roi_ambiguous_series"].fillna(1).eq(0)
    ).to_numpy()

    before_n = len(df)
    after_n = int(strict_mask.sum())
    print(f"[INFO] ROI_STRICT_SENSITIVITY=True，按 ROI 完整性过滤样本: {before_n} -> {after_n}")
    if after_n == 0:
        raise ValueError("严格 ROI 敏感性过滤后无可用样本。")

    df = df.loc[strict_mask].reset_index(drop=True)
    if split_labels is not None:
        split_labels = split_labels[strict_mask]
    if source_labels is not None:
        source_labels = source_labels[strict_mask]
    return df, split_labels, source_labels


def _attach_roi_features(
    df: pd.DataFrame,
    X: pd.DataFrame,
    *,
    roi_df: pd.DataFrame,
    id_col: str,
    roi_feature_cols: List[str],
) -> Tuple[pd.DataFrame, List[str]]:
    roi_lookup = roi_df.set_index("_patient_id_norm")
    match_ids = df[id_col].map(_normalize_patient_id)
    roi_features = roi_lookup.reindex(match_ids.tolist())[roi_feature_cols].copy()
    roi_features.index = df.index

    matched_count = int(roi_features["mr_roi_available"].notna().sum())
    print(f"[INFO] 已对齐 ROI 特征表: matched={matched_count}/{len(df)}")
    X = pd.concat([X, roi_features], axis=1)
    return X, list(roi_feature_cols)


def _compute_roi_sample_valid_mask(X: pd.DataFrame) -> np.ndarray:
    required_cols = ["mr_roi_available", "pet_roi_available", "roi_ambiguous_series"]
    if not all(col in X.columns for col in required_cols):
        return np.ones(len(X), dtype=bool)
    mr_ok = pd.to_numeric(X["mr_roi_available"], errors="coerce").fillna(0).eq(1)
    pet_ok = pd.to_numeric(X["pet_roi_available"], errors="coerce").fillna(0).eq(1)
    amb_ok = pd.to_numeric(X["roi_ambiguous_series"], errors="coerce").fillna(1).eq(0)
    return (mr_ok & pet_ok & amb_ok).to_numpy(dtype=bool)


def _safe_log_ratio(numerator: pd.Series, denominator: pd.Series, eps: float = 1e-3) -> pd.Series:
    return np.log((numerator.astype(float) + eps) / (denominator.astype(float) + eps))


def _safe_ratio(numerator: pd.Series, denominator: pd.Series, eps: float = 1e-3) -> pd.Series:
    return (numerator.astype(float) + eps) / (denominator.astype(float) + eps)


def _add_roi_interaction_features(
    X: pd.DataFrame,
    roi_feature_cols: List[str],
) -> Tuple[pd.DataFrame, List[str], List[str]]:
    if not bool(ROI_ADD_INTERACTIONS):
        return X, roi_feature_cols, []

    X = X.copy()
    added: List[str] = []

    specs = [
        ("roi_cross_log_bbox_ratio", ["pet_bbox_max_dim_mm", "mr_bbox_max_dim_mm"], "log_ratio"),
        ("roi_cross_log_zspan_ratio", ["pet_z_span_mm", "mr_z_span_mm"], "log_ratio"),
        ("roi_cross_log_maxarea_ratio", ["pet_max_slice_area_mm2", "mr_max_slice_area_mm2"], "log_ratio"),
        ("roi_cross_log_pet_suvmax_to_mr_vol", ["pet_suvbw_max", "mr_total_roi_volume_ml"], "log_ratio"),
        ("roi_cross_pet_mr_max_diff", ["pet_rad_original_firstorder_maximum", "mr_rad_original_firstorder_maximum"], "diff"),
        ("roi_cross_pet_mr_p90_diff", ["pet_rad_original_firstorder_90percentile", "mr_rad_original_firstorder_90percentile"], "diff"),
        ("roi_cross_pet_mr_flatness_diff", ["pet_rad_original_shape_flatness", "mr_rad_original_shape_flatness"], "diff"),
        ("roi_cross_pet_mr_elongation_diff", ["pet_rad_original_shape_elongation", "mr_rad_original_shape_elongation"], "diff"),
    ]

    for new_col, parents, mode in specs:
        if new_col in X.columns:
            continue
        if not all(parent in X.columns for parent in parents):
            continue
        left = pd.to_numeric(X[parents[0]], errors="coerce")
        right = pd.to_numeric(X[parents[1]], errors="coerce")
        if mode == "log_ratio":
            X[new_col] = _safe_log_ratio(left, right)
        elif mode == "diff":
            X[new_col] = left - right
        else:
            continue
        added.append(new_col)

    if added:
        roi_feature_cols = list(roi_feature_cols) + added
        print(f"[INFO] 已添加 ROI 跨模态交互特征: {added}")
    return X, roi_feature_cols, added


def _add_roi_primary_composite_features(
    X: pd.DataFrame,
    roi_feature_cols: List[str],
) -> Tuple[pd.DataFrame, List[str], List[str]]:
    if not bool(ROI_ADD_PRIMARY_COMPOSITES):
        return X, roi_feature_cols, []

    X = X.copy()
    added: List[str] = []
    specs = [
        ("roi_comp_pet_metabolic_density", ["pet_tlg", "pet_total_roi_volume_ml"], "ratio"),
        ("roi_comp_pet_suvmax_to_mr_dim", ["pet_suvbw_max", "mr_bbox_max_dim_mm"], "ratio"),
        ("roi_comp_pet_mr_volume_ratio_raw", ["pet_total_roi_volume_ml", "mr_total_roi_volume_ml"], "ratio"),
        ("roi_comp_pet_mr_zspan_ratio", ["pet_z_span_mm", "mr_z_span_mm"], "ratio"),
        ("roi_comp_pet_mr_maxarea_ratio", ["pet_max_slice_area_mm2", "mr_max_slice_area_mm2"], "ratio"),
        ("roi_comp_pet_mr_lesion_delta", ["pet_lesion_count", "mr_lesion_count"], "diff"),
        ("roi_comp_pet_suvmean_to_mr_vol", ["pet_suvbw_mean", "mr_total_roi_volume_ml"], "ratio"),
    ]

    for new_col, parents, mode in specs:
        if new_col in X.columns:
            continue
        if not all(parent in X.columns for parent in parents):
            continue
        left = pd.to_numeric(X[parents[0]], errors="coerce")
        right = pd.to_numeric(X[parents[1]], errors="coerce")
        if mode == "ratio":
            X[new_col] = _safe_ratio(left, right)
        elif mode == "diff":
            X[new_col] = left - right
        else:
            continue
        added.append(new_col)

    if added:
        roi_feature_cols = list(roi_feature_cols) + added
        print(f"[INFO] 已添加 ROI primary composite 特征: {added}")
    return X, roi_feature_cols, added


def _abs_smd_between_groups(values_a: np.ndarray, values_b: np.ndarray) -> float:
    if values_a.size == 0 or values_b.size == 0:
        return np.nan
    mean_a = float(np.mean(values_a))
    mean_b = float(np.mean(values_b))
    std_a = float(np.std(values_a, ddof=1)) if values_a.size > 1 else 0.0
    std_b = float(np.std(values_b, ddof=1)) if values_b.size > 1 else 0.0
    pooled = float(np.sqrt((std_a ** 2 + std_b ** 2) / 2.0))
    if pooled <= 1e-8:
        return 0.0 if abs(mean_b - mean_a) <= 1e-8 else np.inf
    return abs((mean_b - mean_a) / pooled)


def _filter_roi_radiomics_by_batch_stability(
    X: pd.DataFrame,
    train_indices: np.ndarray,
    source_labels: Optional[np.ndarray],
    extra_cols: List[str],
) -> Tuple[List[str], Dict[str, object]]:
    threshold = float(ROI_RAD_MAX_BATCH_SMD or 0.0)
    min_batch_n = int(ROI_RAD_STABILITY_MIN_BATCH_N or 0)
    summary: Dict[str, object] = {
        "stability_filter_enabled": bool(threshold > 0),
        "stability_max_abs_smd_threshold": float(threshold),
        "stability_max_abs_smd_threshold_mr": float(ROI_RAD_MAX_BATCH_SMD_MR or 0.0),
        "stability_max_abs_smd_threshold_pet": float(ROI_RAD_MAX_BATCH_SMD_PET or 0.0),
        "stability_min_batch_n": int(min_batch_n),
        "stability_before_count": int(len(extra_cols)),
        "stability_after_count": int(len(extra_cols)),
        "stability_dropped_count": 0,
        "stability_valid_batches": [],
        "stability_dropped_top20": [],
    }
    if threshold <= 0 or not extra_cols:
        return extra_cols, summary
    if source_labels is None or len(source_labels) != len(X):
        return extra_cols, summary

    train_sources = pd.Series(source_labels[train_indices], index=train_indices).astype(str)
    counts = train_sources.value_counts()
    valid_batches = [
        batch
        for batch, count in counts.items()
        if batch != "__test__" and int(count) >= min_batch_n
    ]
    summary["stability_valid_batches"] = [
        {"label": str(batch), "n": int(counts[batch])} for batch in valid_batches
    ]
    if len(valid_batches) < 2:
        return extra_cols, summary

    dropped_pairs: List[Tuple[str, float]] = []
    kept: List[str] = []
    batch_to_index = {
        batch: train_sources.index[train_sources.eq(batch)].to_numpy()
        for batch in valid_batches
    }

    for col in extra_cols:
        col_threshold = _roi_radiomics_stability_threshold(col, threshold)
        max_abs_smd = 0.0
        has_pair = False
        for i, batch_i in enumerate(valid_batches):
            idx_i = batch_to_index[batch_i]
            vals_i = pd.to_numeric(X.loc[idx_i, col], errors="coerce").dropna().to_numpy(dtype=np.float64)
            if vals_i.size < max(3, min_batch_n // 4):
                continue
            for j in range(i + 1, len(valid_batches)):
                batch_j = valid_batches[j]
                idx_j = batch_to_index[batch_j]
                vals_j = pd.to_numeric(X.loc[idx_j, col], errors="coerce").dropna().to_numpy(dtype=np.float64)
                if vals_j.size < max(3, min_batch_n // 4):
                    continue
                smd = _abs_smd_between_groups(vals_i, vals_j)
                if not np.isfinite(smd):
                    has_pair = True
                    max_abs_smd = np.inf
                    break
                has_pair = True
                if smd > max_abs_smd:
                    max_abs_smd = float(smd)
            if not np.isfinite(max_abs_smd):
                break

        # 如果该特征没有足够可比批次数据，不在这里直接砍掉，交给后续缺失率/方差筛选处理
        if not has_pair or max_abs_smd <= col_threshold:
            kept.append(col)
        else:
            dropped_pairs.append((col, float(max_abs_smd), float(col_threshold)))

    dropped_pairs.sort(key=lambda item: item[1], reverse=True)
    summary["stability_after_count"] = int(len(kept))
    summary["stability_dropped_count"] = int(len(dropped_pairs))
    summary["stability_dropped_top20"] = [
        {"feature": col, "max_abs_smd": round(score, 6), "threshold": round(limit, 6)}
        for col, score, limit in dropped_pairs[:20]
    ]
    print(
        "[INFO] ROI radiomics 批次稳定性筛选: "
        f"before={len(extra_cols)}, after={len(kept)}, dropped={len(dropped_pairs)}, "
        f"threshold={threshold}, mr_threshold={float(ROI_RAD_MAX_BATCH_SMD_MR or 0.0)}, "
        f"pet_threshold={float(ROI_RAD_MAX_BATCH_SMD_PET or 0.0)}"
    )
    return kept, summary


def _select_roi_radiomics_features(
    X: pd.DataFrame,
    y: np.ndarray,
    train_indices: np.ndarray,
    roi_feature_cols: List[str],
    source_labels: Optional[np.ndarray],
) -> Tuple[pd.DataFrame, List[str], Dict[str, object]]:
    allowed_modalities = _normalize_roi_rad_modalities(ROI_RAD_MODALITIES)
    allowed_classes = _normalize_roi_rad_classes(ROI_RAD_CLASSES)
    summary: Dict[str, object] = {
        "mode": ROI_FEATURE_SELECTION_MODE,
        "modalities": sorted(allowed_modalities),
        "classes": sorted(allowed_classes) if allowed_classes is not None else "all",
        "balance_modalities": bool(ROI_RAD_BALANCE_MODALITIES),
        "texture_classes": sorted(_normalize_roi_rad_classes(ROI_RAD_TEXTURE_CLASSES) or []),
        "texture_budget": int(ROI_RAD_TEXTURE_BUDGET or 0),
        "corr_within_modality_only": bool(ROI_RAD_CORR_WITHIN_MODALITY_ONLY),
        "max_pet_features": int(ROI_RAD_MAX_PET_FEATURES or 0),
        "kept_primary_count": 0,
        "candidate_extra_count": 0,
        "after_subset_filter_count": 0,
        "after_stability_filter_count": 0,
        "after_missing_count": 0,
        "after_variance_count": 0,
        "after_correlation_count": 0,
        "selected_extra_count": 0,
        "selected_feature_names": [],
    }
    non_roi_cols = [col for col in X.columns if col not in roi_feature_cols]
    mode = str(ROI_FEATURE_SELECTION_MODE).strip().lower()
    if mode == "none" or not roi_feature_cols:
        return X, roi_feature_cols, summary
    if mode not in {"train_radiomics", "train_radiomics_residual"}:
        raise ValueError(f"Unsupported ROI_FEATURE_SELECTION_MODE: {ROI_FEATURE_SELECTION_MODE!r}")

    primary_cols = [col for col in roi_feature_cols if col in ROI_SELECTION_PRIMARY_SET]
    extra_cols = [
        col
        for col in roi_feature_cols
        if col not in ROI_SELECTION_PRIMARY_SET and _match_roi_radiomics_subset(col, allowed_modalities, allowed_classes)
    ]
    summary["kept_primary_count"] = len(primary_cols)
    summary["candidate_extra_count"] = len([col for col in roi_feature_cols if col not in ROI_SELECTION_PRIMARY_SET])
    summary["after_subset_filter_count"] = len(extra_cols)
    extra_cols, stability_summary = _filter_roi_radiomics_by_batch_stability(
        X,
        train_indices,
        source_labels,
        extra_cols,
    )
    summary["after_stability_filter_count"] = len(extra_cols)
    summary.update(stability_summary)
    if not extra_cols:
        kept = primary_cols
        kept_all = non_roi_cols + kept
        return X[kept_all].copy(), kept, summary

    X_train_extra = X.iloc[train_indices][extra_cols].copy()
    missing_rate = X_train_extra.isna().mean()
    extra_cols = [col for col in extra_cols if missing_rate[col] <= ROI_RAD_MAX_MISSING]
    summary["after_missing_count"] = len(extra_cols)
    if not extra_cols:
        kept = primary_cols
        kept_all = non_roi_cols + kept
        return X[kept_all].copy(), kept, summary

    X_train_extra = X.iloc[train_indices][extra_cols].copy()
    medians = X_train_extra.median(axis=0, skipna=True)
    X_train_imputed = X_train_extra.fillna(medians)
    variances = X_train_imputed.var(axis=0)
    extra_cols = [col for col in extra_cols if pd.notna(variances[col]) and variances[col] > 1e-8]
    summary["after_variance_count"] = len(extra_cols)
    if not extra_cols:
        kept = primary_cols
        kept_all = non_roi_cols + kept
        return X[kept_all].copy(), kept, summary

    X_train_imputed = X.iloc[train_indices][extra_cols].copy().fillna(medians[extra_cols])
    corr = X_train_imputed.corr(method="spearman").abs()
    to_drop = set()
    ordered_cols = list(extra_cols)
    for i, col_i in enumerate(ordered_cols):
        if col_i in to_drop:
            continue
        for j in range(i + 1, len(ordered_cols)):
            col_j = ordered_cols[j]
            if col_j in to_drop:
                continue
            if bool(ROI_RAD_CORR_WITHIN_MODALITY_ONLY):
                modality_i = _roi_radiomics_modality(col_i)
                modality_j = _roi_radiomics_modality(col_j)
                if modality_i and modality_j and modality_i != modality_j:
                    continue
            corr_value = corr.iloc[i, j]
            if pd.notna(corr_value) and corr_value >= ROI_RAD_CORR_THRESHOLD:
                to_drop.add(col_j)
    extra_cols = [col for col in ordered_cols if col not in to_drop]
    summary["after_correlation_count"] = len(extra_cols)
    if not extra_cols:
        kept = primary_cols
        kept_all = non_roi_cols + kept
        return X[kept_all].copy(), kept, summary

    max_features = int(ROI_RAD_MAX_FEATURES) if ROI_RAD_MAX_FEATURES is not None else 0
    if max_features == 0:
        summary["selected_extra_count"] = 0
        summary["selected_feature_names"] = []
        kept_cols = primary_cols
        kept_all = non_roi_cols + kept_cols
        print("[INFO] ROI radiomics 训练集内筛选: max_features=0，仅保留 primary ROI 特征")
        return X[kept_all].copy(), kept_cols, summary

    X_train_imputed = X.iloc[train_indices][extra_cols].copy().fillna(medians.reindex(extra_cols))
    if mode == "train_radiomics_residual":
        score_pairs = _score_roi_radiomics_by_baseline_gain(
            X=X,
            y=y,
            train_indices=train_indices,
            non_roi_cols=non_roi_cols,
            extra_cols=extra_cols,
        )
        summary["score_mode"] = "baseline_residual_gain"
    else:
        scores, _ = f_classif(X_train_imputed.to_numpy(dtype=np.float64), y[train_indices])
        score_pairs = []
        for col, score in zip(extra_cols, scores):
            if pd.isna(score):
                score = -np.inf
            score_pairs.append((col, float(score)))
        summary["score_mode"] = "univariate_f_classif"
    score_pairs.sort(key=lambda item: item[1], reverse=True)

    max_features = int(ROI_RAD_MAX_FEATURES) if ROI_RAD_MAX_FEATURES else 0
    pet_cap = int(ROI_RAD_MAX_PET_FEATURES) if ROI_RAD_MAX_PET_FEATURES else 0
    texture_classes = _normalize_roi_rad_classes(ROI_RAD_TEXTURE_CLASSES) or set()
    texture_budget = int(ROI_RAD_TEXTURE_BUDGET or 0)
    if max_features > 0 and texture_budget > 0 and texture_classes:
        texture_pairs = [(col, score) for col, score in score_pairs if _roi_radiomics_class(col) in texture_classes]
        base_pairs = [(col, score) for col, score in score_pairs if _roi_radiomics_class(col) not in texture_classes]

        selected: List[str] = []
        base_target = max(0, max_features - min(texture_budget, len(texture_pairs)))
        if base_target > 0 and base_pairs:
            selected.extend(_select_roi_score_pairs(base_pairs, base_target, allowed_modalities, pet_cap))

        remaining = max(0, max_features - len(selected))
        if remaining > 0 and texture_pairs:
            selected_set = set(selected)
            texture_pairs = [(col, score) for col, score in texture_pairs if col not in selected_set]
            selected.extend(_select_roi_score_pairs(texture_pairs, remaining, allowed_modalities, pet_cap))

        if len(selected) < max_features:
            selected_set = set(selected)
            for col, _ in score_pairs:
                if col in selected_set:
                    continue
                selected.append(col)
                selected_set.add(col)
                if len(selected) >= max_features:
                    break
        extra_cols = selected[:max_features]
    elif max_features > 0:
        extra_cols = _select_roi_score_pairs(score_pairs, max_features, allowed_modalities, pet_cap)
    else:
        extra_cols = _select_score_pairs_with_pet_cap(score_pairs, len(score_pairs), pet_cap)

    summary["selected_extra_count"] = len(extra_cols)
    summary["selected_feature_names"] = extra_cols

    kept_cols = primary_cols + extra_cols
    kept_all = non_roi_cols + kept_cols
    print(
        "[INFO] ROI radiomics 训练集内筛选: "
        f"primary={len(primary_cols)}, candidate_extra={summary['candidate_extra_count']}, "
        f"after_subset={summary['after_subset_filter_count']}, "
        f"after_stability={summary['after_stability_filter_count']}, "
        f"after_missing={summary['after_missing_count']}, after_variance={summary['after_variance_count']}, "
        f"after_corr={summary['after_correlation_count']}, selected_extra={len(extra_cols)}"
    )
    return X[kept_all].copy(), kept_cols, summary


def _build_residual_gain_pipeline(X_train: pd.DataFrame) -> Pipeline:
    num_cols = X_train.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = [c for c in X_train.columns if c not in num_cols]
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", Pipeline([("imp", SimpleImputer(strategy="median")), ("sc", StandardScaler())]), num_cols),
            ("cat", Pipeline([("imp", SimpleImputer(strategy="most_frequent")), ("oh", OneHotEncoder(handle_unknown="ignore"))]), cat_cols),
        ]
    )
    clf = LogisticRegression(
        max_iter=1000,
        solver="liblinear",
        class_weight="balanced",
        random_state=RANDOM_STATE,
    )
    return Pipeline([("pre", preprocessor), ("clf", clf)])


def _score_roi_radiomics_by_baseline_gain(
    X: pd.DataFrame,
    y: np.ndarray,
    train_indices: np.ndarray,
    non_roi_cols: List[str],
    extra_cols: List[str],
) -> List[Tuple[str, float]]:
    X_train_base = X.iloc[train_indices][non_roi_cols].copy()
    y_train = np.asarray(y[train_indices]).astype(int)
    if X_train_base.empty or len(extra_cols) == 0:
        return [(col, -np.inf) for col in extra_cols]

    cls_counts = pd.Series(y_train).value_counts()
    if cls_counts.shape[0] < 2 or int(cls_counts.min()) < 2:
        scores, _ = f_classif(X.iloc[train_indices][extra_cols].fillna(0.0).to_numpy(dtype=np.float64), y_train)
        return [(col, float(score if pd.notna(score) else -np.inf)) for col, score in zip(extra_cols, scores)]

    n_splits = min(5, int(cls_counts.min()))
    n_splits = max(2, n_splits)
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)

    try:
        base_pipe = _build_residual_gain_pipeline(X_train_base)
        base_prob = cross_val_predict(base_pipe, X_train_base, y_train, cv=cv, method="predict_proba")[:, 1]
        base_prob = np.clip(base_prob, 1e-5, 1 - 1e-5)
        base_loss = float(log_loss(y_train, base_prob))
    except Exception as exc:
        print(f"[WARN] ROI baseline residual scoring失败，回退单变量筛选: {exc}")
        scores, _ = f_classif(X.iloc[train_indices][extra_cols].fillna(0.0).to_numpy(dtype=np.float64), y_train)
        return [(col, float(score if pd.notna(score) else -np.inf)) for col, score in zip(extra_cols, scores)]

    score_pairs: List[Tuple[str, float]] = []
    for col in extra_cols:
        try:
            X_aug = pd.concat([X_train_base, X.iloc[train_indices][[col]].copy()], axis=1)
            aug_pipe = _build_residual_gain_pipeline(X_aug)
            aug_prob = cross_val_predict(aug_pipe, X_aug, y_train, cv=cv, method="predict_proba")[:, 1]
            aug_prob = np.clip(aug_prob, 1e-5, 1 - 1e-5)
            aug_loss = float(log_loss(y_train, aug_prob))
            gain = base_loss - aug_loss
            score_pairs.append((col, gain))
        except Exception:
            score_pairs.append((col, -np.inf))
    return score_pairs


def _harmonize_roi_radiomics_by_source(
    X: pd.DataFrame,
    train_indices: np.ndarray,
    roi_feature_cols: List[str],
    source_labels: Optional[np.ndarray],
) -> Tuple[pd.DataFrame, Dict[str, object]]:
    summary: Dict[str, object] = {
        "enabled": bool(ROI_RAD_HARMONIZE),
        "applied": False,
        "min_batch_n": int(ROI_RAD_HARMONIZE_MIN_BATCH_N),
        "batches": [],
        "extra_feature_count": 0,
    }
    if not bool(ROI_RAD_HARMONIZE):
        return X, summary
    if source_labels is None or len(source_labels) != len(X):
        return X, summary

    extra_cols = [col for col in roi_feature_cols if col not in ROI_SELECTION_PRIMARY_SET]
    summary["extra_feature_count"] = len(extra_cols)
    if not extra_cols:
        return X, summary

    train_sources = pd.Series(source_labels[train_indices], index=train_indices).astype(str)
    batch_counts = train_sources.value_counts()
    valid_batches = [batch for batch, count in batch_counts.items() if batch != "__test__" and count >= int(ROI_RAD_HARMONIZE_MIN_BATCH_N)]
    summary["batches"] = [{"label": batch, "n": int(batch_counts[batch])} for batch in valid_batches]
    if len(valid_batches) < 2:
        return X, summary

    X = X.copy()
    pooled = X.iloc[train_indices][extra_cols].copy()
    pooled_means = pooled.mean(axis=0, skipna=True)
    pooled_stds = pooled.std(axis=0, skipna=True)

    for batch in valid_batches:
        batch_idx = train_sources.index[train_sources.eq(batch)].to_numpy()
        batch_frame = X.loc[batch_idx, extra_cols]
        batch_means = batch_frame.mean(axis=0, skipna=True)
        batch_stds = batch_frame.std(axis=0, skipna=True)
        for col in extra_cols:
            pooled_mean = pooled_means.get(col)
            batch_mean = batch_means.get(col)
            if pd.isna(pooled_mean) or pd.isna(batch_mean):
                continue
            values = X.loc[batch_idx, col]
            mask = values.notna()
            if not mask.any():
                continue
            batch_std = batch_stds.get(col)
            pooled_std = pooled_stds.get(col)
            if pd.notna(batch_std) and pd.notna(pooled_std) and batch_std > 1e-8 and pooled_std > 1e-8:
                X.loc[batch_idx[mask.to_numpy()], col] = ((values[mask] - batch_mean) / batch_std) * pooled_std + pooled_mean
            else:
                X.loc[batch_idx[mask.to_numpy()], col] = values[mask] - batch_mean + pooled_mean

    summary["applied"] = True
    print(
        "[INFO] ROI radiomics batch harmonization: "
        f"batches={[(item['label'], item['n']) for item in summary['batches']]}, "
        f"extra_features={len(extra_cols)}"
    )
    return X, summary


def _clean_special_numeric_strings(s: pd.Series) -> pd.Series:
    """
    将检验值中的特殊字符串统一清洗后再转数值：
    - "<1.2" -> "1.2"
    - ">1000" -> "1000"
    - "*", "未测" -> 缺失
    """
    s = s.astype(str).str.strip()
    s = s.replace(
        {
            "": np.nan,
            "nan": np.nan,
            "NaN": np.nan,
            "*": np.nan,
            "＊": np.nan,
            "未测": np.nan,
        }
    )
    s = s.str.replace(r"^[<>]\s*", "", regex=True)
    return s


def _normalize_image_modalities(value: object) -> List[str]:
    text = str(value or "mr,pet").strip().lower()
    if text in {"", "all"}:
        return ["mr", "pet"]
    items = []
    for part in text.split(","):
        modality = part.strip().lower()
        if modality in {"mr", "pet"} and modality not in items:
            items.append(modality)
    return items or ["mr", "pet"]


def _pool_image_embedding(arr: np.ndarray, pooling: str) -> np.ndarray:
    arr = np.asarray(arr, dtype=np.float32)
    if arr.ndim == 1:
        return arr.reshape(-1)

    pooled_mean = arr.mean(axis=0).reshape(-1)
    mode = str(pooling or "mean").strip().lower()
    if mode == "meanstd":
        pooled_std = arr.std(axis=0).reshape(-1)
        return np.concatenate([pooled_mean, pooled_std], axis=0)
    return pooled_mean


def _load_image_embedding_map(embedding_dir: str, modality: str, pooling: str) -> Dict[str, np.ndarray]:
    suffix = f"_{modality.upper()}.npy"
    id_to_feat: Dict[str, np.ndarray] = {}
    if not embedding_dir or not os.path.isdir(embedding_dir):
        return id_to_feat

    for path in sorted(Path(embedding_dir).glob(f"*{suffix}")):
        pid = str(path.name[: -len(suffix)])
        try:
            arr = np.load(path)
            feat = _pool_image_embedding(arr, pooling)
        except Exception as exc:
            print(f"[WARN] 加载 {modality.upper()} embedding 失败: {path} ({exc})")
            continue
        id_to_feat[pid] = np.asarray(feat, dtype=np.float32)
    return id_to_feat


def _fit_transform_image_pca(
    feature_mat: np.ndarray,
    has_feat: np.ndarray,
    train_indices: np.ndarray,
    target_dim: int,
) -> Tuple[np.ndarray, int]:
    available_train = np.asarray(train_indices)[has_feat[np.asarray(train_indices)]]
    if feature_mat.size == 0 or len(available_train) < 2:
        return np.zeros((feature_mat.shape[0], 0), dtype=np.float32), 0

    train_mat = feature_mat[available_train]
    orig_dim = int(train_mat.shape[1])
    n_components = min(int(target_dim), int(len(available_train)), orig_dim)
    if n_components <= 0:
        return np.zeros((feature_mat.shape[0], 0), dtype=np.float32), 0
    if n_components >= orig_dim:
        return feature_mat.astype(np.float32, copy=False), orig_dim

    pca = PCA(n_components=n_components, random_state=RANDOM_STATE)
    pca.fit(train_mat)
    transformed = np.zeros((feature_mat.shape[0], n_components), dtype=np.float32)
    transformed[has_feat] = pca.transform(feature_mat[has_feat]).astype(np.float32)
    return transformed, n_components


def _attach_image_embedding_features(
    df: pd.DataFrame,
    X: pd.DataFrame,
    train_indices: np.ndarray,
) -> Tuple[pd.DataFrame, List[str], Dict[str, object]]:
    """
    将 embedding_exports 中导出的 MR/PET ViT token embedding 接入结构化特征矩阵。
    流程：
      1) 按患者/影像 ID 对齐 MR/PET .npy 文件；
      2) 对 token 维度做 pooling（默认 mean）得到病例级向量；
      3) 仅用训练集拟合 PCA，将 MR/PET 分别压缩到较低维度；
      4) 将压缩后的 embedding 作为数值列拼接到 X 末尾，供模型辅助分支使用。
    """
    summary: Dict[str, object] = {
        "enabled": bool(USE_IMAGE_EMBEDDINGS or USE_PET_FEATURES),
        "embedding_dir": str(IMAGE_EMBEDDING_DIR),
        "pooling": str(IMAGE_EMBED_POOLING),
        "modalities": [],
        "matched_counts": {},
        "raw_dims": {},
        "reduced_dims": {},
        "feature_names": [],
        "missing_flag_cols": [],
    }
    if not summary["enabled"]:
        return X, [], summary

    img_id_col = find_col(df, ["影像ID", "影像id", "患者id", "patient_id", "ID", "id"])
    if img_id_col is None:
        print("[INFO] 数据中未找到影像/患者ID列，已跳过 image embeddings。")
        return X, [], summary

    embedding_dir = str(IMAGE_EMBEDDING_DIR or "").strip()
    if not embedding_dir or not os.path.isdir(embedding_dir):
        print(f"[INFO] 未找到 image embedding 目录: {embedding_dir}，已跳过。")
        return X, [], summary

    image_ids = df[img_id_col].map(_normalize_patient_id).astype(object).to_numpy()
    modalities = _normalize_image_modalities(IMAGE_EMBED_MODALITIES)
    summary["modalities"] = list(modalities)

    missing_flag_cols: List[str] = []
    appended_cols: List[str] = []

    for modality in modalities:
        id_to_feat = _load_image_embedding_map(embedding_dir, modality, IMAGE_EMBED_POOLING)
        if not id_to_feat:
            print(f"[INFO] 未找到 {modality.upper()} embedding 文件，已跳过该模态。")
            continue

        first = next(iter(id_to_feat.values()))
        raw_dim = int(first.shape[0])
        raw_mat = np.zeros((len(df), raw_dim), dtype=np.float32)
        has_feat = np.zeros(len(df), dtype=bool)
        for i, pid in enumerate(image_ids):
            vec = id_to_feat.get(str(pid))
            if vec is None or vec.shape[0] != raw_dim:
                continue
            raw_mat[i] = vec
            has_feat[i] = True

        summary["matched_counts"][modality] = int(has_feat.sum())
        summary["raw_dims"][modality] = raw_dim
        if int(has_feat.sum()) < 5:
            print(f"[WARN] {modality.upper()} embedding 可用样本数 < 5，已跳过该模态。")
            continue

        target_dim = int(IMAGE_EMBED_MR_DIM if modality == "mr" else IMAGE_EMBED_PET_DIM)
        reduced_mat, reduced_dim = _fit_transform_image_pca(raw_mat, has_feat, train_indices, target_dim)
        if reduced_dim <= 0:
            print(f"[WARN] {modality.upper()} embedding 无法完成训练集内 PCA，已跳过该模态。")
            continue

        summary["reduced_dims"][modality] = int(reduced_dim)
        prefix = f"{modality.upper()}enc"
        embed_cols = [f"{prefix}_{j+1}" for j in range(reduced_dim)]
        embed_df = pd.DataFrame(reduced_mat, columns=embed_cols, index=X.index)

        if bool(IMAGE_EMBED_ADD_MISSING_FLAGS):
            flag_col = f"has_{modality}_embedding"
            X[flag_col] = has_feat.astype(np.float32)
            missing_flag_cols.append(flag_col)

        X = pd.concat([X, embed_df], axis=1)
        appended_cols.extend(embed_cols)
        print(
            f"[INFO] 已接入 {modality.upper()} embeddings: "
            f"matched={int(has_feat.sum())}/{len(df)}, raw_dim={raw_dim}, reduced_dim={reduced_dim}"
        )

    summary["feature_names"] = list(appended_cols)
    summary["missing_flag_cols"] = list(missing_flag_cols)
    return X, appended_cols, summary


def _oof_logistic_scores(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_all: np.ndarray,
    train_indices: np.ndarray,
    available_mask: np.ndarray,
) -> Tuple[np.ndarray, float]:
    model_type = str(IMAGE_RISK_MODEL_TYPE or "logistic").strip().lower()

    def _build_estimator():
        if model_type == "mlp":
            hidden = IMAGE_RISK_MLP_HIDDEN
            if isinstance(hidden, int):
                hidden = (int(hidden),)
            elif isinstance(hidden, (list, tuple)):
                hidden = tuple(int(v) for v in hidden if int(v) > 0)
            else:
                hidden = (16,)
            if not hidden:
                hidden = (16,)
            return make_pipeline(
                StandardScaler(),
                MLPClassifier(
                    hidden_layer_sizes=hidden,
                    activation="relu",
                    solver="lbfgs",
                    alpha=float(IMAGE_RISK_MLP_ALPHA),
                    max_iter=int(IMAGE_RISK_MLP_MAX_ITER),
                    random_state=RANDOM_STATE,
                ),
            )
        return LogisticRegression(
            max_iter=1000,
            class_weight="balanced",
            random_state=RANDOM_STATE,
            solver="liblinear",
        )

    train_available = np.asarray(train_indices)[available_mask[np.asarray(train_indices)]]
    if len(train_available) < 12:
        prevalence = float(np.mean(y_train)) if len(y_train) > 0 else 0.5
        return np.full(X_all.shape[0], prevalence, dtype=np.float32), prevalence

    y_sub = np.asarray(y_train)[available_mask[np.asarray(train_indices)]].astype(int)
    cls_counts = pd.Series(y_sub).value_counts()
    if len(cls_counts) < 2 or int(cls_counts.min()) < 2:
        prevalence = float(np.mean(y_sub)) if len(y_sub) > 0 else 0.5
        scores = np.full(X_all.shape[0], prevalence, dtype=np.float32)
        clf = _build_estimator()
        clf.fit(X_train[available_mask[np.asarray(train_indices)]], y_sub)
        scores[available_mask] = clf.predict_proba(X_all[available_mask])[:, 1].astype(np.float32)
        return scores, prevalence

    n_splits = min(5, int(cls_counts.min()))
    n_splits = max(2, n_splits)
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)
    clf = _build_estimator()

    scores = np.full(X_all.shape[0], float(np.mean(y_sub)), dtype=np.float32)
    oof = cross_val_predict(
        clf,
        X_train[available_mask[np.asarray(train_indices)]],
        y_sub,
        cv=cv,
        method="predict_proba",
    )[:, 1].astype(np.float32)
    scores[train_available] = oof

    clf.fit(X_train[available_mask[np.asarray(train_indices)]], y_sub)
    all_available_idx = np.where(available_mask)[0]
    scores[all_available_idx] = clf.predict_proba(X_all[all_available_idx])[:, 1].astype(np.float32)
    scores[train_available] = oof
    return scores, float(np.mean(y_sub))


def _safe_feature_token(value: object) -> str:
    text = str(value or "").strip().lower()
    text = re.sub(r"[^a-z0-9]+", "_", text)
    text = re.sub(r"_+", "_", text).strip("_")
    return text or "group"


def _oof_logistic_scores_from_global_subset(
    X_all: np.ndarray,
    y_all: np.ndarray,
    subset_train_indices: np.ndarray,
    available_mask: np.ndarray,
) -> Tuple[np.ndarray, float]:
    model_type = str(IMAGE_RISK_MODEL_TYPE or "logistic").strip().lower()

    def _build_estimator():
        if model_type == "mlp":
            hidden = IMAGE_RISK_MLP_HIDDEN
            if isinstance(hidden, int):
                hidden = (int(hidden),)
            elif isinstance(hidden, (list, tuple)):
                hidden = tuple(int(v) for v in hidden if int(v) > 0)
            else:
                hidden = (16,)
            if not hidden:
                hidden = (16,)
            return make_pipeline(
                StandardScaler(),
                MLPClassifier(
                    hidden_layer_sizes=hidden,
                    activation="relu",
                    solver="lbfgs",
                    alpha=float(IMAGE_RISK_MLP_ALPHA),
                    max_iter=int(IMAGE_RISK_MLP_MAX_ITER),
                    random_state=RANDOM_STATE,
                ),
            )
        return LogisticRegression(
            max_iter=1000,
            class_weight="balanced",
            random_state=RANDOM_STATE,
            solver="liblinear",
        )

    subset_train_indices = np.asarray(subset_train_indices, dtype=int)
    train_available = subset_train_indices[available_mask[subset_train_indices]]
    if len(train_available) < 12:
        prevalence = float(np.mean(y_all[subset_train_indices])) if len(subset_train_indices) > 0 else 0.5
        return np.full(X_all.shape[0], prevalence, dtype=np.float32), prevalence

    y_sub = np.asarray(y_all)[train_available].astype(int)
    cls_counts = pd.Series(y_sub).value_counts()
    if len(cls_counts) < 2 or int(cls_counts.min()) < 2:
        prevalence = float(np.mean(y_sub)) if len(y_sub) > 0 else 0.5
        scores = np.full(X_all.shape[0], prevalence, dtype=np.float32)
        clf = _build_estimator()
        clf.fit(X_all[train_available], y_sub)
        all_available_idx = np.where(available_mask)[0]
        scores[all_available_idx] = clf.predict_proba(X_all[all_available_idx])[:, 1].astype(np.float32)
        return scores, prevalence

    n_splits = min(5, int(cls_counts.min()))
    n_splits = max(2, n_splits)
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)
    clf = _build_estimator()

    scores = np.full(X_all.shape[0], float(np.mean(y_sub)), dtype=np.float32)
    oof = cross_val_predict(
        clf,
        X_all[train_available],
        y_sub,
        cv=cv,
        method="predict_proba",
    )[:, 1].astype(np.float32)
    scores[train_available] = oof

    clf.fit(X_all[train_available], y_sub)
    all_available_idx = np.where(available_mask)[0]
    scores[all_available_idx] = clf.predict_proba(X_all[all_available_idx])[:, 1].astype(np.float32)
    scores[train_available] = oof
    return scores, float(np.mean(y_sub))


def _attach_image_risk_score_features(
    df: pd.DataFrame,
    X: pd.DataFrame,
    y: np.ndarray,
    t: np.ndarray,
    keep_treatments: List[str],
    train_indices: np.ndarray,
) -> Tuple[pd.DataFrame, List[str], Dict[str, object]]:
    summary: Dict[str, object] = {
        "enabled": bool(USE_IMAGE_RISK_SCORES),
        "embedding_dir": str(IMAGE_EMBEDDING_DIR),
        "pooling": str(IMAGE_EMBED_POOLING),
        "score_mode": str(IMAGE_RISK_SCORE_MODE),
        "modalities": [],
        "matched_counts": {},
        "reduced_dims": {},
        "risk_cols": [],
        "branch_feature_names": [],
        "missing_flag_cols": [],
    }
    if not bool(USE_IMAGE_RISK_SCORES):
        return X, [], summary

    img_id_col = find_col(df, ["影像ID", "影像id", "患者id", "patient_id", "ID", "id"])
    if img_id_col is None:
        print("[INFO] 数据中未找到影像/患者ID列，已跳过 image risk scores。")
        return X, [], summary

    embedding_dir = str(IMAGE_EMBEDDING_DIR or "").strip()
    if not embedding_dir or not os.path.isdir(embedding_dir):
        print(f"[INFO] 未找到 image embedding 目录: {embedding_dir}，已跳过 image risk scores。")
        return X, [], summary

    image_ids = df[img_id_col].map(_normalize_patient_id).astype(object).to_numpy()
    modalities = _normalize_image_modalities(IMAGE_EMBED_MODALITIES)
    summary["modalities"] = list(modalities)

    risk_cols: List[str] = []
    branch_feature_cols: List[str] = []
    missing_flag_cols: List[str] = []
    modality_scores: List[np.ndarray] = []
    branch_mode = str(IMAGE_RISK_BRANCH_FEATURE_MODE or "raw").strip().lower()
    score_mode = str(IMAGE_RISK_SCORE_MODE or "global").strip().lower()
    if branch_mode != "raw":
        branch_mode = "raw"
    if score_mode not in {"global", "treatment_aware"}:
        score_mode = "global"

    for modality in modalities:
        id_to_feat = _load_image_embedding_map(embedding_dir, modality, IMAGE_EMBED_POOLING)
        if not id_to_feat:
            continue
        first = next(iter(id_to_feat.values()))
        raw_dim = int(first.shape[0])
        raw_mat = np.zeros((len(df), raw_dim), dtype=np.float32)
        has_feat = np.zeros(len(df), dtype=bool)
        for i, pid in enumerate(image_ids):
            vec = id_to_feat.get(str(pid))
            if vec is None or vec.shape[0] != raw_dim:
                continue
            raw_mat[i] = vec
            has_feat[i] = True

        summary["matched_counts"][modality] = int(has_feat.sum())
        if int(has_feat.sum()) < 5:
            continue

        target_dim = int(IMAGE_EMBED_MR_DIM if modality == "mr" else IMAGE_EMBED_PET_DIM)
        reduced_mat, reduced_dim = _fit_transform_image_pca(raw_mat, has_feat, train_indices, target_dim)
        if reduced_dim <= 0:
            continue
        summary["reduced_dims"][modality] = int(reduced_dim)

        if score_mode == "treatment_aware":
            treatment_score_cols: List[str] = []
            treatment_prevalences: List[float] = []
            for treatment_idx, treatment_name in enumerate(keep_treatments):
                subset_train_indices = np.asarray(train_indices)[t[np.asarray(train_indices)] == int(treatment_idx)]
                if len(subset_train_indices) < 12:
                    continue
                score_all, prevalence = _oof_logistic_scores_from_global_subset(
                    X_all=reduced_mat,
                    y_all=np.asarray(y),
                    subset_train_indices=subset_train_indices,
                    available_mask=has_feat,
                )
                token = _safe_feature_token(treatment_name)
                risk_col = f"{modality}_img_risk_{token}"
                X[risk_col] = score_all.astype(np.float32)
                risk_cols.append(risk_col)
                treatment_score_cols.append(risk_col)
                treatment_prevalences.append(float(prevalence))
                modality_scores.append(score_all.astype(np.float32))

                branch_feature_cols.append(risk_col)

                print(
                    f"[INFO] 已生成 {modality.upper()} treatment-aware image score[{treatment_name}]: "
                    f"train_n={len(subset_train_indices)}, matched={int(has_feat.sum())}/{len(df)}, "
                    f"reduced_dim={reduced_dim}, prevalence={prevalence:.4f}"
                )

            if len(treatment_score_cols) >= 2:
                score_mat = X[treatment_score_cols].to_numpy(dtype=np.float32)
                spread_col = f"{modality}_img_risk_spread"
                X[spread_col] = (score_mat.max(axis=1) - score_mat.min(axis=1)).astype(np.float32)
                branch_feature_cols.append(spread_col)
                risk_cols.append(spread_col)
        else:
            score_all, prevalence = _oof_logistic_scores(
                X_train=reduced_mat[np.asarray(train_indices)],
                y_train=np.asarray(y)[np.asarray(train_indices)],
                X_all=reduced_mat,
                train_indices=np.asarray(train_indices),
                available_mask=has_feat,
            )
            risk_col = f"{modality}_img_risk"
            X[risk_col] = score_all.astype(np.float32)
            risk_cols.append(risk_col)
            modality_scores.append(score_all.astype(np.float32))

            branch_feature_cols.append(risk_col)

            print(
                f"[INFO] 已生成 {modality.upper()} image risk score: "
                f"matched={int(has_feat.sum())}/{len(df)}, reduced_dim={reduced_dim}, prevalence={prevalence:.4f}"
            )

        if bool(IMAGE_EMBED_ADD_MISSING_FLAGS):
            flag_col = f"has_{modality}_embedding"
            if flag_col not in X.columns:
                X[flag_col] = has_feat.astype(np.float32)
            missing_flag_cols.append(flag_col)

    if score_mode == "global" and len(modality_scores) > 1:
        stacked = np.vstack(modality_scores)
        X["img_risk_mean"] = stacked.mean(axis=0).astype(np.float32)
        risk_cols.append("img_risk_mean")

    summary["risk_cols"] = list(risk_cols)
    summary["branch_feature_names"] = list(dict.fromkeys(branch_feature_cols))
    summary["missing_flag_cols"] = list(missing_flag_cols)
    return X, risk_cols, summary


def _resolve_train_split_paths() -> List[str]:
    paths: List[str] = []
    configured = TRAIN_DATA_PATHS if "TRAIN_DATA_PATHS" in globals() else None
    if isinstance(configured, (list, tuple)):
        paths.extend([str(path) for path in configured if str(path).strip()])
    elif TRAIN_DATA_PATH:
        paths.append(str(TRAIN_DATA_PATH))

    deduped: List[str] = []
    seen = set()
    for path in paths:
        norm = os.path.abspath(path)
        if norm in seen:
            continue
        seen.add(norm)
        deduped.append(path)
    return deduped


def _concat_fixed_train_frames(train_paths: List[str]) -> pd.DataFrame:
    frames: List[pd.DataFrame] = []
    for path in train_paths:
        if not os.path.exists(path):
            raise FileNotFoundError(f"未找到训练集文件: {path}")
        print(f"[INFO] Loading fixed train split: {path}")
        df_part = pd.read_excel(path, engine="openpyxl").copy()
        df_part["_source_train_file"] = os.path.basename(path)
        frames.append(df_part)

    if not frames:
        raise ValueError("未配置任何训练集文件。")

    df_train = pd.concat(frames, ignore_index=True)
    id_col = _resolve_roi_id_column(df_train)
    if id_col is None:
        print("[WARN] 训练集未找到影像ID列，无法按患者去重；将直接拼接多个训练集文件。")
        return df_train

    df_train["_split_patient_id_norm"] = df_train[id_col].map(_normalize_patient_id)
    before = len(df_train)
    df_train = df_train.drop_duplicates("_split_patient_id_norm", keep="first").reset_index(drop=True)
    removed = before - len(df_train)
    if removed > 0:
        print(f"[INFO] 多训练集文件按 {id_col} 去重: {before} -> {len(df_train)} (移除 {removed} 条重复)")
    return df_train

def load_and_prepare():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    use_fixed_split_files = bool(USE_FIXED_SPLIT_FILES)
    source_labels: Optional[np.ndarray] = None

    if use_fixed_split_files:
        train_paths = _resolve_train_split_paths()
        if not (train_paths and TEST_DATA_PATH):
            raise ValueError("已启用固定划分文件模式，但 TRAIN_DATA_PATHS / TEST_DATA_PATH 未配置。")
        if not os.path.exists(TEST_DATA_PATH):
            raise FileNotFoundError(f"未找到测试集文件: {TEST_DATA_PATH}")

        df_train = _concat_fixed_train_frames(train_paths)
        print(f"[INFO] Loading fixed test split: {TEST_DATA_PATH}")
        df_test = pd.read_excel(TEST_DATA_PATH, engine="openpyxl").copy()
        df_test["_source_train_file"] = "__test__"

        train_id_col = _resolve_roi_id_column(df_train)
        test_id_col = _resolve_roi_id_column(df_test)
        if train_id_col and test_id_col:
            df_test["_split_patient_id_norm"] = df_test[test_id_col].map(_normalize_patient_id)
            test_before = len(df_test)
            df_test = df_test.drop_duplicates("_split_patient_id_norm", keep="first").reset_index(drop=True)
            if test_before != len(df_test):
                print(f"[INFO] 测试集按 {test_id_col} 去重: {test_before} -> {len(df_test)}")

            train_ids = set(df_train["_split_patient_id_norm"]) if "_split_patient_id_norm" in df_train.columns else set(df_train[train_id_col].map(_normalize_patient_id))
            overlap_mask = df_test["_split_patient_id_norm"].isin(train_ids)
            overlap_count = int(overlap_mask.sum())
            if overlap_count > 0:
                df_train_before = len(df_train)
                if "_split_patient_id_norm" not in df_train.columns:
                    df_train["_split_patient_id_norm"] = df_train[train_id_col].map(_normalize_patient_id)
                df_train = df_train.loc[~df_train["_split_patient_id_norm"].isin(set(df_test.loc[overlap_mask, "_split_patient_id_norm"]))].reset_index(drop=True)
                print(f"[WARN] 发现 train/test 重叠病例 {overlap_count} 例；已从训练集中移除，避免数据泄漏: {df_train_before} -> {len(df_train)}")

        split_labels = np.array(["train"] * len(df_train) + ["test"] * len(df_test), dtype=object)
        df = pd.concat([df_train, df_test], ignore_index=True)
        if "_source_train_file" in df.columns:
            source_labels = df["_source_train_file"].fillna("__test__").astype(str).to_numpy()
            df["__source_batch__"] = source_labels
        for col in ["_split_patient_id_norm", "_source_train_file"]:
            if col in df.columns:
                df = df.drop(columns=col)
    else:
        print(f"[INFO] Loading: {DATA_PATH}")
        df = pd.read_excel(DATA_PATH, engine="openpyxl")
        split_labels = None
        source_labels = np.array(["__all__"] * len(df), dtype=object)
        df["__source_batch__"] = source_labels

    roi_df = None
    roi_feature_cols = []
    roi_id_col = None
    if USE_ROI_FEATURES or ROI_STRICT_SENSITIVITY:
        roi_id_col = _resolve_roi_id_column(df)
        if roi_id_col is None:
            raise ValueError("未找到影像/患者ID列，无法对齐 ROI 特征表。")
        roi_df, roi_feature_cols = _load_roi_feature_frame()
        if ROI_STRICT_SENSITIVITY:
            df, split_labels, source_labels = _apply_roi_strict_filter(
                df,
                split_labels,
                source_labels,
                roi_df=roi_df,
                id_col=roi_id_col,
            )

    y_col, y_series = auto_detect_outcome_col(df)
    t_col = auto_detect_treatment_col(df)
    print(f"[INFO] Outcome: {y_col} | Treatment: {t_col}")

    # 基本排除列：结局列本身、治疗列以及各种 ID 列
    exclude = {y_col.split("->")[0], t_col, "__source_batch__"}
    for c in ["id", "患者id", "住院号", "门诊号", "姓名", "病理号"]:
        cc = find_col(df, [c])
        if cc:
            exclude.add(cc)
    
    # 排除包含未来信息的列
    leak_cols = [
        "MP分级",
        "MP",
        "MP_grade",
        "Miller-Payne",
        "MillerPayne",
        "RCB分级",
        "bpCR",
        "手术完整病理",
        "手术日期",
        "原发灶完整病理",
        "淋巴结",
    ]
    for c in leak_cols:
        cc = find_col(df, [c])
        if cc: 
            exclude.add(cc)
            print(f"[WARNING] 排除数据泄漏列: {cc} (与pCR结果高度相关)")
    
    # 其它显式不使用的特征列
    exclude_features = [
        "PET/CT",
        "淋巴结位置（0腋窝/1锁骨）",
        "HRD相关通路",
        "DNA修复相关通路",
        "G2M检查点相关通路",
        "PI3K-AKT-MTOR相关通路",
        "干扰素γ相关通路",
        "血管生成相关通路",
        "Post-SUVmax",
    ]
    # 若当前实验不使用 C2 后随访特征，则一并排除（后续只需在 config 中改开关）
    if not USE_C2_FEATURES:
        exclude_features.extend(["C2后Size-MR", "C2后Size-PET", "C2后最大径", "C2后SUVmax"])
    for c in exclude_features:
        cc = find_col(df, [c])
        if cc:
            exclude.add(cc)
            print(f"[INFO] 排除特征列: {cc}")

    feats = []
    whitelist = list(TABULAR_FEATURE_WHITELIST) if TABULAR_FEATURE_WHITELIST else []
    if whitelist:
        missing = []
        ordered_feats = []
        seen = set()
        for name in whitelist:
            cc = find_col(df, [name])
            if cc is None or cc in exclude:
                missing.append(name)
                continue
            if cc in seen:
                continue
            seen.add(cc)
            ordered_feats.append(cc)
        feats = ordered_feats
        print(f"[INFO] 启用表格特征白名单，命中 {len(feats)}/{len(whitelist)} 列。")
        if missing:
            print(f"[WARN] 白名单中以下列未命中或已被排除: {missing}")
        if not feats:
            print("[WARN] 表格特征白名单未命中，改为变量池/自动选择。")

    if not feats:
        varpool = load_varpool_names(VARPOOL_PATH)
        if varpool:
            feats = [c for c in df.columns if (c in varpool and c not in exclude)]
            if not feats:
                print("[WARN] 变量池列未命中，改为自动选择")

    if not feats:
        # 自动筛列：剔除高缺失/唯一值
        feats = []
        for c in df.columns:
            if c in exclude: continue
            miss = df[c].isna().mean()
            nunique = df[c].nunique(dropna=True)
            if miss >= 0.4: continue
            if nunique <= 1: continue
            feats.append(c)

    # 处理结局二值化
    y_raw = y_series.copy()
    if not pd.api.types.is_numeric_dtype(y_raw):
        # 处理原发灶疗效列：pCR=1, non-pCR=0
        mapping = {"1":1,"yes":1,"y":1,"true":1,"是":1,"阳性":1,"pcr":1,"达到":1,"t0n0":1,"non-pcr":0,"nonpcr":0}
        y = y_raw.astype(str).str.strip().str.lower().map(mapping)
        # 如果映射后还有NaN值，说明是新的值，需要特殊处理
        if y.isna().any():
            print(f"[INFO] 发现新的结局值: {y_raw[y.isna()].unique()}")
            # 对于原发灶疗效列，pCR=1，其他=0
            y = (y_raw.astype(str).str.strip().str.lower() == "pcr").astype(int)
        y = y.fillna(0).astype(int).values
    else:
        y = (y_raw.astype(float) > 0).astype(int).values

    # 自动识别所有治疗方案
    t_raw = df[t_col].astype(str).str.strip().replace({"nan": np.nan})
    # 获取所有治疗方案，按频次排序
    all_treatments = t_raw.value_counts(dropna=True)
    print(f"[INFO] 发现 {len(all_treatments)} 种治疗方案:")
    for i, (treatment, count) in enumerate(all_treatments.items()):
        print(f"  {i+1:2d}. {treatment}: {count} 例")
    
    # 使用所有治疗方案（不限制TOP_K）
    keep = all_treatments.index.tolist()
    mask = t_raw.isin(keep)
    df = df.loc[mask].reset_index(drop=True)
    y = y[mask.values]
    if split_labels is not None:
        split_labels = split_labels[mask.values]
    if source_labels is not None:
        source_labels = source_labels[mask.values]
    t_raw = df[t_col].astype(str).str.strip()
    t_map = {name:i for i,name in enumerate(keep)}
    t = t_raw.map(t_map).astype(int).values
    
    print(f"[INFO] 最终使用 {len(keep)} 种治疗方案进行训练")

    X = df[feats].copy()

    if USE_ROI_FEATURES:
        if roi_df is None or roi_id_col is None:
            raise ValueError("ROI 特征开关已开启，但 ROI 特征表或 ID 列未初始化。")
        X, roi_feature_cols = _attach_roi_features(
            df,
            X,
            roi_df=roi_df,
            id_col=roi_id_col,
            roi_feature_cols=roi_feature_cols,
        )
        X, roi_feature_cols, roi_interaction_cols = _add_roi_interaction_features(X, roi_feature_cols)
        X, roi_feature_cols, roi_primary_composite_cols = _add_roi_primary_composite_features(X, roi_feature_cols)
    else:
        print("[INFO] USE_ROI_FEATURES=False，跳过 ROI 特征拼接。")
        roi_interaction_cols = []
        roi_primary_composite_cols = []

    roi_sample_valid_mask = np.ones(len(X), dtype=bool)
    if USE_ROI_FEATURES and roi_feature_cols:
        roi_sample_valid_mask = _compute_roi_sample_valid_mask(X)

    image_feature_cols: List[str] = []
    image_risk_cols: List[str] = []
    image_embedding_summary: Dict[str, object] = {"enabled": bool(USE_IMAGE_EMBEDDINGS or USE_PET_FEATURES)}
    image_risk_summary: Dict[str, object] = {"enabled": bool(USE_IMAGE_RISK_SCORES)}
    
    # 预处理混合数据类型列
    for col in X.columns:
        if X[col].dtype == 'object':
            # 先清洗特殊检验值，再尝试转换为数值，失败则保持为字符串
            try:
                cleaned = _clean_special_numeric_strings(X[col])
                X[col] = pd.to_numeric(cleaned, errors='coerce')
            except:
                pass
    
    # 重新分类数值和分类列
    num_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = [c for c in X.columns if c not in num_cols]
    
    print(f"[INFO] 数值列数量: {len(num_cols)}, 分类列数量: {len(cat_cols)}")
    print(f"[INFO] 数值列: {num_cols[:5]}...")
    print(f"[INFO] 分类列: {cat_cols[:5]}...")

    # 优化的分层分割：确保测试集中各个治疗方案分布更加均匀
    from collections import Counter
    import random
    
    # 设置随机种子
    random.seed(RANDOM_STATE)
    np.random.seed(RANDOM_STATE)
    
    # 统计每个治疗方案的样本数
    treatment_counts = Counter(t)
    print(f"[INFO] 各治疗方案样本数:")
    for treatment, count in sorted(treatment_counts.items()):
        name = keep[treatment] if (isinstance(treatment, (int, np.integer)) and 0 <= treatment < len(keep)) else str(treatment)
        print(f"  {name}: {count} 例")

    if use_fixed_split_files:
        train_indices = np.where(split_labels == "train")[0]
        test_indices = np.where(split_labels == "test")[0]
        print(f"[INFO] 使用固定 train/test 文件划分。训练集大小: {len(train_indices)} | 测试集大小: {len(test_indices)}")
    else:
        # 计算目标测试集大小（按 7:3 划分；在编码之前按原始样本数计算）
        target_test_size = int(round(0.30 * len(X)))
        print(f"[INFO] 目标测试集大小: {target_test_size}")

        # 为每个治疗方案计算理想的测试集分配数量
        test_indices = []
        train_indices = []

        # 按样本数排序治疗方案
        sorted_treatments = sorted(treatment_counts.items(), key=lambda x: x[1], reverse=True)

        # 第一阶段：为每个治疗方案分配基础测试集样本
        for treatment, count in sorted_treatments:
            treatment_mask = (t == treatment)
            treatment_indices = np.where(treatment_mask)[0]

            if count >= 4:
                n_test = max(1, min(int(round(count * 0.30)), count - 1))
            elif count >= 2:
                n_test = 1
            else:
                n_test = 1 if random.random() < 0.3 else 0

            if n_test > 0:
                test_samples = random.sample(list(treatment_indices), n_test)
                test_indices.extend(test_samples)
                train_indices.extend([idx for idx in treatment_indices if idx not in test_samples])
            else:
                train_indices.extend(treatment_indices)

        # 第二阶段：调整测试集大小到目标大小
        current_test_size = len(test_indices)

        if current_test_size < target_test_size:
            needed = target_test_size - current_test_size
            for treatment, count in sorted_treatments:
                if needed <= 0:
                    break
                treatment_mask = (t == treatment)
                treatment_indices = np.where(treatment_mask)[0]
                available_train = [idx for idx in treatment_indices if idx in train_indices]
                if len(available_train) > 0:
                    n_add = min(needed, len(available_train))
                    additional_samples = random.sample(available_train, n_add)
                    test_indices.extend(additional_samples)
                    train_indices = [idx for idx in train_indices if idx not in additional_samples]
                    needed -= n_add

        elif current_test_size > target_test_size:
            excess = current_test_size - target_test_size
            for treatment, count in sorted_treatments:
                if excess <= 0:
                    break
                treatment_mask = (t == treatment)
                treatment_indices = np.where(treatment_mask)[0]
                test_samples = [idx for idx in treatment_indices if idx in test_indices]
                if len(test_samples) > 1:
                    n_remove = min(excess, len(test_samples) - 1)
                    remove_samples = random.sample(test_samples, n_remove)
                    test_indices = [idx for idx in test_indices if idx not in remove_samples]
                    train_indices.extend(remove_samples)
                    excess -= n_remove

        test_indices = np.array(sorted(test_indices))
        train_indices = np.array(sorted(train_indices))

    if USE_IMAGE_EMBEDDINGS or USE_PET_FEATURES:
        X, image_feature_cols, image_embedding_summary = _attach_image_embedding_features(df, X, train_indices)
    else:
        print("[INFO] USE_IMAGE_EMBEDDINGS=False，跳过 MR/PET image embeddings。")

    if USE_IMAGE_RISK_SCORES:
        X, image_risk_cols, image_risk_summary = _attach_image_risk_score_features(df, X, y, t, keep, train_indices)
        if bool(IMAGE_RISK_USE_SEPARATE_BRANCH):
            image_feature_cols = list(dict.fromkeys(image_feature_cols + list(image_risk_summary.get("branch_feature_names") or [])))
    else:
        print("[INFO] USE_IMAGE_RISK_SCORES=False，跳过 image risk scores。")

    num_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = [c for c in X.columns if c not in num_cols]
    print(f"[INFO] image features 接入后，数值列数量: {len(num_cols)}, 分类列数量: {len(cat_cols)}")

    roi_selection_summary: Dict[str, object] = {"mode": ROI_FEATURE_SELECTION_MODE}
    if USE_ROI_FEATURES and roi_feature_cols:
        X, roi_feature_cols, roi_selection_summary = _select_roi_radiomics_features(
            X,
            y,
            train_indices,
            roi_feature_cols,
            source_labels,
        )
        num_cols = [c for c in num_cols if c in X.columns]
        cat_cols = [c for c in cat_cols if c in X.columns]
        if bool(ROI_USE_SEPARATE_BRANCH):
            num_cols = (
                [c for c in num_cols if c not in set(roi_feature_cols) and c not in set(image_feature_cols)]
                + [c for c in num_cols if c in set(image_feature_cols)]
                + [c for c in num_cols if c in set(roi_feature_cols)]
            )

    roi_harmonization_summary: Dict[str, object] = {"enabled": bool(ROI_RAD_HARMONIZE), "applied": False}
    if USE_ROI_FEATURES and roi_feature_cols:
        X, roi_harmonization_summary = _harmonize_roi_radiomics_by_source(
            X,
            train_indices,
            roi_feature_cols,
            source_labels,
        )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", Pipeline([("imp", SimpleImputer(strategy="median")), ("sc", StandardScaler())]), num_cols),
            ("cat", Pipeline([("imp", SimpleImputer(strategy="most_frequent")), ("oh", OneHotEncoder(handle_unknown="ignore"))]), cat_cols),
        ]
    )

    # 拆分原始X后再拟合预处理（防止数据泄漏）
    Xtr_raw = X.iloc[train_indices].copy()
    Xva_raw = X.iloc[test_indices].copy()

    Xtr_enc = preprocessor.fit_transform(Xtr_raw)
    Xva_enc = preprocessor.transform(Xva_raw)
    if hasattr(Xtr_enc, "toarray"): Xtr_enc = Xtr_enc.toarray()
    if hasattr(Xva_enc, "toarray"): Xva_enc = Xva_enc.toarray()

    # 同步得到全量编码（用于全数据预测导出）
    X_enc = preprocessor.transform(X)
    if hasattr(X_enc, "toarray"): X_enc = X_enc.toarray()

    roi_branch_policy = str(ROI_BRANCH_MISSING_POLICY).strip().lower()
    roi_branch_enabled = bool(ROI_USE_SEPARATE_BRANCH and USE_ROI_FEATURES and len(roi_feature_cols) > 0)
    encoded_feature_names = []
    try:
        encoded_feature_names = [str(name) for name in preprocessor.get_feature_names_out()]
    except Exception:
        encoded_feature_names = []
    encoded_num_cols = [name.split("__", 1)[1] for name in encoded_feature_names if name.startswith("num__")]
    encoded_cat_cols = [name for name in encoded_feature_names if name.startswith("cat__")]
    if not encoded_num_cols:
        encoded_num_cols = list(num_cols)
    roi_num_cols = [c for c in encoded_num_cols if c in set(roi_feature_cols)]
    image_num_cols = [c for c in encoded_num_cols if c in set(image_feature_cols)]
    roi_branch_feature_dim = int(len(roi_num_cols) if roi_branch_enabled else 0)
    if roi_branch_enabled and roi_branch_policy == "gate" and roi_branch_feature_dim > 0:
        roi_start = len(encoded_num_cols) - roi_branch_feature_dim
        roi_end = len(encoded_num_cols)
        invalid_mask_all = ~roi_sample_valid_mask
        invalid_train = invalid_mask_all[train_indices]
        invalid_test = invalid_mask_all[test_indices]
        if invalid_train.any():
            Xtr_enc[invalid_train, roi_start:roi_end] = 0.0
        if invalid_test.any():
            Xva_enc[invalid_test, roi_start:roi_end] = 0.0
        if invalid_mask_all.any():
            X_enc[invalid_mask_all, roi_start:roi_end] = 0.0
        print(
            "[INFO] ROI branch gating 已启用: "
            f"policy={roi_branch_policy}, train_invalid={int(invalid_train.sum())}, "
            f"test_invalid={int(invalid_test.sum())}, roi_branch_dim={roi_branch_feature_dim}"
        )

    # 分割数据
    Xtr = Xtr_enc
    Xva = Xva_enc
    ttr = t[train_indices]
    tva = t[test_indices]
    ytr = y[train_indices]
    yva = y[test_indices]
    
    print(f"[INFO] 训练集大小: {len(Xtr)}, 测试集大小: {len(Xva)}")
    print(f"[INFO] 测试集中治疗方案分布:")
    test_treatment_counts = Counter(tva)
    for treatment, count in sorted(test_treatment_counts.items()):
        name = keep[treatment] if (isinstance(treatment, (int, np.integer)) and 0 <= treatment < len(keep)) else str(treatment)
        print(f"  {name}: {count} 例")
    
    print(f"[INFO] 训练集中治疗方案分布:")
    train_treatment_counts = Counter(ttr)
    for treatment, count in sorted(train_treatment_counts.items()):
        name = keep[treatment] if (isinstance(treatment, (int, np.integer)) and 0 <= treatment < len(keep)) else str(treatment)
        print(f"  {name}: {count} 例")

    # 保存预处理与映射
    joblib.dump(preprocessor, os.path.join(OUTPUT_DIR, "preprocessor.joblib"))
    with open(os.path.join(OUTPUT_DIR, "treatment_map.json"), "w", encoding="utf-8") as f:
        json.dump({"keep_order": keep, "map": t_map}, f, ensure_ascii=False, indent=2)

    # 统计图像 / ROI 分支维度（以编码后真实输出为准，避免全缺失列被预处理裁掉时维度错位）
    image_feature_dim = int(len(image_num_cols))
    secondary_feature_dim = int(image_feature_dim + roi_branch_feature_dim if roi_branch_enabled else image_feature_dim)
    meta = {
        "features_used": feats,
        "num_features": encoded_num_cols,
        "cat_features": encoded_cat_cols if encoded_cat_cols else cat_cols,
        "outcome_col": y_col,
        "treatment_col": t_col,
        "keep_treatments": keep,
        "pet_feature_dim": int(secondary_feature_dim),
        "image_feature_names": image_feature_cols,
        "image_feature_dim": int(image_feature_dim),
        "image_embedding_summary": image_embedding_summary,
        "image_risk_feature_names": image_risk_cols,
        "image_risk_summary": image_risk_summary,
        "image_risk_branch_enabled": bool(IMAGE_RISK_USE_SEPARATE_BRANCH),
        "secondary_feature_dim": int(secondary_feature_dim),
        "roi_feature_names": roi_feature_cols,
        "roi_feature_dim": len(roi_feature_cols),
        "roi_branch_enabled": bool(roi_branch_enabled),
        "roi_branch_feature_dim": int(roi_branch_feature_dim),
        "roi_nonbranch_num_feature_dim": int(max(len(encoded_num_cols) - len(roi_num_cols), 0)),
        "roi_branch_missing_policy": roi_branch_policy,
        "roi_valid_sample_count": int(roi_sample_valid_mask.sum()),
        "roi_invalid_sample_count": int((~roi_sample_valid_mask).sum()),
        "roi_invalid_train_count": int((~roi_sample_valid_mask[train_indices]).sum()),
        "roi_invalid_test_count": int((~roi_sample_valid_mask[test_indices]).sum()),
        "roi_strict_sensitivity": bool(ROI_STRICT_SENSITIVITY),
        "roi_feature_selection": roi_selection_summary,
        "roi_harmonization": roi_harmonization_summary,
        "roi_interaction_features": roi_interaction_cols,
        "roi_primary_composite_features": roi_primary_composite_cols,
    }
    with open(os.path.join(OUTPUT_DIR, "data_meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    # 返回训练/测试在过滤后数据框 df 中的原始行索引，便于下游严格对齐到原表行
    return (Xtr, Xva, ttr, tva, ytr, yva, X_enc, df, t_map, keep, y, train_indices, test_indices)
