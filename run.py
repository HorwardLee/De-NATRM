import os, json
import random
import warnings
from datetime import datetime

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, WeightedRandomSampler

# 忽略警告信息
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings(
    "ignore",
    message="invalid value encountered in cast",
    category=RuntimeWarning,
    module="pandas.core.util.hashing",
)

from config import (
    OUTPUT_DIR,
    DATA_PATH,
    USE_FIXED_SPLIT_FILES,
    TRAIN_DATA_PATHS,
    TEST_DATA_PATH,
    TOP_K_TREATMENTS,
    EPOCHS,
    BATCH_SIZE,
    LR,
    WEIGHT_DECAY,
    ALPHA_MMD,
    BETA_CF,
    MMD_SIGMA,
    CF_METHOD,
    CF_K,
    CF_DISTANCE,
    CF_LABEL_MODE,
    CF_BLEND_LAMBDA,
    CF_DISTANCE_WEIGHTED,
    CF_LABEL_SMOOTHING,
    CF_WARMUP_EPOCHS,
    CF_RAMP_EPOCHS,
    REP_DIM,
    ENC_HIDDEN,
    HEAD_HIDDEN,
    DROPOUT,
    BN,
    RANDOM_STATE,
    EXPLICIT_OUTCOME_COL,
    EXPLICIT_TREATMENT_COL,
    THRESHOLD_STRATEGY,
    TARGET_RECALL,
    TARGET_PRECISION,
    ENABLE_VAL_EARLY_STOPPING,
    VAL_SPLIT_RATIO,
    EARLY_STOPPING_PATIENCE,
    USE_BATCH_BALANCED_SAMPLER,
    BATCH_SAMPLER_GROUP_BY_Y,
    BATCH_SAMPLER_POWER,
    AUTO_AFTER_TRAINING_VIS,
    CODE_VERSION,
    BOOTSTRAP_AUROC_CI,
    BOOTSTRAP_N,
    BOOTSTRAP_CI_LEVEL,
    FACTUAL_LOSS_TYPE,
    FACTUAL_POS_WEIGHT,
    FACTUAL_NEG_WEIGHT,
    FACTUAL_FOCAL_GAMMA,
    FACTUAL_FOCAL_ALPHA,
)


def _public_data_guidance_message(exc: Exception) -> str:
    docs_path = os.path.join("docs", "github-release", "data_availability.md")
    lines = [
        f"[ERROR] {exc}",
        "",
        "De-NATRM is released as paper-linked open-source code, but the protected clinical training tables are not distributed in this public repository.",
        f"See {docs_path} for the public data boundary and expected user-supplied assets.",
        "",
        "Supported environment variable overrides:",
        "  DE_NATRM_DATA_PATH=/path/to/authorized_dataset.xlsx",
        "  DE_NATRM_USE_FIXED_SPLIT_FILES=0|1",
        "  DE_NATRM_TRAIN_DATA_PATHS=/path/train_batch1.xlsx:/path/train_batch2.xlsx:/path/train_batch3.xlsx",
        "  DE_NATRM_TEST_DATA_PATH=/path/test.xlsx",
        "  DE_NATRM_OUTPUT_DIR=/path/to/output_dir",
        "",
    ]
    if bool(USE_FIXED_SPLIT_FILES):
        lines.append("Current runtime is in fixed-split mode and expects user-supplied authorized split files.")
        for idx, path in enumerate(TRAIN_DATA_PATHS or [], start=1):
            lines.append(f"  train split {idx}: {path}")
        lines.append(f"  test split: {TEST_DATA_PATH}")
    else:
        lines.append("Current runtime is in single-table mode and expects one authorized dataset table.")
        lines.append(f"  dataset table: {DATA_PATH}")
    return "\n".join(lines)

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # 确保PyTorch的确定性行为
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f"[INFO] 随机种子已设置为: {seed}")
from data import load_and_prepare
from data import auto_detect_treatment_col
from model import CausalNet
from train import TorchDataset, Trainer
from utils import evaluate_and_plots
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score, roc_curve, precision_recall_curve


def build_model(input_dim: int, num_treatments: int) -> CausalNet:
    """
    构建因果模型 CausalNet。

    读取 data_meta.json 中记录的特征信息（数值列总数 / ROI或图像分支维度），
    以便在存在独立 ROI branch 或 PET 特征时自动构建双分支结构。
    """
    meta_path = os.path.join(OUTPUT_DIR, "data_meta.json")
    num_features = None
    image_dim = 0
    secondary_dim = 0
    pet_dim = 0
    roi_dim = 0
    use_roi_branch = False
    try:
        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
        # num_features 记录的是数值特征列名列表
        num_features_list = meta.get("num_features") or []
        num_features = len(num_features_list)
        image_dim = int(meta.get("image_feature_dim", 0) or 0)
        secondary_dim = int(meta.get("secondary_feature_dim", 0) or 0)
        # 兼容老版本：若不存在 pet_feature_dim，则退化为单分支
        pet_dim = int(meta.get("pet_feature_dim", 0) or 0)
        roi_dim = int(meta.get("roi_branch_feature_dim", 0) or 0)
        use_roi_branch = bool(meta.get("roi_branch_enabled", False))
    except Exception:
        print(f"[WARN] 无法从 {meta_path} 读取特征元数据，将使用单分支 CausalNet。")
        num_features = None
        image_dim = 0
        secondary_dim = 0
        pet_dim = 0
        roi_dim = 0
        use_roi_branch = False

    branch_desc = "single"
    if image_dim > 0 and roi_dim > 0 and use_roi_branch:
        branch_desc = f"tab+image+roi(image_dim={image_dim}, roi_dim={roi_dim})"
    elif secondary_dim > 0:
        branch_desc = f"tab+aux(aux_dim={secondary_dim}, roi_dim={roi_dim})"
    elif use_roi_branch and roi_dim > 0:
        branch_desc = f"clinical+roi(roi_dim={roi_dim})"
    elif pet_dim > 0:
        branch_desc = f"tab+pet(pet_dim={pet_dim})"
    print(
        f"[INFO] 构建模型，治疗方案数量: {num_treatments}，输入维度={input_dim}，"
        f"数值列={num_features}，image维度={image_dim}，aux维度={secondary_dim}，PET维度={pet_dim}，ROI维度={roi_dim}，branch={branch_desc}"
    )
    return CausalNet(
        in_dim=input_dim,
        rep_dim=REP_DIM,
        enc_hidden=ENC_HIDDEN,
        head_hidden=HEAD_HIDDEN,
        num_treatments=num_treatments,
        dropout=DROPOUT,
        bn=BN,
        num_features=num_features,
        image_feature_dim=image_dim,
        secondary_feature_dim=secondary_dim,
        pet_feature_dim=pet_dim,
        roi_feature_dim=roi_dim,
        use_roi_branch=use_roi_branch,
    )


def build_trainer(model: CausalNet) -> Trainer:
    """
    根据全局配置构建 Trainer, 对应损失:
    L_total = L_pred + α*MMD + β*反事实损失
    """
    return Trainer(
        model,
        alpha=ALPHA_MMD,
        beta=BETA_CF,
        mmd_sigma=MMD_SIGMA,
        lr=LR,
        weight_decay=WEIGHT_DECAY,
        cf_method=CF_METHOD,
        cf_k=CF_K,
        cf_distance=CF_DISTANCE,
        cf_label_mode=CF_LABEL_MODE,
        cf_blend_lambda=CF_BLEND_LAMBDA,
        cf_distance_weighted=CF_DISTANCE_WEIGHTED,
        cf_label_smoothing=CF_LABEL_SMOOTHING,
        cf_warmup_epochs=CF_WARMUP_EPOCHS,
        cf_ramp_epochs=CF_RAMP_EPOCHS,
        factual_loss_type=FACTUAL_LOSS_TYPE,
        factual_pos_weight=FACTUAL_POS_WEIGHT,
        factual_neg_weight=FACTUAL_NEG_WEIGHT,
        factual_focal_gamma=FACTUAL_FOCAL_GAMMA,
        factual_focal_alpha=FACTUAL_FOCAL_ALPHA,
    )

def main():
    """
    训练因果模型、评估性能，并导出各类预测/推荐结果与可视化。
    """
    # 1. 设置随机种子与加载数据
    set_seed(RANDOM_STATE)
    print("[INFO] 加载数据...")
    try:
        Xtr, Xva, ttr, tva, ytr, yva, X_all, df_model, t_map, keep, y_all, train_indices, test_indices = load_and_prepare()
    except (FileNotFoundError, ValueError) as exc:
        raise SystemExit(_public_data_guidance_message(exc)) from exc

    # 获取全数据的治疗方案（使用自动识别列，避免中英文列名不一致）
    treatment_col_used = auto_detect_treatment_col(df_model)
    t_all = df_model[treatment_col_used].astype(str).str.strip().values

    # 可选：在训练集内部再切一小块验证集做早停（不改最终测试集）
    tr_idx = np.arange(len(Xtr))
    va_idx = np.array([], dtype=int)
    val_loader = None
    if bool(ENABLE_VAL_EARLY_STOPPING) and len(Xtr) >= 40:
        split_ratio = float(VAL_SPLIT_RATIO)
        split_ratio = min(max(split_ratio, 0.05), 0.40)

        strat_labels = np.array([f"{int(tt)}_{int(yy)}" for tt, yy in zip(ttr, ytr)], dtype=object)
        strat_series = pd.Series(strat_labels)
        # 仅当每个分层桶样本数>=2时才做分层切分
        if len(strat_series.value_counts()) >= 2 and int(strat_series.value_counts().min()) >= 2:
            stratify = strat_labels
        else:
            stratify = None

        try:
            tr_idx, va_idx = train_test_split(
                np.arange(len(Xtr)),
                test_size=split_ratio,
                random_state=RANDOM_STATE,
                stratify=stratify,
            )
        except Exception as exc:
            print(f"[WARN] 训练集内分层验证切分失败，退化为随机切分: {exc}")
            tr_idx, va_idx = train_test_split(
                np.arange(len(Xtr)),
                test_size=split_ratio,
                random_state=RANDOM_STATE,
                stratify=None,
            )

        dval = TorchDataset(Xtr[va_idx], ttr[va_idx], ytr[va_idx])
        val_loader = DataLoader(dval, batch_size=BATCH_SIZE, shuffle=False, drop_last=False)
        print(f"[INFO] 启用训练集内 early stopping。sub-train={len(tr_idx)} | sub-val={len(va_idx)}")

    dtr = TorchDataset(Xtr[tr_idx], ttr[tr_idx], ytr[tr_idx])

    sampler = None
    sampler_group_count = 0
    if bool(USE_BATCH_BALANCED_SAMPLER):
        if "__source_batch__" in df_model.columns:
            source_all = df_model["__source_batch__"].astype(str).to_numpy()
            source_train = source_all[train_indices][tr_idx]
            y_sub = ytr[tr_idx]
            if bool(BATCH_SAMPLER_GROUP_BY_Y):
                keys = np.array([f"{src}|y={int(lbl)}" for src, lbl in zip(source_train, y_sub)], dtype=object)
            else:
                keys = np.array(source_train, dtype=object)
            key_counts = pd.Series(keys).value_counts()
            weights = np.array(
                [1.0 / max(float(key_counts.get(k, 1.0)), 1.0) for k in keys],
                dtype=np.float64,
            )
            power = float(BATCH_SAMPLER_POWER)
            if abs(power - 1.0) > 1e-8:
                weights = np.power(weights, power)
            sampler = WeightedRandomSampler(
                weights=torch.as_tensor(weights, dtype=torch.double),
                num_samples=len(weights),
                replacement=True,
            )
            sampler_group_count = int(key_counts.shape[0])
            print(
                "[INFO] 启用 batch balanced sampler: "
                f"groups={sampler_group_count}, group_by_y={bool(BATCH_SAMPLER_GROUP_BY_Y)}, power={power:.3f}"
            )
        else:
            print("[WARN] USE_BATCH_BALANCED_SAMPLER=True 但数据中没有 __source_batch__，已退化为普通 shuffle。")

    # 2. 构建模型与训练器
    actual_num_treatments = len(keep)
    model = build_model(Xtr.shape[1], actual_num_treatments)
    trainer = build_trainer(model)

    # 3. 训练 Foundation 模型，优化 L_pred + α*MMD + β*反事实损失
    # 启用 BN 时，训练态下 batch 内样本数必须 >1，否则最后一批若只有 1 条会报错：
    # ValueError: Expected more than 1 value per channel when training
    ltr = DataLoader(
        dtr,
        batch_size=BATCH_SIZE,
        shuffle=(sampler is None),
        sampler=sampler,
        drop_last=bool(BN),
    )
    print(f"[INFO] 开始训练，训练轮数: {EPOCHS}")
    trainer.fit(
        ltr,
        epochs=EPOCHS,
        val_loader=val_loader,
        early_stopping_patience=int(EARLY_STOPPING_PATIENCE) if val_loader is not None else None,
    )

    # 模型评估 
    # 训练集评估：计算AUROC、AUPRC、Brier Score等指标
    print("\n[INFO] 评估训练集性能...")
    ytr_prob = trainer.predict_proba(Xtr, ttr)
    tr_auroc, tr_auprc, tr_brier, tr_auroc_ci_lo, tr_auroc_ci_hi = evaluate_and_plots(
        ytr,
        ytr_prob,
        OUTPUT_DIR,
        prefix="train",
        bootstrap_auroc_ci=BOOTSTRAP_AUROC_CI,
        bootstrap_n=BOOTSTRAP_N,
        bootstrap_random_state=RANDOM_STATE,
        bootstrap_ci_level=BOOTSTRAP_CI_LEVEL,
    )
    # 测试集评估：计算AUROC、AUPRC、Brier Score等指标
    print("\n[INFO] 评估测试集性能...")
    yva_prob = trainer.predict_proba(Xva, tva)
    te_auroc, te_auprc, te_brier, te_auroc_ci_lo, te_auroc_ci_hi = evaluate_and_plots(
        yva,
        yva_prob,
        OUTPUT_DIR,
        prefix="test",
        bootstrap_auroc_ci=BOOTSTRAP_AUROC_CI,
        bootstrap_n=BOOTSTRAP_N,
        bootstrap_random_state=RANDOM_STATE + 1,
        bootstrap_ci_level=BOOTSTRAP_CI_LEVEL,
    )
    if len(va_idx) > 0:
        print("\n[INFO] 评估训练集内验证子集性能...")
        ysubval_prob = trainer.predict_proba(Xtr[va_idx], ttr[va_idx])
        evaluate_and_plots(
            ytr[va_idx],
            ysubval_prob,
            OUTPUT_DIR,
            prefix="subval",
            bootstrap_auroc_ci=False,
        )
    # 阈值优化
    def pick_threshold(y_true, y_prob):
        if THRESHOLD_STRATEGY == 'fixed':
            return 0.5
        if THRESHOLD_STRATEGY == 'youden':
            fpr, tpr, thr = roc_curve(y_true, y_prob)
            j = tpr - fpr
            return float(thr[np.argmax(j)])
        if THRESHOLD_STRATEGY == 'f1':
            prec, rec, thr = precision_recall_curve(y_true, y_prob)
            f1 = 2*prec*rec/(prec+rec+1e-12)
            thr = np.append(thr, 1.0)  # 对齐长度
            return float(thr[np.nanargmax(f1)])
        if THRESHOLD_STRATEGY == 'precision_at_recall':
            prec, rec, thr = precision_recall_curve(y_true, y_prob)
            ok = np.where(rec >= TARGET_RECALL)[0]
            if len(ok)==0: return 0.5
            best = ok[np.argmax(prec[ok])]
            thr = np.append(thr, 1.0)
            return float(thr[best])
        if THRESHOLD_STRATEGY == 'recall_at_precision':
            prec, rec, thr = precision_recall_curve(y_true, y_prob)
            ok = np.where(prec >= TARGET_PRECISION)[0]
            if len(ok)==0: return 0.5
            best = ok[np.argmax(rec[ok])]
            thr = np.append(thr, 1.0)
            return float(thr[best])
        return 0.5

    # 阈值只在训练集上选择，再固定到测试集评估，避免直接用测试集挑阈值
    best_thr = pick_threshold(ytr, ytr_prob)
    ytr_pred = (ytr_prob >= best_thr).astype(int)
    tr_p = precision_score(ytr, ytr_pred, zero_division=0)
    tr_r = recall_score(ytr, ytr_pred, zero_division=0)
    print(f"[INFO] Threshold strategy={THRESHOLD_STRATEGY} | selected threshold={best_thr:.6f}")
    print(f"[train] Precision={tr_p:.4f} | Recall={tr_r:.4f}")

    yva_pred = (yva_prob >= best_thr).astype(int)
    te_p = precision_score(yva, yva_pred, zero_division=0)
    te_r = recall_score(yva, yva_pred, zero_division=0)
    print(f"[test ] Precision={te_p:.4f} | Recall={te_r:.4f}")

    # 追加实验结果到当前 tpCR 汇总 CSV，并记录代码版本
    try:
        summary_row = {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "code_version": str(CODE_VERSION),
            "data_file": os.path.basename(DATA_PATH),
            "outcome_col": str(EXPLICIT_OUTCOME_COL),
            "treatment_col": str(EXPLICIT_TREATMENT_COL),
            "train_n": int(len(Xtr)),
            "test_n": int(len(Xva)),
            "n_treatments": int(len(keep)),
            "treatment_names": json.dumps(list(keep), ensure_ascii=False),
            "epochs": int(EPOCHS),
            "batch_size": int(BATCH_SIZE),
            "lr": float(LR),
            "weight_decay": float(WEIGHT_DECAY),
            "alpha_mmd": float(ALPHA_MMD),
            "beta_cf": float(BETA_CF),
            "mmd_sigma": float(MMD_SIGMA),
            "cf_method": str(CF_METHOD),
            "cf_k": int(CF_K),
            "cf_distance": str(CF_DISTANCE),
            "cf_label_mode": str(CF_LABEL_MODE),
            "cf_blend_lambda": float(CF_BLEND_LAMBDA),
            "cf_distance_weighted": bool(CF_DISTANCE_WEIGHTED),
            "cf_label_smoothing": float(CF_LABEL_SMOOTHING),
            "cf_warmup_epochs": int(CF_WARMUP_EPOCHS),
            "cf_ramp_epochs": int(CF_RAMP_EPOCHS),
            "rep_dim": int(REP_DIM),
            "enc_hidden": json.dumps(list(ENC_HIDDEN), ensure_ascii=False),
            "head_hidden": json.dumps(list(HEAD_HIDDEN), ensure_ascii=False),
            "dropout": float(DROPOUT),
            "bn": bool(BN),
            "factual_loss_type": str(FACTUAL_LOSS_TYPE),
            "factual_pos_weight": float(FACTUAL_POS_WEIGHT),
            "factual_neg_weight": float(FACTUAL_NEG_WEIGHT),
            "factual_focal_gamma": float(FACTUAL_FOCAL_GAMMA),
            "factual_focal_alpha": float(FACTUAL_FOCAL_ALPHA),
            "threshold_strategy": str(THRESHOLD_STRATEGY),
            "enable_val_early_stopping": bool(ENABLE_VAL_EARLY_STOPPING),
            "val_split_ratio": float(VAL_SPLIT_RATIO),
            "early_stopping_patience": int(EARLY_STOPPING_PATIENCE),
            "sub_train_n": int(len(tr_idx)),
            "sub_val_n": int(len(va_idx)),
            "use_batch_balanced_sampler": bool(USE_BATCH_BALANCED_SAMPLER),
            "batch_sampler_group_by_y": bool(BATCH_SAMPLER_GROUP_BY_Y),
            "batch_sampler_power": float(BATCH_SAMPLER_POWER),
            "batch_sampler_group_count": int(sampler_group_count),
            "selected_threshold": float(best_thr),
            "train_auroc": float(tr_auroc),
            "train_auroc_ci_low": float(tr_auroc_ci_lo),
            "train_auroc_ci_high": float(tr_auroc_ci_hi),
            "train_auprc": float(tr_auprc),
            "train_brier": float(tr_brier),
            "train_precision": float(tr_p),
            "train_recall": float(tr_r),
            "test_auroc": float(te_auroc),
            "test_auroc_ci_low": float(te_auroc_ci_lo),
            "test_auroc_ci_high": float(te_auroc_ci_hi),
            "test_auprc": float(te_auprc),
            "test_brier": float(te_brier),
            "test_precision": float(te_p),
            "test_recall": float(te_r),
        }
        summary_csv = os.path.join(OUTPUT_DIR, "experiment_results_summary.csv")
        row_df = pd.DataFrame([summary_row])
        if os.path.exists(summary_csv):
            old_df = pd.read_csv(summary_csv)
            row_df = pd.concat([old_df, row_df], ignore_index=True)
        row_df.to_csv(summary_csv, index=False, encoding="utf-8-sig")
        print(f"[INFO] 实验结果已追加到汇总表: {summary_csv}")
    except Exception as e:
        print(f"[WARN] 写入实验结果汇总 CSV 失败（不影响主流程）: {e}")

    # 反事实推断 
    # 预测所有治疗下的概率：φ → [h1(φ), h2(φ), ..., hK(φ)] → [p1, p2, ..., pK]
    print("\n[INFO] 预测每个病人在所有治疗方案下的pCR率...")
    
    # 分别预测训练集和测试集
    train_probs = trainer.predict_all(Xtr)
    test_probs = trainer.predict_all(Xva)

    # 计算并保存每个治疗方案的 Precision / Recall（仅在实际接受该方案的患者上评估）
    def save_per_treatment_precision_recall(y, t, probs_all, keep_names, prefix, threshold):
        rows = []
        for idx, name in enumerate(keep_names):
            mask = (t == idx)
            n = int(mask.sum())
            if n == 0:
                continue
            y_true = y[mask]
            y_prob = probs_all[mask, idx]
            y_pred = (y_prob >= threshold).astype(int)
            p = precision_score(y_true, y_pred, zero_division=0)
            r = recall_score(y_true, y_pred, zero_division=0)
            rows.append({
                "treatment": name,
                "n_samples": n,
                "positives": int(y_true.sum()),
                "negatives": int(n - y_true.sum()),
                "precision": round(float(p), 4),
                "recall": round(float(r), 4),
            })
        # 追加Overall一行：基于每位患者其实际治疗的预测
        if len(y) > 0:
            import numpy as _np
            y_true_all = y
            y_prob_all = probs_all[_np.arange(len(y)), t]
            y_pred_all = (y_prob_all >= threshold).astype(int)
            p_all = precision_score(y_true_all, y_pred_all, zero_division=0)
            r_all = recall_score(y_true_all, y_pred_all, zero_division=0)
            rows.append({
                "treatment": "Overall",
                "n_samples": int(len(y_true_all)),
                "positives": int(y_true_all.sum()),
                "negatives": int(len(y_true_all) - y_true_all.sum()),
                "precision": round(float(p_all), 4),
                "recall": round(float(r_all), 4),
            })
        df_pr = pd.DataFrame(rows)
        out_path = os.path.join(OUTPUT_DIR, f"{prefix}_per_treatment_precision_recall.csv")
        df_pr.to_csv(out_path, index=False, encoding="utf-8-sig")
        if not df_pr.empty:
            print(f"[INFO] 每个治疗方案的Precision/Recall({prefix})已保存到: {out_path}")
        else:
            print(f"[WARN] {prefix}集中无可计算Precision/Recall的治疗方案样本")

    save_per_treatment_precision_recall(ytr, ttr, train_probs, keep, "train", best_thr)
    save_per_treatment_precision_recall(yva, tva, test_probs, keep, "test", best_thr)
    
    # 获取实际治疗方案
    # 注意：这里不要再 import DATA_PATH（会在函数内触发局部变量遮蔽），避免 summary 写入时出现
    # "local variable 'DATA_PATH' referenced before assignment" 的 warning。
    df_original = df_model  # 使用已过滤并重置索引后的数据框，保证与划分索引对齐
    
    def create_prediction_table(X, y, probs, t, prefix="", keep=None, t_map=None, row_indices=None, df_source=None):
        """创建预测结果表的通用函数"""
        if df_source is None:
            df_source = df_original
        # 1. 基础信息
        n = len(probs)
        result_data = {
            "患者ID": list(range(1, n + 1)),
            "实际pCR结果": np.asarray(y)[:n]
        }

        # 获取实际治疗方案名称和影像ID
        # 使用分割后的治疗方案索引，而不是原始数据的顺序
        if prefix in ("train", "test", "subval"):
            # 使用传入的划分索引严格对齐原始（过滤后）数据
            actual_treatment_names = [keep[t_val] for t_val in t]
            idx = np.asarray(row_indices) if row_indices is not None else np.arange(n)
            image_ids = (
                df_source.iloc[idx]["影像ID"].values
                if "影像ID" in df_source.columns else (np.arange(len(probs)) + 1)
            )
        else:  # all data
            actual_treatment_names = [keep[t_val] for t_val in t]
            image_ids = (
                df_source["影像ID"].values if "影像ID" in df_source.columns else (np.arange(len(probs)) + 1)
            )

        # 统一长度
        image_ids = np.asarray(image_ids)[:n]
        actual_treatment_names = list(actual_treatment_names)[:n]
        result_data["影像ID"] = image_ids
        result_data["实际治疗方案"] = actual_treatment_names

        # 计算实际治疗方案的预测pCR率
        actual_treatment_probs = []
        for i, actual_treatment in enumerate(actual_treatment_names):
            if actual_treatment in t_map:
                treatment_idx = t_map[actual_treatment]
                actual_treatment_probs.append(probs[i, treatment_idx])
            else:
                actual_treatment_probs.append(np.nan)
        result_data["实际方案预测pCR率"] = actual_treatment_probs

        # 找出每个病人的最佳治疗方案
        best_idx = probs.argmax(axis=1)
        best_treatment_names = [keep[idx] for idx in best_idx]
        best_probs = [probs[i, best_idx[i]] for i in range(len(best_idx))]

        result_data["推荐治疗方案"] = best_treatment_names[:n]
        result_data["推荐方案预测pCR率"] = np.asarray(best_probs)[:n]

        # 计算提升空间
        uplift = []
        for i in range(len(best_probs)):
            if pd.notna(actual_treatment_probs[i]):
                uplift.append(best_probs[i] - actual_treatment_probs[i])
            else:
                uplift.append(np.nan)
        result_data["提升空间"] = np.asarray(uplift)[:n]

        # 添加所有治疗方案的预测pCR率
        for idx, treatment_name in enumerate(keep):
            result_data[f"pCR率[{treatment_name}]"] = probs[:n, idx]

        return result_data
    
    # 创建训练集预测结果表
    print("[INFO] 生成训练集预测结果表...")
    train_result_data = create_prediction_table(Xtr, ytr, train_probs, ttr, "train", keep=keep, t_map=t_map, row_indices=train_indices, df_source=df_original)
    train_df = pd.DataFrame(train_result_data)
    
    # 创建测试集预测结果表
    print("[INFO] 生成测试集预测结果表...")
    test_result_data = create_prediction_table(Xva, yva, test_probs, tva, "test", keep=keep, t_map=t_map, row_indices=test_indices, df_source=df_original)
    test_df = pd.DataFrame(test_result_data)
    
    subval_df = None
    if len(va_idx) > 0:
        print("[INFO] 生成训练集内验证子集预测结果表...")
        subval_probs = trainer.predict_all(Xtr[va_idx])
        subval_row_indices = np.asarray(train_indices)[va_idx]
        subval_result_data = create_prediction_table(
            Xtr[va_idx],
            ytr[va_idx],
            subval_probs,
            ttr[va_idx],
            "subval",
            keep=keep,
            t_map=t_map,
            row_indices=subval_row_indices,
            df_source=df_original,
        )
        subval_df = pd.DataFrame(subval_result_data)
    
    # 保存训练集详细结果
    train_detailed_path = os.path.join(OUTPUT_DIR, "train_per_treatment_probs_detailed.csv")
    train_df.to_csv(train_detailed_path, index=False, encoding="utf-8-sig")
    print(f"[INFO] 训练集详细预测结果已保存到: {train_detailed_path}")
    
    # 保存测试集详细结果
    test_detailed_path = os.path.join(OUTPUT_DIR, "test_per_treatment_probs_detailed.csv")
    test_df.to_csv(test_detailed_path, index=False, encoding="utf-8-sig")
    print(f"[INFO] 测试集详细预测结果已保存到: {test_detailed_path}")
    
    if subval_df is not None:
        subval_detailed_path = os.path.join(OUTPUT_DIR, "subval_per_treatment_probs_detailed.csv")
        subval_df.to_csv(subval_detailed_path, index=False, encoding="utf-8-sig")
        print(f"[INFO] 训练集内验证子集详细预测结果已保存到: {subval_detailed_path}")
    
    # 保存训练集简化版
    train_simple_df = pd.DataFrame(train_probs, columns=[f"pCR率[{name}]" for name in keep])
    train_simple_df.insert(0, "患者ID", range(1, len(train_probs) + 1))
    train_simple_df.insert(
        1,
        "影像ID",
        df_original.iloc[train_indices]["影像ID"].values if "影像ID" in df_original.columns else (np.arange(len(train_probs)) + 1),
    )
    train_simple_df.insert(2, "实际pCR结果", ytr)
    train_simple_df.insert(3, "实际治疗方案", df_original.iloc[train_indices][treatment_col_used].values)
    train_simple_df.insert(4, "实际方案预测pCR率", train_result_data["实际方案预测pCR率"])
    train_simple_df.insert(5, "推荐治疗方案", train_result_data["推荐治疗方案"])
    train_simple_df.insert(6, "推荐方案预测pCR率", train_result_data["推荐方案预测pCR率"])
    train_simple_df.insert(7, "提升空间", train_result_data["提升空间"])
    
    train_simple_path = os.path.join(OUTPUT_DIR, "train_per_treatment_probs.csv")
    train_simple_df.to_csv(train_simple_path, index=False, encoding="utf-8-sig")
    print(f"[INFO] 训练集简化预测结果已保存到: {train_simple_path}")
    
    # 保存测试集简化版
    test_simple_df = pd.DataFrame(test_probs, columns=[f"pCR率[{name}]" for name in keep])
    test_simple_df.insert(0, "患者ID", range(1, len(test_probs) + 1))
    test_simple_df.insert(
        1,
        "影像ID",
        df_original.iloc[test_indices]["影像ID"].values if "影像ID" in df_original.columns else (np.arange(len(test_probs)) + 1),
    )
    test_simple_df.insert(2, "实际pCR结果", yva)
    test_simple_df.insert(3, "实际治疗方案", df_original.iloc[test_indices][treatment_col_used].values)
    test_simple_df.insert(4, "实际方案预测pCR率", test_result_data["实际方案预测pCR率"])
    test_simple_df.insert(5, "推荐治疗方案", test_result_data["推荐治疗方案"])
    test_simple_df.insert(6, "推荐方案预测pCR率", test_result_data["推荐方案预测pCR率"])
    test_simple_df.insert(7, "提升空间", test_result_data["提升空间"])
    
    test_simple_path = os.path.join(OUTPUT_DIR, "test_per_treatment_probs.csv")
    test_simple_df.to_csv(test_simple_path, index=False, encoding="utf-8-sig")
    print(f"[INFO] 测试集简化预测结果已保存到: {test_simple_path}")
    
    if subval_df is not None:
        subval_simple_df = pd.DataFrame(subval_probs, columns=[f"pCR率[{name}]" for name in keep])
        subval_simple_df.insert(0, "患者ID", range(1, len(subval_probs) + 1))
        subval_simple_df.insert(
            1,
            "影像ID",
            df_original.iloc[subval_row_indices]["影像ID"].values if "影像ID" in df_original.columns else (np.arange(len(subval_probs)) + 1),
        )
        subval_simple_df.insert(2, "实际pCR结果", ytr[va_idx])
        subval_simple_df.insert(3, "实际治疗方案", df_original.iloc[subval_row_indices][treatment_col_used].values)
        subval_simple_df.insert(4, "实际方案预测pCR率", subval_result_data["实际方案预测pCR率"])
        subval_simple_df.insert(5, "推荐治疗方案", subval_result_data["推荐治疗方案"])
        subval_simple_df.insert(6, "推荐方案预测pCR率", subval_result_data["推荐方案预测pCR率"])
        subval_simple_df.insert(7, "提升空间", subval_result_data["提升空间"])
        subval_simple_path = os.path.join(OUTPUT_DIR, "subval_per_treatment_probs.csv")
        subval_simple_df.to_csv(subval_simple_path, index=False, encoding="utf-8-sig")
        print(f"[INFO] 训练集内验证子集简化预测结果已保存到: {subval_simple_path}")
    
    # 使用训练好的模型在全数据上进行预测
    print("[INFO] 使用训练好的模型在全数据上预测...")
    all_probs = trainer.predict_all(X_all)
    
    # 创建全数据预测结果表
    # 将治疗方案名称转换为索引
    t_all_indices = np.array([t_map[treatment] for treatment in t_all])
    all_result_data = create_prediction_table(X_all, y_all, all_probs, t_all_indices, "all", keep=keep, t_map=t_map)
    all_df = pd.DataFrame(all_result_data)
    
    # 保存全数据详细预测结果
    all_detailed_path = os.path.join(OUTPUT_DIR, "all_data_predictions_detailed.csv")
    all_df.to_csv(all_detailed_path, index=False, encoding="utf-8-sig")
    print(f"[INFO] 全数据详细预测结果已保存到: {all_detailed_path}")
    
    # 保存全数据简化预测结果
    all_simple_df = pd.DataFrame(all_probs, columns=[f"pCR率[{name}]" for name in keep])
    all_simple_df.insert(0, "患者ID", range(1, len(all_probs) + 1))
    all_simple_df.insert(1, "影像ID", df_original["影像ID"].values if "影像ID" in df_original.columns else (np.arange(len(all_probs)) + 1))
    all_simple_df.insert(2, "实际pCR结果", y_all)
    all_simple_df.insert(3, "实际治疗方案", df_original[treatment_col_used].values)
    all_simple_df.insert(4, "实际方案预测pCR率", all_result_data["实际方案预测pCR率"])
    all_simple_df.insert(5, "推荐治疗方案", all_result_data["推荐治疗方案"])
    all_simple_df.insert(6, "推荐方案预测pCR率", all_result_data["推荐方案预测pCR率"])
    all_simple_df.insert(7, "提升空间", all_result_data["提升空间"])
    
    all_simple_path = os.path.join(OUTPUT_DIR, "all_data_predictions.csv")
    all_simple_df.to_csv(all_simple_path, index=False, encoding="utf-8-sig")
    print(f"[INFO] 全数据简化预测结果已保存到: {all_simple_path}")
    
    # 打印统计摘要
    print("\n" + "="*80)
    print("预测结果统计摘要")
    print("="*80)
    print(f"训练集患者数: {len(train_probs)}")
    print(f"测试集患者数: {len(test_probs)}")
    print(f"治疗方案数: {len(keep)}")
    
    print(f"\n训练集各治疗方案的平均预测pCR率:")
    for idx, treatment_name in enumerate(keep):
        avg_prob = train_probs[:, idx].mean()
        print(f"  {treatment_name:25s}: {avg_prob:.4f}")
    
    print(f"\n测试集各治疗方案的平均预测pCR率:")
    for idx, treatment_name in enumerate(keep):
        avg_prob = test_probs[:, idx].mean()
        print(f"  {treatment_name:25s}: {avg_prob:.4f}")
    
    # 统计训练集推荐情况
    train_best_treatment_names = [keep[idx] for idx in train_probs.argmax(axis=1)]
    print(f"\n训练集推荐治疗方案分布:")
    train_recommendation_counts = pd.Series(train_best_treatment_names).value_counts()
    for treatment, count in train_recommendation_counts.items():
        percentage = (count / len(train_best_treatment_names)) * 100
        print(f"  {treatment:25s}: {count:4d} 例 ({percentage:5.1f}%)")
    
    # 统计测试集推荐情况
    test_best_treatment_names = [keep[idx] for idx in test_probs.argmax(axis=1)]
    print(f"\n测试集推荐治疗方案分布:")
    test_recommendation_counts = pd.Series(test_best_treatment_names).value_counts()
    for treatment, count in test_recommendation_counts.items():
        percentage = (count / len(test_best_treatment_names)) * 100
        print(f"  {treatment:25s}: {count:4d} 例 ({percentage:5.1f}%)")
    
    # 统计训练集提升空间
    train_uplift = train_result_data["提升空间"]
    valid_train_uplift = [u for u in train_uplift if pd.notna(u)]
    if valid_train_uplift:
        positive_train_uplift = [u for u in valid_train_uplift if u > 0.05]  # 提升>5%才算显著
        print(f"\n训练集治疗优化潜力:")
        print(f"  可通过更换治疗方案显著改善的患者: {len(positive_train_uplift)} 例 ({len(positive_train_uplift)/len(valid_train_uplift)*100:.1f}%)")
        print(f"  平均提升空间: {np.mean(valid_train_uplift):.4f}")
        print(f"  最大提升空间: {np.max(valid_train_uplift):.4f}")
    
    # 统计测试集提升空间
    test_uplift = test_result_data["提升空间"]
    valid_test_uplift = [u for u in test_uplift if pd.notna(u)]
    if valid_test_uplift:
        positive_test_uplift = [u for u in valid_test_uplift if u > 0.05]  # 提升>5%才算显著
        print(f"\n测试集治疗优化潜力:")
        print(f"  可通过更换治疗方案显著改善的患者: {len(positive_test_uplift)} 例 ({len(positive_test_uplift)/len(valid_test_uplift)*100:.1f}%)")
        print(f"  平均提升空间: {np.mean(valid_test_uplift):.4f}")
        print(f"  最大提升空间: {np.max(valid_test_uplift):.4f}")
    print("="*80 + "\n")

    # 个体化治疗推荐报告 - 训练集
    train_rec = pd.DataFrame({
        "患者ID": range(1, len(train_probs) + 1),
        "影像ID": (df_original.iloc[train_indices]["影像ID"].values if "影像ID" in df_original.columns else (np.arange(len(train_probs)) + 1)),
        "实际pCR结果": ytr,
        "实际治疗方案": df_original.iloc[train_indices][treatment_col_used].values,
        "实际方案预测pCR率": train_result_data["实际方案预测pCR率"],
        "推荐治疗方案": train_best_treatment_names,
        "推荐方案预测pCR率": train_result_data["推荐方案预测pCR率"],
        "提升空间": train_uplift,
    })
    train_rec.to_csv(os.path.join(OUTPUT_DIR, "train_recommendations.csv"), index=False, encoding="utf-8-sig")
    
    # 个体化治疗推荐报告 - 测试集
    test_rec = pd.DataFrame({
        "患者ID": range(1, len(test_probs) + 1),
        "影像ID": (df_original.iloc[test_indices]["影像ID"].values if "影像ID" in df_original.columns else (np.arange(len(test_probs)) + 1)),
        "实际pCR结果": yva,
        "实际治疗方案": df_original.iloc[test_indices][treatment_col_used].values,
        "实际方案预测pCR率": test_result_data["实际方案预测pCR率"],
        "推荐治疗方案": test_best_treatment_names,
        "推荐方案预测pCR率": test_result_data["推荐方案预测pCR率"],
        "提升空间": test_uplift,
    })
    test_rec.to_csv(os.path.join(OUTPUT_DIR, "test_recommendations.csv"), index=False, encoding="utf-8-sig")
    #print(f"[INFO] 治疗推荐报告已保存到: {os.path.join(OUTPUT_DIR, 'recommendations.csv')}")

    # 模型保存 
    # 保存model
    torch.save(model.state_dict(), os.path.join(OUTPUT_DIR, "causal_net.pt"))
    with open(os.path.join(OUTPUT_DIR, "treatments_keep.json"), "w", encoding="utf-8") as f:
        json.dump(keep, f, ensure_ascii=False, indent=2)

    # 生成训练集热力图矩阵（患者×治疗方案）
    train_treatment_cols = [col for col in train_df.columns if col.startswith('pCR率[')]
    train_treatment_names = [col.replace('pCR率[', '').replace(']', '') for col in train_treatment_cols]
    
    # 创建训练集矩阵
    train_matrix_data = train_df[train_treatment_cols].values
    train_patient_labels = [f"患者{int(row['患者ID'])}" for _, row in train_df.iterrows()]
    
    train_matrix_df = pd.DataFrame(
        train_matrix_data,
        index=train_patient_labels,
        columns=train_treatment_names
    )
    
    # 保存训练集矩阵CSV
    train_matrix_path = os.path.join(OUTPUT_DIR, "train_heatmap_matrix.csv")
    train_matrix_df.to_csv(train_matrix_path, encoding='utf-8-sig')
    
    # 生成测试集热力图矩阵（患者×治疗方案）
    test_treatment_cols = [col for col in test_df.columns if col.startswith('pCR率[')]
    test_treatment_names = [col.replace('pCR率[', '').replace(']', '') for col in test_treatment_cols]
    
    # 创建测试集矩阵
    test_matrix_data = test_df[test_treatment_cols].values
    test_patient_labels = [f"患者{int(row['患者ID'])}" for _, row in test_df.iterrows()]
    
    test_matrix_df = pd.DataFrame(
        test_matrix_data,
        index=test_patient_labels,
        columns=test_treatment_names
    )
    
    # 保存测试集矩阵CSV
    test_matrix_path = os.path.join(OUTPUT_DIR, "test_heatmap_matrix.csv")
    test_matrix_df.to_csv(test_matrix_path, encoding='utf-8-sig')

    # 训练结束后自动生成散点图与指标
    if bool(AUTO_AFTER_TRAINING_VIS):
        try:
            from visualize_embeddings import auto_after_training
            print("[INFO] 自动生成散点图与嵌入质量指标...")
            auto_after_training(seed=RANDOM_STATE)
        except Exception as e:
            print(f"[WARN] 自动可视化阶段跳过：{e}")
    else:
        print("[INFO] AUTO_AFTER_TRAINING_VIS=False，跳过自动可视化。")

    print(f"\n{'='*80}")
    print(f"所有结果已保存到: {OUTPUT_DIR}")
    print(f"{'='*80}\n")

if __name__ == "__main__":
    main()
