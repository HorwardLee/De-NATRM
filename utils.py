"""
工具模块 - 提供各种实用功能
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss, roc_curve, precision_recall_curve


def _bootstrap_auroc_ci(y_true, y_prob, n_boot=1000, random_state=None, ci_level=0.95, min_valid=50):
    """
    对 AUROC 做样本级有放回 bootstrap，返回 (ci_low, ci_high)。
    若正负类不足或有效 bootstrap 次数过少，返回 (nan, nan)。
    """
    y_true = np.asarray(y_true, dtype=float).ravel()
    y_prob = np.asarray(y_prob, dtype=float).ravel()
    n = len(y_true)
    if n < 2 or len(np.unique(y_true)) < 2:
        return float("nan"), float("nan")
    rng = np.random.default_rng(random_state)
    scores = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        yt = y_true[idx]
        yp = y_prob[idx]
        if len(np.unique(yt)) < 2:
            continue
        try:
            scores.append(roc_auc_score(yt, yp))
        except ValueError:
            continue
    if len(scores) < min_valid:
        return float("nan"), float("nan")
    alpha = (1.0 - ci_level) / 2.0
    lo, hi = np.quantile(scores, [alpha, 1.0 - alpha])
    return float(lo), float(hi)


def evaluate_and_plots(
    y_true,
    y_prob,
    outdir,
    prefix="val",
    *,
    bootstrap_auroc_ci=False,
    bootstrap_n=1000,
    bootstrap_random_state=None,
    bootstrap_ci_level=0.95,
):
    """
    评估和可视化
    评估模型在事实预测上的性能：L_pred = L(h_k(φ), y^F = X_{i,c1})
    
    Args:
        y_true: 真实标签 (X_{i,c1})
        y_prob: 预测概率 (h_k(φ)的输出)
        outdir: 输出目录
        prefix: 文件前缀
    """
    os.makedirs(outdir, exist_ok=True)
    
    # 计算评估指标
    auroc = roc_auc_score(y_true, y_prob)        # ROC曲线下面积
    # 标准 AP（基于所有唯一分数阈值）
    auprc_std = average_precision_score(y_true, y_prob)
    # 按固定阈值步长0.05近似的 AUPRC（用户需求）
    thresholds = np.arange(0.0, 1.0001, 0.05)
    precisions, recalls = [], []
    for t in thresholds:
        y_pred = (y_prob >= t).astype(int)
        tp = int(((y_pred == 1) & (y_true == 1)).sum())
        fp = int(((y_pred == 1) & (y_true == 0)).sum())
        fn = int(((y_pred == 0) & (y_true == 1)).sum())
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        precisions.append(prec); recalls.append(rec)
    r = np.array(recalls); p = np.array(precisions)
    # 使用“阶梯法”与 sklearn AP 对齐：用后一段 precision 乘以 recall 增量
    order = np.argsort(r)  # 从小到大
    r_sorted = r[order]; p_sorted = p[order]
    auprc_step05 = float(np.sum((r_sorted[1:] - r_sorted[:-1]) * p_sorted[1:]))
    # 报告值改为固定步长0.05版本
    auprc = auprc_step05
    brier = brier_score_loss(y_true, y_prob)     # Brier Score (预测校准度)
    
    auroc_ci_lo, auroc_ci_hi = float("nan"), float("nan")
    if bootstrap_auroc_ci:
        auroc_ci_lo, auroc_ci_hi = _bootstrap_auroc_ci(
            y_true,
            y_prob,
            n_boot=int(bootstrap_n),
            random_state=bootstrap_random_state,
            ci_level=float(bootstrap_ci_level),
        )

    # 保存评估指标
    metrics_path = os.path.join(outdir, f"{prefix}_metrics.txt")
    with open(metrics_path, "w", encoding="utf-8") as f:
        f.write(f"AUROC={auroc:.4f}\nAUPRC={auprc:.4f}\nBrier={brier:.4f}\n")
        if bootstrap_auroc_ci and not (np.isnan(auroc_ci_lo) or np.isnan(auroc_ci_hi)):
            pct = int(round(100 * float(bootstrap_ci_level)))
            f.write(f"AUROC_{pct}pct_CI_low={auroc_ci_lo:.4f}\n")
            f.write(f"AUROC_{pct}pct_CI_high={auroc_ci_hi:.4f}\n")
        elif bootstrap_auroc_ci:
            f.write("AUROC_bootstrap_CI=NA\n")
    if bootstrap_auroc_ci and not (np.isnan(auroc_ci_lo) or np.isnan(auroc_ci_hi)):
        pct = int(round(100 * float(bootstrap_ci_level)))
        print(
            f"[{prefix}] AUROC={auroc:.4f} ({pct}% CI {auroc_ci_lo:.4f}-{auroc_ci_hi:.4f}) | "
            f"AUPRC={auprc:.4f} | Brier={brier:.4f}"
        )
    else:
        print(f"[{prefix}] AUROC={auroc:.4f} | AUPRC={auprc:.4f} | Brier={brier:.4f}")
        if bootstrap_auroc_ci:
            print(f"[{prefix}] [WARN] AUROC bootstrap CI 无法可靠估计（有效重抽样过少或类别不足），见 {metrics_path}")

    # ROC曲线
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    plt.figure()
    plt.plot(fpr, tpr, label=f"AUC={auroc:.3f}")
    plt.plot([0,1],[0,1],"--")  # 随机分类器基线
    plt.xlabel("FPR"); plt.ylabel("TPR"); plt.title(f"ROC - {prefix}")
    plt.legend(); plt.savefig(os.path.join(outdir, f"{prefix}_roc.png"), dpi=200); plt.close()

    # 精确率-召回率曲线
    prec, rec, _ = precision_recall_curve(y_true, y_prob)
    plt.figure()
    plt.plot(rec, prec, label=f"AP(step=0.05)={auprc:.3f}")
    plt.xlabel("Recall"); plt.ylabel("Precision"); plt.title(f"PR - {prefix}")
    plt.legend(); plt.savefig(os.path.join(outdir, f"{prefix}_pr.png"), dpi=200); plt.close()

    # 校准曲线 (预测概率 vs 实际概率)
    dfc = pd.DataFrame({"y":y_true, "p":y_prob})
    dfc["bin"] = pd.qcut(dfc["p"], q=10, duplicates="drop")
    cal = dfc.groupby("bin").agg(obs=("y","mean"), pred=("p","mean"))
    plt.figure()
    plt.plot([0,1],[0,1],"--")  # 完美校准线
    plt.plot(cal["pred"], cal["obs"], marker="o")
    plt.xlabel("Predicted"); plt.ylabel("Observed"); plt.title(f"Calibration - {prefix}")
    plt.savefig(os.path.join(outdir, f"{prefix}_calibration.png"), dpi=200); plt.close()
    return auroc, auprc, brier, auroc_ci_lo, auroc_ci_hi