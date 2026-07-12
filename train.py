import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import Dataset, DataLoader
from model import mmd_rbf_multi

class TorchDataset(Dataset):
    """
    因果推理数据集
    包含特征X、治疗t、结果y
    """
    def __init__(self, X, t, y):
        import numpy as np
        assert isinstance(X, (list, tuple)) or hasattr(X, "shape")
        self.X = torch.tensor(X, dtype=torch.float32)  # 特征矩阵
        self.t = torch.tensor(t, dtype=torch.long)     # 治疗标签
        self.y = torch.tensor(y, dtype=torch.float32)  # 结果标签
    def __len__(self): return self.X.shape[0]
    def __getitem__(self, i): return self.X[i], self.t[i], self.y[i]

class Trainer:
    """
    Foundation Model 
    损失函数：
    L_total = L_pred + α*dis(p_φ^F, p_φ^CF) + β*L(h_j(φ), y^CF)
    """
    def __init__(
        self,
        model,
        alpha=1.0,
        beta=0.0,
        mmd_sigma=1.0,
        lr=1e-3,
        weight_decay=1e-4,
        device=None,
        cf_method='knn',
        cf_k=1,
        cf_distance='cosine',
        cf_label_mode='observed',
        cf_blend_lambda=0.2,
        cf_distance_weighted=True,
        cf_label_smoothing=0.05,
        cf_warmup_epochs=0,
        cf_ramp_epochs=0,
        factual_loss_type='bce',
        factual_pos_weight=1.0,
        factual_neg_weight=1.0,
        factual_focal_gamma=2.0,
        factual_focal_alpha=0.5,
    ):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)
        # 损失函数权重参数
        self.alpha = alpha      # MMD分布对齐权重 (α)
        self.beta = beta        # 反事实损失权重 (β)
        self.mmd_sigma = mmd_sigma  # MMD核函数参数
        # 反事实近邻参数
        self.cf_method = str(cf_method).strip().lower()
        self.cf_k = int(cf_k)
        self.cf_distance = cf_distance
        self.cf_label_mode = str(cf_label_mode).strip().lower()
        self.cf_blend_lambda = float(cf_blend_lambda)
        self.cf_distance_weighted = bool(cf_distance_weighted)
        self.cf_label_smoothing = float(cf_label_smoothing)
        self.cf_warmup_epochs = max(0, int(cf_warmup_epochs))
        self.cf_ramp_epochs = max(0, int(cf_ramp_epochs))
        self.current_beta = float(beta)
        self.factual_loss_type = str(factual_loss_type).strip().lower()
        self.factual_pos_weight = float(factual_pos_weight)
        self.factual_neg_weight = float(factual_neg_weight)
        self.factual_focal_gamma = float(factual_focal_gamma)
        self.factual_focal_alpha = float(factual_focal_alpha)
        self.opt = torch.optim.Adam(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        self.crit = nn.BCEWithLogitsLoss()

    def _classification_loss(self, logits, targets):
        targets = targets.float()
        loss_type = self.factual_loss_type
        if loss_type == 'bce':
            return self.crit(logits, targets)

        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')

        if loss_type == 'weighted_bce':
            pos_w = max(self.factual_pos_weight, 0.0)
            neg_w = max(self.factual_neg_weight, 0.0)
            weights = neg_w + (pos_w - neg_w) * targets
            return (bce * weights).mean()

        if loss_type == 'focal':
            prob = torch.sigmoid(logits)
            pt = prob * targets + (1.0 - prob) * (1.0 - targets)
            gamma = max(self.factual_focal_gamma, 0.0)
            focal_factor = torch.pow((1.0 - pt).clamp(min=1e-6), gamma)
            alpha = min(max(self.factual_focal_alpha, 0.0), 1.0)
            alpha_t = alpha * targets + (1.0 - alpha) * (1.0 - targets)
            return (alpha_t * focal_factor * bce).mean()

        raise ValueError(f"Unsupported factual loss type: {self.factual_loss_type}")

    def _build_neighbor_weights(self, distances, k):
        if k <= 0:
            return None
        if distances is None or not self.cf_distance_weighted:
            return torch.full((k,), 1.0 / k, device=self.device)
        d = distances.reshape(-1)[:k]
        w = 1.0 / (d + 1e-6)
        w = w / (w.sum() + 1e-8)
        return w

    def _smooth_label(self, value):
        eps = float(self.cf_label_smoothing)
        if eps <= 0:
            return value.clamp(0.0, 1.0)
        return value.clamp(0.0, 1.0) * (1.0 - 2.0 * eps) + eps

    def _beta_for_epoch(self, epoch_idx):
        target = float(self.beta)
        if target <= 0:
            return 0.0
        if epoch_idx <= self.cf_warmup_epochs:
            return 0.0
        if self.cf_ramp_epochs <= 0:
            return target
        progress = min(max(epoch_idx - self.cf_warmup_epochs, 0), self.cf_ramp_epochs)
        return target * (progress / self.cf_ramp_epochs)

    def compute_distance(self, phi1, phi2, method='cosine'):
        """计算样本间的距离"""
        if method == 'euclidean':
            return torch.cdist(phi1.unsqueeze(0), phi2.unsqueeze(0)).squeeze()
        elif method == 'cosine':
            # 确保phi1是2D张量
            if phi1.dim() == 1:
                phi1 = phi1.unsqueeze(0)
            if phi2.dim() == 1:
                phi2 = phi2.unsqueeze(0)
            phi1_norm = phi1 / (phi1.norm(dim=1, keepdim=True) + 1e-8)
            phi2_norm = phi2 / (phi2.norm(dim=1, keepdim=True) + 1e-8)
            cosine_sim = torch.mm(phi1_norm, phi2_norm.t())
            return 1 - cosine_sim
        else:
            raise ValueError(f"Unknown distance method: {method}")

    def find_counterfactual_neighbors(self, phi, t, y, all_logits):
        """为每个样本找到反事实近邻"""
        batch_size = phi.size(0)
        cf_treatment = []
        cf_labels = []
        
        for i in range(batch_size):
            current_phi = phi[i].unsqueeze(0)
            current_t = t[i].item()
            
            # 找到其他治疗组的样本；same_outcome 模式优先使用相同结局的匹配对象
            other_treatment_mask = (t != current_t)
            if self.cf_method == 'knn_same_outcome':
                same_outcome_mask = other_treatment_mask & (y == y[i])
                if same_outcome_mask.sum() > 0:
                    other_treatment_mask = same_outcome_mask

            if other_treatment_mask.sum() == 0:
                # 如果没有其他治疗组，随机选择
                cf_t = (current_t + 1) % self.model.num_treatments
                cf_treatment.append(cf_t)
                with torch.no_grad():
                    cf_labels.append(torch.sigmoid(all_logits[i, cf_t]))
                continue
            
            other_phi = phi[other_treatment_mask]
            other_t = t[other_treatment_mask]

            other_y = y[other_treatment_mask]

            if self.cf_method == 'random':
                k = min(max(1, self.cf_k), len(other_phi))
                topk_indices = torch.randperm(len(other_phi), device=phi.device)[:k]
                distances_topk = None
            else:
                distances = self.compute_distance(current_phi, other_phi, method=self.cf_distance)

                # 选择最近的K个邻居
                k = min(max(1, self.cf_k), len(other_phi))
                if k == 0:
                    # 如果该治疗组没有样本，随机选择一个不同的治疗方案
                    available_treatments = [t for t in range(self.model.num_treatments) if t != current_t]
                    if available_treatments:
                        cf_t = np.random.choice(available_treatments)
                        cf_treatment.append(cf_t)
                        cf_labels.append(0.5)  # 中性预测
                        continue
                    else:
                        continue

                topk_distances, topk_indices = torch.topk(distances.reshape(-1), k, largest=False)
                distances_topk = topk_distances

            # 从K个邻居中选择最常见的治疗方案
            neighbor_treatments = other_t[topk_indices]
            cf_t = neighbor_treatments.mode().values.item()
            cf_treatment.append(cf_t)

            selected = (neighbor_treatments == cf_t)
            if not selected.any():
                selected = torch.ones_like(neighbor_treatments, dtype=torch.bool)

            selected_indices = topk_indices[selected]
            selected_y = other_y[selected_indices]
            selected_pred = torch.sigmoid(all_logits[other_treatment_mask][selected_indices, cf_t])
            if distances_topk is None:
                selected_distances = None
            else:
                selected_distances = distances_topk[selected]

            weights = self._build_neighbor_weights(selected_distances, len(selected_indices))
            observed_label = (selected_y * weights).sum()
            predicted_label = (selected_pred * weights).sum()

            with torch.no_grad():
                if self.cf_label_mode == 'predicted':
                    label = predicted_label
                elif self.cf_label_mode == 'blended':
                    lam = min(max(self.cf_blend_lambda, 0.0), 1.0)
                    label = (1.0 - lam) * observed_label + lam * predicted_label
                else:
                    label = observed_label
                cf_labels.append(self._smooth_label(label))
        
        cf_treatment = torch.tensor(cf_treatment, device=self.device)
        cf_labels = torch.tensor(cf_labels, device=self.device)
        return cf_treatment, cf_labels

    def _step(self, batch):
        """
        单步训练，计算完整损失函数
        对应图片中的：L_pred + α*MMD + β*反事实损失
        """
        x, t, y = [z.to(self.device) for z in batch]
        
        # 事实预测损失：L_pred = L(h_k(φ), y^F = X_{i,c1})
        phi, factual_logits, all_logits = self.model.factual_logits(x, t)
        loss_pred = self._classification_loss(factual_logits, y)
        
        # MMD分布对齐损失：α dis(p_φ^F, p_φ^CF)
        # 按治疗分组，计算不同治疗组间潜在表示φ的分布差异
        if self.alpha > 0:
            groups = [phi[t==k] for k in range(self.model.num_treatments) if (t==k).any()]
            loss_mmd = mmd_rbf_multi(groups, sigma=self.mmd_sigma)
        else:
            loss_mmd = phi.new_tensor(0.0)
        
        # 反事实一致性损失（使用KNN匹配）
        loss_cf = 0.0
        if self.current_beta > 0:
            cf_treatment, cf_labels = self.find_counterfactual_neighbors(phi, t, y, all_logits)
            cf_logits = all_logits.gather(1, cf_treatment.unsqueeze(1)).squeeze(1)
            loss_cf = self._classification_loss(cf_logits, cf_labels)

        # 总损失：L_pred + α*MMD + β*反事实一致性损失
        loss = loss_pred + self.alpha * loss_mmd + self.current_beta * loss_cf
        
        return loss, dict(
            pred=float(loss_pred.item()), 
            mmd=float(loss_mmd.item()), 
            cf=float(loss_cf.item()) if isinstance(loss_cf, torch.Tensor) else 0.0,
            total=float(loss.item())
        )

    def fit(self, loader: DataLoader, epochs=20, val_loader: DataLoader=None, early_stopping_patience: int=None):
        """
        训练模型
        支持可选的验证集早停：当验证损失在 patience 个epoch内无改善则提前停止，并恢复最佳权重。
        """
        self.model.train()
        best_state = None
        best_val = float('inf')
        epochs_no_improve = 0
        for ep in range(1, epochs+1):
            self.current_beta = self._beta_for_epoch(ep)
            agg, n = {"pred":0.0,"mmd":0.0,"cf":0.0,"total":0.0}, 0
            for batch in loader:
                self.opt.zero_grad()
                loss, parts = self._step(batch)
                loss.backward()
                self.opt.step()
                for k in agg: agg[k]+=parts[k]
                n+=1
            log_msg = f"[Epoch {ep:03d}] loss={agg['total']/n:.4f} | pred={agg['pred']/n:.4f} | mmd={agg['mmd']/n:.4f}"
            if self.beta > 0:
                log_msg += f" | cf={agg['cf']/n:.4f} | beta={self.current_beta:.4f}"
            # 可选验证
            if val_loader is not None:
                self.model.eval()
                with torch.no_grad():
                    v_total, v_n = 0.0, 0
                    for bx, bt, by in val_loader:
                        bx, bt, by = bx.to(self.device), bt.to(self.device), by.to(self.device)
                        phi, factual_logits, _ = self.model.factual_logits(bx, bt)
                        v_loss = self._classification_loss(factual_logits, by)
                        v_total += float(v_loss.item()); v_n += 1
                val_loss = v_total / max(1, v_n)
                log_msg += f" || val={val_loss:.4f}"
                # 早停
                if early_stopping_patience is not None:
                    if val_loss < best_val - 1e-6:
                        best_val = val_loss
                        epochs_no_improve = 0
                        best_state = {k: v.detach().cpu().clone() for k, v in self.model.state_dict().items()}
                self.model.train()
            print(log_msg)
        # 训练结束后若有更优权重则恢复
        if val_loader is not None and best_state is not None:
            self.model.load_state_dict(best_state)
    @torch.no_grad()
    def predict_proba(self, X, t):
        """
        预测事实概率
        h_k(φ) → p(y^F)
        """
        self.model.eval()
        X = torch.tensor(X, dtype=torch.float32).to(self.device)
        t = torch.tensor(t, dtype=torch.long).to(self.device)
        _, logit, _ = self.model.factual_logits(X, t)
        return torch.sigmoid(logit).cpu().numpy()  # 转换为概率

    @torch.no_grad()
    def predict_all(self, X):
        """
        预测所有治疗下的概率
         φ → [h1(φ), h2(φ), ..., hK(φ)] → [p1, p2, ..., pK]
        用于反事实推断和个体化治疗推荐
        """
        self.model.eval()
        X = torch.tensor(X, dtype=torch.float32).to(self.device)
        _, logits_all = self.model.forward(X)
        import torch as th
        return th.cat([th.sigmoid(z) for z in logits_all], dim=1).cpu().numpy()
