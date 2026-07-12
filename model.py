from typing import List, Optional
import torch
import torch.nn as nn


def mlp(in_dim: int, hidden: List[int], out_dim: int, dropout: float = 0.0, bn: bool = False) -> nn.Sequential:
    """
    构建一个简单的多层感知机模块：Linear → BN? → ReLU → Dropout → … → Linear(out_dim)
    """
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
    """
    因果表征学习网络（支持单分支 / 双分支结构）

    - 单分支：所有特征先拼在一起，经一个 encoder 得到 φ，再接 K 个 treatment head。
    - 双分支：将输入特征拆分为表格分支和图像分支，
      分别通过 encoder_tab / encoder_img 得到 φ_tab, φ_img，
      在隐藏层拼接得到 φ = [φ_tab, φ_img]，再接 K 个 head。
    """

    def __init__(
        self,
        in_dim: int,
        rep_dim: int = 128,
        enc_hidden: List[int] = [256, 128],
        head_hidden: List[int] = [64, 32],
        num_treatments: int = 3,
        dropout: float = 0.1,
        bn: bool = True,
        # 下述参数用于多分支：
        # - image_feature_dim: 独立 image branch 的数值维度
        # - roi_feature_dim: 独立 ROI branch 的数值维度
        # - secondary_feature_dim / pet_feature_dim / use_roi_branch: 兼容旧配置
        num_features: Optional[int] = None,
        image_feature_dim: int = 0,
        secondary_feature_dim: int = 0,
        pet_feature_dim: int = 0,
        roi_feature_dim: int = 0,
        use_roi_branch: bool = False,
    ) -> None:
        super().__init__()
        
        self.num_treatments = num_treatments
        self.num_features = num_features  # 数值列总数（经预处理前）
        self.image_feature_dim = int(image_feature_dim) if image_feature_dim is not None else 0
        self.secondary_feature_dim = int(secondary_feature_dim) if secondary_feature_dim is not None else 0
        self.pet_feature_dim = int(pet_feature_dim) if pet_feature_dim is not None else 0
        self.roi_feature_dim = int(roi_feature_dim) if roi_feature_dim is not None else 0
        self.use_roi_branch = bool(use_roi_branch)

        self.branch_mode = "single"
        # 新逻辑优先使用显式 image+roi 多分支
        explicit_image_dim = 0
        if self.num_features is not None and 0 < self.image_feature_dim < self.num_features:
            explicit_image_dim = self.image_feature_dim

        explicit_roi_dim = 0
        if (
            self.use_roi_branch
            and self.num_features is not None
            and 0 < self.roi_feature_dim < self.num_features
        ):
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
        elif (
            self.num_features is not None
            and 0 < self.secondary_feature_dim < self.num_features
        ):
            self.branch_mode = "secondary"
            self.image_feature_dim = self.secondary_feature_dim
            self.roi_feature_dim = 0
        elif (
            self.num_features is not None
            and 0 < self.pet_feature_dim < self.num_features
        ):
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
                # 数值列中，前 tab_num_dim 维为主干数值，后 secondary_feature_dim 维为辅助分支
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
            # 退化为单分支结构：所有特征一起送入一个 encoder
            self.encoder = mlp(in_dim, enc_hidden, rep_dim, dropout=dropout, bn=bn)
            self.rep_dim = rep_dim
            head_in_dim = rep_dim

        # K 个解码头：φ → logit(y) for each treatment
        self.heads = nn.ModuleList(
            [mlp(head_in_dim, head_hidden, 1, dropout=dropout, bn=bn) for _ in range(num_treatments)]
        )

    def forward(self, x: torch.Tensor):
        """
        前向传播：X → φ → K 个输出

        返回:
            phi: 组合后的潜在表示 φ，形状 (B, rep_dim 或 rep_tab+rep_img)
            logits_all: 长度为 K 的列表，每个元素是形状 (B,1) 的 logit
        """
        if self.use_multi_branch:
            num_total = self.num_features
            x_num = x[:, :num_total]
            x_cat = x[:, num_total:] if self.cat_dim > 0 else None

            if self.branch_mode == "tri":
                x_tab_num = x_num[:, : self.tab_num_dim]
                x_img = x_num[:, self.tab_num_dim : self.tab_num_dim + self.image_feature_dim]
                x_roi = x_num[:, self.tab_num_dim + self.image_feature_dim : self.tab_num_dim + self.image_feature_dim + self.roi_feature_dim]

                if x_cat is not None and x_cat.numel() > 0:
                    x_tab_in = torch.cat([x_tab_num, x_cat], dim=1)
                else:
                    x_tab_in = x_tab_num

                phi_tab = self.encoder_tab(x_tab_in)
                phi_img = self.encoder_img(x_img)
                phi_roi = self.encoder_roi(x_roi)
                phi = torch.cat([phi_tab, phi_img, phi_roi], dim=1)
            else:
                x_tab_num = x_num[:, : self.tab_num_dim]
                x_img = x_num[:, self.tab_num_dim : self.tab_num_dim + self.secondary_feature_dim]

                if x_cat is not None and x_cat.numel() > 0:
                    x_tab_in = torch.cat([x_tab_num, x_cat], dim=1)
                else:
                    x_tab_in = x_tab_num

                phi_tab = self.encoder_tab(x_tab_in)
                phi_img = self.encoder_img(x_img)
                phi = torch.cat([phi_tab, phi_img], dim=1)
        else:
            phi = self.encoder(x)

        logits_all = [head(phi) for head in self.heads]
        return phi, logits_all

    def factual_logits(self, x, t):
        """
        计算事实预测的logits
         L_pred = L(h_k(φ), y^F = X_{i,c1})
        """
        phi, logits_all = self.forward(x)
        # 将所有治疗的输出堆叠成 (B, K) 矩阵
        out = torch.stack([z.squeeze(1) for z in logits_all], dim=1)  # (B,K)
        # 根据实际治疗t选择对应的logit
        factual = out.gather(1, t.view(-1,1)).squeeze(1)
        return phi, factual, out

def gaussian_kernel(x, y, sigma=1.0):
    """
    RBF高斯核函数
    用于MMD计算中的核函数
    """
    x = x.unsqueeze(1); y = y.unsqueeze(0)
    dist2 = ((x - y) ** 2).sum(dim=2)  # 计算欧氏距离的平方
    return torch.exp(-dist2 / (2.0 * sigma ** 2))  # RBF核

def mmd_rbf_multi(groups: List[torch.Tensor], sigma=1.0, eps=1e-8):
    """
    多组MMD (Maximum Mean Discrepancy) 计算
    对应 α dis(p_φ^F, p_φ^CF) 分布对齐项
    
    目标：使不同治疗组的潜在表示φ分布尽可能相似，消除混淆偏差
    """
    if len(groups) <= 1:
        return torch.tensor(0.0, device=groups[0].device if groups else "cpu")
    
    mmd_total, pairs = 0.0, 0
    # 计算所有治疗组之间的pairwise MMD
    for i in range(len(groups)):
        for j in range(i + 1, len(groups)):
            xi, xj = groups[i], groups[j]  # 两个治疗组的潜在表示
            if xi.shape[0] < 2 or xj.shape[0] < 2:
                continue
            
            # 计算核矩阵
            k_xx = gaussian_kernel(xi, xi, sigma)  # 组内核矩阵
            k_yy = gaussian_kernel(xj, xj, sigma)  # 组内核矩阵
            k_xy = gaussian_kernel(xi, xj, sigma)  # 组间核矩阵
            
            n, m = xi.size(0), xj.size(0)
            # MMD无偏估计公式
            mmd = (k_xx.sum() - k_xx.diag().sum()) / (n * (n - 1) + eps)
            mmd += (k_yy.sum() - k_yy.diag().sum()) / (m * (m - 1) + eps)
            mmd -= 2 * k_xy.mean()
            # 使用max(0, mmd)确保非负，或使用mmd^2
            mmd = torch.relu(mmd)  # 负数截断为0
            mmd_total += mmd; pairs += 1
    
    return mmd_total / pairs if pairs > 0 else torch.tensor(0.0, device=groups[0].device)
