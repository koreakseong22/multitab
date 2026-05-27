"""
libs/ptarl.py
=============
PTaRL: Prototype-based Tabular Representation Learning via Space Calibration
Paper: Hangting Ye et al., ICLR 2024
Official code: https://github.com/HangtingYe/PTaRL

원본 코드 (train_final_version.py, models.py) 기반 충실 구현.

핵심 구조 (원본 그대로):
  - Stage 1: Encoder(MLP backbone) + Head 일반 학습
  - K-Means: Stage1 완료 후 encoder hidden으로 prototype 초기화
  - Stage 2: OT loss + Diversity loss (contrastive) + Orthogonality loss

원본 loss 수식:
  OT:          loss -= ot_weight * mean(sum(r * cosine(hidden, topic) / norm))
  Diversity:   contrastive on r coordinates (same-class positive pairs)
  Orthogonal:  l1/l2 + 0.5 * |K - l1|   (topic matrix sparsity)

n_clusters = ceil(log2(n_num_features + n_cat_features))  [원본]

search_space.py 파라미터:
  d_hidden, n_blocks, dropout,
  lambda_div (diversity_weight), lambda_orth (r_weight),
  lr, lr_s2 (stage2 lr), weight_decay,
  stage1_epochs, stage2_epochs, early_stopping_rounds
"""

import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.cluster import KMeans
from tqdm import tqdm

from libs.supervised import CallbackContainer, EarlyStopping, CosineAnnealingLR_Warmup
from libs.data import get_batch_size


# ─────────────────────────────────────────────────────────────
# Encoder backbone (원본 Models/mlp.py 구조 기반)
# ─────────────────────────────────────────────────────────────

class ResBlock(nn.Module):
    def __init__(self, d: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(d),
            nn.Linear(d, d * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d * 2, d),
            nn.Dropout(dropout),
        )
    def forward(self, x):
        return x + self.net(x)


class PTaRLEncoder(nn.Module):
    """원본의 MLP encoder (backbone). hidden representation 반환."""
    def __init__(self, input_dim: int, d_hidden: int, n_blocks: int, dropout: float):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(input_dim, d_hidden),
            nn.LayerNorm(d_hidden),
            nn.GELU(),
        )
        # n_blocks=0이면 빈 ModuleList → loop 자체 미실행
        self.blocks = nn.ModuleList([ResBlock(d_hidden, dropout) for _ in range(n_blocks)])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.proj(x)
        for blk in self.blocks:
            z = blk(z)
        return z


class PTaRLNet(nn.Module):
    """
    원본 models.py의 Model 클래스에 대응.

    구성 (원본과 동일):
      encoder  : PTaRLEncoder  (backbone Gf)
      head     : Linear(d_hidden → output_dim)  (Gh)
      reduce   : 3-layer MLP → topic_num  (estimator ϕ, 원본과 동일한 3 hidden layer)
      topic    : nn.Parameter (K, d_hidden)  — K-Means 초기화 후 설정
    """
    def __init__(self, input_dim: int, output_dim: int,
                 d_hidden: int, n_blocks: int, n_proto: int, dropout: float):
        super().__init__()
        self.n_proto   = n_proto
        self.d_hidden  = d_hidden

        self.encoder = PTaRLEncoder(input_dim, d_hidden, n_blocks, dropout)
        self.head    = nn.Linear(d_hidden, output_dim)

        # 원본 reduce 네트워크 (3 hidden layers, 동일한 구조)
        self.reduce = nn.Sequential(
            nn.Linear(d_hidden, d_hidden), nn.GELU(), nn.Dropout(0.1),
            nn.Linear(d_hidden, d_hidden), nn.GELU(), nn.Dropout(0.1),
            nn.Linear(d_hidden, d_hidden), nn.GELU(), nn.Dropout(0.1),
            nn.Linear(d_hidden, n_proto),
        )

        # topic (prototype matrix): K-Means 초기화 전 임시 zeros
        self.topic = nn.Parameter(
            torch.zeros(n_proto, d_hidden), requires_grad=True
        )

    def forward(self, x: torch.Tensor):
        """Returns (logits, r, hidden)."""
        hidden = self.encoder(x)                        # (B, D)
        r      = torch.softmax(self.reduce(hidden), dim=1)  # (B, K) — 원본과 동일
        logits = self.head(hidden)                      # (B, out)
        return logits, r, hidden

    def init_topic_from_kmeans(self, centers: np.ndarray):
        """K-Means cluster centers로 topic 초기화 (원본 generate_topic 대응)."""
        with torch.no_grad():
            self.topic.copy_(
                torch.tensor(centers, dtype=torch.float32).to(self.topic.device)
            )


# ─────────────────────────────────────────────────────────────
# Loss 함수 (원본 run_one_epoch 내부 수식 그대로)
# ─────────────────────────────────────────────────────────────

def ot_loss(hidden: torch.Tensor, r: torch.Tensor, topic: torch.Tensor,
            ot_weight: float) -> torch.Tensor:
    """
    원본:
        norm = sqrt(sum(hidden^2)) * sqrt(sum(topic.T^2))
        loss_ot = mean(sum(r * (hidden @ topic.T / norm)))
        loss -= ot_weight * loss_ot
    → 반환값을 loss에서 빼야 하므로 음수로 반환.
    """
    norm = (
        torch.sqrt((hidden ** 2).sum(dim=1, keepdim=True)) *
        torch.sqrt((topic.T ** 2).sum(dim=0, keepdim=True))
    ).clamp(min=1e-8)                                          # (B, K)
    cos_ht    = (hidden.float() @ topic.T.float()) / norm      # (B, K)
    loss_ot   = torch.mean(torch.sum(r * cos_ht, dim=1))
    return -ot_weight * loss_ot   # loss에 더하면 됨 (= loss -= ot_weight * loss_ot)


def diversity_loss(r: torch.Tensor, y: torch.Tensor,
                   tasktype: str, div_weight: float) -> torch.Tensor:
    """
    원본 Coordinates Diversifying Constraint.
    - 50% 랜덤 샘플링
    - positive pair = 같은 클래스(분류) 또는 같은 bin(회귀)
    - contrastive: -sum(pos_mask * (cos - log_denom)) / pos_count
    """
    n = r.shape[0]
    idx = np.random.choice(n, max(int(n * 0.5), 2), replace=False)
    r_sel = r[idx]

    coord     = F.normalize(r_sel.float(), dim=1)
    cos_sim   = torch.clamp(coord @ coord.T, -1.0, 1.0)       # (m, m)

    if tasktype != "regression":
        y_sel  = y.reshape(-1)[idx]
        pos_mask = (y_sel.unsqueeze(1) == y_sel.unsqueeze(0)).float()
    else:
        y_flat = y.reshape(-1)
        y_min, y_max = y_flat.min(), y_flat.max()
        num_bin = max(1 + int(math.log2(n)), 2)
        interval = (y_max - y_min) / num_bin + 1e-8
        y_assign = torch.clamp(
            ((y_flat - y_min) / interval).long(), 0, num_bin - 1
        )
        y_sel    = y_assign[idx]
        pos_mask = (y_sel.unsqueeze(1) == y_sel.unsqueeze(0)).float()

    pos_count = pos_mask.sum().clamp(min=1.0)
    log_denom = torch.logsumexp(cos_sim.reshape(-1), dim=0)
    loss_div  = -(pos_mask * (cos_sim - log_denom)).sum() / pos_count
    return div_weight * loss_div


def orthogonality_loss(topic: torch.Tensor, r_weight: float) -> torch.Tensor:
    """
    원본 Matrix Orthogonalization Constraint (Eq. 7):
        r1 = sqrt(sum(topic^2, dim=1, keepdim=True))
        topic_matrix = (topic @ topic.T) / (r1 @ r1.T)
        topic_matrix = clamp(abs(topic_matrix), 0, 1)
        l1 = sum(abs(topic_matrix))
        l2 = sum(topic_matrix^2)
        loss_sparse = l1 / l2
        loss_constraint = abs(l1 - K)
        r_loss = loss_sparse + 0.5 * loss_constraint
    """
    r1           = torch.sqrt((topic.float() ** 2).sum(dim=1, keepdim=True)).clamp(min=1e-8)
    topic_matrix = (topic.float() @ topic.T.float()) / (r1 @ r1.T)
    topic_matrix = torch.clamp(topic_matrix.abs(), 0.0, 1.0)
    l1           = topic_matrix.abs().sum()
    l2           = (topic_matrix ** 2).sum().clamp(min=1e-8)
    loss_sparse  = l1 / l2
    loss_const   = (l1 - topic_matrix.shape[0]).abs()
    return r_weight * (loss_sparse + 0.5 * loss_const)


# ─────────────────────────────────────────────────────────────
# 헬퍼
# ─────────────────────────────────────────────────────────────

def _n_proto(n_num: int, n_cat: int) -> int:
    """원본: n_clusters = ceil(log2(n_num + n_cat))."""
    total = max(n_num + n_cat, 2)
    return max(int(math.ceil(math.log2(total))), 2)


# ─────────────────────────────────────────────────────────────
# MultiTab 래퍼
# ─────────────────────────────────────────────────────────────

class PTaRLMethod:
    """
    MultiTab supmodel 인터페이스 (fit / predict / predict_proba).

    model.py 호출 형식:
        PTaRLMethod(params, tasktype, dataset.X_num, dataset.X_cat,
                    input_dim=input_dim, output_dim=output_dim,
                    device=device, data_id=openml_id)
    """

    def __init__(self, params, tasktype,
                 num_cols=[], cat_features=[],
                 input_dim=0, output_dim=1,
                 device="cuda", data_id=None, modelname="ptarl"):
        self.params      = params
        self.tasktype    = tasktype
        self.device      = device
        self.data_id     = data_id
        self.modelname   = modelname
        self._input_dim  = input_dim
        self._output_dim = output_dim
        self._n_num      = len(num_cols)
        self._n_cat      = len(cat_features)
        self.model       = None

    def _build_model(self, n_train: int):
        p       = self.params
        n_proto = _n_proto(self._n_num, self._n_cat)
        self.model = PTaRLNet(
            input_dim  = self._input_dim,
            output_dim = self._output_dim,
            d_hidden   = p["d_hidden"],
            n_blocks   = p["n_blocks"],
            n_proto    = n_proto,
            dropout    = p["dropout"],
        ).to(self.device)

    def _get_loss_fn(self):
        if self.tasktype == "regression":
            return F.mse_loss
        elif self.tasktype == "binclass":
            return F.binary_cross_entropy_with_logits
        return F.cross_entropy

    def _make_loaders(self, X_train, y_train, X_val, y_val, batch_size):
        drop_last    = (len(X_train) % batch_size == 1)
        train_loader = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(X_train, y_train),
            batch_size=batch_size, shuffle=True, drop_last=drop_last)
        val_loader   = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(X_val, y_val),
            batch_size=batch_size, shuffle=False)
        return train_loader, val_loader

    # ── fit ─────────────────────────────────────────────────

    def fit(self, X_train, y_train, X_val, y_val):
        if y_train.ndim == 1:
            y_train = y_train.unsqueeze(1)
            y_val   = y_val.unsqueeze(1)

        self._build_model(len(X_train))

        p          = self.params
        device     = self.device
        batch_size = get_batch_size(len(X_train))
        loss_fn    = self._get_loss_fn()

        train_loader, val_loader = self._make_loaders(
            X_train, y_train, X_val, y_val, batch_size)

        # ── Stage 1: 일반 supervised 학습 ───────────────────
        # 원본: lr=1e-4 고정, weight_decay만 config에서
        s1_opt = torch.optim.AdamW(
            self.model.parameters(),
            lr=p["lr"], weight_decay=p["weight_decay"]
        )
        s1_epochs = p["stage1_epochs"]

        pbar = tqdm(range(1, s1_epochs + 1))
        pbar.set_description("PTaRL Stage1")
        for epoch in pbar:
            self.model.train()
            ep_loss = 0.0
            for x_b, y_b in train_loader:
                x_b, y_b = x_b.to(device), y_b.to(device)
                s1_opt.zero_grad()
                logits, _, _ = self.model(x_b)
                if logits.size() != y_b.size():
                    logits = logits.view(y_b.size())
                loss = loss_fn(logits, y_b)
                loss.backward()
                s1_opt.step()
                ep_loss = loss.item()
            pbar.set_postfix_str(f"data_id:{self.data_id}, loss:{ep_loss:.5f}")

        # ── K-Means prototype 초기화 (원본 generate_topic) ──
        self.model.eval()
        hiddens = []
        with torch.no_grad():
            for x_b, _ in train_loader:
                h = self.model.encoder(x_b.to(device))
                hiddens.append(h.cpu().numpy())
        hiddens = np.concatenate(hiddens, axis=0)

        kmeans  = KMeans(n_clusters=self.model.n_proto, n_init=10, random_state=0)
        centers = kmeans.fit(hiddens).cluster_centers_   # (K, d_hidden)
        self.model.init_topic_from_kmeans(centers)

        # ── Stage 2: OT + Diversity + Orthogonality loss ──
        s2_opt    = torch.optim.AdamW(
            self.model.parameters(),
            lr=p["lr_s2"], weight_decay=p["weight_decay"]
        )
        s2_epochs = p["stage2_epochs"]
        patience  = p["early_stopping_rounds"]

        callback = CallbackContainer([
            EarlyStopping(early_stopping_metric="val_loss", patience=patience)
        ])

        pbar2 = tqdm(range(1, s2_epochs + 1))
        pbar2.set_description("PTaRL Stage2")
        for epoch in pbar2:
            self.model.train()
            ep_loss = 0.0
            for x_b, y_b in train_loader:
                x_b, y_b = x_b.to(device), y_b.to(device)
                s2_opt.zero_grad()

                logits, r, hidden = self.model(x_b)
                if logits.size() != y_b.size():
                    logits = logits.view(y_b.size())

                # task loss
                loss = loss_fn(logits, y_b)
                # OT loss (원본: loss -= ot_weight * loss_ot)
                loss = loss + ot_loss(hidden, r, self.model.topic, p["lambda_div"])
                # Diversity loss (contrastive on r)
                loss = loss + diversity_loss(r, y_b, self.tasktype, p["lambda_div"])
                # Orthogonality loss
                loss = loss + orthogonality_loss(self.model.topic, p["lambda_orth"])

                loss.backward()
                s2_opt.step()
                ep_loss = loss.item()

            pbar2.set_postfix_str(f"data_id:{self.data_id}, loss:{ep_loss:.5f}")

            # Validation
            self.model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for x_v, y_v in val_loader:
                    x_v, y_v = x_v.to(device), y_v.to(device)
                    logits, _, _ = self.model(x_v)
                    if logits.size() != y_v.size():
                        logits = logits.view(y_v.size())
                    val_loss += loss_fn(logits, y_v).item()
            val_loss /= len(val_loader)

            callback.on_epoch_end(epoch, {"val_loss": val_loss, "epoch": epoch})
            if any(cb.should_stop for cb in callback.callbacks):
                print(f"Early stopping at epoch {epoch}")
                break

        self.model.eval()

    # ── predict ─────────────────────────────────────────────

    def predict(self, X_test):
        self.model.eval()
        with torch.no_grad():
            logits = self._forward_batched(X_test)
            if self.tasktype == "binclass":
                return torch.sigmoid(logits).round().detach().cpu().numpy()
            elif self.tasktype == "regression":
                return logits.detach().cpu().numpy()
            else:
                return torch.argmax(logits, dim=1).detach().cpu().numpy()

    def predict_proba(self, X_test, logit=False):
        self.model.eval()
        with torch.no_grad():
            logits = self._forward_batched(X_test)
            if logit:
                return logits.detach().cpu().numpy()
            if self.tasktype == "binclass":
                return torch.sigmoid(logits).detach().cpu().numpy()
            elif self.tasktype == "multiclass":
                return F.softmax(logits, dim=-1).detach().cpu().numpy()
            return None

    def _forward_batched(self, X, batch_size: int = 1024):
        parts = []
        for i in range(0, len(X), batch_size):
            xb       = X[i: i + batch_size].to(self.device)
            logits, _, _ = self.model(xb)
            parts.append(logits.cpu())
        return torch.cat(parts, dim=0)