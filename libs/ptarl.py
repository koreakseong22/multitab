"""
libs/ptarl.py
=============
PTaRL: Prototype-based Tabular Representation Learning via Space Calibration
Paper: Hangting Ye et al., ICLR 2024
Official code: https://github.com/HangtingYe/PTaRL  (models.py, train_final_version.py)

이 버전은 원본 코드의 핵심 학습 절차를 충실히 따른다:

  - Encoder: vanilla MLP (Linear -> ReLU -> Dropout 스택), LayerNorm/residual 없음
            (원본 Models/mlp.py와 동일)
  - Stage 1: "source model" (topic/reduce 존재하지만 forward/loss에서 미사용)을
             처음부터 학습 (task loss만)
  - K-Means: Stage1 모델의 encoder hidden representation으로 cluster centers 계산
             (원본 generate_topic)
  - Stage 2: 모델을 처음부터 다시 초기화(가중치 재시작), topic만 K-means centers로
             초기화 후 OT + Diversity + Orthogonality loss와 함께 학습
             (원본: _set_seed(seed) 재호출 후 새 Model 인스턴스 생성)
  - lr = 1e-4 고정 (원본: config의 lr은 무시되고 하드코딩됨), Stage1/Stage2 동일
  - n_epochs: 원본은 1e9(사실상 무제한) + early stopping(patience=20)만으로 종료.
              여기서는 무한 루프 방지를 위해 max_epochs 상한을 두되 충분히 크게 설정.
  - ot_weight / diversity_weight / r_weight: 원본처럼 독립적인 3개 가중치
              (원본 default = 모두 0.25, CLI 인자이며 Optuna 탐색 대상이 아니었음.
               여기서는 Optuna 탐색 대상으로 두되 독립적으로 분리함)
  - n_clusters = ceil(log2(n_num_features + n_cat_features))  [원본과 동일]

MultiTab supmodel 인터페이스 (fit / predict / predict_proba) 유지.
"""

import copy
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.cluster import KMeans
from tqdm import tqdm

from libs.data import get_batch_size


# ─────────────────────────────────────────────────────────────
# Encoder (원본 Models/mlp.py 와 동일한 vanilla MLP)
# ─────────────────────────────────────────────────────────────

class PTaRLEncoder(nn.Module):
    """
    원본: Linear -> ReLU -> Dropout 을 d_layers 개수만큼 반복.
    LayerNorm/residual 없음.
    """
    def __init__(self, input_dim: int, d_layers: list, dropout: float):
        super().__init__()
        dims = [input_dim] + list(d_layers)
        self.layers = nn.ModuleList([
            nn.Linear(dims[i], dims[i + 1]) for i in range(len(d_layers))
        ])
        self.dropout = dropout

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x)
            x = F.relu(x)
            if self.dropout:
                x = F.dropout(x, self.dropout, self.training)
        return x


# ─────────────────────────────────────────────────────────────
# Model (원본 models.py 의 Model 클래스에 대응)
# ─────────────────────────────────────────────────────────────

class PTaRLNet(nn.Module):
    """
    원본과 동일하게 topic/reduce/head/encoder를 항상 생성한다.
    `use_ot=False`이면 forward는 logits만 반환 (Stage1 "source model" 역할).
    `use_ot=True`이면 (logits, r, hidden)을 반환 (Stage2).

    cluster_centers: (n_proto, d_hidden) 또는 None
        None이면 topic을 0으로 초기화 (원본: Stage1 source model 시점,
        cluster_centers_ = np.zeros([n_clusters, 1]) 와 동등한 placeholder).
    """
    def __init__(self, input_dim: int, output_dim: int,
                 d_layers: list, dropout: float, n_proto: int,
                 cluster_centers=None, use_ot: bool = False):
        super().__init__()
        self.n_proto = n_proto
        self.use_ot  = use_ot
        d_last = d_layers[-1]

        self.encoder = PTaRLEncoder(input_dim, d_layers, dropout)
        self.head    = nn.Linear(d_last, output_dim)

        # 원본 reduce: 3 hidden layers (GELU + Dropout(0.1)) -> topic_num
        self.reduce = nn.Sequential(
            nn.Linear(d_last, d_last), nn.GELU(), nn.Dropout(0.1),
            nn.Linear(d_last, d_last), nn.GELU(), nn.Dropout(0.1),
            nn.Linear(d_last, d_last), nn.GELU(), nn.Dropout(0.1),
            nn.Linear(d_last, n_proto),
        )

        if cluster_centers is not None:
            topic_init = torch.tensor(cluster_centers, dtype=torch.float32)
        else:
            # 원본: cluster_centers_ = np.zeros([n_clusters, 1]) 인 상태로
            # Model이 생성되지만, 이 시점(source model)은 forward에서
            # topic을 사용하지 않으므로 shape만 맞춰 placeholder로 둔다.
            topic_init = torch.zeros(n_proto, d_last)

        self.topic = nn.Parameter(topic_init, requires_grad=True)

    def forward(self, x: torch.Tensor):
        hidden = self.encoder(x)
        logits = self.head(hidden)
        if self.use_ot:
            r = torch.softmax(self.reduce(hidden), dim=1)
            return logits, r, hidden
        return logits


# ─────────────────────────────────────────────────────────────
# Loss 함수 (원본 run_one_epoch 내부 수식 그대로)
# ─────────────────────────────────────────────────────────────

def ot_loss(hidden: torch.Tensor, r: torch.Tensor, topic: torch.Tensor) -> torch.Tensor:
    """
    원본:
        norm = sqrt(sum(hidden^2)) @ sqrt(sum(topic.T^2))
        loss_ot = mean(sum(r * (hidden @ topic.T / norm), dim=1))
        loss -= ot_weight * loss_ot
    → ot_weight는 호출부에서 곱하므로 여기서는 loss_ot 자체(빼기 전 값)를 반환.
    """
    norm = torch.mm(
        torch.sqrt((hidden ** 2).sum(dim=1, keepdim=True)),
        torch.sqrt((topic.T ** 2).sum(dim=0, keepdim=True))
    ).clamp(min=1e-8)
    cos_ht  = (hidden.float() @ topic.T.float()) / norm
    loss_ot = torch.mean(torch.sum(r * cos_ht, dim=1))
    return loss_ot


def diversity_loss(r: torch.Tensor, y: torch.Tensor, tasktype: str) -> torch.Tensor:
    """
    원본 Coordinates Diversifying Constraint.
    50% 랜덤 샘플링 -> coord = normalize(r) -> cos_sim -> positive_mask 기반 contrastive.
    """
    n = r.shape[0]
    n_sel = max(int(n * 0.5), 2)
    idx = np.random.choice(n, n_sel, replace=False)
    r_sel = r[idx]

    coord   = F.normalize(r_sel.float(), dim=1)
    cos_sim = torch.clamp(coord @ coord.T, -1.0, 1.0)

    y_flat = y.reshape(-1)
    if tasktype != "regression":
        y_sel = y_flat[idx]
        positive_mask = (y_sel.unsqueeze(1) == y_sel.unsqueeze(0)).float()
    else:
        y_min, y_max = y_flat.min(), y_flat.max()
        num_bin = max(1 + int(math.log2(n)), 1)
        interval = (y_max - y_min) / num_bin
        interval = interval if interval != 0 else torch.tensor(1e-8, device=y.device)
        y_assign = torch.clamp(((y_flat - y_min) / interval).long(), 0, num_bin - 1)
        y_sel = y_assign[idx]
        positive_mask = (y_sel.unsqueeze(1) == y_sel.unsqueeze(0)).float()

    positive_count = positive_mask.sum().clamp(min=1.0)
    log_denom = torch.logsumexp(cos_sim.reshape(-1), dim=0)
    loss_diversity = -(positive_mask * (cos_sim - log_denom)).sum() / positive_count
    return loss_diversity


def orthogonality_loss(topic: torch.Tensor) -> torch.Tensor:
    """
    원본 Matrix Orthogonalization Constraint (Eq. 7):
        r1 = sqrt(sum(topic^2, dim=1, keepdim=True))
        topic_matrix = clamp(abs((topic @ topic.T) / (r1 @ r1.T)), 0, 1)
        l1 = sum(abs(topic_matrix)); l2 = sum(topic_matrix^2)
        r_loss = l1/l2 + 0.5 * abs(l1 - K)
    """
    r1 = torch.sqrt((topic.float() ** 2).sum(dim=1, keepdim=True)).clamp(min=1e-8)
    topic_matrix = (topic.float() @ topic.T.float()) / (r1 @ r1.T)
    topic_matrix = torch.clamp(topic_matrix.abs(), 0.0, 1.0)

    l1 = topic_matrix.abs().sum()
    l2 = (topic_matrix ** 2).sum().clamp(min=1e-8)

    loss_sparse = l1 / l2
    loss_const  = (l1 - topic_matrix.shape[0]).abs()
    return loss_sparse + 0.5 * loss_const


# ─────────────────────────────────────────────────────────────
# 헬퍼
# ─────────────────────────────────────────────────────────────

def _n_clusters(n_num: int, n_cat: int) -> int:
    """원본: n_clusters = ceil(log2(n_num + n_cat))."""
    total = max(n_num + n_cat, 2)
    return max(int(math.ceil(math.log2(total))), 2)


def _make_d_layers(d_hidden: int, n_blocks: int) -> list:
    """
    원본 toml의 `d_layers`는 [256, 256, 256] 같은 동일 폭 리스트.
    여기서는 (n_blocks + 1)개의 동일 폭 레이어로 구성한다 (최소 1개).
    """
    n_layers = max(n_blocks + 1, 1)
    return [d_hidden] * n_layers


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

    필요한 params 키:
        d_hidden, n_blocks, dropout,
        ot_weight, diversity_weight, r_weight,
        weight_decay,
        early_stopping_rounds  (원본: patience=20)
        max_epochs              (원본: 1e9, 여기서는 안전상 상한 — 기본 1000)
    """

    LR = 1e-4  # 원본 하드코딩 값 (config의 lr은 무시됨)

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

    def _get_loss_fn(self):
        if self.tasktype == "regression":
            return F.mse_loss
        elif self.tasktype == "binclass":
            return F.binary_cross_entropy_with_logits
        return F.cross_entropy

    def _make_loaders(self, X_train, y_train, X_val, y_val, batch_size):
        drop_last = (len(X_train) % batch_size == 1)
        train_loader = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(X_train, y_train),
            batch_size=batch_size, shuffle=False, drop_last=drop_last)
        val_loader = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(X_val, y_val),
            batch_size=batch_size, shuffle=False)
        return train_loader, val_loader

    def _build_model(self, d_layers, n_proto, cluster_centers, use_ot):
        return PTaRLNet(
            input_dim=self._input_dim,
            output_dim=self._output_dim,
            d_layers=d_layers,
            dropout=self.params["dropout"],
            n_proto=n_proto,
            cluster_centers=cluster_centers,
            use_ot=use_ot,
        ).to(self.device)

    # ── 공용 학습 루프 (원본 fit() 대응) ─────────────────────
    def _train_loop(self, model, train_loader, val_loader, loss_fn,
                     ot_weight, diversity_weight, r_weight,
                     max_epochs, patience, desc):
        weight_decay = self.params["weight_decay"]
        optimizer = torch.optim.AdamW(model.parameters(), lr=self.LR,
                                       weight_decay=weight_decay)

        best_val_loss = float("inf")
        best_state = None
        cur_patience = patience

        pbar = tqdm(range(1, max_epochs + 1))
        pbar.set_description(desc)
        for epoch in pbar:
            model.train()
            running_loss = 0.0
            n_batches = 0
            for x_b, y_b in train_loader:
                x_b, y_b = x_b.to(self.device), y_b.to(self.device)
                optimizer.zero_grad()

                if model.use_ot:
                    logits, r, hidden = model(x_b)
                else:
                    logits = model(x_b)

                if loss_fn == F.cross_entropy:
                    loss = loss_fn(logits, y_b)
                else:
                    loss = loss_fn(logits.view(y_b.shape), y_b)

                if model.use_ot:
                    loss = loss - ot_weight * ot_loss(hidden, r, model.topic)
                    loss = loss + diversity_weight * diversity_loss(r, y_b, self.tasktype)
                    loss = loss + r_weight * orthogonality_loss(model.topic)

                loss.backward()
                optimizer.step()
                running_loss += loss.item()
                n_batches += 1

            train_loss = running_loss / max(n_batches, 1)

            # ── validation ──────────────────────────────────
            model.eval()
            val_running = 0.0
            n_val_batches = 0
            with torch.no_grad():
                for x_v, y_v in val_loader:
                    x_v, y_v = x_v.to(self.device), y_v.to(self.device)
                    if model.use_ot:
                        logits_v, _, _ = model(x_v)
                    else:
                        logits_v = model(x_v)

                    if loss_fn == F.cross_entropy:
                        vloss = loss_fn(logits_v, y_v)
                    else:
                        vloss = loss_fn(logits_v.view(y_v.shape), y_v)
                    val_running += vloss.item()
                    n_val_batches += 1
            val_loss = val_running / max(n_val_batches, 1)

            pbar.set_postfix_str(
                f"data_id:{self.data_id}, train:{train_loss:.5f}, val:{val_loss:.5f}"
            )

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_state = copy.deepcopy(model.state_dict())
                cur_patience = patience
            else:
                cur_patience -= 1

            if cur_patience <= 0:
                break

        if best_state is not None:
            model.load_state_dict(best_state)
        return model

    # ── fit ─────────────────────────────────────────────────

    def fit(self, X_train, y_train, X_val, y_val):
        if (self.tasktype != "multiclass") and y_train.ndim == 1:
            y_train_ = y_train.float().unsqueeze(1)
            y_val_   = y_val.float().unsqueeze(1)
        else:
            y_train_ = y_train
            y_val_   = y_val

        p = self.params
        batch_size = get_batch_size(len(X_train))
        loss_fn = self._get_loss_fn()
        train_loader, val_loader = self._make_loaders(
            X_train, y_train_, X_val, y_val_, batch_size)

        n_proto  = _n_clusters(self._n_num, self._n_cat)
        d_layers = _make_d_layers(p["d_hidden"], p["n_blocks"])

        max_epochs = p.get("max_epochs", 1000)
        patience   = p["early_stopping_rounds"]

        # ── Stage 1: source model (use_ot=False, topic 미사용) ──
        stage1_model = self._build_model(
            d_layers, n_proto, cluster_centers=None, use_ot=False
        )
        stage1_model = self._train_loop(
            stage1_model, train_loader, val_loader, loss_fn,
            ot_weight=0, diversity_weight=0, r_weight=0,
            max_epochs=max_epochs, patience=patience,
            desc="PTaRL Stage1 (source)",
        )

        # ── K-Means: stage1 encoder hidden -> cluster centers ──
        stage1_model.eval()
        hiddens = []
        with torch.no_grad():
            for x_b, _ in train_loader:
                h = stage1_model.encoder(x_b.to(self.device))
                hiddens.append(h.cpu().numpy())
        hiddens = np.concatenate(hiddens, axis=0)

        kmeans = KMeans(n_clusters=n_proto, n_init=10, random_state=0)
        cluster_centers = kmeans.fit(hiddens).cluster_centers_  # (n_proto, d_hidden)

        # ── Stage 2: 모델 재초기화 (원본: _set_seed 후 새 Model) ──
        # cluster_centers만 전달, 가중치는 새로 초기화됨
        self.model = self._build_model(
            d_layers, n_proto, cluster_centers=cluster_centers, use_ot=True
        )
        self.model = self._train_loop(
            self.model, train_loader, val_loader, loss_fn,
            ot_weight=p["ot_weight"],
            diversity_weight=p["diversity_weight"],
            r_weight=p["r_weight"],
            max_epochs=max_epochs, patience=patience,
            desc="PTaRL Stage2 (OT)",
        )
        self.model.eval()

    # ── predict ─────────────────────────────────────────────

    def predict(self, X_test):
        self.model.eval()
        with torch.no_grad():
            logits = self._forward_batched(X_test)
            if self.tasktype == "binclass":
                return torch.sigmoid(logits).round().detach().cpu().numpy().reshape(-1)
            elif self.tasktype == "regression":
                return logits.detach().cpu().numpy().reshape(-1)
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
            xb = X[i: i + batch_size].to(self.device)
            out = self.model(xb)
            logits = out[0] if self.model.use_ot else out
            parts.append(logits.cpu())
        return torch.cat(parts, dim=0)