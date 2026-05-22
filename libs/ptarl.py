## Reference
## PTaRL: Prototype-based Tabular Representation Learning via Space Calibration
## Hangting Ye et al., ICLR 2024
## https://arxiv.org/abs/2407.05364
##
## Implementation follows the ModernNCA pattern in MultiTab (modernnca.py).
## PTaRL is a model-agnostic two-stage pipeline:
##   Stage 1: Train backbone (FT-Transformer style MLP) normally
##   Stage 2: Freeze backbone, add PTaRL prototype projection head,
##            re-train projection with OT + diversity + orthogonalization losses
##
## Backbone: simple ResNet-style MLP (same family as MultiTab's resnet.py)
## HPO: follows original paper (Appendix A) — backbone HPs inherited in stage 2,
##      PTaRL-specific HPs: n_prototype (=ceil(log(n_features))), lambda_div, lambda_orth

import math
import abc
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
from sklearn.cluster import KMeans

from libs.data import get_batch_size


# ─────────────────────────────────────────────────────────────
# Backbone: ResNet-style MLP (PTaRL paper uses FTT / ResNet)
# We use a simple Pre-LN ResidualMLP stack — consistent with MultiTab
# ─────────────────────────────────────────────────────────────

class ResidualBlock(nn.Module):
    def __init__(self, d_in: int, d_hidden: int, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(d_in),
            nn.Linear(d_in, d_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_hidden, d_in),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return x + self.net(x)


class BackboneMLP(nn.Module):
    """
    Simple residual MLP backbone.
    PTaRL paper uses FT-Transformer or ResNet as backbone;
    here we use a lighter residual MLP consistent with MultiTab's style.
    """
    def __init__(
        self,
        d_in: int,
        d_hidden: int,
        n_blocks: int,
        d_out: int,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.proj = nn.Sequential(
            nn.LayerNorm(d_in),
            nn.Linear(d_in, d_hidden),
        )
        self.blocks = nn.Sequential(
            *[ResidualBlock(d_hidden, d_hidden * 2, dropout) for _ in range(n_blocks)]
        )
        self.head = nn.Sequential(
            nn.LayerNorm(d_hidden),
            nn.Linear(d_hidden, d_out),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.blocks(self.proj(x)))

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Return representation before final head (for PTaRL projection)."""
        return self.blocks(self.proj(x))


# ─────────────────────────────────────────────────────────────
# PTaRL Projection Components
# ─────────────────────────────────────────────────────────────

class PrototypeEstimator(nn.Module):
    """
    3-layer MLP estimator φ(·; γ) that maps representations to P-Space coordinates.
    Paper §3.2: "The estimator φ(·; γ) is a simple 3-layer fully-connected MLP."
    """
    def __init__(self, d_rep: int, n_prototype: int):
        super().__init__()
        d_hidden = max(d_rep, n_prototype * 2)
        self.net = nn.Sequential(
            nn.Linear(d_rep, d_hidden),
            nn.ReLU(),
            nn.Linear(d_hidden, d_hidden),
            nn.ReLU(),
            nn.Linear(d_hidden, n_prototype),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """Returns coordinates in P-Space: (B, K)"""
        return self.net(z)


class PTaRLModel(nn.Module):
    """
    Full PTaRL model: Backbone + Prototype Projection Head.

    Forward returns logits (for prediction) and p_space_coords (for OT loss).
    In Stage 1, only backbone is used.
    In Stage 2, backbone is frozen; estimator + prediction head trained.
    """
    def __init__(
        self,
        d_in: int,
        d_hidden: int,
        n_blocks: int,
        d_out: int,
        n_prototype: int,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.backbone = BackboneMLP(d_in, d_hidden, n_blocks, d_out, dropout)
        self.d_rep = d_hidden
        self.n_prototype = n_prototype
        self.d_out = d_out

        # PTaRL components (Stage 2)
        self.estimator = PrototypeEstimator(d_hidden, n_prototype)
        # Prediction head: from P-Space coords to output
        self.ptarl_head = nn.Linear(n_prototype, d_out)

        # Global prototypes B ∈ R^{K×d_rep}: learnable basis vectors
        self.prototypes = nn.Parameter(torch.randn(n_prototype, d_hidden))

    def forward_backbone(self, x: torch.Tensor):
        """Stage 1: standard backbone prediction."""
        return self.backbone(x)

    def forward_ptarl(self, x: torch.Tensor):
        """Stage 2: PTaRL projection prediction."""
        with torch.no_grad():
            z = self.backbone.encode(x)          # (B, d_rep)
        coords = self.estimator(z)               # (B, K) — P-Space coordinates
        logits = self.ptarl_head(coords)         # (B, d_out)
        return logits, coords, z

    def forward(self, x: torch.Tensor, stage: int = 2):
        if stage == 1:
            return self.forward_backbone(x)
        else:
            logits, coords, z = self.forward_ptarl(x)
            return logits

    def init_prototypes_kmeans(self, z_all: torch.Tensor):
        """
        Initialize global prototypes with K-Means on training representations.
        Paper §3.1: 'We initialize the global prototypes with K-Means clustering.'
        """
        z_np = z_all.detach().cpu().numpy()
        k = self.n_prototype
        kmeans = KMeans(n_clusters=k, n_init=10, random_state=42)
        kmeans.fit(z_np)
        centers = torch.tensor(kmeans.cluster_centers_, dtype=torch.float32)
        with torch.no_grad():
            self.prototypes.copy_(centers)


# ─────────────────────────────────────────────────────────────
# OT Loss (Sinkhorn approximation)
# ─────────────────────────────────────────────────────────────

def sinkhorn(cost: torch.Tensor, eps: float = 0.1, n_iter: int = 20) -> torch.Tensor:
    """
    Sinkhorn algorithm for OT plan.
    cost: (B, K) cost matrix
    Returns transport plan T: (B, K)
    """
    B, K = cost.shape
    # Uniform marginals
    a = torch.ones(B, device=cost.device) / B
    b = torch.ones(K, device=cost.device) / K

    log_M = -cost / eps
    log_u = torch.zeros(B, device=cost.device)
    log_v = torch.zeros(K, device=cost.device)

    for _ in range(n_iter):
        log_u = torch.log(a) - torch.logsumexp(log_M + log_v.unsqueeze(0), dim=1)
        log_v = torch.log(b) - torch.logsumexp(log_M + log_u.unsqueeze(1), dim=0)

    T = torch.exp(log_M + log_u.unsqueeze(1) + log_v.unsqueeze(0))
    return T


def ot_loss(coords: torch.Tensor, z: torch.Tensor, prototypes: torch.Tensor) -> torch.Tensor:
    """
    OT loss: align P-Space coordinates with prototype distances.
    Paper eq.(3): minimize transport cost between representation distribution
    and prototype distribution.

    coords:     (B, K) — estimated P-Space coordinates (softmax normalized)
    z:          (B, d_rep) — backbone representations
    prototypes: (K, d_rep) — global prototype vectors
    """
    # Cost matrix: L2 distance between each sample and each prototype
    # cost[i,j] = ||z_i - B_j||^2
    cost = torch.cdist(z, prototypes, p=2).pow(2)   # (B, K)

    # Transport plan
    T = sinkhorn(cost.detach(), eps=0.1, n_iter=20)  # (B, K)

    # OT loss: <T, cost>
    loss = (T * cost).sum()
    return loss


def diversity_loss(coords: torch.Tensor) -> torch.Tensor:
    """
    Coordinates Diversifying Constraint (paper §3.3):
    push representations of different samples apart in P-Space.
    Implemented as negative mean pairwise cosine similarity (contrastive style).
    """
    coords_norm = F.normalize(coords, dim=1)        # (B, K)
    sim = coords_norm @ coords_norm.T               # (B, B)
    B = coords.size(0)
    mask = 1.0 - torch.eye(B, device=coords.device)
    loss = (sim * mask).sum() / (B * (B - 1) + 1e-8)
    return loss


def orthogonality_loss(prototypes: torch.Tensor) -> torch.Tensor:
    """
    Matrix Orthogonalization Constraint (paper §3.3):
    make prototype vectors orthogonal to ensure independence.
    ||B B^T - I||_F^2
    """
    K = prototypes.size(0)
    proto_norm = F.normalize(prototypes, dim=1)     # (K, d_rep)
    gram = proto_norm @ proto_norm.T                # (K, K)
    identity = torch.eye(K, device=prototypes.device)
    loss = (gram - identity).pow(2).sum()
    return loss


# ─────────────────────────────────────────────────────────────
# PTaRLMethod: MultiTab-compatible wrapper (modernnca.py 패턴)
# ─────────────────────────────────────────────────────────────

class PTaRLMethod(object, metaclass=abc.ABCMeta):
    """
    PTaRL wrapper following ModernNCAMethod interface in MultiTab.

    Exposes: fit(X_train, y_train, X_val, y_val)
             predict(X_test) -> np.ndarray
             predict_proba(X_test, logit=False) -> np.ndarray

    Two-stage training:
      Stage 1 (stage1_epochs): train backbone normally with task loss
      Stage 2 (stage2_epochs): freeze backbone, train PTaRL projection
                                with task loss + OT + diversity + orthogonality
    """

    def __init__(
        self,
        params: dict,
        tasktype: str,
        num_cols=None,
        cat_features=None,
        input_dim: int = 0,
        output_dim: int = 0,
        device: str = "cuda",
        data_id=None,
        modelname: str = "ptarl",
    ):
        super().__init__()

        self.num_cols     = num_cols if num_cols is not None else []
        self.cat_features = cat_features if cat_features is not None else []
        self.tasktype     = tasktype
        self.params       = params
        self.data_id      = data_id
        self.device       = device

        self.is_binclass   = (tasktype == "binclass")
        self.is_multiclass = (tasktype == "multiclass")
        self.is_regression = (tasktype == "regression")

        self.max_epoch     = params.get("early_stopping_rounds", 20) * 5  # 총 에폭 상한
        self.stage1_epochs = params.get("stage1_epochs", 50)
        self.stage2_epochs = params.get("stage2_epochs", 50)
        self.patience      = params.get("early_stopping_rounds", 20)

        # n_prototype: paper sets K = ceil(log(n_features))
        # We expose it as tunable but default to paper's rule
        n_features_all = len(self.num_cols) + len(self.cat_features)
        self.n_prototype = params.get(
            "n_prototype",
            max(2, math.ceil(math.log(max(n_features_all, 2))))
        )

        # Build model
        self.model = PTaRLModel(
            d_in        = input_dim,
            d_hidden    = params["d_hidden"],
            n_blocks    = params["n_blocks"],
            d_out       = output_dim,
            n_prototype = self.n_prototype,
            dropout     = params["dropout"],
        ).to(device)
        self.model.float()

        # Loss weights (PTaRL §3.3)
        self.lambda_div  = params.get("lambda_div",  0.1)
        self.lambda_orth = params.get("lambda_orth", 0.1)

        # Training log
        self.trlog = {
            "args": params,
            "train_loss_s1": [],
            "train_loss_s2": [],
            "best_epoch_s1": 0,
            "best_epoch_s2": 0,
            "best_res": 1e10 if self.is_regression else 0,
        }

    # ── 내부 헬퍼 ─────────────────────────────────────────────

    def _get_criterion(self):
        if self.is_regression:
            return F.mse_loss
        elif self.is_multiclass:
            return F.cross_entropy
        else:
            return F.binary_cross_entropy_with_logits

    def _prep_y(self, y: torch.Tensor) -> torch.Tensor:
        """multiclass: argmax, else: squeeze."""
        if self.is_multiclass:
            return torch.argmax(y, dim=1) if y.ndim == 2 else y.long()
        return y.squeeze(-1) if y.ndim == 2 else y

    def _is_better(self, new_val: float, best_val: float) -> bool:
        if self.is_regression:
            return new_val < best_val
        return new_val > best_val

    def _forward_x(self, X: torch.Tensor, stage: int):
        """Runs model on full input (no num/cat split; MultiTab pre-stacks them)."""
        return self.model(X.float(), stage=stage)

    @torch.no_grad()
    def _encode_all(self, X: torch.Tensor, batch_size: int = 512) -> torch.Tensor:
        """Encode training set for K-Means init."""
        self.model.eval()
        zs = []
        for i in range(0, X.size(0), batch_size):
            zs.append(self.model.backbone.encode(X[i:i+batch_size].float()))
        return torch.cat(zs, dim=0)

    # ── Stage 1: Backbone Pre-training ────────────────────────

    def _stage1(self, X_train, y_train, X_val, y_val):
        criterion  = self._get_criterion()
        optimizer  = torch.optim.AdamW(
            self.model.backbone.parameters(),
            lr=self.params["lr"],
            weight_decay=self.params["weight_decay"],
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=self.stage1_epochs
        )

        best_val  = 1e10 if self.is_regression else 0.0
        best_state = None
        no_improve = 0
        batch_size = get_batch_size(len(X_train))

        pbar = tqdm(range(1, self.stage1_epochs + 1),
                    desc="PTaRL S1", ncols=80, leave=False)

        for epoch in pbar:
            self.model.train()
            perm  = torch.randperm(len(X_train), device=self.device)

            for i in range(0, len(X_train), batch_size):
                idx    = perm[i:i+batch_size]
                xb     = X_train[idx].float()
                yb     = self._prep_y(y_train[idx])
                logits = self.model(xb, stage=1)
                if self.is_binclass:
                    logits = logits.squeeze(-1)
                    yb = yb.float()
                loss = criterion(logits, yb)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            scheduler.step()
            self.trlog["train_loss_s1"].append(loss.item())

            # Validation
            val_metric = self._eval_metric(X_val, y_val, stage=1)
            if self._is_better(val_metric, best_val):
                best_val   = val_metric
                best_state = {k: v.clone() for k, v in self.model.state_dict().items()}
                self.trlog["best_epoch_s1"] = epoch
                no_improve = 0
            else:
                no_improve += 1
            if no_improve >= self.patience:
                break

            pbar.set_description(f"PTaRL S1 e{epoch} val={val_metric:.4f}")

        # Restore best backbone
        if best_state is not None:
            self.model.load_state_dict(best_state)

    # ── Stage 2: PTaRL Projection Training ───────────────────

    def _stage2(self, X_train, y_train, X_val, y_val):
        # Freeze backbone
        for p in self.model.backbone.parameters():
            p.requires_grad_(False)

        # K-Means init for prototypes
        z_all = self._encode_all(X_train)
        self.model.init_prototypes_kmeans(z_all)

        # Only train estimator, ptarl_head, prototypes
        s2_params = (
            list(self.model.estimator.parameters())
            + list(self.model.ptarl_head.parameters())
            + [self.model.prototypes]
        )
        criterion = self._get_criterion()
        optimizer = torch.optim.AdamW(
            s2_params,
            lr=self.params.get("lr_s2", self.params["lr"]),
            weight_decay=self.params["weight_decay"],
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=self.stage2_epochs
        )

        best_val   = 1e10 if self.is_regression else 0.0
        best_state = None
        no_improve = 0
        batch_size = get_batch_size(len(X_train))

        pbar = tqdm(range(1, self.stage2_epochs + 1),
                    desc="PTaRL S2", ncols=80, leave=False)

        for epoch in pbar:
            self.model.train()
            # backbone remains frozen
            self.model.backbone.eval()

            perm = torch.randperm(len(X_train), device=self.device)
            epoch_loss = 0.0

            for i in range(0, len(X_train), batch_size):
                idx    = perm[i:i+batch_size]
                xb     = X_train[idx].float()
                yb     = self._prep_y(y_train[idx])

                logits, coords, z = self.model.forward_ptarl(xb)
                if self.is_binclass:
                    logits = logits.squeeze(-1)
                    yb = yb.float()

                # Task loss
                loss_task = criterion(logits, yb)

                # Softmax coords for OT (treat as probability distribution over K)
                coords_soft = F.softmax(coords, dim=1)

                # OT loss
                loss_ot = ot_loss(coords_soft, z.detach(), self.model.prototypes)

                # Diversity loss (coordinates diversifying constraint)
                loss_div = diversity_loss(coords_soft)

                # Orthogonality loss (matrix orthogonalization constraint)
                loss_orth = orthogonality_loss(self.model.prototypes)

                loss = (
                    loss_task
                    + loss_ot
                    + self.lambda_div  * loss_div
                    + self.lambda_orth * loss_orth
                )

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                epoch_loss = loss.item()

            scheduler.step()
            self.trlog["train_loss_s2"].append(epoch_loss)

            # Validation
            val_metric = self._eval_metric(X_val, y_val, stage=2)
            if self._is_better(val_metric, best_val):
                best_val   = val_metric
                best_state = {k: v.clone() for k, v in self.model.state_dict().items()}
                self.trlog["best_epoch_s2"] = epoch
                no_improve = 0
            else:
                no_improve += 1
            if no_improve >= self.patience:
                break

            pbar.set_description(f"PTaRL S2 e{epoch} val={val_metric:.4f}")

        # Restore best
        if best_state is not None:
            self.model.load_state_dict(best_state)

        # Unfreeze backbone (for predict compatibility)
        for p in self.model.backbone.parameters():
            p.requires_grad_(True)

    # ── Metric helper ─────────────────────────────────────────

    @torch.no_grad()
    def _eval_metric(self, X_val, y_val, stage: int = 2) -> float:
        self.model.eval()
        logits_list = []
        batch_size = 512

        for i in range(0, X_val.size(0), batch_size):
            xb = X_val[i:i+batch_size].float()
            if stage == 1:
                out = self.model(xb, stage=1)
            else:
                out = self.model(xb, stage=2)
            logits_list.append(out)

        logits = torch.cat(logits_list, dim=0)
        y = self._prep_y(y_val)

        if self.is_regression:
            if logits.ndim > 1:
                logits = logits.squeeze(-1)
            return float(F.mse_loss(logits, y.float()).sqrt().item())
        elif self.is_multiclass:
            preds = logits.argmax(dim=1)
            return float((preds == y).float().mean().item())
        else:
            preds = (torch.sigmoid(logits.squeeze(-1)) > 0.5).long()
            return float((preds == y.long()).float().mean().item())

    # ── Public interface (ModernNCA 호환) ─────────────────────

    def fit(self, X_train, y_train, X_val, y_val):
        """Two-stage PTaRL training."""
        self.batch_size = get_batch_size(len(X_train))

        # Stage 1: backbone pre-training
        self._stage1(X_train, y_train, X_val, y_val)

        # Stage 2: PTaRL projection
        self._stage2(X_train, y_train, X_val, y_val)

    def predict(self, X_test) -> np.ndarray:
        self.model.eval()
        logits_list = []
        batch_size = 512

        with torch.no_grad():
            for i in range(0, X_test.size(0), batch_size):
                xb = X_test[i:i+batch_size].float()
                logits_list.append(self.model(xb, stage=2))

        logits = torch.cat(logits_list, dim=0)

        if self.is_regression:
            return logits.squeeze(-1).cpu().numpy()
        elif self.is_multiclass:
            return logits.argmax(dim=1).cpu().numpy()
        else:
            return (torch.sigmoid(logits.squeeze(-1)) > 0.5).long().cpu().numpy()

    def predict_proba(self, X_test, logit: bool = False) -> np.ndarray:
        self.model.eval()
        logits_list = []
        batch_size = 512

        with torch.no_grad():
            for i in range(0, X_test.size(0), batch_size):
                xb = X_test[i:i+batch_size].float()
                logits_list.append(self.model(xb, stage=2))

        logits = torch.cat(logits_list, dim=0)

        if logit:
            return logits.cpu().numpy()

        if self.is_multiclass:
            # overflow 방지: logits를 max 기준으로 shift 후 softmax
            logits = logits - logits.max(dim=1, keepdim=True).values
            probs = F.softmax(logits, dim=1).cpu().numpy()
        else:
            prob_pos = torch.sigmoid(logits.squeeze(-1)).cpu().numpy()
            probs = np.stack([1 - prob_pos, prob_pos], axis=1)

        # nan/inf → 0으로 대체 후 행 합 재정규화
        probs = np.nan_to_num(probs, nan=0.0, posinf=0.0, neginf=0.0)
        row_sum = probs.sum(axis=1, keepdims=True)
        row_sum = np.where(row_sum == 0, 1.0, row_sum)
        probs = probs / row_sum
        return probs