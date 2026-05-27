"""
libs/tabm.py
============
TabM: Advancing Tabular Deep Learning With Parameter-Efficient Ensembling
Paper: Gorishniy et al., ICLR 2025
공식 패키지: pip install tabm  (저자 Yury Gorishniy 직접 배포)

공식 패키지의 make_tabm_backbone + EnsembleView 사용.
→ LinearEfficientEnsemble(BatchEnsemble + tabm_init) 원본 그대로.

Forward:
    x (B, F)
    → EnsembleView    → (B, k, F)
    → backbone        → (B, k, d_block)   [LinearEfficientEnsemble × n_blocks]
    → head (Linear)   → (B, k, output_dim)
    → mean(dim=1)     → (B, output_dim)

start_scaling_init='random-signs': 임베딩 없을 때 원본 권장값
start_scaling_init_chunks=[1]*n_features: 특성별 1개 스칼라 (원본 default)

search_space.py 파라미터:
    k=32(고정), d_block, n_blocks, dropout,
    lr, weight_decay, lr_scheduler, n_epochs, early_stopping_rounds
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

try:
    import tabm as tabm_pkg
except ImportError:
    raise ImportError(
        "TabM 공식 패키지가 필요합니다: pip install tabm"
    )

from libs.supervised import CallbackContainer, EarlyStopping, CosineAnnealingLR_Warmup
from libs.data import get_batch_size


# ─────────────────────────────────────────────────────────────
# TabM 네트워크
# ─────────────────────────────────────────────────────────────

class TabMNet(nn.Module):
    """공식 tabm 패키지 기반 TabM 구현."""

    def __init__(
        self,
        input_dim:  int,
        output_dim: int,
        d_block:    int   = 256,
        n_blocks:   int   = 2,
        k:          int   = 32,
        dropout:    float = 0.0,
    ):
        super().__init__()
        self.k          = k
        self.output_dim = output_dim

        # (B, F) → (B, k, F)
        self.ensemble_view = tabm_pkg.EnsembleView(k=k)

        # BatchEnsemble MLP backbone (원본 tabm_init=True: 첫 레이어만 random-signs)
        self.backbone = tabm_pkg.make_tabm_backbone(
            d_in                    = input_dim,
            n_blocks                = n_blocks,
            d_block                 = d_block,
            dropout                 = dropout,
            k                       = k,
            arch_type               = 'tabm',
            start_scaling_init      = 'random-signs',
            # feature별 1개 스칼라 초기화 (원본 권장 default)
            start_scaling_init_chunks = [1] * input_dim,
        )

        # 출력 헤드: 공유 Linear (원본과 동일)
        self.head = nn.Linear(d_block, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.ensemble_view(x)   # (B, k, F)
        x = self.backbone(x)        # (B, k, d_block)
        x = self.head(x)            # (B, k, output_dim)
        return x.mean(dim=1)        # (B, output_dim)


# ─────────────────────────────────────────────────────────────
# MultiTab 래퍼
# ─────────────────────────────────────────────────────────────

class TabMMethod:
    """
    MultiTab supmodel 인터페이스 (fit / predict / predict_proba).

    model.py 호출 형식:
        TabMMethod(params, tasktype, dataset.X_num, dataset.X_cat,
                   input_dim=input_dim, output_dim=output_dim,
                   device=device, data_id=openml_id)
    """

    def __init__(
        self,
        params:       dict,
        tasktype:     str,
        num_cols:     list = [],
        cat_features: list = [],
        input_dim:    int  = 0,
        output_dim:   int  = 1,
        device:       str  = "cuda",
        data_id              = None,
        modelname:    str  = "tabm",
    ):
        self.params    = params
        self.tasktype  = tasktype
        self.device    = device
        self.data_id   = data_id
        self.modelname = modelname

        self.model = TabMNet(
            input_dim  = input_dim,
            output_dim = output_dim,
            d_block    = params["d_block"],
            n_blocks   = params["n_blocks"],
            k          = params.get("k", 32),
            dropout    = params.get("dropout", 0.0),
        ).to(device)

        self._callback_container = CallbackContainer([
            EarlyStopping(
                early_stopping_metric="val_loss",
                patience=params.get("early_stopping_rounds", 20),
            )
        ])

    # ── fit ─────────────────────────────────────────────────

    def fit(self, X_train, y_train, X_val, y_val):
        if y_train.ndim == 1:
            y_train = y_train.unsqueeze(1)
            y_val   = y_val.unsqueeze(1)

        p          = self.params
        device     = self.device
        batch_size = get_batch_size(len(X_train))

        if self.tasktype == "regression":
            loss_fn = F.mse_loss
        elif self.tasktype == "binclass":
            loss_fn = F.binary_cross_entropy_with_logits
        else:
            loss_fn = F.cross_entropy

        # 원본 파라미터 그룹:
        #   no weight_decay: bias, s (output scaling adapter)
        #   weight_decay 적용: weight, r (input scaling adapter)
        # 실제 파라미터 이름: blocks.N.0.{weight, r, s, bias}
        decay, no_decay = [], []
        for name, param in self.model.named_parameters():
            if name.endswith('.s') or name.endswith('.bias') or name.endswith('bias'):
                no_decay.append(param)
            else:
                decay.append(param)

        optimizer = torch.optim.AdamW([
            {'params': decay,    'weight_decay': p["weight_decay"]},
            {'params': no_decay, 'weight_decay': 0.0},
        ], lr=p["lr"])

        train_ds     = torch.utils.data.TensorDataset(X_train, y_train)
        val_ds       = torch.utils.data.TensorDataset(X_val,   y_val)
        drop_last    = (len(train_ds) % batch_size == 1)
        train_loader = torch.utils.data.DataLoader(
            train_ds, batch_size=batch_size, shuffle=True, drop_last=drop_last)
        val_loader   = torch.utils.data.DataLoader(
            val_ds, batch_size=batch_size, shuffle=False)

        scheduler = None
        if p.get("lr_scheduler", False):
            scheduler = CosineAnnealingLR_Warmup(
                optimizer,
                base_lr=p["lr"],
                warmup_epochs=10,
                T_max=p.get("n_epochs", 100),
                iter_per_epoch=len(train_loader),
                warmup_lr=1e-6,
                eta_min=0,
                last_epoch=-1,
            )

        n_epochs = p.get("n_epochs", 100)
        pbar = tqdm(range(1, n_epochs + 1))
        for epoch in pbar:
            self.model.train()
            ep_loss = 0.0
            for x_b, y_b in train_loader:
                x_b, y_b = x_b.to(device), y_b.to(device)
                optimizer.zero_grad()

                logits = self.model(x_b)
                if logits.size() != y_b.size():
                    logits = logits.view(y_b.size())

                loss = loss_fn(logits, y_b)
                loss.backward()
                optimizer.step()
                if scheduler:
                    scheduler.step()
                ep_loss = loss.item()

            pbar.set_description("EPOCH: %i" % epoch)
            pbar.set_postfix_str(
                f"data_id: {self.data_id}, Model: {self.modelname}, "
                f"Tr loss: {ep_loss:.5f}")

            # Validation
            self.model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for x_v, y_v in val_loader:
                    x_v, y_v = x_v.to(device), y_v.to(device)
                    logits = self.model(x_v)
                    if logits.size() != y_v.size():
                        logits = logits.view(y_v.size())
                    val_loss += loss_fn(logits, y_v).item()
            val_loss /= len(val_loader)

            self._callback_container.on_epoch_end(
                epoch, {"val_loss": val_loss, "epoch": epoch})
            if any(cb.should_stop for cb in self._callback_container.callbacks):
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

    def _forward_batched(self, X, batch_size: int = 256):
        parts = []
        for i in range(0, len(X), batch_size):
            xb = X[i: i + batch_size].to(self.device)
            parts.append(self.model(xb).cpu())
        return torch.cat(parts, dim=0)