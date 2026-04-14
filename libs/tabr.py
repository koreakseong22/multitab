import torch
import torch.nn as nn
import torch.nn.functional as F

import faiss
import delu
from scipy.special import expit

import abc
import math, time
import statistics
from functools import partial
from typing import Any, Callable, Optional, Union, cast
import typing as ty
from torch import Tensor
from torch.nn.parameter import Parameter
from tqdm import tqdm

from libs.data import get_batch_size
from libs.supervised import CallbackContainer, EarlyStopping, CosineAnnealingLR_Warmup

class Averager():
    """
    A simple averager.

    """
    def __init__(self):
        self.n = 0
        self.v = 0

    def add(self, x):
        """
        
        :x: float, value to be added
        """
        self.v = (self.v * self.n + x) / (self.n + 1)
        self.n += 1

    def item(self):
        return self.v

class PeriodicEmbeddings(nn.Module):
    def __init__(
        self, n_features: int, n_frequencies: int, frequency_scale: float
    ) -> None:
        super().__init__()
        self.frequencies = Parameter(
            torch.normal(0.0, frequency_scale, (n_features, n_frequencies))
        )

    def forward(self, x: Tensor) -> Tensor:
        assert x.ndim == 2
        x = 2 * torch.pi * self.frequencies[None] * x[..., None]
        x = torch.cat([torch.cos(x), torch.sin(x)], -1)
        return x

class PLREmbeddings(nn.Sequential):
    """The PLR embeddings from the paper 'On Embeddings for Numerical Features in Tabular Deep Learning'.

    Additionally, the 'lite' option is added. Setting it to `False` gives you the original PLR
    embedding from the above paper. We noticed that `lite=True` makes the embeddings
    noticeably more lightweight without critical performance loss, and we used that for our model.
    """  # noqa: E501

    def __init__(
        self,
        n_features: int,
        n_frequencies: int = 48,
        frequency_scale: float = 0.01,
        d_embedding: int = 32,
        **kwargs, # [추가] n_bins 등 불필요한 인자 흡수
    ) -> None:
        actual_n_freq = kwargs.get("n_frequencies", n_frequencies)
        actual_scale = kwargs.get("frequency_scale", frequency_scale)
        actual_d_emb = kwargs.get("d_embedding", d_embedding)
        
        super().__init__(
            PeriodicEmbeddings(n_features, actual_n_freq, actual_scale),
            (
                nn.Linear(2 * actual_n_freq, actual_d_emb)
                # if lite
                # else NLinear(n_features, 2 * actual_n_freq, actual_d_emb)
            ),
            nn.ReLU(),
        )
        
_CUSTOM_MODULES = {
    x.__name__: x
    for x in [
        PLREmbeddings
    ]
}

def make_random_batches(
    train_size: int, batch_size: int, device: Optional[torch.device] = None
) :
    permutation = torch.randperm(train_size, device=device)
    batches = permutation.split(batch_size)
    # this function is borrowed from tabr
    # Below, we check that we do not face this issue:
    # https://github.com/pytorch/vision/issues/3816
    # This is still noticeably faster than running randperm on CPU.
    # UPDATE: after thousands of experiments, we faced the issue zero times,
    # so maybe we should remove the assert.
    # assert torch.equal(
    #     torch.arange(train_size, device=device), permutation.sort().values
    # )
    return batches  # type: ignore[code]


# [추가] Gumbel-Retriever
class GumbelRetriever(nn.Module):
    def __init__(self, d_main: int, n_centroids: int = 64):
        super().__init__()
        # 1단계 & 2단계를 위한 Centroid 설정
        self.centroids = nn.Parameter(torch.randn(n_centroids, d_main))
        self.temperature = nn.Parameter(torch.tensor(1.0), requires_grad=False)
        
    def forward(self, k: Tensor, is_train: bool):
        # k: [Batch, d_main], centroids: [Num_Centroids, d_main]
        # 코사인 유사도 혹은 L2 거리 기반 로직 (여기서는 Logits 생성을 위해 유사도 사용)
        logits = torch.matmul(k, self.centroids.t()) # [Batch, Num_Centroids]
        
        if is_train:
            # Gumbel-Softmax sampling (Hard=False로 미분 가능하게 유지)
            # 3단계: self.temperature가 외부에서 annealing됨
            selection = F.gumbel_softmax(logits, tau=self.temperature, hard=True)
        else:
            # 추론 시에는 가장 가까운 Centroid 선택
            selection = F.softmax(logits / self.temperature, dim=-1)
            
        return selection, logits


class TabR(nn.Module):
    def __init__(
        self,
        *,
        n_num_features: int,
        n_cat_features: int,
        n_classes: Optional[int],
        num_embeddings: Optional[dict],
        d_main: int,
        d_multiplier: float,
        encoder_n_blocks: int,
        predictor_n_blocks: int,
        mixer_normalization,
        context_dropout: float,
        dropout0: float,
        dropout1,
        normalization: str,
        activation: str,
        memory_efficient: bool = False,
        candidate_encoding_batch_size: Optional[int] = None,
        **kwargs, # [해결] TypeError 방지용: feature_interaction 등 예상치 못한 인자를 흡수
    ) -> None:
        super().__init__()
        if not memory_efficient:
            assert candidate_encoding_batch_size is None
        if mixer_normalization == 'auto':
            mixer_normalization = encoder_n_blocks > 0
        if encoder_n_blocks == 0:
            assert not mixer_normalization
        if dropout1 == 'dropout0':
            dropout1 = dropout0
        self.n_classes = n_classes

        self.num_embeddings = (
            None
            if num_embeddings is None
            else PLREmbeddings(**num_embeddings, n_features=n_num_features)
        )

        self.n_num_features = n_num_features
        d_in = (
            n_num_features * (1 if num_embeddings is None else num_embeddings['d_embedding'])
            + n_cat_features
        )
        d_block = int(d_main * d_multiplier)
        Normalization = getattr(nn, normalization)
        Activation = getattr(nn, activation)

        def make_block(prenorm: bool) -> nn.Sequential:
            return nn.Sequential(
                *([Normalization(d_main)] if prenorm else []),
                nn.Linear(d_main, d_block),
                Activation(),
                nn.Dropout(dropout0),
                nn.Linear(d_block, d_main),
                nn.Dropout(dropout1),
            )

        self.linear = nn.Linear(d_in, d_main)
        self.blocks0 = nn.ModuleList([make_block(i > 0) for i in range(encoder_n_blocks)])

        self.normalization = Normalization(d_main) if mixer_normalization else None
        self.label_encoder = (
            nn.Linear(1, d_main)
            if n_classes == 1
            else nn.Sequential(
                nn.Embedding(n_classes, d_main),
                delu.nn.Lambda(lambda x: x.squeeze(-2))
            )
        )
        self.K = nn.Linear(d_main, d_main)
        self.T = nn.Sequential(
            nn.Linear(d_main, d_block),
            Activation(),
            nn.Dropout(dropout0),
            nn.Linear(d_block, d_main, bias=False),
        )
        self.dropout = nn.Dropout(context_dropout)

        self.blocks1 = nn.ModuleList([make_block(True) for _ in range(predictor_n_blocks)])
        self.head = nn.Sequential(
            Normalization(d_main),
            Activation(),
            nn.Linear(d_main, n_classes),
        )

        self.search_index = None
        self.memory_efficient = False
        self.candidate_encoding_batch_size = candidate_encoding_batch_size
        self.reset_parameters()

        # Cached candidates
        self.cached_candidate_k = None
        self.cached_candidate_y = None

        # [실험 추가] Gumbel-Retriever 도입
        self.retriever = GumbelRetriever(d_main, n_centroids=128)

        # 1단계 : Centroid 고정(Freeze) -> 이후 Learnable하게 전환 예정(True)
        self.retriever.centroids.requires_grad = False

    def update_index(self, x_num=None, x_cat=None, y=None) -> None:
        """Update FAISS search index (CPU/GPU fallback version)."""
        import faiss
        self.eval()

        target_x_num = x_num if x_num is not None else getattr(self, 'candidate_x_num', None)
        target_x_cat = x_cat if x_cat is not None else getattr(self, 'candidate_x_cat', None)

        if target_x_num is None and target_x_cat is None:
            return

        # [수정된 부분] 인덱스와 짝이 맞는 라벨을 모델에 저장
        if y is not None:
            # nn.Module에는 .device가 없으므로 파라미터 장치를 참조합니다.
            current_device = next(self.parameters()).device
            self.cached_candidate_y = y.detach().to(current_device)

        with torch.no_grad():
            _, k = self._encode(target_x_num, target_x_cat)
            self.cached_candidate_k = k.detach() 
            
            if y is not None:
                current_device = next(self.parameters()).device
                self.cached_candidate_y = y.detach().to(current_device)
            
            # FAISS용 numpy 변환
            embeddings = k.cpu().numpy().astype('float32')
        
        d = embeddings.shape[1]
        
        try:
            res = faiss.StandardGpuResources()
            cfg = faiss.GpuIndexFlatConfig()
            # 장치 번호 자동 할당
            cfg.device = next(self.parameters()).device.index if next(self.parameters()).is_cuda else 0
            
            gpu_index = faiss.GpuIndexFlatL2(res, d, cfg)
            gpu_index.add(embeddings)
            self.search_index = gpu_index
            
        except (AttributeError, Exception):
            cpu_index = faiss.IndexFlatL2(d)
            cpu_index.add(embeddings)
            self.search_index = cpu_index

    def reset_parameters(self):
        if isinstance(self.label_encoder, nn.Linear):
            bound = 1 / math.sqrt(2.0)
            nn.init.uniform_(self.label_encoder.weight, -bound, bound)
            nn.init.uniform_(self.label_encoder.bias, -bound, bound)
        else:
            nn.init.uniform_(self.label_encoder[0].weight, -1.0, 1.0)

    def _encode(self, x_num, x_cat):
        x = []
        if x_num is None:
            self.num_embeddings = None
        else:
            x.append(
                x_num if self.num_embeddings is None else self.num_embeddings(x_num).flatten(1)
            )
        if x_cat is not None:
            x.append(x_cat)
        x = torch.cat(x, dim=1)
        x = self.linear(x)
        for block in self.blocks0:
            x = x + block(x)
        k = self.K(x if self.normalization is None else self.normalization(x))
        return x, k

    def forward(
        self,
        *,
        x_num: Tensor,
        x_cat: ty.Optional[Tensor],
        y: Optional[Tensor],
        candidate_x_num: ty.Optional[Tensor],
        candidate_x_cat: ty.Optional[Tensor],
        candidate_y: Tensor, # 훈련 시 전달되지만, 안전을 위해 내부 캐시를 우선 사용함
        context_size: int,
        is_train: bool,
    ) -> Tensor:
        device = x_num.device if x_num is not None else x_cat.device

        # 1. 메인 특징 인코딩
        x, k = self._encode(x_num, x_cat)
        batch_size = k.shape[0]

        # 2. FAISS 검색 (인덱스 업데이트 시 저장된 캐시 데이터 기반)
        # 훈련 시에는 자기 자신을 제외하기 위해 context_size + 1개를 찾습니다.
        with torch.no_grad():
            distances, context_idx_raw = self.search_index.search(
                k.to(torch.float32).detach().cpu().numpy(), context_size + 1
            )
            context_idx = torch.tensor(context_idx_raw, device=device)
            
            # [안전 장치] FAISS의 -1 리턴 방지 및 인덱스 범위 클램핑
            context_idx = torch.clamp(context_idx, min=0, max=self.cached_candidate_y.size(0) - 1)

            # 훈련 시 자기 자신 제외 (Top-1 이 자기 자신인 경우 제거)
            if is_train:
                # 간단한 방식: 자기 자신과 같은 인덱스가 있으면 맨 뒤로 밀어버림
                self_mask = (context_idx == torch.arange(batch_size, device=device)[:, None])
                distances_tmp = torch.tensor(distances, device=device)
                distances_tmp[self_mask] = torch.inf
                context_idx = context_idx.gather(-1, distances_tmp.argsort()[:, :-1])
            else:
                context_idx = context_idx[:, :context_size]

        # 3. 인덱스와 완벽히 일치하는 데이터 추출
        # update_index에서 저장한 cached_candidate_k와 y를 사용함
        context_k = self.cached_candidate_k[context_idx] # [Batch, Context, D]
        context_y = self.cached_candidate_y[context_idx] # [Batch, Context]

        # 4. Gumbel-Softmax 라우팅 및 유사도 계산
        # 가이드라인: self.retriever를 통해 어떤 Centroid 구역으로 갈지 결정
        selection, _ = self.retriever(k, is_train)
        
        # L2 거리 기반 유사도 산출 (Temperature Annealing 적용)
        similarities = (
            -k.square().sum(-1, keepdim=True)
            + (2 * (k[..., None, :] @ context_k.transpose(-1, -2))).squeeze(-2)
            - context_k.square().sum(-1)
        ) / (self.retriever.temperature + 1e-8)

        probs = F.softmax(similarities, dim=-1)

        # 5. Label Encoding & Aggregation
        if self.n_classes > 1:
            # Classification: 타겟을 long 타입 인덱스로 변환하여 임베딩
            context_y_emb = self.label_encoder(context_y.long())
        else:
            # Regression: 타겟 값 그대로 사용
            context_y_emb = self.label_encoder(context_y[..., None])

        # Residual connection with context
        values = context_y_emb + self.T(k[:, None] - context_k)
        context_x = (probs[:, None] @ values).squeeze(1)
        x = x + context_x

        # 6. Predictor blocks
        for block in self.blocks1:
            x = x + block(x)
        
        return self.head(x)

class TabRMethod(object, metaclass=abc.ABCMeta):
    def __init__(self, params, tasktype, num_cols=[], cat_features=[], input_dim=0, output_dim=0, device='cuda', data_id=None, modelname="tabr"):
        super().__init__()

        self.num_cols = num_cols
        self.cat_features = cat_features
        self.tasktype = tasktype
        self.params = params
        self.data_id = data_id
        self.device = device
        self.max_epoch = 100

        self.is_binclass = (self.tasktype == "binclass")
        self.is_multiclass = (self.tasktype == "multiclass")
        self.is_regression = (self.tasktype == "regression")
        self.n_num_features = len(self.num_cols)
        self.n_cat_features = len(self.cat_features)

        self.train_step = 0
        self.context_size = 96

        self.model = TabR(
            n_num_features = self.n_num_features,
            n_cat_features = self.n_cat_features,
            n_classes = output_dim,
            **params["model"]
        ).to(device)
        self.model.float()

        self._callback_container = CallbackContainer([EarlyStopping(
            early_stopping_metric="val_loss",
            patience=params["early_stopping_rounds"],
        )])

        self.trlog = {}
        self.trlog['args'] = params
        self.trlog['train_loss'] = []
        self.trlog['best_epoch'] = 0
        if self.is_regression:
            self.trlog['best_res'] = 1e10
        else:
            self.trlog['best_res'] = 0

        # [1단계] : Centroid 고정(Freeze) - 이미 TabR 클래스 내에서 구현되어 있음
        for name, param in self.model.retriever.named_parameters():
            if "centroids" in name:
                param.requires_grad = False

    def fit(self, X_train, y_train, X_val, y_val):
        self.N = X_train[:, self.num_cols]
        self.C = X_train[:, self.cat_features]
        
        # [데이터 매핑] Wine Quality 등에서 발생하는 Index Out of Bounds 방지
        if self.tasktype == "multiclass":
            y_t = torch.argmax(y_train, dim=1) if y_train.dim() > 1 else y_train
            y_v = torch.argmax(y_val, dim=1) if y_val.dim() > 1 else y_val
            
            unique_labels = torch.unique(torch.cat([y_t.cpu(), y_v.cpu()]))
            label_map = {val.item(): i for i, val in enumerate(unique_labels)}
            
            self.y = torch.tensor([label_map[v.item()] for v in y_t], device=self.device).long()
            self.y_val_mapped = torch.tensor([label_map[v.item()] for v in y_v], device=self.device).long()
            print(f"[*] Target Mapped: {len(unique_labels)} classes identified.")
        else:
            self.y = y_train
            self.y_val_mapped = y_val

        # [가이드라인] C = sqrt(N) 동적 설정
        self.train_size = self.N.shape[0] if self.N is not None else self.C.shape[0]
        self.n_centroids = int(math.sqrt(self.train_size))
        
        # [Centroid 초기화] 훈련 데이터 임베딩을 샘플링하여 초기 포인트 설정
        with torch.no_grad():
            sample_idx = torch.randperm(self.train_size)[:self.n_centroids]
            s_n = self.N[sample_idx].float().to(self.device) if self.N is not None else None
            s_c = self.C[sample_idx].to(self.device) if self.C is not None else None
            _, initial_centroids = self.model._encode(s_n, s_c)
            self.model.retriever.centroids.data = initial_centroids
            print(f"[*] Centroids initialized with {self.n_centroids} samples (sqrt(N)).")

        # Optimizer 및 학습 설정
        self.batch_size = get_batch_size(self.train_size)
        self.criterion = F.cross_entropy if self.is_multiclass else (F.mse_loss if self.is_regression else F.binary_cross_entropy_with_logits)
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.params['lr'], weight_decay=self.params['weight_decay'])
        
        if self.params.get("lr_scheduler") and (self.train_size > self.batch_size):
            self.scheduler = CosineAnnealingLR_Warmup(self.optimizer, warmup_epochs=10, T_max=100, iter_per_epoch=self.train_size//self.batch_size, 
                                                     base_lr=self.params['lr'], warmup_lr=1e-6, eta_min=0, last_epoch=-1)
        
        self.train_indices = torch.arange(self.train_size, device=self.device)

        # 메인 학습 루프
        pbar = tqdm(range(1, self.max_epoch+1))
        for epoch in pbar:
            pbar.set_description(f"EPOCH: {epoch}")
            tic = time.time()
            loss = self.train_epoch(epoch)
            self.validate(epoch, X_val, self.y_val_mapped) 
            elapsed = time.time() - tic
            pbar.set_postfix_str(f'Time: {elapsed:.2f}s, Loss: {loss:.4f}')
            if not self.continue_training: break

    def train_epoch(self, epoch):
        self.model.train()
        tl = Averager()
        entropy_log = Averager()
        active_log = Averager()

        # 1. FAISS 인덱스 & 검색용 라벨 동기화 업데이트 (매우 중요)
        # 훈련 데이터 중 최대 5만개를 검색 후보군(Memory Bank)으로 설정
        c_n = self.N[:50000].float().to(self.device) if self.N is not None else None
        c_c = self.C[:50000].to(self.device) if self.C is not None else None
        c_y = self.y[:50000].to(self.device)
        
        # [수정] update_index가 y도 받아서 내부에 저장하도록 tabr.py를 고쳐야 함
        self.model.update_index(c_n, c_c, c_y)

        # 2. 가이드라인: Temperature Annealing 및 Centroid 학습 제어
        # 지수적 감소: 1.0 -> 0.1 (r=0.05 설정 시 약 45에폭에서 0.1 도달)
        new_tau = max(0.1, 1.0 * math.exp(-0.05 * (epoch - 1)))
        self.model.retriever.temperature.data = torch.tensor(new_tau).to(self.device)
        
        # 10에폭 이후부터 Centroid 위치 최적화 시작
        self.model.retriever.centroids.requires_grad = (epoch > 10)

        # 3. 배치 학습
        for batch_idx in make_random_batches(self.train_size, self.batch_size, self.device):
            self.train_step += 1
            
            X_num = self.N[batch_idx].float() if self.N is not None else None
            X_cat = self.C[batch_idx] if self.C is not None else None
            y = self.y[batch_idx]

            # Forward
            logits = self.model(
                x_num=X_num, x_cat=X_cat, y=y, 
                candidate_x_num=None, candidate_x_cat=None, candidate_y=None, # 내부 캐시 사용
                context_size=self.context_size,
                is_train=True
            )

            loss = self.criterion(logits, y)
            
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            if hasattr(self, 'scheduler'):
                self.scheduler.step()

            # [가이드라인 검증 지표 트래킹]
            with torch.no_grad():
                # 현재 배치의 쿼리들이 어떤 Centroid에 쏠리는지 계산
                _, cluster_logits = self.model.retriever(self.model._encode(X_num, X_cat)[1], is_train=False)
                w = F.softmax(cluster_logits / new_tau, dim=-1)
                
                # Sparsity(엔트로피): 낮을수록 특정 클러스터에 집중됨
                entropy = -(w * torch.log(w + 1e-8)).sum(-1).mean()
                entropy_log.add(entropy.item())
                
                # Active Ratio: 1% 이상의 확률을 가진 클러스터 비율
                active_ratio = (w > 0.01).float().sum(-1).mean() / self.n_centroids
                active_log.add(active_ratio.item())

            tl.add(loss.item())

        # 결과 출력 (교수님께 보고할 핵심 지표)
        avg_loss = tl.item()
        print(f"\n[Epoch {epoch:03d}] Tau: {new_tau:.3f} | Loss: {avg_loss:.4f} | "
              f"Entropy: {entropy_log.item():.4f} | Active: {active_log.item()*100:.1f}%")
        
        self.trlog['train_loss'].append(avg_loss)
        return avg_loss


    def validate(self, epoch, X_val, y_val_mapped):
        self.model.eval()
        val_loss = 0.0
        with torch.no_grad():
            # 후보군은 항상 훈련 데이터(self.y)에서 가져옵니다.
            candidate_y = self.y[:50000]
            candidate_x_num = self.N[:50000].float() if self.N is not None else None
            candidate_x_cat = self.C[:50000] if self.C is not None else None

            logits = []
            iters = (X_val.shape[0] + 9999) // 10000 
            for i in range(iters):
                batch_X = X_val[10000*i:10000*(i+1)]
                X_num = batch_X[:, self.num_cols].float() if len(self.num_cols) > 0 else None
                X_cat = batch_X[:, self.cat_features] if len(self.cat_features) > 0 else None
                
                val_pred = self.model(
                    x_num=X_num, x_cat=X_cat, y=None, 
                    candidate_x_num=candidate_x_num,
                    candidate_x_cat=candidate_x_cat,
                    candidate_y=candidate_y,
                    context_size=self.context_size,
                    is_train=False,
                ).squeeze(-1)
                logits.append(val_pred)

            logits = torch.cat(logits, dim=0)
            val_loss = self.criterion(logits, y_val_mapped).item()
        
        self._callback_container.on_epoch_end(epoch, {"val_loss": val_loss, "epoch": epoch})
        self.continue_training = not any([cb.should_stop for cb in self._callback_container.callbacks])
        
        self._callback_container.on_epoch_end(epoch, {"val_loss": val_loss, "epoch": epoch})
        if any([cb.should_stop for cb in self._callback_container.callbacks]):
            self.continue_training = False
        else:
            self.continue_training = True
                
    def predict(self, X_test):        
        self.model.eval()
        with torch.no_grad():
            candidate_x_num = self.N[:50000] if self.N is not None else None
            candidate_x_cat = self.C[:50000] if self.C is not None else None
            candidate_y = self.y[:50000]
            candidate_x_num = candidate_x_num.float() if candidate_x_num is not None else None
            candidate_x_cat = candidate_x_cat.float() if candidate_x_cat is not None else None
            if self.is_regression:
                candidate_y = candidate_y.float()

            logits = []
            iters = X_test.shape[0] // 10000 + 1
            for i in range(iters):
                N = X_test[10000*i:10000*(i+1), self.num_cols]
                C = X_test[10000*i:10000*(i+1), self.cat_features]
                if len(self.num_cols) == 0:
                    X_num, X_cat = None, C
                elif len(self.cat_features) == 0:
                    X_num, X_cat = N, None
                else:
                    X_num, X_cat = N, C
                
                X_num = X_num.float() if X_num is not None else None
                X_cat = X_cat.float() if X_cat is not None else None
    
                if X_cat is None and X_num is not None:
                    x, candidate_x = X_num, candidate_x_num
                elif X_cat is not None and X_num is None:
                    x, candidate_x = X_cat, candidate_x_cat
                else:
                    x, candidate_x = torch.cat([X_num, X_cat], dim=1),torch.cat([candidate_x_num, candidate_x_cat], dim=1)
    
                val_pred = self.model(
                    x_num=x[:,:self.n_num_features], x_cat=x[:,self.n_num_features:], y=None, 
                    candidate_x_num=candidate_x[:,:self.n_num_features],
                    candidate_x_cat=candidate_x[:,self.n_num_features:],
                    candidate_y=candidate_y,
                    context_size=self.context_size,
                    is_train=False,
                ).squeeze(-1)

                logits.append(val_pred)

            logits = torch.concatenate(logits, dim=0)

        if self.tasktype == "binclass":
            return torch.round(torch.sigmoid(logits)).detach().cpu().numpy()
        elif self.tasktype == "regression":
            return logits.detach().cpu().numpy()
        else:
            return torch.argmax(logits, dim=1).detach().cpu().numpy()

    def predict_proba(self, X_test, logit=False):
        self.model.eval()
        with torch.no_grad():
            candidate_x_num = self.N[:50000] if self.N is not None else None
            candidate_x_cat = self.C[:50000] if self.C is not None else None
            candidate_y = self.y[:50000]
            candidate_x_num = candidate_x_num.float() if candidate_x_num is not None else None
            candidate_x_cat = candidate_x_cat.float() if candidate_x_cat is not None else None
            if self.is_regression:
                candidate_y = candidate_y.float()

            logits = []
            iters = X_test.shape[0] // 10000 + 1
            for i in range(iters):
                N = X_test[10000*i:10000*(i+1), self.num_cols]
                C = X_test[10000*i:10000*(i+1), self.cat_features]
                if len(self.num_cols) == 0:
                    X_num, X_cat = None, C
                elif len(self.cat_features) == 0:
                    X_num, X_cat = N, None
                else:
                    X_num, X_cat = N, C
                
                X_num = X_num.float() if X_num is not None else None
                X_cat = X_cat.float() if X_cat is not None else None
    
                if X_cat is None and X_num is not None:
                    x, candidate_x = X_num, candidate_x_num
                elif X_cat is not None and X_num is None:
                    x, candidate_x = X_cat, candidate_x_cat
                else:
                    x, candidate_x = torch.cat([X_num, X_cat], dim=1),torch.cat([candidate_x_num, candidate_x_cat], dim=1)
    
                val_pred = self.model(
                    x_num=x[:,:self.n_num_features], x_cat=x[:,self.n_num_features:], y=None, 
                    candidate_x_num=candidate_x[:,:self.n_num_features],
                    candidate_x_cat=candidate_x[:,self.n_num_features:],
                    candidate_y=candidate_y,
                    context_size=self.context_size,
                    is_train=False,
                ).squeeze(-1)
                    
                logits.append(val_pred)

            logits = torch.concatenate(logits, dim=0)

        if logit:
            return logits.detach().cpu().numpy()
        else:
            return torch.nn.functional.softmax(logits).detach().cpu().numpy()
