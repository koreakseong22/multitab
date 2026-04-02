import torch
import torch.nn as nn
import torch.nn.functional as F

# import faiss
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


# --- 추가: Wasserstein 계산을 위한 Sinkhorn 알고리즘 (Simplified) ---
def sinkhorn_distance(x, y, eps=0.1, n_iters=5):
    """
    x, y: (N, d_main) 형태의 벡터 (N = Batch * Fetch_Size)
    """
    # 1. 확률 분포로 변환 및 정규화 (각 샘플의 합이 1이 되도록)
    mu = F.softplus(x)
    mu = mu / (mu.sum(dim=-1, keepdim=True) + 1e-8)
    nu = F.softplus(y)
    nu = nu / (nu.sum(dim=-1, keepdim=True) + 1e-8)

    # 2. 비용 행렬 C와 커널 행렬 K 생성 (d_main x d_main)
    d = mu.shape[-1]
    indices = torch.arange(d, device=mu.device).float()
    C = torch.abs(indices.unsqueeze(0) - indices.unsqueeze(1)) # (D, D)
    K = torch.exp(-C / eps) # (D, D)
    
    # 3. Sinkhorn Iterations
    u = torch.ones_like(mu) # (N, D)
    
    for _ in range(n_iters):
        # v = nu / (u @ K)
        v = nu / (torch.matmul(u, K) + 1e-8) # (N, D)
        # u = mu / (v @ K.T)
        u = mu / (torch.matmul(v, K.t()) + 1e-8) # (N, D)
        
    # 4. [수정된 부분] 최종 거리 계산 (N, D) -> (N,)
    # Transport Plan P = diag(u) @ K @ diag(v)
    # Cost = Tr(P.T @ C) = sum(u * ((K * C) @ v.T).T)
    K_C = K * C # Kernel과 Cost의 요소별 곱 (D, D)
    
    # u와 (v @ K_C.T)를 행렬곱 연산으로 처리하여 차원 불일치 해결
    dist = torch.sum(u * torch.matmul(v, K_C.t()), dim=-1)
    
    return dist

# --- 상단에 반드시 포함되어야 할 Loss 함수 ---
def compute_wasserstein_contrastive_loss(anchor_k, context_k, anchor_y, context_y, margin=1.0):
    batch_size, context_size, d_main = context_k.shape
    
    # 1. Wasserstein 거리 계산 (기존 sinkhorn_distance 활용)
    k_expanded = anchor_k.unsqueeze(1).expand(-1, context_size, -1).reshape(-1, d_main)
    ctx_expanded = context_k.reshape(-1, d_main)
    
    dists = sinkhorn_distance(k_expanded, ctx_expanded)
    dists = dists.view(batch_size, context_size)
    
    # 2. Positive/Negative 마스크 생성 (같은 클래스면 Positive)
    mask_pos = (anchor_y.unsqueeze(1) == context_y).float()
    mask_neg = 1 - mask_pos
    
    # 3. Contrastive Loss 계산
    loss_pos = mask_pos * torch.pow(dists, 2)
    loss_neg = mask_neg * torch.pow(torch.clamp(margin - dists, min=0.0), 2)
    
    loss = (loss_pos.sum() + loss_neg.sum()) / (batch_size * context_size + 1e-8)
    return loss


# [추가] 미분가능한 Retriever 모듈.
class NeuralRetriever(nn.Module):
    def __init__(self, d_main, n_anchors=64, temperature=1.0):
        super().__init__()
        # 학습 가능한 앵커 포인트 (Neural Cluster Routing의 핵심)
        self.anchors = nn.Parameter(torch.randn(n_anchors, d_main))
        self.temperature = nn.Parameter(torch.tensor(temperature))
        
    def forward(self, query_k, candidate_k):
        # 1. Query와 모든 후보군 간의 유사도 계산 (전체 미분 가능)
        # similarities: (Batch, N_candidates)
        logits = torch.matmul(query_k, candidate_k.t()) 
        
        # 2. Gumbel-Softmax 또는 Softmax로 가중치 계산
        # 여기서는 부드러운 최적화를 위해 Softmax를 사용합니다.
        probs = F.softmax(logits / self.temperature, dim=-1)
        
        return probs, logits


# --- 추가 : Feature Interaction을 위한 모듈 ---
class FeatureInteraction(nn.Module):
    def __init__(self, d_embedding: int, n_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        # [수정] d_embedding이 n_heads로 나누어떨어지도록 n_heads를 동적으로 설정.
        self.attn_dim = ((d_embedding + n_heads - 1) // n_heads) * n_heads

        # 입력 차원은 Attention 가능 차원으로 투영.
        self.input_proj = nn.Linear(d_embedding, self.attn_dim)
        
        # 피처 토큰 간의 관계를 학습하기 위한 Multi-Head Attention
        self.mha = nn.MultiheadAttention(self.attn_dim, n_heads, dropout=dropout, batch_first=True)
        
        # 다시 원래 d_embedding 차원으로 복구
        self.output_proj = nn.Linear(self.attn_dim, d_embedding)
        
        self.ln = nn.LayerNorm(d_embedding)
        self.ffn = nn.Sequential(
            nn.Linear(d_embedding, d_embedding * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_embedding * 2, d_embedding),
        )
        self.ln2 = nn.LayerNorm(d_embedding)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: Tensor) -> Tensor:
        # x: (Batch, n_features, d_embedding)
        residual = x
        
        # 입력을 Attention 가능 차원으로 투영
        x_proj = self.input_proj(x)
        
        # 1. Self-Attention: 어떤 피처가 다른 피처와 관련이 있는지 학습
        attn_out, _ = self.mha(x_proj, x_proj, x_proj)
        attn_out = self.output_proj(attn_out)  # 다시 원래 차원으로 복구
        x = x + self.dropout(attn_out)
        x = self.ln(x)
        
        # 2. Feed-Forward: 각 피처 토큰의 표현력을 강화
        ffn_out = self.ffn(x)
        x = x + self.dropout(ffn_out)
        x = self.ln2(x)
        return x
    # def forward(self, x: Tensor) -> Tensor:
    #     # x: (Batch, n_features, d_embedding)
    #     residual = x
    #     # 1. Self-Attention: 어떤 피처가 다른 피처와 관련이 있는지 학습
    #     attn_out, _ = self.mha(x, x, x)
    #     x = x + self.dropout(attn_out)
    #     x = self.ln(x)
        
    #     # 2. Feed-Forward: 각 피처 토큰의 표현력을 강화
    #     ffn_out = self.ffn(x)
    #     x = x + self.dropout(ffn_out)
    #     x = self.ln2(x)
    #     return x

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

# --- 추가: Soft Binning Embeddings ---
class SoftBinningEmbeddings(nn.Module):
    def __init__(self, n_features: int, n_bins: int, d_embedding: int):
        super().__init__()
        self.n_bins = n_bins
        # 각 피처별로 n_bins개의 중심점을 학습 가능한 파라미터로 설정
        # 초기값은 -2.0에서 2.0 사이로 균등하게 분포 (데이터가 정규화되었다고 가정)
        self.centers = Parameter(torch.linspace(-2.0, 2.0, n_bins).repeat(n_features, 1))
        
        # 생성된 bin 확률 분포를 모델의 임베딩 차원으로 투영
        self.projection = nn.Linear(n_bins, d_embedding)
        self.activation = nn.ReLU()

    def forward(self, x: Tensor) -> Tensor:
        # x: (Batch, n_features)
        # 1. 거리 계산: (Batch, n_features, 1) - (1, n_features, n_bins)
        # -> (Batch, n_features, n_bins)
        diff = x.unsqueeze(-1) - self.centers.unsqueeze(0)
        dist = torch.abs(diff)
        
        # 2. Soft-assignment: 거리가 가까울수록 높은 확률 (Temperature 0.1 적용)
        # 이 부분이 '이산화(Binning)'를 미분 가능하게 만드는 핵심입니다.
        bin_weights = F.softmax(-dist * 10.0, dim=-1) 
        
        # 3. 토큰화 완성: 각 피처별로 d_embedding 크기의 벡터 생성
        # (Batch, n_features, n_bins) -> (Batch, n_features, d_embedding)
        x = self.projection(bin_weights)
        x = self.activation(x)
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
        n_frequencies: int,
        frequency_scale: float,
        d_embedding: int,
    ) -> None:
        super().__init__(
            PeriodicEmbeddings(n_features, n_frequencies, frequency_scale),
            (
                nn.Linear(2 * n_frequencies, d_embedding)
                # if lite
                # else NLinear(n_features, 2 * n_frequencies, d_embedding)
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
    assert torch.equal(
        torch.arange(train_size, device=device), permutation.sort().values
    )
    return batches  # type: ignore[code]


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
        metric: str = 'l2',  # 추가: 'l1', 'cosine', 'mahalanobis', 'wasserstein', 'kl'
        feature_interaction: bool = True,  # 추가: Feature Interaction 레이어 사용 여부
        **kwargs,
    ) -> None:
        super().__init__()
        self.metric = metric

        # [추가] Feature Interaction 레이어 정의
        self.user_interaction = feature_interaction
        if self.user_interaction and num_embeddings is not None:
            self.feature_interaction_layer = FeatureInteraction(
                d_embedding = num_embeddings['d_embedding'],
                # n_heads = 4,
            )

        # [추가] Mahalanobis를 위한 학습 가능 행렬 (d_main x d_main)
        if self.metric == 'mahalanobis':
            self.W = Parameter(torch.eye(d_main))  # 초기값은 단위 행렬로 설정
        
        # Gumbel-Softmax를 위한 temperature 파라미터 (학습 가능하게 설정 가능)
        self.temperature = Parameter(torch.tensor(1.0))  # 초기값은 1.0, 필요에 따라 조정 가능 

        if not memory_efficient:
            assert candidate_encoding_batch_size is None
        if mixer_normalization == 'auto':
            mixer_normalization = encoder_n_blocks > 0
        if encoder_n_blocks == 0:
            assert not mixer_normalization
        if dropout1 == 'dropout0':
            dropout1 = dropout0
        self.n_classes = n_classes

        # [수정] num_embeddings이 soft_binning인 경우 별도의 SoftBinningEmbeddings 클래스를 사용하도록 분기
        if num_embeddings is not None:
            token_type = num_embeddings.get('type')
            
            if token_type == 'soft_binning':
                # Soft-Binning에 필요한 인자만 추출
                self.num_embeddings = SoftBinningEmbeddings(
                    n_features=n_num_features,
                    n_bins=num_embeddings.get('n_bins', 32),
                    d_embedding=num_embeddings.get('d_embedding', 128)
                )
            else:
                # PLR 방식일 때는 n_bins와 type을 제외한 필요한 인자만 추출하여 전달
                # PLREmbeddings의 __init__은 n_frequencies, frequency_scale, d_embedding만 받습니다.
                plr_params = {
                    'n_frequencies': num_embeddings.get('n_frequencies'),
                    'frequency_scale': num_embeddings.get('frequency_scale'),
                    'd_embedding': num_embeddings.get('d_embedding')
                }
                # None인 값들은 제외 (기본값 사용 유도)
                plr_params = {k: v for k, v in plr_params.items() if v is not None}
                
                self.num_embeddings = PLREmbeddings(**plr_params, n_features=n_num_features)
        else:
            self.num_embeddings = None
        # self.num_embeddings = (
        #     None
        #     if num_embeddings is None
        #     else PLREmbeddings(**num_embeddings, n_features=n_num_features)
        # )

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

        # self.search_index = None

        # [추가] Neural Retriever 모듈 초기화 (FAISS 제거) - d_main 차원의 벡터를 처리할 수 있도록 설정
        self.retriever = NeuralRetriever(d_main) # 추가

        self.memory_efficient = False
        self.candidate_encoding_batch_size = candidate_encoding_batch_size
        self.reset_parameters()

        # Cached candidates
        self.cached_candidate_k = None
        self.cached_candidate_y = None

    def update_index(self):
        pass # FAISS 제거
        """Update FAISS search index once per epoch."""
        # if self.cached_candidate_k is None or self.cached_candidate_y is None:
        #     return
        # d_main = self.cached_candidate_k.shape[1]
        # # if self.search_index is None:
        # try:
        #     res = faiss.StandardGpuResources()
        #     cfg = faiss.GpuIndexFlatConfig()
        #     cfg.device = torch.cuda.current_device()

        #     # [수정] Cosine 유사도 계산을 위한 Inner Product 인덱스 사용, L1 등은 L2로 후보를 뽑는 방식 유지
        #     if self.metric == 'cosine':
        #         self.search_index = faiss.GpuIndexFlatIP(res, d_main, cfg)
        #     else:
        #         # L1, Mahalanobis, KL, Wasserstein 등은 1차적으로 L2로 후보를 뽑음 (Coarse Search)
        #         self.search_index = faiss.GpuIndexFlatL2(res, d_main, cfg)
        # except AttributeError:

        #         # GPU가 없는 경우 CPU 인덱스 사용
        #         if self.metric == 'cosine':
        #             self.search_index = faiss.IndexFlatIP(d_main)
        #         else:
        #             self.search_index = faiss.IndexFlatL2(d_main)
        
        # self.search_index.reset()
        # # self.search_index.add(self.cached_candidate_k.to(torch.float32).detach().cpu().numpy())
        # # [수정] 코사인일 경우 미리 정규화해서 저장
        # data = self.cached_candidate_k.detach().cpu().numpy().astype('float32')
        # if self.metric == 'cosine':
        #     faiss.normalize_L2(data)
            
        # self.search_index.add(data)

    def _compute_similarity(self, k, context_k):
        """다양한 지표를 계산하는 핵심 로직"""
        if self.metric == 'l2':
            return (-k.square().sum(-1, keepdim=True) 
                    + (2 * (k[..., None, :] @ context_k.transpose(-1, -2))).squeeze(-2) 
                    - context_k.square().sum(-1))
        
        elif self.metric == 'l1':
            # (Batch, 1, Dim) - (Batch, Context, Dim) -> (Batch, Context, Dim)
            return -torch.abs(k.unsqueeze(1) - context_k).sum(-1)

        elif self.metric == 'cosine':
            k_norm = F.normalize(k, p=2, dim=-1)
            context_k_norm = F.normalize(context_k, p=2, dim=-1)
            return (k_norm.unsqueeze(1) @ context_k_norm.transpose(-1, -2)).squeeze(-2)

        elif self.metric == 'mahalanobis':
            # W 행렬을 통과시켜 공간 왜곡 후 L2 계산
            k_w = k @ self.W
            context_k_w = context_k @ self.W
            return (-k_w.square().sum(-1, keepdim=True) 
                    + (2 * (k_w[..., None, :] @ context_k_w.transpose(-1, -2))).squeeze(-2) 
                    - context_k_w.square().sum(-1))

        elif self.metric == 'kl':
            # 각 벡터를 확률 분포로 해석 (Softmax)
            p = F.softmax(k.unsqueeze(1), dim=-1) # (B, 1, D)
            q = F.softmax(context_k, dim=-1)      # (B, C, D)
            # KL Divergence는 낮을수록 유사하므로 마이너스
            return -torch.sum(p * (torch.log(p + 1e-8) - torch.log(q + 1e-8)), dim=-1)

        elif self.metric == 'wasserstein':
            # Sinkhorn 알고리즘을 사용한 정밀 거리 계산
            batch_size, context_size, d_main = context_k.shape
            k_expanded = k.unsqueeze(1).expand(-1, context_size, -1).reshape(-1, d_main)
            ctx_expanded = context_k.reshape(-1, d_main)
            
            dists = sinkhorn_distance(k_expanded, ctx_expanded)
            return -dists.view(batch_size, context_size)

        return None


    def reset_parameters(self):
        if isinstance(self.label_encoder, nn.Linear):
            bound = 1 / math.sqrt(2.0)
            nn.init.uniform_(self.label_encoder.weight, -bound, bound)
            nn.init.uniform_(self.label_encoder.bias, -bound, bound)
        else:
            nn.init.uniform_(self.label_encoder[0].weight, -1.0, 1.0)

    # def _encode(self, x_num, x_cat):
    #     x = []
    #     if x_num is None:
    #         self.num_embeddings = None
    #     else:
    #         x.append(
    #             x_num if self.num_embeddings is None else self.num_embeddings(x_num).flatten(1)
    #         )
    #     if x_cat is not None:
    #         x.append(x_cat)
    #     x = torch.cat(x, dim=1)
    #     x = self.linear(x)
    #     for block in self.blocks0:
    #         x = x + block(x)
    #     k = self.K(x if self.normalization is None else self.normalization(x))
    #     return x, k
    def _encode(self, x_num, x_cat):
        x = []
        if x_num is not None:
            # 1. 개별 feature embedding (Soft-binning etc.)
            # tokens shape : (Batch, n_features, d_embedding)
            tokens = self.num_embeddings(x_num)

            # 2. [추가] Feature Interaction 레이어 적용 (피처 간의 관계 학습)
            if self.user_interaction and hasattr(self, 'feature_interaction_layer'):
                tokens = self.feature_interaction_layer(tokens)
            
            # 3. Retriever 입력을 위한 flatten 작업
            x.append(tokens.flatten(1))
        if x_cat is not None:
            x.append(x_cat)
        
        x = torch.cat(x, dim=1)
        x = self.linear(x)
        for block in self.blocks0:
            x = x + block(x)
        k = self.K(x if self.normalization is None else self.normalization(x))
        return x, k

    # def forward(
    #     self,
    #     *,
    #     x_num: Tensor,
    #     x_cat: ty.Optional[Tensor],
    #     y: Optional[Tensor],
    #     candidate_x_num: ty.Optional[Tensor],
    #     candidate_x_cat: ty.Optional[Tensor],
    #     candidate_y: Tensor,
    #     context_size: int,
    #     is_train: bool,
    # ) -> Tensor:

    #     device = x_num.device if x_num is not None else x_cat.device

    #     if self.cached_candidate_k is None:
    #         with torch.no_grad():
    #             self.cached_candidate_k = (
    #                 self._encode(candidate_x_num, candidate_x_cat)[1]
    #                 if self.candidate_encoding_batch_size is None
    #                 else torch.cat([
    #                     self._encode(xn, xc)[1]
    #                     for xn, xc in delu.iter_batches(
    #                         (candidate_x_num, candidate_x_cat), self.candidate_encoding_batch_size
    #                     )
    #                 ])
    #             )
    #             self.cached_candidate_y = candidate_y

    #     x, k = self._encode(x_num, x_cat)
    #     if is_train:
    #         assert y is not None
    #         candidate_k = torch.cat([k, self.cached_candidate_k])
    #         candidate_y = torch.cat([y, self.cached_candidate_y])
    #     else:
    #         candidate_k = self.cached_candidate_k
    #         candidate_y = self.cached_candidate_y

    #     batch_size, d_main = k.shape
    
    # ### ---- 수정 ---- ###
    #     # 1단계 : FAISS 검색(Coarse Search)
    #     with torch.no_grad():
    #         search_k = k.to(torch.float32).detach().cpu().numpy()
    #         if self.metric == 'cosine':
    #             faiss.normalize_L2(search_k)
            
    #         # 2단계 재정렬을 위해 context_size보다 더 많은 후보를 검색 (예: context_size * 2)
    #         fetch_size = context_size * 2
    #         distances, context_idx = self.search_index.search(search_k, fetch_size + 1 if is_train else fetch_size)  # +1은 자기 자신을 제외하기 위함
    #             # k.to(torch.float32).detach().cpu().numpy(), context_size + (1 if is_train else 0)            )
    #         distances = torch.tensor(distances, device=device)
    #         context_idx = torch.tensor(context_idx, device=device)
    #         if is_train:
    #             distances[context_idx == torch.arange(batch_size, device=device)[:, None]] = torch.inf
    #             context_idx = context_idx.gather(-1, distances.argsort()[:, :-1])

    #     context_k = candidate_k[context_idx]

    #     # [수정] 위에서 정의한 다양한 지표로 유사도 계산
    #     similarities = self._compute_similarity(k, context_k)
    #     # similarities = (
    #     #     -k.square().sum(-1, keepdim=True)
    #     #     + (2 * (k[..., None, :] @ context_k.transpose(-1, -2))).squeeze(-2)
    #     #     - context_k.square().sum(-1)
    #     # )
        
    #     # 1. 2단계 재정렬: 192개 중 실제 사용할 96개(context_size)의 상위 이웃 선별
    #     top_sim, top_indices = similarities.topk(context_size, dim=-1)

    #     # 2. 확률 계산: 반드시 잘라낸 'top_sim'을 사용하여 softmax 계산 (차원: 96)
    #     probs = F.softmax(top_sim, dim=-1)
    #     probs = self.dropout(probs)

    #     # [수정] 데이터 슬라이싱 : context_idx, context_k, context_y_emb 등을 top_indices에 맞춰 재정렬하여 상위 context_size 후보들로만 구성되도록 함
    #     # 3. 데이터 필터링 : 192개 후보 인덱스 중 실제 사용할 96개(context_size)의 인덱스만 선별하여 context_idx, context_k, context_y_emb 등을 재구성
    #     # context_idx : [Batch, 192] -> [Batch, 96]
    #     context_idx = context_idx.gather(-1, top_indices)

    #     # 4. 필터링된 인덱스로 이웃의 k와 y 정보를 다시 가져옴.
    #     context_k = candidate_k[context_idx] # [Batch, 96, d_main]
    #     context_y = candidate_y[context_idx] # [Batch, 96]

    #     # 5. Label Embedding 계산
    #     if self.n_classes > 1:
    #         context_y_emb = self.label_encoder(context_y[..., None].long())
    #     else:
    #         context_y_emb = self.label_encoder(context_y[..., None])
    #         if len(context_y_emb.shape) == 4:
    #             context_y_emb = context_y_emb[:, :, 0, :]

    #     # if self.n_classes > 1:
    #     #     context_y_emb = self.label_encoder(candidate_y[context_idx][..., None].long())
    #     # else:
    #     #     context_y_emb = self.label_encoder(candidate_y[context_idx][..., None])
    #     #     if len(context_y_emb.shape) == 4:
    #     #         context_y_emb = context_y_emb[:, :, 0, :]

    #     # 6. Values 계산 (모든 tensor가 96차원으로 통일)
    #     values = context_y_emb + self.T(k[:, None] - context_k)

    #     # 7. 최종 가중합 : [Batch, 1, 96] @ [Batch, 96, d_main] -> [Batch, d_main]
    #     context_x = (probs[:, None] @ values).squeeze(1)
    #     x = x + context_x

    #     for block in self.blocks1:
    #         x = x + block(x)
    #     x = self.head(x)
    #     return x
    def forward(
        self,
        *,
        x_num: Tensor,
        x_cat: ty.Optional[Tensor],
        y: Optional[Tensor],
        candidate_x_num: ty.Optional[Tensor],
        candidate_x_cat: ty.Optional[Tensor],
        candidate_y: Tensor,
        context_size: int,
        is_train: bool,
    ) -> Union[Tensor, ty.Tuple[Tensor, Tensor, Tensor, Tensor]]:

        device = x_num.device if x_num is not None else x_cat.device

        # 1. Encoding: Soft-Binning 및 Feature Interaction 포함
        # x: MLP 입력용, k: 리트리버 검색용 임베딩
        x, k = self._encode(x_num, x_cat)

        # 2. Candidate Pool 설정
        # 훈련 시에는 현재 배치를 후보군에 포함시켜 Contrastive 효과를 극대화합니다.
        if is_train:
            assert y is not None
            candidate_k = torch.cat([k, self.cached_candidate_k])
            candidate_y = torch.cat([y, self.cached_candidate_y])
        else:
            candidate_k = self.cached_candidate_k
            candidate_y = self.cached_candidate_y

        batch_size, d_main = k.shape

        # 3. Neural Retrieval (Differentiable)
        # FAISS 대신 NeuralRetriever 모듈을 사용하여 유사도 계산
        # similarities: (Batch, N_candidates)
        probs, similarities = self.retriever(k, candidate_k)

        # 4. Self-Masking (훈련 시 자기 자신을 제외)
        if is_train:
            # candidate_k의 앞부분이 현재 배치(k)이므로 대각 성분을 -inf로 마스킹
            mask = torch.eye(batch_size, device=device)
            if candidate_k.size(0) > batch_size:
                # 패딩 처리 (배치 사이즈보다 후보군이 클 경우)
                padding = torch.zeros(batch_size, candidate_k.size(0) - batch_size, device=device)
                mask = torch.cat([mask, padding], dim=1)
            
            similarities = similarities.masked_fill(mask.bool(), -1e9)
            # 마스킹된 유사도로 확률 다시 계산
            probs = F.softmax(similarities / self.retriever.temperature, dim=-1)

        # 5. Top-K Selection & Re-normalization
        # 미분 흐름을 유지하면서 연산 효율성을 위해 상위 정예 이웃만 추출
        top_probs, top_indices = probs.topk(context_size, dim=-1)
        
        # 선택된 이웃들의 정보 슬라이싱
        context_k = candidate_k[top_indices]    # (Batch, 96, d_main)
        context_y = candidate_y[top_indices]    # (Batch, 96)
        
        # 선택된 96개 이웃에 대해서만 확률 합이 1이 되도록 재정규화 (Soft-selection)
        top_probs = top_probs / (top_probs.sum(dim=-1, keepdim=True) + 1e-8)
        top_probs = self.dropout(top_probs)

        # 6. Label Embedding & Value Computation
        if self.n_classes > 1:
            context_y_emb = self.label_encoder(context_y[..., None].long())
        else:
            context_y_emb = self.label_encoder(context_y[..., None])
            if len(context_y_emb.shape) == 4:
                context_y_emb = context_y_emb[:, :, 0, :]

        # 이웃의 라벨 정보와 Query-Neighbor 간의 잔차(Residual) 정보를 결합
        values = context_y_emb + self.T(k[:, None] - context_k)

        # 7. 최종 가중합 (Weighted Sum)
        # (Batch, 1, 96) @ (Batch, 96, d_main) -> (Batch, d_main)
        context_x = (top_probs[:, None] @ values).squeeze(1)
        x = x + context_x

        # 8. Predictor MLP & Head
        for block in self.blocks1:
            x = x + block(x)
        logits = self.head(x)

        # 훈련 시에는 Contrastive Loss 계산을 위해 중간 텐서들을 반환
        if is_train:
            return logits, k, context_k, context_y
        return logits


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

    def fit(self, X_train, y_train, X_val, y_val):
        self.N = X_train[:, self.num_cols]
        self.C = X_train[:, self.cat_features]
        if self.tasktype == "multiclass":
            self.y = torch.argmax(y_train, dim=1)
        else:
            self.y = y_train

        self.batch_size = get_batch_size(len(X_train))

        if self.tasktype == "regression":
            self.criterion = F.mse_loss 
        elif self.tasktype == "multiclass":
            self.criterion = F.cross_entropy
        else:
            self.criterion = F.binary_cross_entropy_with_logits
            
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(), 
            lr=self.params['lr'], 
            weight_decay=self.params['weight_decay']
        )
        
        if self.params["lr_scheduler"] & (len(X_train) > self.batch_size):
            self.scheduler = CosineAnnealingLR_Warmup(self.optimizer, warmup_epochs=10, T_max=100, iter_per_epoch=len(X_train)//self.batch_size, 
                                                      base_lr=self.params['lr'], warmup_lr=1e-6, eta_min=0, last_epoch=-1)
            
        self.train_size = self.N.shape[0] if self.N is not None else self.C.shape[0]
        self.train_indices = torch.arange(min([self.train_size, 50000]), device=self.device)

        pbar = tqdm(range(1, self.max_epoch+1))
        for epoch in pbar:
            pbar.set_description("EPOCH: %i" %epoch)
            tic = time.time()
            loss = self.train_epoch(epoch)
            self.validate(epoch, X_val, y_val)
            elapsed = time.time() - tic
            pbar.set_postfix_str(f'Time cost: {elapsed}, Tr loss: {loss:.5f}')
            if not self.continue_training:
                break

    # def train_epoch(self, epoch):
    #     self.model.train()
    #     tl = Averager()

    #     if self.model.cached_candidate_k is None:
    #         candidate_x_num = self.N[:50000].float().to(self.device) if self.N is not None else None
    #         candidate_x_cat = self.C[:50000].float().to(self.device) if self.C is not None else None
    #         candidate_y = self.y[:50000].float().to(self.device) if self.is_regression else self.y[:50000].to(self.device)
    #         with torch.no_grad():
    #             self.model.cached_candidate_k = self.model._encode(candidate_x_num, candidate_x_cat)[1]
    #             self.model.cached_candidate_y = candidate_y

    #     self.model.update_index()

    #     for batch_idx in make_random_batches(self.train_size, self.batch_size, self.device):
    #         self.train_step = self.train_step + 1
            
    #         X_num = self.N[batch_idx] if self.N is not None else None
    #         X_cat = self.C[batch_idx] if self.C is not None else None
    #         y = self.y[batch_idx]

    #         candidate_indices = self.train_indices
    #         candidate_indices = candidate_indices[~torch.isin(candidate_indices, batch_idx)]

    #         candidate_x_num = self.N[candidate_indices] if self.N is not None else None
    #         candidate_x_cat = self.C[candidate_indices] if self.C is not None else None
    #         candidate_y = self.y[candidate_indices]
    #         X_num = X_num.float() if X_num is not None else None
    #         X_cat = X_cat.float()   if X_cat is not None else None
    #         candidate_x_num = candidate_x_num.float() if candidate_x_num is not None else None
    #         candidate_x_cat = candidate_x_cat.float() if candidate_x_cat is not None else None
    #         if self.is_regression:
    #             candidate_y = candidate_y.float()
    #             y = y.float()
    #         if X_cat is None and X_num is not None:
    #             x, candidate_x = X_num, candidate_x_num
    #         elif X_cat is not None and X_num is None:
    #             x, candidate_x = X_cat, candidate_x_cat
    #         else:
    #             x, candidate_x = torch.cat([X_num, X_cat], dim=1),torch.cat([candidate_x_num, candidate_x_cat], dim=1)

    #         if x.size(0) > 1:
    #             pred = self.model(
    #                 x_num=x[:,:self.n_num_features], x_cat=x[:,self.n_num_features:], y=y, 
    #                 candidate_x_num=candidate_x_num,
    #                 candidate_x_cat=candidate_x_cat,
    #                 candidate_y=candidate_y,
    #                 context_size=self.context_size,
    #                 is_train=True,
    #             ).squeeze(-1)

    #             loss = self.criterion(pred, y)
    #             tl.add(loss.item())
    #             self.optimizer.zero_grad()
    #             loss.backward()
    #             self.optimizer.step()
    #             if self.params["lr_scheduler"] & (len(self.y) > self.batch_size):
    #                 self.scheduler.step()

    #     tl = tl.item()
    #     self.trlog['train_loss'].append(tl)

    #     return tl
    def train_epoch(self, epoch):
        self.model.train()
        tl = Averager()

        # 1. 후보군(Memory Bank) 최신화
        # FAISS를 사용하지 않으므로 인덱스 업데이트 과정은 생략되거나 
        # 임베딩 공간의 최신 상태를 반영하기 위한 최소한의 연산만 수행합니다.
        if self.model.cached_candidate_k is None:
            candidate_x_num = self.N[:50000].float().to(self.device) if self.N is not None else None
            candidate_x_cat = self.C[:50000].float().to(self.device) if self.C is not None else None
            candidate_y = self.y[:50000].float().to(self.device) if self.is_regression else self.y[:50000].to(self.device)
            with torch.no_grad():
                # 초기 1회 후보군 임베딩 생성
                self.model.cached_candidate_k = self.model._encode(candidate_x_num, candidate_x_cat)[1]
                self.model.cached_candidate_y = candidate_y

        # 미분 가능한 구조에서는 매 에폭 FAISS 인덱스를 빌드할 필요가 없으므로 pass 처리된 함수 호출
        self.model.update_index()

        # 2. 배치 학습 루프
        for batch_idx in make_random_batches(self.train_size, self.batch_size, self.device):
            self.train_step = self.train_step + 1
            
            # 현재 배치 데이터 준비
            X_num = self.N[batch_idx].float() if self.N is not None else None
            X_cat = self.C[batch_idx].float() if self.C is not None else None
            y = self.y[batch_idx]

            # 리트리버 후보군에서 현재 배치 제외 (Data Leakage 방지)
            candidate_indices = self.train_indices
            candidate_indices = candidate_indices[~torch.isin(candidate_indices, batch_idx)]

            candidate_x_num = self.N[candidate_indices].float() if self.N is not None else None
            candidate_x_cat = self.C[candidate_indices].float() if self.C is not None else None
            candidate_y_pool = self.y[candidate_indices]
            
            if self.is_regression:
                candidate_y_pool = candidate_y_pool.float()
                y = y.float()

            # 3. 모델 순전파 (Forward Pass)
            if X_num.size(0) > 1:
                # [중요] 새로운 forward 구조에 맞춰 4개의 결과값을 Unpacking 합니다.
                # pred: 예측 로그잇, k: 앵커 임베딩, context_k: 선택된 이웃 임베딩, context_y_neighbors: 이웃 라벨
                pred, k, context_k, context_y_neighbors = self.model(
                    x_num=X_num, 
                    x_cat=X_cat, 
                    y=y, 
                    candidate_x_num=candidate_x_num,
                    candidate_x_cat=candidate_x_cat,
                    candidate_y=candidate_y_pool,
                    context_size=self.context_size,
                    is_train=True,
                )
                pred = pred.squeeze(-1)

                # 4. 복합 손실 함수 계산 (Multi-objective Loss)
                # 4.1. 기본 분류/회귀 손실 (Cross-Entropy 등)
                ce_loss = self.criterion(pred, y)

                # 4.2. Wasserstein-Contrastive Loss (공간 정렬 손실)
                # 하이퍼파라미터 lambda_w와 margin을 적용합니다.
                lambda_w = self.params.get('lambda_w', 0.1)
                margin = self.params.get('margin', 1.0)
                
                w_loss = compute_wasserstein_contrastive_loss(
                    k, context_k, y, context_y_neighbors, margin=margin
                )

                # 4.3. 최종 통합 손실
                loss = ce_loss + lambda_w * w_loss

                # 5. 역전파 및 최적화 (Backpropagation)
                # 이제 Retriever의 파라미터(anchors, temperature)도 이 과정에서 함께 학습됩니다.
                tl.add(loss.item())
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                if self.params["lr_scheduler"] and (len(self.y) > self.batch_size):
                    self.scheduler.step()

        # 에폭 손실 기록
        tl_result = tl.item()
        self.trlog['train_loss'].append(tl_result)

        return tl_result

    def validate(self, epoch, X_val, y_val):
        if self.tasktype == "multiclass":
            y_val = torch.argmax(y_val, dim=1)
        else:
            y_val = y_val
            
        self.model.eval()
        val_loss = 0.0
        with torch.no_grad():
            candidate_x_num = self.N[:50000] if self.N is not None else None
            candidate_x_cat = self.C[:50000] if self.C is not None else None
            candidate_y = self.y[:50000]
            candidate_x_num = candidate_x_num.float() if candidate_x_num is not None else None
            candidate_x_cat = candidate_x_cat.float() if candidate_x_cat is not None else None
            if self.is_regression:
                candidate_y = candidate_y.float()

            logits = []
            iters = X_val.shape[0] // 10000 + 1
            for i in range(iters):
                N = X_val[10000*i:10000*(i+1), self.num_cols]
                C = X_val[10000*i:10000*(i+1), self.cat_features]
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
            val_loss = self.criterion(logits, y_val).item()
        
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