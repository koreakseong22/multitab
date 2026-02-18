import torch
import numpy as np
import torch.nn.functional as F
from sklearn.metrics import log_loss, accuracy_score
from xgboost import XGBClassifier
from tabpfn import TabPFNClassifier
from libs.data import TabularDataset
from libs.model import TabRMethod

# 1. 환경 설정
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
DATA_ID = "43986"
SEED = 1

dataset = TabularDataset(int(DATA_ID), "multiclass", device=DEVICE, seed=SEED)
(X_train, y_train), (X_val, y_val), (X_test, y_test) = dataset._indv_dataset()

# Numpy 변환 (XGB, PFN용)
X_train_np, y_train_np = X_train.cpu().numpy(), y_train.argmax(1).cpu().numpy()
X_val_np, y_val_np = X_val.cpu().numpy(), y_val.argmax(1).cpu().numpy()
X_test_np, y_test_np = X_test.cpu().numpy(), y_test.argmax(1).cpu().numpy()

# ---------------------------------------------------------
# 2. 모델 학습 및 검증/테스트 확률 추출
# ---------------------------------------------------------
# (A) XGBoost
print("🌲 XGBoost 학습 중...")
xgb = XGBClassifier(random_state=SEED).fit(X_train_np, y_train_np)
xgb_val_probs = xgb.predict_proba(X_val_np)
xgb_test_probs = xgb.predict_proba(X_test_np)

# (B) TabPFN
print("🧠 TabPFN 실행 중...")
pfn = TabPFNClassifier(device=DEVICE).fit(X_train_np, y_train_np)
pfn_val_probs = pfn.predict_proba(X_val_np)
pfn_test_probs = pfn.predict_proba(X_test_np)

# (C) TabR (Optuna Optimized)
print("📚 TabR 학습 중...")
params = {
    "lr": 0.004989260796631495,            # Best Params에서 가져옴
    "weight_decay": 0.00041677445463749905, # Best Params에서 가져옴
    "lr_scheduler": True,
    "early_stopping_rounds": 15,
    "model": {
        "d_main": 128,                     # Best Params
        "d_multiplier": 2.0,               # 기본값 (고정)
        "encoder_n_blocks": 1,
        "predictor_n_blocks": 1,
        "mixer_normalization": 'auto',
        "context_dropout": 0.3799302892069304, # Best Params
        "dropout0": 0.42118083801897405,      # Best Params
        "dropout1": 0.42118083801897405,      # dropout0와 동일하게 설정 권장
        "normalization": "LayerNorm",
        "activation": "ReLU",
        "num_embeddings": None                # 필수 인자 (에러 방지)
    }
}
tabr = TabRMethod(params=params, tasktype="multiclass", num_cols=list(range(11)), cat_features=[], input_dim=11, output_dim=y_train.shape[1], device=DEVICE, data_id=DATA_ID)
tabr.fit(X_train, y_train, X_val, y_val)
tabr_val_logits = tabr.model(
    x_num=X_val, 
    x_cat=None,
    candidate_x_num=X_train,     # x_context 대신 candidate_x_num
    candidate_x_cat=None,         # candidate_x_cat도 None으로 명시
    candidate_y=y_train.argmax(1) # y_context 대신 candidate_y (보통 라벨 인덱스를 기대함)
)
tabr_test_logits = tabr.model(
    x_num=X_test, 
    x_cat=None, 
    candidate_x_num=X_train, 
    candidate_x_cat=None,
    candidate_y=y_train.argmax(1)
)

tabr_val_probs = F.softmax(tabr_val_logits, dim=1).detach().cpu().numpy()
tabr_test_probs = F.softmax(tabr_test_logits, dim=1).detach().cpu().numpy()
# ---------------------------------------------------------
# 3. Log-loss 기반 자동 가중치 산출 (Inverse Scaling)
# ---------------------------------------------------------
print("\n⚖️ 모델별 검증 데이터 Log-loss 측정 및 가중치 계산...")

losses = {
    "XGBoost": log_loss(y_val_np, xgb_val_probs),
    "TabPFN":  log_loss(y_val_np, pfn_val_probs),
    "TabR":    log_loss(y_val_np, tabr_val_probs)
}

# 역수 취하기: 가중치 w = (1/loss) / sum(1/loss)
# 
inv_losses = {k: 1.0 / v for k, v in losses.items()}
total_inv_loss = sum(inv_losses.values())
auto_weights = {k: v / total_inv_loss for k, v in inv_losses.items()}

for m, w in auto_weights.items():
    print(f" - {m}: 가중치 {w:.4f} (Loss: {losses[m]:.4f})")

# ---------------------------------------------------------
# 4. 최종 앙상블 (Weighted Soft Voting)
# ---------------------------------------------------------
final_test_probs = (
    xgb_test_probs * auto_weights["XGBoost"] +
    pfn_test_probs * auto_weights["TabPFN"] +
    tabr_test_probs * auto_weights["TabR"]
)
final_preds = np.argmax(final_test_probs, axis=1)

print("\n" + "="*35)
print(f"🏆 최종 자동 앙상블 Accuracy: {accuracy_score(y_test_np, final_preds):.4f}")
print("="*35)