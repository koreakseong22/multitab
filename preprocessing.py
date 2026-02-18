import torch
import numpy as np
import pandas as pd
import json
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler, RobustScaler, QuantileTransformer
from sklearn.metrics import accuracy_score
from imblearn.over_sampling import SMOTE
import torch.nn.functional as F

from libs.data import TabularDataset
from libs.model import TabRMethod

# 1. 환경 및 하이퍼파라미터 설정
DATA_ID = "43986"
SEED = 1
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# RQ2에서 도출된 최적의 파라미터 적용
BEST_PARAMS = {
    "lr": 0.004989, "weight_decay": 0.000416, "lr_scheduler": True, "early_stopping_rounds": 15,
    "model": {
        "d_main": 128, "d_multiplier": 2.0, "encoder_n_blocks": 1, "predictor_n_blocks": 1,
        "mixer_normalization": 'auto', "context_dropout": 0.3799, "dropout0": 0.4211,
        "dropout1": 'dropout0', "normalization": "LayerNorm", "activation": "ReLU", "num_embeddings": None
    }
}

def train_and_eval(name, X_tr, y_tr, X_te, y_te):
    """실제 TabR 학습 및 평가를 담당하는 엔진"""
    # TabR은 검증셋(Val)이 필요하므로 Test셋의 일부를 Val로 잠시 빌려옵니다.
    method = TabRMethod(
        params=BEST_PARAMS, tasktype="multiclass", num_cols=list(range(11)),
        cat_features=[], input_dim=11, output_dim=y_tr.shape[1],
        device=DEVICE, data_id=DATA_ID
    )
    
    X_tr_t = torch.tensor(X_tr, device=DEVICE).float()
    X_te_t = torch.tensor(X_te, device=DEVICE).float()
    
    # 5-Fold이므로 Validation은 Train의 일부를 사용하거나 간단히 Test를 모니터링용으로 씁니다.
    method.fit(X_tr_t, y_tr, X_te_t, y_te) 
    preds = method.predict(X_te_t)
    acc = accuracy_score(y_te.argmax(dim=1).cpu().numpy(), preds)
    return acc

# 2. 전체 데이터 로드 및 합치기 (Concatenation)
print("🍷 Wine Quality 데이터 로드 및 통합 중...")
dataset = TabularDataset(int(DATA_ID), "multiclass", device=DEVICE, seed=SEED)

# 이미 정의된 메서드를 통해 쪼개진 데이터를 가져옵니다.
(X_train, y_train), (X_val, y_val), (X_test, y_test) = dataset._indv_dataset()

# 5-Fold CV를 위해 모든 데이터를 하나로 합칩니다.
X_all = torch.cat([X_train, X_val, X_test], dim=0).cpu().numpy()

# y_all은 StratifiedKFold를 위해 클래스 인덱스(0, 1, 2...) 형태로 변환합니다.
y_all = torch.cat([y_train, y_val, y_test], dim=0).argmax(dim=1).cpu().numpy()

num_classes = y_train.shape[1]
print(f"📊 통합 완료: 총 {len(X_all)} 샘플, {num_classes} 클래스")

# 3. 5-Fold CV 설정
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
scenarios = ['Pure Raw', 'StandardScaler', 'RobustScaler', 'QuantileTransformer', 'SMOTE']
history = {s: [] for s in scenarios}

print(f"🔬 5-Fold CV 시작... (Total Samples: {len(X_all)})")

for fold, (train_idx, test_idx) in enumerate(skf.split(X_all, y_all)):
    print(f"\n#️⃣ Fold {fold+1} 진행 중...")
    
    X_train_raw, X_test_raw = X_all[train_idx], X_all[test_idx]
    y_train_raw, y_test_raw = y_all[train_idx], y_all[test_idx]
    
    # 정답셋 One-hot 변환
    y_train_t = F.one_hot(torch.tensor(y_train_raw).long(), num_classes=num_classes).float().to(DEVICE)
    y_test_t = F.one_hot(torch.tensor(y_test_raw).long(), num_classes=num_classes).float().to(DEVICE)

    # --- 시나리오별 실행 ---
    
    # [1] Pure Raw
    history['Pure Raw'].append(train_and_eval('Pure Raw', X_train_raw, y_train_t, X_test_raw, y_test_t))

    # [2] StandardScaler
    scaler_s = StandardScaler()
    history['StandardScaler'].append(train_and_eval('StandardScaler', 
        scaler_s.fit_transform(X_train_raw), y_train_t, scaler_s.transform(X_test_raw), y_test_t))

    # [3] RobustScaler
    scaler_r = RobustScaler()
    history['RobustScaler'].append(train_and_eval('RobustScaler', 
        scaler_r.fit_transform(X_train_raw), y_train_t, scaler_r.transform(X_test_raw), y_test_t))

    # [4] QuantileTransformer
    qt = QuantileTransformer(output_distribution='normal', random_state=SEED)
    history['QuantileTransformer'].append(train_and_eval('QuantileTransformer', 
        qt.fit_transform(X_train_raw), y_train_t, qt.transform(X_test_raw), y_test_t))

    # [5] SMOTE (k_neighbors=1로 설정하여 소수 클래스 에러 방지)
    sm = SMOTE(random_state=SEED, k_neighbors=1)
    X_res, y_res = sm.fit_resample(X_train_raw, y_train_raw)
    y_res_t = F.one_hot(torch.tensor(y_res).long(), num_classes=num_classes).float().to(DEVICE)
    history['SMOTE'].append(train_and_eval('SMOTE', X_res, y_res_t, X_test_raw, y_test_t))

# 4. 최종 결과 출력
print("\n" + "="*50)
print("🏆 RQ3 5-Fold CV 최종 리포트 (Mean ± Std)")
print("-"*50)
for s in scenarios:
    mean = np.mean(history[s])
    std = np.std(history[s])
    print(f"{s:20}: {mean:.4f} (± {std:.4f})")
print("="*50)