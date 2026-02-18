import torch
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, RobustScaler, QuantileTransformer
from sklearn.metrics import accuracy_score
from imblearn.over_sampling import SMOTE
from libs.data import TabularDataset
from libs.model import TabRMethod
import torch.nn.functional as F

# 1. 환경 설정
DATA_ID = "43986"
SEED = 1
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# RQ2에서 도출된 최적의 하이퍼파라미터 적용 (Regularized Retrieval 상태 유지)
BEST_PARAMS = {
    "lr": 0.004989, "weight_decay": 0.000416, "lr_scheduler": True, "early_stopping_rounds": 15,
    "model": {
        "d_main": 128, "d_multiplier": 2.0, "encoder_n_blocks": 1, "predictor_n_blocks": 1,
        "mixer_normalization": 'auto', "context_dropout": 0.3799, "dropout0": 0.4211,
        "dropout1": 'dropout0', "normalization": "LayerNorm", "activation": "ReLU", "num_embeddings": None
    }
}

def run_tabr(name, X_tr, y_tr, X_v, y_v, X_te, y_te):
    """TabR 학습 및 평가 엔진"""
    print(f"\n🚀 Running Scenario: [{name}]")
    method = TabRMethod(
        params=BEST_PARAMS, tasktype="multiclass", num_cols=list(range(11)),
        cat_features=[], input_dim=11, output_dim=y_tr.shape[1],
        device=DEVICE, data_id=DATA_ID
    )
    
    # 텐서 변환 및 학습
    X_tr_t = torch.tensor(X_tr, device=DEVICE).float()
    X_v_t = torch.tensor(X_v, device=DEVICE).float()
    X_te_t = torch.tensor(X_te, device=DEVICE).float()
    
    method.fit(X_tr_t, y_tr, X_v_t, y_v)
    preds = method.predict(X_te_t)
    acc = accuracy_score(y_te.argmax(dim=1).cpu().numpy(), preds)
    return acc

# 데이터 로드
dataset = TabularDataset(int(DATA_ID), "multiclass", device=DEVICE, seed=SEED)
(X_train, y_train), (X_val, y_val), (X_test, y_test) = dataset._indv_dataset()

X_tr_np, X_v_np, X_te_np = X_train.cpu().numpy(), X_val.cpu().numpy(), X_test.cpu().numpy()
results = {}

# --- [RQ3 시나리오 루프] ---

# 1. Pure Raw (기준점)
results['Pure Raw'] = run_tabr('Pure Raw', X_tr_np, y_train, X_v_np, y_val, X_te_np, y_test)

# 2. StandardScaler (전통적 스케일링)
ss = StandardScaler()
results['StandardScaler'] = run_tabr('StandardScaler', 
    ss.fit_transform(X_tr_np), y_train, ss.transform(X_v_np), y_val, ss.transform(X_te_np), y_test)

# 3. RobustScaler (기하학적 보존)
rs = RobustScaler()
results['RobustScaler'] = run_tabr('RobustScaler', 
    rs.fit_transform(X_tr_np), y_train, rs.transform(X_v_np), y_val, rs.transform(X_te_np), y_test)

# 4. QuantileTransformer (위상 붕괴 위험)
qt = QuantileTransformer(output_distribution='normal', random_state=SEED)
results['QuantileTransformer'] = run_tabr('QuantileTransformer', 
    qt.fit_transform(X_tr_np), y_train, qt.transform(X_v_np), y_val, qt.transform(X_te_np), y_test)

# 5. SMOTE (매니폴드 침범)
try:
    # 9등급 같은 극소수 클래스를 위해 이웃 수를 1~2로 제한합니다.
    sm = SMOTE(random_state=SEED, k_neighbors=1) 
    X_res, y_res = sm.fit_resample(X_tr_np, y_train.argmax(dim=1).cpu().numpy())
    
    # [이하 동일]
    y_res_t = F.one_hot(torch.tensor(y_res).long(), num_classes=y_train.shape[1]).float().to(DEVICE)
    results['SMOTE'] = run_tabr('SMOTE', X_res, y_res_t, X_v_np, y_val, X_te_np, y_test)
except Exception as e:
    print(f"❌ SMOTE 실행 실패: {e}")


# 최종 리포트 출력
print("\n" + "="*40)
print("🏆 RQ3: Preprocessing Paradox Results")
print("-"*40)
for name, acc in results.items():
    print(f"{name:20}: {acc:.4f}")
print("="*40)