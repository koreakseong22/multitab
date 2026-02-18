import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler
from sklearn.utils.class_weight import compute_class_weight
from libs.data import TabularDataset
from libs.model import TabRMethod  # 원본 클래스 로드
import libs.model  # 내부 F.cross_entropy 접근용
from sklearn.metrics import accuracy_score
from functools import partial
import json

# 1. 환경 설정
DATA_ID = "43986"
SEED = 1
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

dataset = TabularDataset(int(DATA_ID), "multiclass", device=DEVICE, seed=SEED)
(X_train, y_train), (X_val, y_val), (X_test, y_test) = dataset._indv_dataset()

# 2. [전처리] Robust Scaling (이상치 방어 및 거리 해상도 확보)
print("🛠️ Pipeline B: Robust Scaling 적용 중...")
scaler = RobustScaler()
X_train_np = X_train.cpu().numpy()
X_train_scaled = torch.tensor(scaler.fit_transform(X_train_np), device=DEVICE).float()
X_val_scaled = torch.tensor(scaler.transform(X_val.cpu().numpy()), device=DEVICE).float()
X_test_scaled = torch.tensor(scaler.transform(X_test.cpu().numpy()), device=DEVICE).float()

# 3. [전략] Class Weight 계산
#의 불균형을 해결하기 위해 소수 클래스에 더 큰 가중치 부여
y_train_label = y_train.argmax(dim=1).cpu().numpy()
classes = np.unique(y_train_label)
weights = compute_class_weight(class_weight='balanced', classes=classes, y=y_train_label)
weights_tensor = torch.tensor(weights, dtype=torch.float).to(DEVICE)

print(f"⚖️ 계산된 클래스 가중치: {weights}")

# 4. [Hacker's Tip] 원본 코드를 수정하지 않고 Loss에 가중치 주입하기
# libs.model 내의 F.cross_entropy를 가중치가 포함된 함수로 일시적으로 교체합니다.
original_ce = F.cross_entropy
F.cross_entropy = partial(F.cross_entropy, weight=weights_tensor)

# 5. TabR 학습
print("\n🚀 Pipeline B (Robust + Class Weights) 학습 시작...")
params = {
    "lr": 1e-3, "weight_decay": 1e-4, "lr_scheduler": True, "early_stopping_rounds": 15,
    "model": {
        "d_main": 128, "d_multiplier": 2.0, "encoder_n_blocks": 1, "predictor_n_blocks": 1,
        "mixer_normalization": 'auto', "context_dropout": 0.2, "dropout0": 0.1,
        "dropout1": 'dropout0', "normalization": "LayerNorm", "activation": "ReLU", "num_embeddings": None
    }
}

method = TabRMethod(
    params=params, tasktype="multiclass", num_cols=list(range(11)),
    cat_features=[], input_dim=11, output_dim=y_train.shape[1],
    device=DEVICE, data_id=DATA_ID
)

try:
    method.fit(X_train_scaled, y_train, X_val_scaled, y_val)
finally:
    # 학습 종료 후 다른 모델에 영향을 주지 않도록 원상복구
    F.cross_entropy = original_ce

# 6. 결과 도출
y_pred = method.predict(X_test_scaled)
final_acc = accuracy_score(y_test.argmax(dim=1).cpu().numpy(), y_pred)
print(f"\n🏆 Pipeline B 최종 결과 Accuracy: {final_acc:.6f}")