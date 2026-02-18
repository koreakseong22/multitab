import optuna
import torch
import numpy as np
import os
import json
from libs.data import TabularDataset
from libs.model import getmodel, add_default_params
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
import torch.nn.functional as F

# 1. 데이터 로드 환경 설정
OPENML_ID = "43986"
SEED = 1
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

with open('dataset_id.json', 'r') as file:
    data_info = json.load(file)
tasktype = data_info[OPENML_ID]["tasktype"]

def objective(trial):
    # --- 2. 탐색할 하이퍼파라미터 범위 정의 ---
    lr = trial.suggest_float("lr", 1e-4, 5e-3, log=True)
    weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-3, log=True)
    batch_size = trial.suggest_categorical("batch_size", [32, 64, 128])
    context_size = trial.suggest_int("context_size", 24, 192, step=24)
    
    # 모델 내부 파라미터
    d_main = trial.suggest_int("d_main", 64, 256, step=32)
    dropout0 = trial.suggest_float("dropout0", 0.1, 0.5)
    context_dropout = trial.suggest_float("context_dropout", 0.1, 0.5)
    
    # --- 3. 모델 세팅 ---
    dataset = TabularDataset(int(OPENML_ID), tasktype, device=DEVICE, seed=SEED)
    (X_train, y_train), (X_val, y_val), _ = dataset._indv_dataset()
    
    # 라벨 인코딩 (0~N-1)
    le = LabelEncoder()
    y_train_raw = y_train.cpu().numpy().argmax(axis=1) if y_train.ndim > 1 else y_train.cpu().numpy()
    y_val_raw = y_val.cpu().numpy().argmax(axis=1) if y_val.ndim > 1 else y_val.cpu().numpy()
    
    y_train_enc = torch.tensor(le.fit_transform(y_train_raw), device=DEVICE)
    y_val_enc = torch.tensor(le.transform(y_val_raw), device=DEVICE)
    
    # TabR은 원-핫 형태를 내부에서 argmax하므로 다시 원-핫으로 전달
    num_classes = len(le.classes_)
    y_train_final = F.one_hot(y_train_enc, num_classes=num_classes).float()
    y_val_final = F.one_hot(y_val_enc, num_classes=num_classes).float()

    params = {
        "lr": lr,
        "weight_decay": weight_decay,
        "lr_scheduler": True,
        "early_stopping_rounds": 15,
        "model": {
            "d_main": d_main,
            "d_multiplier": 2.0,
            "encoder_n_blocks": 0,
            "predictor_n_blocks": 1,
            "mixer_normalization": "auto",
            "context_dropout": context_dropout,
            "dropout0": dropout0,
            "dropout1": "dropout0",
            "normalization": "LayerNorm",
            "activation": "ReLU",
            "num_embeddings": None
        }
    }

    try:
        model = getmodel("tabr", params, tasktype, dataset, OPENML_ID, X_train.shape[1], num_classes, DEVICE)
        model.context_size = context_size # 하드코딩 덮어쓰기
        model.batch_size = batch_size

        # 학습
        model.fit(X_train, y_train_final, X_val, y_val_final)
        
        # 검증 성능 평가
        y_val_pred = model.predict(X_val)
        acc = accuracy_score(y_val_enc.cpu().numpy(), y_val_pred)
        
        # 메모리 정리
        del model
        torch.cuda.empty_cache()
        
        return acc # 목적: Accuracy 최대화

    except Exception as e:
        print(f"Trial failed: {e}")
        return 0.0

# --- 4. Optuna 실행 ---
study = optuna.create_study(direction="maximize") # 정확도를 높이는 방향
print("🚀 TabR 하이퍼파라미터 최적화 시작 (30회 시도)...")
study.optimize(objective, n_trials=30)

print("\n🏆 최적의 결과:")
print(f"  Best Accuracy: {study.best_value:.4f}")
print(f"  Best Params: {json.dumps(study.best_params, indent=2)}")

# 최적 파라미터 저장
with open("best_tabr_params.json", "w") as f:
    json.dump(study.best_params, f, indent=2)