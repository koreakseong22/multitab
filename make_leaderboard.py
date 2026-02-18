import torch
import numpy as np
import os, json
from libs.data import TabularDataset
from libs.model import getmodel
from sklearn.preprocessing import LabelEncoder
import torch.nn.functional as F

# 1. 설정 및 Optuna 결과 주입
data_id = "43986"
seed = 1
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# [Potato님 결과 주입]
best_params = {
  "lr": 0.00409683748402871,
  "weight_decay": 7.404676079744137e-06,
  "batch_size": 128,
  "context_size": 192,
  "d_main": 128,
  "dropout0": 0.39927951312965715,
  "context_dropout": 0.14097174256003486
}

with open('dataset_id.json', 'r') as file:
    data_info = json.load(file)
tasktype = data_info[data_id]["tasktype"]

dataset = TabularDataset(int(data_id), tasktype, device=device, seed=seed)
(X_train, y_train), (X_val, y_val), (X_test, y_test) = dataset._indv_dataset()

# 라벨 전처리
le = LabelEncoder()
y_train_raw = y_train.cpu().numpy().argmax(axis=1) if y_train.ndim > 1 else y_train.cpu().numpy()
y_train_final = F.one_hot(torch.tensor(le.fit_transform(y_train_raw)), num_classes=len(le.classes_)).float().to(device)
y_val_raw = y_val.cpu().numpy().argmax(axis=1) if y_val.ndim > 1 else y_val.cpu().numpy()
y_val_final = F.one_hot(torch.tensor(le.transform(y_val_raw)), num_classes=len(le.classes_)).float().to(device)

# 모델 생성 (Best Params 반영)
params = {
    "lr": best_params["lr"],
    "weight_decay": best_params["weight_decay"],
    "lr_scheduler": True,
    "early_stopping_rounds": 20,
    "model": {
        "d_main": best_params["d_main"],
        "d_multiplier": 2.0,
        "encoder_n_blocks": 0,
        "predictor_n_blocks": 1,
        "mixer_normalization": "auto",
        "context_dropout": best_params["context_dropout"],
        "dropout0": best_params["dropout0"],
        "dropout1": "dropout0",
        "normalization": "LayerNorm",
        "activation": "ReLU",
        "num_embeddings": None
    }
}

print(f"🚀 최적화된 TabR 최종 학습 시작 (Context Size: {best_params['context_size']})")
model = getmodel("tabr", params, tasktype, dataset, data_id, X_train.shape[1], len(le.classes_), device)
model.context_size = best_params["context_size"]
model.batch_size = best_params["batch_size"]

model.fit(X_train, y_train_final, X_val, y_val_final)
final_probs = model.predict_proba(X_test)

# 파일 저장 (리더보드용 경로)
save_dir = f"results/reproduce_logs/seed={seed}/data={data_id}"
os.makedirs(save_dir, exist_ok=True)
save_path = os.path.join(save_dir, "model=tabr..init_hps=True..deep=0..hyper=0.npy")
np.save(save_path, {"Probability": final_probs})
print(f"✅ 리더보드용 TabR 결과 업데이트 완료!")