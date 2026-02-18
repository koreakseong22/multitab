import argparse
import os
import torch
import numpy as np
import json
from libs.data import TabularDataset
from libs.model import getmodel, add_default_params
from sklearn.preprocessing import LabelEncoder
import torch.nn.functional as F

# 1. 설정
parser = argparse.ArgumentParser()
parser.add_argument('--gpu_id', type=int, default=0)
parser.add_argument('--openml_id', type=str, default="43986")
parser.add_argument('--seed', type=int, default=1)
parser.add_argument('--model', type=str, default="tabr")
parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--context_size', type=int, default=96)
args = parser.parse_args()

device = f'cuda:{args.gpu_id}' if torch.cuda.is_available() else 'cpu'
with open('dataset_id.json', 'r') as file:
    data_info = json.load(file)
tasktype = data_info[args.openml_id]["tasktype"]

# 2. 데이터 로드 및 피처 선택 (Pruning)
dataset = TabularDataset(int(args.openml_id), tasktype, device=device, seed=args.seed)
(X_train, y_train), (X_val, y_val), (X_test, y_test) = dataset._indv_dataset()

# [중요] 피처를 5개로 줄이고, 인덱스 정보 강제 업데이트
top_5_indices = [10, 1, 9, 3, 7] # alcohol, volatile_acidity, sulphates, residual_sugar, density
X_train = X_train[:, top_5_indices]
X_val = X_val[:, top_5_indices]
X_test = X_test[:, top_5_indices]

# 모델이 내부에서 사용할 인덱스를 [0, 1, 2, 3, 4]로 재세팅
dataset.X_num = list(range(len(top_5_indices)))
dataset.X_cat = []
print(f"✂️ Feature Pruning: {len(top_5_indices)}개 피처로 인덱스 재구성 완료")

# 3. 라벨 인코딩 (생략 - 기존 로직 유지)
def get_label_encoded_y(y):
    y_np = y.cpu().numpy()
    if y_np.ndim > 1: y_np = y_np.argmax(axis=1)
    return y_np

le = LabelEncoder()
y_train_le = le.fit_transform(get_label_encoded_y(y_train))
y_val_le = le.transform(get_label_encoded_y(y_val))

num_classes = len(le.classes_)
y_train_final = F.one_hot(torch.tensor(y_train_le), num_classes=num_classes).float().to(device)
y_val_final = F.one_hot(torch.tensor(y_val_le), num_classes=num_classes).float().to(device)

# 4. 모델 생성 및 파이라미터 주입
input_dim = X_train.shape[1]
output_dim = num_classes

params = {}
params = add_default_params(args.model, params, args.openml_id)
params.update({"batch_size": args.batch_size, "early_stopping_rounds": 20, "lr": 0.001, "weight_decay": 1e-6, "lr_scheduler": True})
params["model"] = {"d_main": 96, "d_multiplier": 2.0, "encoder_n_blocks": 0, "predictor_n_blocks": 1, 
                   "mixer_normalization": "auto", "context_dropout": 0.2, "dropout0": 0.2, "dropout1": "dropout0", 
                   "normalization": "LayerNorm", "activation": "ReLU", "num_embeddings": None}

# dataset.X_num이 [0..4]이므로 에러가 나지 않음
model = getmodel(args.model, params, tasktype, dataset, args.openml_id, input_dim, output_dim, device)
model.context_size = args.context_size

# 5. 학습
try:
    print(f"🔄 TabR (Pruned) 학습 시작...")
    model.fit(X_train, y_train_final, X_val, y_val_final)
    prob = model.predict_proba(X_test)
    
    save_dir = f"results/reproduce_logs/seed={args.seed}/data={args.openml_id}"
    os.makedirs(save_dir, exist_ok=True)
    save_name = f"model=tabr_pruned..init_hps=True..deep=0..hyper=0.npy"
    np.save(os.path.join(save_dir, save_name), {"Probability": prob})
    print(f"✅ 저장 성공: {save_name}")
except Exception as e:
    import traceback
    traceback.print_exc()