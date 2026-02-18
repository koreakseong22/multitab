import argparse
import os
import torch
import numpy as np
import json
from libs.data import TabularDataset
from libs.model import getmodel, add_default_params
from sklearn.preprocessing import LabelEncoder, StandardScaler, QuantileTransformer
import torch.nn.functional as F # 원-핫 변환용


# 1. 설정 (생략 - 기존과 동일)
parser = argparse.ArgumentParser()
parser.add_argument('--gpu_id', type=int, default=0)
parser.add_argument('--openml_id', type=str, default="43986")
parser.add_argument('--seed', type=int, default=1)
parser.add_argument('--model', type=str, required=True)
parser.add_argument('--savepath', type=str, default='results')
parser.add_argument('--batch_size', type=int, default=128)
parser.add_argument('--context_size', type=int, default=96) # [수정] 인자 추가
args = parser.parse_args()

device = f'cuda:{args.gpu_id}' if torch.cuda.is_available() else 'cpu'
with open('dataset_id.json', 'r') as file:
    data_info = json.load(file)
tasktype = data_info[args.openml_id]["tasktype"]

dataset = TabularDataset(int(args.openml_id), tasktype, device=device, seed=args.seed)
(X_train, y_train), (X_val, y_val), (X_test, y_test) = dataset._indv_dataset()

# TabR Feature Scaling
# scaler = StandardScaler()
scaler = QuantileTransformer(output_distribution='normal', n_quantiles=1000, random_state=args.seed)
X_train_scaled = torch.tensor(scaler.fit_transform(X_train.cpu().numpy()), device=device).float()
X_val_scaled = torch.tensor(scaler.transform(X_val.cpu().numpy()), device=device).float()
X_test_scaled = torch.tensor(scaler.transform(X_test.cpu().numpy()), device=device).float()

# [수정] 1. 라벨 인코딩 (3, 5, 8 -> 0, 1, 2)
def get_label_encoded_y(y):
    y_np = y.cpu().numpy()
    if y_np.ndim > 1: y_np = y_np.argmax(axis=1)
    return y_np

y_train_raw = get_label_encoded_y(y_train)
y_val_raw = get_label_encoded_y(y_val)

le = LabelEncoder()
y_train_le = le.fit_transform(y_train_raw)
y_val_le = le.transform(y_val_raw)

# [수정] 2. 다시 원-핫(2차원)으로 변환 (tabr.py 내부의 argmax 대응)
num_classes = len(le.classes_)
y_train_final = F.one_hot(torch.tensor(y_train_le), num_classes=num_classes).float().to(device)
y_val_final = F.one_hot(torch.tensor(y_val_le), num_classes=num_classes).float().to(device)

input_dim = X_train.shape[1]
output_dim = num_classes

# 3. 모델 생성 (기존과 동일)
print(f"🚀 실행 모델: {args.model} | 클래스 개수: {output_dim}")
params = {}
params = add_default_params(args.model, params, args.openml_id)
if args.model == "tabr":
    params.update({"batch_size": args.batch_size, "early_stopping_rounds": 20, "lr": 0.001, "weight_decay": 1e-6, "lr_scheduler": True})
    if "model" not in params:
        params["model"] = {"d_main": 96, "d_multiplier": 2.0, "encoder_n_blocks": 0, "predictor_n_blocks": 1, "mixer_normalization": "auto", "context_dropout": 0.2, "dropout0": 0.2, "dropout1": "dropout0", "normalization": "LayerNorm", "activation": "ReLU", "num_embeddings": None}

model = getmodel(args.model, params, tasktype, dataset, args.openml_id, input_dim, output_dim, device)

# 4. 학습 및 추론
result = {}
try:
    print(f"🔄 {args.model} 학습 시작...")
    # 2차원 원-핫 텐서를 넘겨줌으로써 Dimension out of range 에러 방지
    model.fit(X_train, y_train_final, X_val, y_val_final)
    
    print(f"🧪 {args.model} 추론 중...")
    result["Probability"] = model.predict_proba(X_test)
except Exception as e:
    import traceback
    traceback.print_exc()

# 5. 저장 (기존과 동일)
save_dir = os.path.join(args.savepath, 'reproduce_logs', f'seed={args.seed}', f'data={args.openml_id}')
os.makedirs(save_dir, exist_ok=True)
save_name = f"model={args.model}..init_hps=True..deep=0..hyper=0.npy"
np.save(os.path.join(save_dir, save_name), result)
print(f"✅ 저장 성공: {save_name}")