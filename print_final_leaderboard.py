import pandas as pd
import numpy as np
import os, json, glob
import torch
from libs.data import TabularDataset
from libs.eval import calculate_metric
from scipy.special import expit, softmax

# 1. 설정
basepath = 'results'
data_id = "43986"
seed = 1
models = ["lr", "randomforest", "xgboost", "catboost", "lightgbm", "mlp", "embedmlp", "mlpplr", "resnet", "ftt", "t2gformer", "saint", "tabr", "tabpfn"]
device = 'cuda' if torch.cuda.is_available() else 'cpu'

with open('dataset_id.json', 'r') as file:
    data_info = json.load(file)
tasktype = data_info[data_id]["tasktype"]

# 2. 데이터셋 로드
dataset = TabularDataset(int(data_id), tasktype, device=device, seed=seed)
_, _, (_, y_test) = dataset._indv_dataset()
y_std = dataset.y_std

summary_list = []

print(f"📊 {data_id} 최종 리더보드 산출 중...")

for m in models:
    # 저장된 모든 .npy 파일을 찾음
    pattern = os.path.join(basepath, 'reproduce_logs', f'seed={seed}', f'data={data_id}', f'*model={m}*deep=0*hyper=0*.npy')
    file_matches = glob.glob(pattern)
    
    if file_matches:
        fname = file_matches[0]
        try:
            f_raw = np.load(fname, allow_pickle=True)
            f = f_raw.item() if f_raw.dtype == 'O' else f_raw
            
            key = "Prediction" if tasktype == "regression" else "Probability"
            preds = f[key]
            
            # 메트릭 계산 로직
            if tasktype != "regression":
                if tasktype == "binclass":
                    p_proc = expit(preds) if (preds.min() < 0 or preds.max() > 1) else preds
                    c_proc = np.round(p_proc)
                else:
                    if preds.ndim == 1: 
                        c_proc = preds
                        p_proc = np.eye(len(np.unique(y_test)))[preds.astype(int)]
                    else:
                        p_proc = softmax(preds, axis=1) if (preds.min() < 0 or preds.max() > 1) else preds
                        c_proc = np.argmax(p_proc, axis=1)
                
                perf = calculate_metric(y_test, c_proc, p_proc, tasktype, "test", prob=True)
                score = perf.get("acc_test")
            else:
                perf = calculate_metric(y_test * y_std, preds * y_std, preds * y_std, tasktype, "test", prob=True)
                score = perf.get("rmse_test")
            
            summary_list.append({
                "Model": m, 
                "Score": score, 
                "AUROC": perf.get("auroc_test"),
                "Type": "Tabular-DL" if m in ["tabr", "tabpfn", "ftt", "saint", "t2gformer"] else ("GBDT" if m in ["xgboost", "catboost", "lightgbm", "randomforest"] else "General/Baseline")
            })
            
        except Exception as e:
            continue

# 3. 리더보드 출력
if summary_list:
    df = pd.DataFrame(summary_list)
    # 점수 기준 내림차순 정렬
    df = df.sort_values(by="Score", ascending=False).reset_index(drop=True)

    print("\n" + "="*60)
    print("🏆 FINAL RESEARCH LEADERBOARD (Wine Quality)")
    print("="*60)
    print(df[['Type', 'Model', 'Score', 'AUROC']].to_string(index=False))
    print("="*60)
else:
    print("❌ 결과 파일이 하나도 없습니다. 경로를 확인하세요.")