import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from xgboost import XGBClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import accuracy_score
from sklearn.base import BaseEstimator, ClassifierMixin
from libs.data import TabularDataset
from libs.model import getmodel
import json

# --- 1. Sklearn 호환을 위한 Wrapper 클래스 ---
class TabRSklearnWrapper(BaseEstimator, ClassifierMixin):
    def __init__(self, model_wrapper, device):
        self.model_wrapper = model_wrapper
        self.device = device
        self.classes_ = None # 사이킷런 규격

    def fit(self, X, y):
        # 이미 학습된 모델을 사용하므로 아무것도 하지 않음
        self.classes_ = np.unique(y)
        return self

    def predict(self, X):
        # 텐서 변환 및 예측
        X_tensor = torch.tensor(X).float().to(self.device)
        return self.model_wrapper.predict(X_tensor)

    def score(self, X, y):
        # 점수 계산
        y_pred = self.predict(X)
        return accuracy_score(y, y_pred)
    
# --- 2. 데이터 로드 및 학습 로직 (기존과 동일) ---
OPENML_ID = "43986"
SEED = 1
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

with open('dataset_id.json', 'r') as file:
    data_info = json.load(file)
tasktype = data_info[OPENML_ID]["tasktype"]

dataset = TabularDataset(int(OPENML_ID), tasktype, device=DEVICE, seed=SEED)
(X_train, y_train), (_, _), (X_test, y_test) = dataset._indv_dataset()

feature_names = ["fixed_acidity", "volatile_acidity", "citric_acid", "residual_sugar", 
                 "chlorides", "free_sulfur_dioxide", "total_sulfur_dioxide", "density", 
                 "pH", "sulphates", "alcohol"]

y_train_label = y_train.cpu().numpy().argmax(axis=1) if y_train.ndim > 1 else y_train.cpu().numpy()
y_test_label = y_test.cpu().numpy().argmax(axis=1) if y_test.ndim > 1 else y_test.cpu().numpy()

# XGBoost 학습
print("🌲 XGBoost 학습 중...")
xgb = XGBClassifier(random_state=SEED)
xgb.fit(X_train.cpu().numpy(), y_train_label)
xgb_importance = xgb.feature_importances_

# TabR 학습
print("🚀 TabR 학습 중...")
params = {"batch_size": 32, "early_stopping_rounds": 20, "lr": 0.001, "weight_decay": 1e-6, "lr_scheduler": True}
params["model"] = {"d_main": 96, "d_multiplier": 2.0, "encoder_n_blocks": 0, "predictor_n_blocks": 1, 
                   "mixer_normalization": "auto", "context_dropout": 0.2, "dropout0": 0.2, "dropout1": "dropout0", 
                   "normalization": "LayerNorm", "activation": "ReLU", "num_embeddings": None}

model_raw = getmodel("tabr", params, tasktype, dataset, OPENML_ID, X_train.shape[1], len(np.unique(y_train_label)), DEVICE)
y_train_final = torch.nn.functional.one_hot(torch.tensor(y_train_label), num_classes=len(np.unique(y_train_label))).float().to(DEVICE)
model_raw.fit(X_train, y_train_final, X_train, y_train_final)

# --- 3. Wrapper로 감싸기 ---
wrapped_tabr = TabRSklearnWrapper(model_raw, DEVICE)

# --- 4. Importance 계산 ---
print("🧪 TabR Permutation Importance 계산 중 (이 작업은 피처별로 데이터를 섞으므로 다소 오래 걸립니다)...")
r = permutation_importance(wrapped_tabr, X_test.cpu().numpy(), y_test_label,
                        n_repeats=5, random_state=SEED, scoring='accuracy')
tabr_importance = r.importances_mean

# --- 5. 결과 시각화 ---
df = pd.DataFrame({
    'Feature': feature_names,
    'XGBoost': xgb_importance,
    'TabR': tabr_importance
})

# 정규화 (최댓값 대비 상대적 비율)
df['XGBoost'] = df['XGBoost'] / df['XGBoost'].max()
df['TabR'] = df['TabR'] / (df['TabR'].max() + 1e-9)

df_melted = df.melt(id_vars='Feature', var_name='Model', value_name='Relative Importance')

plt.figure(figsize=(12, 6))
sns.barplot(data=df_melted, x='Feature', y='Relative Importance', hue='Model')
plt.title('Feature Importance Comparison: XGBoost vs TabR (Normalized)')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('importance_comparison.png')
print("✅ 분석 완료! 'importance_comparison.png' 파일을 확인하세요.")
plt.show()