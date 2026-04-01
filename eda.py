import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import fetch_openml
from sklearn.preprocessing import StandardScaler

# 1. 데이터 로드 (OpenML ID: 1504 - Steel Plates Fault)
# Tip: data_id만 바꾸면 다른 데이터셋도 바로 분석 가능합니다.
data_id = 54 
print(f"🚀 vehicle(ID: {data_id}) 로딩 중...")
data = fetch_openml(data_id=data_id, as_frame=True, parser='auto')
df = data.frame

# [핵심] 타겟 컬럼 자동 지정 (마지막 컬럼을 타겟으로 가정)
target_col = df.columns[-1] 
print(f"✅ 분석 타겟 컬럼: [{target_col}]")

# 2. 기본 구조 파악
print("\n--- [1] 데이터 기본 정보 ---")
print(df.info())

# 3. 변수 타입 분리 및 결측치 확인
num_cols = df.select_dtypes(include=np.number).columns.tolist()
# 타겟이 수치형으로 분류되어 있다면 리스트에서 제거 (시각화 목적)
if target_col in num_cols: num_cols.remove(target_col)

print("\n--- [2] 결측치 확인 ---")
print(df.isnull().sum().sum(), "개의 결측치가 있습니다.")

# 4. 타겟 변수 분포 확인
plt.figure(figsize=(10, 5))
sns.countplot(x=target_col, data=df, hue=target_col, palette='viridis', legend=False)
plt.title(f'Distribution of {target_col} (Target)')
plt.xticks(rotation=45)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()

# 5. 피처별 분포
df[num_cols[:]].hist(figsize=(15, 10), bins=30, edgecolor='black')
plt.suptitle('Feature Distributions', fontsize=16)
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()

# 6. 타겟에 따른 주요 Feature의 분포 차이 (Violin Plot)
# plt.figure(figsize=(15, 10))
# features_to_plot = num_cols[:6] 
# for i, col in enumerate(features_to_plot):
#     plt.subplot(2, 3, i+1)
#     sns.violinplot(x=target_col, y=col, data=df, palette='muted')
#     plt.title(f'{target_col} vs {col}')
# plt.tight_layout()
# plt.show()

# 7. 상관관계 분석 (수치형 변수들만)
plt.figure(figsize=(12, 10))
corr = df[num_cols + ([target_col] if df[target_col].dtype != 'category' else [])].corr()
mask = np.triu(np.ones_like(corr, dtype=bool))
sns.heatmap(corr, mask=mask, annot=False, cmap='coolwarm', center=0) # annot=False로 가독성 높임
plt.title('Feature Correlation Heatmap')
plt.show()

# 8. 이상치 탐지 (표준화 적용 후 Boxplot)
plt.figure(figsize=(15, 8))
scaler = StandardScaler()
df_scaled = pd.DataFrame(scaler.fit_transform(df[num_cols]), columns=num_cols)

sns.boxplot(data=df_scaled.iloc[:, :15], palette='Set2') # 너무 많으면 보기 힘드니 15개만
plt.xticks(rotation=45)
plt.title('Feature Boxplots (Standardized Scale - Top 15)')
plt.axhline(0, color='red', linestyle='--', alpha=0.5)
plt.show()