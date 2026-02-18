import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import fetch_openml

# 1. 데이터 로드 (OpenML ID: 43986 - Wine Quality)
print("🍷 데이터 로딩 중...")
data = fetch_openml(data_id=43986, as_frame=True, parser='auto')
df = data.frame

# 2. 기본 구조 파악
print("\n--- [1] 데이터 기본 정보 ---")
print(df.info())

print("\n--- [2] 기술 통계량 ---")
# .T를 사용하면 피처가 많을 때 보기 편합니다.
print(df.describe().T)

# 3. 타겟 변수(quality) 분포 확인
plt.figure(figsize=(8, 5))
sns.countplot(x='quality', data=df, hue='quality', palette='viridis', legend=False)
plt.title('Distribution of Wine Quality (Target)')
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()

# 

# 4. 피처별 분포 (히스토그램 & 밀도 그래프)
df.hist(figsize=(15, 10), bins=30, edgecolor='black')
plt.suptitle('Feature Distributions', fontsize=16)
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()

# 

# 5. 상관관계 분석 (Heatmap)
plt.figure(figsize=(12, 10))
corr = df.corr()
mask = np.triu(np.ones_like(corr, dtype=bool)) # 상삼각형 가리기
sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", cmap='coolwarm', center=0)
plt.title('Feature Correlation Heatmap')
plt.show()

# 

# 6. 이상치 탐지 (Boxplot)
plt.figure(figsize=(15, 8))
sns.boxplot(data=df.drop('quality', axis=1))
plt.xticks(rotation=45)
plt.title('Feature Boxplots (Checking for Outliers)')
plt.show()