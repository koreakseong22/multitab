import os
import numpy as np
import pandas as pd

# 1. 결과가 저장된 상대 경로 설정
base_dir = os.path.join('results', 'reproduce_logs', 'seed=1', 'data=31')

if not os.path.exists(base_dir):
    print(f"❌ 경로를 찾을 수 없습니다: {base_dir}")
    exit()

summary_data = []

# 2. 폴더 내 모든 .npy 파일을 읽어서 성능 요약
for file in os.listdir(base_dir):
    if file.endswith('.npy'):
        path = os.path.join(base_dir, file)
        try:
            # allow_pickle=True는 numpy 객체를 읽을 때 필수입니다.
            res = np.load(path, allow_pickle=True).item()
            
            # 파일명 분석 (예: model=saint..init_hps=False...)
            parts = file.replace('.npy', '').split('..')
            model_name = parts[0].split('=')[1]
            
            # 튜닝 여부 확인
            is_tuned = "Tuned (Best)" if "init_hps=False" in file else "Default"
            
            # 앙상블 여부 확인
            is_deep = "Deep" if "deep=1" in file else ""
            is_hyper = "Hyper" if "hyper=1" in file else ""
            setting = f"{is_tuned} {is_deep} {is_hyper}".strip()
            
            # 성능 지표 추출
            perf = res.get('Performance', {})
            summary_data.append({
                'Model': model_name,
                'Setting': setting,
                'Accuracy': perf.get('acc_test'),
                'AUROC': perf.get('auroc_test'),
                'F1': perf.get('f1_test')
            })
        except Exception as e:
            print(f"⚠️ {file} 읽기 실패: {e}")

# 3. 데이터프레임으로 변환 및 정렬 (정확도 높은 순)
df = pd.DataFrame(summary_data)
if not df.empty:
    df = df.sort_values(by='Accuracy', ascending=False)

    print("\n🏆 --- MultiTab 최종 리더보드 (Credit-G) --- 🏆")
    print(df.to_string(index=False))

    # 4. 엑셀에서 보기 편하게 CSV로도 저장
    df.to_csv('final_leaderboard_31.csv', index=False)
    print(f"\n✅ 요약 완료! 'final_leaderboard_31.csv' 파일이 생성되었습니다.")
else:
    print("데이터가 없습니다.")