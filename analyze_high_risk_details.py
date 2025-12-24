# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# 데이터 경로
BW_DIR = Path("/mnt/hdd1/jeon/bass/BW 데이터")
SESSION_FILE = BW_DIR / "bw_aging_sessions.csv"
RESULTS_CSV = BW_DIR / "bw_strict_aging_results.csv"

# 9대의 정밀 점검 대상 차량 ID
TARGET_DEVICES = [
    'V012BE0000', 'V000BE0017', 'V004BL0002', 'V004AK0000', 
    'V023AL0000', 'V004BG0006', 'V004CA0000', 'V013BC0000', 'V004BA0030'
]

def deep_dive_analysis():
    print("Step 1: Loading Session Data for High-Risk Vehicles...")
    df_all = pd.read_csv(SESSION_FILE)
    df_res = pd.read_csv(RESULTS_CSV)
    
    # 대상 차량 세션만 추출
    df_targets = df_all[df_all['dev_id'].isin(TARGET_DEVICES)].copy()
    df_targets['time'] = pd.to_datetime(df_targets['time'])
    
    # 대조군(Fleet) 통계 (상위 90% 이상 효율 차량들과 비교하기 위함)
    fleet_avg_speed = df_all['avg_speed'].mean()
    fleet_avg_temp = df_all['avg_temp'].mean()
    
    report_data = []
    
    print("Step 2: Analyzing Individual Driving Patterns...")
    for dev_id in TARGET_DEVICES:
        sessions = df_targets[df_targets['dev_id'] == dev_id]
        res = df_res[df_res['dev_id'] == dev_id].iloc[0]
        
        # 1. 속도 패턴 분석 (과속 여부)
        avg_speed = sessions['avg_speed'].mean()
        high_speed_ratio = len(sessions[sessions['avg_speed'] > 100]) / len(sessions) * 100
        
        # 2. 온도 노출 분석 (혹한/혹서기 주행 비중)
        extreme_temp_ratio = len(sessions[(sessions['avg_temp'] < 0) | (sessions['avg_temp'] > 30)]) / len(sessions) * 100
        
        # 3. 효율 변동성 (주행의 일관성 - 급가속 등 추정)
        # 속도와 온도로 보정된 값의 변동성(std)이 크면 운전 습관이 거칠 가능성 높음
        eff_std = sessions['kWh_per_km'].std()
        
        # 4. 원인 추정 로직
        primary_cause = "배터리 노화 의심 (Aging)"
        if high_speed_ratio > 30:
            primary_cause = "고속 주행 위주 패턴 (High-Speed Driving)"
        elif extreme_temp_ratio > 40:
            primary_cause = "극한 온도 노출 (Extreme Environment)"
        elif eff_std > df_all['kWh_per_km'].std() * 1.5:
            primary_cause = "운전 습관 불규칙 (Aggressive/Irregular Driving)"
            
        report_data.append({
            'dev_id': dev_id,
            'car_type': res.car_type,
            'perf_index': res.performance_index,
            'avg_speed': round(avg_speed, 1),
            'high_speed_ratio': round(high_speed_ratio, 1),
            'extreme_temp_ratio': round(extreme_temp_ratio, 1),
            'eff_variance': round(eff_std, 3),
            'primary_cause_candidate': primary_cause
        })
        
    report_df = pd.DataFrame(report_data).sort_values('perf_index')
    
    print("\n" + "="*120)
    print(" [정밀 점검 대상 9대 - 원인 분석 및 Deep-Dive 리포트] ")
    print("="*120)
    print(report_df.to_markdown(index=False))
    print("="*120)
    
    # 시각화: 개별 차량의 속도 vs 전비 산점도 (Fleet과 비교)
    plt.figure(figsize=(15, 10))
    plt.switch_backend('Agg')
    
    for i, dev_id in enumerate(TARGET_DEVICES[:6]): # 상위 6대만 시각화
        plt.subplot(2, 3, i+1)
        sns.scatterplot(data=df_all.sample(1000), x='avg_speed', y='kWh_per_km', color='gray', alpha=0.1, label='Fleet')
        sns.scatterplot(data=df_targets[df_targets['dev_id'] == dev_id], x='avg_speed', y='kWh_per_km', color='red', alpha=0.6, label=dev_id)
        plt.title(f"{dev_id} ({report_df[report_df['dev_id']==dev_id]['car_type'].values[0]})")
        plt.ylim(0, 0.5)
        
    plt.tight_layout()
    plt.savefig(BW_DIR / "high_risk_deep_dive.jpg")
    print(f"\n상세 분석 그래프 저장 완료: {BW_DIR / 'high_risk_deep_dive.jpg'}")

if __name__ == "__main__":
    deep_dive_analysis()

