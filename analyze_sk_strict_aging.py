# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.linear_model import HuberRegressor
from datetime import datetime

# [공통 설정] 계절 정의
SEASONS = {12: 'Winter', 1: 'Winter', 2: 'Winter', 3: 'Spring', 4: 'Spring', 5: 'Spring',
           6: 'Summer', 7: 'Summer', 8: 'Summer', 9: 'Autumn', 10: 'Autumn', 11: 'Autumn'}

# [Case 2를 위한 차종 세그먼트 매핑 - SK 데이터 기준]
SEGMENT_MAP = {
    'EV6': 'MID_SUV_EGMP',
    'IONIQ5': 'MID_SUV_EGMP',
    'Kona': 'SMALL_SUV'
}

SK_DIR = Path("/mnt/hdd1/jeon/bass/SK 데이터/analysis")
SESSION_FILE = SK_DIR / "sk_aging_sessions.csv"
RESULTS_CSV = SK_DIR / "sk_strict_aging_results.csv"

def revin_normalization(data, fleet_mean, fleet_std):
    """[Case 3] ReVIN: Instance Normalization & Mapping"""
    if len(data) == 0: return fleet_mean
    eps = 1e-5
    mean = np.mean(data)
    std = np.std(data) + eps
    # 1. Normalize
    norm_data = (data - mean) / std
    # 2. Map to Fleet Space & Denormalize
    return (norm_data * fleet_std + fleet_mean).mean()

def analyze_sk_hierarchical():
    print(f"Step 1: Loading and Preprocessing SK Data...")
    if not SESSION_FILE.exists(): 
        print(f"Error: {SESSION_FILE} not found.")
        return
    df = pd.read_csv(SESSION_FILE)
    df['time'] = pd.to_datetime(df['time'])
    df['month'] = df['time'].dt.month
    df['season'] = df['month'].map(SEASONS)
    df['segment'] = df['car_type'].map(lambda x: SEGMENT_MAP.get(x, 'UNKNOWN'))
    
    # 환경 보정 (세그먼트 그룹별로 분리하여 보정)
    df = df[(df['avg_speed'] <= 150) & (df['avg_speed'] > 5)].dropna()
    df = df[(df['kWh_per_km'] > 0.05) & (df['kWh_per_km'] < 1.0)]
    
    # 보정 모델 그룹화 (물리적 특성이 유사한 그룹끼리 묶음)
    def get_norm_group(seg):
        if 'COMMERCIAL' in seg: return 'COMMERCIAL'
        if 'LARGE' in seg or 'PREMIUM' in seg or 'PERFORMANCE' in seg: return 'HEAVY_LUXURY'
        if 'MID_SUV' in seg: return 'MID_SUV'
        return 'SEDAN_LIGHT'
    
    df['norm_group'] = df['segment'].map(get_norm_group)
    df['norm_kwh'] = 0.0
    
    for ng, group_df in df.groupby('norm_group'):
        if len(group_df) < 10:
            df.loc[group_df.index, 'norm_kwh'] = group_df['kWh_per_km'] # 데이터 부족시 원본 유지
            continue
        X_g = group_df[['avg_temp', 'avg_speed']].copy()
        X_g['temp_sq'] = X_g['avg_temp'] ** 2
        model_g = HuberRegressor().fit(X_g, group_df['kWh_per_km'])
        df.loc[group_df.index, 'norm_kwh'] = group_df['kWh_per_km'] - model_g.predict(X_g) + group_df['kWh_per_km'].mean()

    # --- [Hierarchy Baseline Engine] ---
    print("Step 2: Pre-calculating Hierarchical Baselines (using New Car data < 10,000km)...")
    
    # Baseline candidate: Only new cars to set the standard
    df_new = df[df['cum_mileage'] < 10000].copy()
    if len(df_new) < 50: # SK 데이터는 신차 비중이 적을 수 있음
        df_new = df[df['cum_mileage'] < 20000].copy()

    # Global Statistics (for Case 4)
    global_mean = df['norm_kwh'].median()
    global_std = df['norm_kwh'].std()

    # Segment-level Statistics (from New Cars)
    def get_robust_median(x):
        if len(x) < 5: return x.median()
        low, high = x.quantile([0.1, 0.9])
        return x[(x >= low) & (x <= high)].median()

    segment_stats = df_new.groupby('segment')['norm_kwh'].agg(['median', 'std']).to_dict('index')
    segment_baselines = df_new.groupby(['segment', 'season'])['norm_kwh'].apply(get_robust_median).to_dict()

    # Case 1: Car Type Baselines
    type_new_counts = df_new.groupby('car_type')['norm_kwh'].count()
    type_new_counts_dict = type_new_counts.to_dict()
    type_baselines_raw = df_new.groupby(['car_type', 'season'])['norm_kwh'].apply(get_robust_median).to_dict()

    results = []

    print("Step 3: Evaluating Vehicles using Priority Rules...")
    for (car_type, dev_id), group in df.groupby(['car_type', 'dev_id']):
        # 최소 세션 수 필터링
        if len(group) < 15:
            continue
            
        group = group.sort_values('time')
        current_eff = group['norm_kwh'].median()
        current_season = group.iloc[-1]['season']
        segment = SEGMENT_MAP.get(car_type, 'UNKNOWN')
        
        applied_case = ""
        baseline_val = 0
        confidence = "높음"
        method = ""
        sufficiency = "충분"

        # [Weighted Baseline Engine]
        n_type = type_new_counts_dict.get(car_type, 0)
        b_type = type_baselines_raw.get((car_type, current_season))
        b_seg = segment_baselines.get((segment, current_season))

        if b_type and b_seg:
            weight = min(1.0, n_type / 500.0)
            baseline_val = (b_type * weight) + (b_seg * (1.0 - weight))
            if weight > 0.9:
                applied_case = "Case 1"
                method = f"차종({car_type}) 중심 가중 베이스라인 (w={weight:.2f})"
            else:
                applied_case = "Case 2"
                method = f"세그먼트({segment}) 중심 가중 베이스라인 (w={weight:.2f})"
            confidence = "높음" if weight > 0.5 else "보통"
            
        elif b_seg:
            applied_case = "Case 2"
            baseline_val = b_seg
            method = f"유사 세그먼트({segment}) 계절별 평균 연비"
            confidence = "보통"
            
        elif b_type:
            applied_case = "Case 1"
            baseline_val = b_type
            method = f"차종({car_type}) 계절별 평균 연비"
            confidence = "보통 (세그먼트 정보 없음)"
            
        elif len(group) > 5:
            applied_case = "Case 3"
            target_stats = segment_stats.get(segment, {'median': global_mean, 'std': global_std})
            baseline_val = revin_normalization(group['norm_kwh'].values, target_stats['median'], target_stats['std'])
            method = f"ReVIN (Segment:{segment} 매핑)"
            confidence = "낮음"
            sufficiency = "부족"
            
        else:
            applied_case = "Case 4"
            baseline_val = global_mean
            method = "Fallback (전체 차종 단순 평균)"
            confidence = "매우 낮음"
            sufficiency = "매우 부족"

        # 성능 평가 (%) - Confidence-based Bayesian Shrinkage 적용
        # 데이터가 많을수록(신뢰도↑) 실제 측정값을 더 많이 반영, 적을수록 100%로 강하게 수축
        obs_days = (group['time'].max() - group['time'].min()).days
        w_sessions = min(1.0, len(group) / 100.0) # 100세션 기준
        w_days = min(1.0, obs_days / 365.0)       # 1년 기준
        confidence_score = (w_sessions + w_days) / 2.0
        
        # 신뢰도에 따라 수축 계수 가변 (0.15 ~ 0.70)
        shrink_factor = 0.15 + (0.55 * confidence_score)
        
        raw_perf_index = (baseline_val / current_eff) * 100
        perf_index = 100 + (raw_perf_index - 100) * shrink_factor
        
        ensemble_match = 0
        if obs_days >= 365:
            month_stats = group.groupby([group['time'].dt.year, 'month'])['norm_kwh'].median().unstack()
            for m in range(1, 13):
                if m in month_stats.columns and month_stats[m].count() >= 2:
                    ensemble_match += 1

        results.append({
            'dev_id': dev_id, 'car_type': car_type, 'applied_case': applied_case,
            'baseline_method': method, 'data_sufficiency': sufficiency, 'confidence': confidence,
            'current_efficiency': round(1/current_eff, 2), # km/kWh
            'performance_index': round(perf_index, 2),
            'observation_days': obs_days, 'ensemble_matches': ensemble_match
        })

    res_df = pd.DataFrame(results).sort_values('performance_index')
    res_df.to_csv(RESULTS_CSV, index=False, encoding='utf-8-sig')
    
    # --- [Visualization] ---
    plt.figure(figsize=(12, 7))
    plt.switch_backend('Agg')
    sns.set_style("whitegrid")
    
    # 차량 대수 기준 필터링 (최소 1대 이상 - SK는 종류가 적으므로 일단 다 포함)
    counts = res_df['car_type'].value_counts()
    plot_df = res_df.copy()
    order = plot_df.groupby('car_type')['performance_index'].median().sort_values().index
    
    sns.boxplot(data=plot_df, x='car_type', y='performance_index', order=order, palette='Set2', hue='car_type', legend=False)
    sns.stripplot(data=plot_df, x='car_type', y='performance_index', order=order, color='black', size=4, jitter=0.2, alpha=0.6)
    
    plt.axhline(100, color='red', linestyle='--', alpha=0.7, label='Standard (100%)')
    plt.title('SK Vehicle Performance Index Distribution (Calibrated)', fontsize=16)
    plt.xticks(rotation=0, fontsize=12)
    plt.ylabel('Performance Index (%)', fontsize=12)
    plt.ylim(80, 120)
    plt.tight_layout()
    
    PLOT_PATH = SK_DIR / "sk_strict_aging_analysis.jpg"
    plt.savefig(PLOT_PATH, dpi=300)
    print(f"Plot saved to: {PLOT_PATH}")

    print("\n" + "="*110)
    print(f" [SK 데이터 전비 분석 베이스라인 결정 보고서] - 총 {len(res_df)}대")
    print("="*110)
    print(res_df[['dev_id', 'car_type', 'applied_case', 'performance_index', 'confidence', 'baseline_method']].head(20).to_markdown(index=False))
    print("="*110)

if __name__ == "__main__":
    analyze_sk_hierarchical()
