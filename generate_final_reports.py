import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.linear_model import HuberRegressor
import warnings

warnings.filterwarnings('ignore')

# 설정
ROOT_DIR = Path("/mnt/hdd1/jeon/bass")
BW_SES_PATH = ROOT_DIR / "BW 데이터" / "bw_aging_sessions.csv"
SK_SES_PATH = ROOT_DIR / "SK 데이터" / "analysis" / "sk_aging_sessions.csv"

def get_payload_stress(df):
    """차종별/속도구간별 기준 전비 대비 실제 전비의 비율 산출"""
    df['speed_bin'] = (df['avg_speed'] // 10) * 10
    # 차종별/속도구간별 중앙값 계산
    baselines = df.groupby(['car_type', 'speed_bin'])['norm_kwh'].transform('median')
    df['payload_stress_index'] = df['norm_kwh'] / baselines
    return df

def process_data(ses_path, label):
    print(f"Processing {label} data...")
    df = pd.read_csv(ses_path)
    df['time'] = pd.to_datetime(df['time'])
    
    # 1. 환경 보정 (전역)
    X = df[['avg_temp', 'avg_speed']].copy()
    X['temp_sq'] = X['avg_temp'] ** 2
    model = HuberRegressor().fit(X, df['kWh_per_km'])
    df['norm_kwh'] = df['kWh_per_km'] - model.predict(X) + df['kWh_per_km'].mean()
    
    # 2. 스트레스 지수 계산
    df = get_payload_stress(df)
    
    results = []
    for dev_id, group in df.groupby('dev_id'):
        group = group.sort_values('time')
        obs_days = (group['time'].max() - group['time'].min()).days
        
        # 최소 조건: 200일 이상, 50세션 이상
        if len(group) < 50 or obs_days < 200:
            continue
            
        group['smooth_kwh'] = group['norm_kwh'].rolling(window=30, min_periods=5).mean()
        group_clean = group.dropna(subset=['smooth_kwh'])
        
        if len(group_clean) < 20: continue
        
        # [Best-Ever Baseline] 초기 100세션 중 가장 효율이 좋았던(낮은 kWh/km) 20세션의 중앙값
        # 이를 통해 운전자가 가장 잘 탔을 때를 100%로 설정하여, 이후의 '회춘' 현상을 억제함
        first_100 = group_clean.head(100)
        initial_kwh = first_100['smooth_kwh'].nsmallest(20).median()
        
        current_kwh = group_clean.tail(15)['smooth_kwh'].mean()
        
        current_retention = (initial_kwh / current_kwh) * 100
        total_drop = current_retention - 100
        
        # 주행 스트레스 (해당 차량의 평균 스트레스)
        avg_stress = group['payload_stress_index'].mean()
        
        # 일일 변화율 (Regression) - Best-Ever Baseline 대비 상대 성능으로 계산
        days = (group_clean['time'] - group_clean['time'].min()).dt.days.values.reshape(-1, 1)
        rel_perf_series = (initial_kwh / group_clean['smooth_kwh']) * 100
        huber = HuberRegressor().fit(days, rel_perf_series)
        daily_slope = huber.coef_[0]
        
        results.append({
            'dev_id': dev_id,
            'car_type': group['car_type'].iloc[0],
            'obs_days': obs_days,
            'total_drop_pct': total_drop,
            'current_retention': current_retention,
            'avg_payload_stress': avg_stress,
            'daily_slope': daily_slope,
            'data_source': label
        })
        
    return pd.DataFrame(results), df

def generate_visuals():
    bw_res, bw_df = process_data(BW_SES_PATH, "BW")
    sk_res, sk_df = process_data(SK_SES_PATH, "SK")
    
    all_res = pd.concat([bw_res, sk_res])
    
    plt.switch_backend('Agg')
    sns.set_style('whitegrid')

    # 1. final_refined_aging_report.jpg (분포 중심)
    plt.figure(figsize=(16, 8))
    plt.subplot(1, 2, 1)
    sns.histplot(all_res['total_drop_pct'], kde=True, color='teal', bins=30)
    plt.axvline(0, color='red', linestyle='--')
    plt.title('Distribution of Total Observed Performance Change (%)', fontsize=14)
    plt.xlabel('Total Change (%) during Observation Period')
    
    plt.subplot(1, 2, 2)
    sns.scatterplot(data=all_res, x='obs_days', y='total_drop_pct', hue='data_source', alpha=0.6)
    plt.axhline(0, color='red', linestyle='--')
    plt.title('Observation Days vs. Total Change (%)', fontsize=14)
    plt.xlabel('Observation Days')
    plt.savefig(ROOT_DIR / "final_refined_aging_report.jpg", dpi=300)
    print("Regenerated: final_refined_aging_report.jpg")

    # 2. trend_based_aging_analysis.jpg (개별 추세)
    targets = pd.concat([all_res.sort_values('daily_slope').head(3), 
                         all_res.sort_values('daily_slope', ascending=False).head(3)])
    plt.figure(figsize=(24, 14))
    for i, (idx, row) in enumerate(targets.iterrows()):
        dev_id, src = row['dev_id'], row['data_source']
        df_full = bw_df if src == "BW" else sk_df
        v_ses = df_full[df_full['dev_id'] == dev_id].sort_values('time').copy()
        v_ses['smooth_kwh'] = v_ses['norm_kwh'].rolling(window=30, min_periods=5).mean()
        v_ses = v_ses.dropna(subset=['smooth_kwh'])
        
        # Best-Ever Baseline logic for individual plots
        first_100_v = v_ses.head(100)
        init_v = first_100_v['smooth_kwh'].nsmallest(20).median()
        
        v_ses['rel_perf'] = (init_v / v_ses['smooth_kwh']) * 100
        v_ses['days'] = (v_ses['time'] - v_ses['time'].min()).dt.days
        
        plt.subplot(2, 3, i+1)
        plt.scatter((df_full[df_full['dev_id']==dev_id]['time']-df_full[df_full['dev_id']==dev_id]['time'].min()).dt.days,
                    (init_v/df_full[df_full['dev_id']==dev_id]['norm_kwh'])*100, alpha=0.1, color='gray', s=10)
        plt.plot(v_ses['days'], v_ses['rel_perf'], color='blue', alpha=0.6, linewidth=2)
        sns.regplot(data=v_ses, x='days', y='rel_perf', scatter=False, color='red' if i < 3 else 'green')
        plt.title(f"{dev_id} ({row['car_type']})\nTotal Change: {row['total_drop_pct']:.2f}% ({row['obs_days']} days)", fontsize=16)
        plt.ylim(85, 115)
    plt.tight_layout()
    plt.savefig(ROOT_DIR / "trend_based_aging_analysis.jpg", dpi=300)
    print("Regenerated: trend_based_aging_analysis.jpg")

    # 3. car_type_aging_trend_diagnostic.jpg (사격판 진단 지도)
    # X축: 현재 성능 유지율(Status), Y축: 연간 노화 추세(Velocity)
    all_res['annual_rate'] = all_res['daily_slope'] * 365
    plt.figure(figsize=(16, 12))
    sns.scatterplot(data=all_res, x='current_retention', y='annual_rate', 
                    hue='car_type', size='obs_days', sizes=(50, 500), alpha=0.6)
    
    plt.axvline(100, color='gray', linestyle='--', alpha=0.5)
    plt.axhline(0, color='gray', linestyle='--', alpha=0.5)
    
    # 영역 구분 (사격판 논리)
    plt.text(102, 1, 'Stable & Best', fontsize=15, color='green', weight='bold')
    plt.text(90, -5, 'High Risk (Fast Aging)', fontsize=15, color='red', weight='bold')
    plt.text(102, -5, 'Monitoring Needed', fontsize=15, color='orange', weight='bold')
    
    plt.title('Vehicle Diagnostic Target Map (Status vs. Velocity)', fontsize=20)
    plt.xlabel('Current Retention (%) [Health Status]', fontsize=14)
    plt.ylabel('Annual Aging Rate (%) [Degradation Velocity]', fontsize=14)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize='small', ncol=2)
    plt.grid(True, which='both', linestyle='--', alpha=0.3)
    plt.tight_layout()
    plt.savefig(ROOT_DIR / "car_type_aging_trend_diagnostic.jpg", dpi=300)
    print("Regenerated: car_type_aging_trend_diagnostic.jpg")

    # 4. pi_calibration_comparison.jpg (보정 효과)
    # Raw Retention vs. Stress-Adjusted Retention
    all_res['calibrated_retention'] = 100 + (all_res['current_retention'] - 100) * (all_res['avg_payload_stress']**0.5)
    
    plt.figure(figsize=(18, 10))
    plot_data = pd.melt(all_res, id_vars=['car_type'], value_vars=['current_retention', 'calibrated_retention'],
                        var_name='Metric', value_name='Value')
    plot_data['Metric'] = plot_data['Metric'].replace({'current_retention': 'Raw Retention', 'calibrated_retention': 'Calibrated (Stress-Adj)'})
    
    order = all_res.groupby('car_type')['current_retention'].median().sort_values().index
    sns.boxplot(data=plot_data, x='car_type', y='Value', hue='Metric', order=order, palette='Set2')
    plt.xticks(rotation=45, ha='right')
    plt.axhline(100, color='red', linestyle='--', alpha=0.6)
    plt.title('Effect of Payload Stress Calibration on Performance Index', fontsize=18)
    plt.tight_layout()
    plt.savefig(ROOT_DIR / "pi_calibration_comparison.jpg", dpi=300)
    print("Regenerated: pi_calibration_comparison.jpg")

    # 5. stress_impact_analysis.jpg (스트레스 상관관계)
    plt.figure(figsize=(12, 8))
    sns.regplot(data=all_res, x='avg_payload_stress', y='current_retention', 
                scatter_kws={'alpha':0.4, 's':60}, line_kws={'color':'red'})
    plt.axhline(100, color='blue', linestyle='--', alpha=0.3)
    plt.axvline(1.0, color='blue', linestyle='--', alpha=0.3)
    plt.title('Correlation: Payload Stress Index vs. Observed Retention', fontsize=16)
    plt.xlabel('Average Payload Stress Index (1.0 = Standard Load)')
    plt.ylabel('Observed Retention (%)')
    plt.savefig(ROOT_DIR / "stress_impact_analysis.jpg", dpi=300)
    print("Regenerated: stress_impact_analysis.jpg")

    # 6. car_type_individual_diagnostic.jpg (차종별 분포 상세 - 추가)
    plt.figure(figsize=(20, 10))
    sns.violinplot(data=all_res, x='car_type', y='current_retention', order=order, inner="quart", palette='pastel')
    sns.stripplot(data=all_res, x='car_type', y='current_retention', order=order, color='black', size=3, alpha=0.3)
    plt.axhline(100, color='red', linestyle='--', alpha=0.6)
    plt.title('Detailed Performance Distribution by Car Type', fontsize=18)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(ROOT_DIR / "car_type_individual_diagnostic.jpg", dpi=300)
    print("Regenerated: car_type_individual_diagnostic.jpg")

    all_res.to_csv(ROOT_DIR / "final_aging_summary_all_v2.csv", index=False)
    print("All tasks completed.")

if __name__ == "__main__":
    generate_visuals()
