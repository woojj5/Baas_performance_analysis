# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm
import csv
import os

DATA_ROOT = Path("/mnt/hdd1/jeon/bass/SK 데이터/analysis/db_datasets")
OUTPUT_FILE = Path("/mnt/hdd1/jeon/bass/SK 데이터/analysis/sk_aging_sessions.csv")
TEMP_DIR = Path("/mnt/hdd1/jeon/bass/SK 데이터/analysis/temp_dev_data")

SESSION_BREAK_MINUTES = 30
MIN_SESSION_DISTANCE_KM = 10

def extract_sessions_optimized(file_path):
    car_type = file_path.stem
    print(f"\n[info] Processing {car_type} (Optimized Single-Pass)...")
    
    required_cols = ['dev_id', 'coll_dt', 'c_mileage', 'b_accum_discharg_power_quan', 'v_car_speed']
    temp_cols = [f'b_modul_{i}_temp' for i in range(1, 5)]
    all_cols = required_cols + temp_cols
    
    if not TEMP_DIR.exists():
        TEMP_DIR.mkdir(parents=True)
        
    # Step 1: Split into per-device small files (Single Pass)
    dev_files = {}
    
    print(f" Splitting {file_path.name} into per-device files...")
    reader = pd.read_csv(file_path, usecols=all_cols, chunksize=1000000, low_memory=False)
    
    for chunk in tqdm(reader, desc="Splitting"):
        for dev_id, dev_group in chunk.groupby('dev_id'):
            dev_file_path = TEMP_DIR / f"{dev_id}.csv"
            dev_group.to_csv(dev_file_path, mode='a', index=False, header=not dev_file_path.exists())
            dev_files[dev_id] = dev_file_path

    # Step 2: Process each device file
    sessions = []
    print(f" Extracting sessions from {len(dev_files)} devices...")
    for dev_id, dev_file_path in tqdm(dev_files.items(), desc="Processing Devices"):
        try:
            dev_data = pd.read_csv(dev_file_path)
            dev_data['coll_dt'] = pd.to_datetime(dev_data['coll_dt'])
            dev_data = dev_data.sort_values('coll_dt')
            
            # Filter driving only
            drive_data = dev_data[dev_data['v_car_speed'] > 0].copy()
            if drive_data.empty: continue
            
            drive_data['t_diff'] = drive_data['coll_dt'].diff().dt.total_seconds() / 60.0
            drive_data['d_diff'] = drive_data['c_mileage'].diff()
            
            is_new = (drive_data['t_diff'] > SESSION_BREAK_MINUTES) | (drive_data['d_diff'] < 0)
            drive_data['session_id'] = is_new.cumsum()
            
            for sid, sdata in drive_data.groupby('session_id'):
                if len(sdata) < 5: continue
                
                dist = sdata['c_mileage'].iloc[-1] - sdata['c_mileage'].iloc[0]
                if dist < MIN_SESSION_DISTANCE_KM: continue
                
                energy = sdata['b_accum_discharg_power_quan'].iloc[-1] - sdata['b_accum_discharg_power_quan'].iloc[0]
                if energy <= 0.1: continue
                
                kwh_km = energy / dist
                if not (0.05 < kwh_km < 1.0): continue
                
                avg_temp = sdata[temp_cols].mean(axis=1).mean()
                duration = (sdata['coll_dt'].iloc[-1] - sdata['coll_dt'].iloc[0]).total_seconds() / 3600.0
                avg_speed = dist / duration if duration > 0 else 0
                
                sessions.append([
                    car_type,
                    dev_id,
                    sdata['coll_dt'].iloc[-1].strftime("%Y-%m-%d %H:%M:%S"),
                    round(sdata['c_mileage'].iloc[-1], 2),
                    round(kwh_km, 4),
                    round(avg_temp, 2),
                    round(avg_speed, 2)
                ])
        finally:
            if dev_file_path.exists():
                os.remove(dev_file_path)
                
    return sessions

def main():
    # Only processing EV6, IONIQ5, Kona first to save time. 
    # Niro (72GB) is excluded for now or can be included if we have time.
    target_files = ["EV6.csv", "IONIQ5.csv", "Kona.csv"]
    
    if not OUTPUT_FILE.parent.exists():
        OUTPUT_FILE.parent.mkdir(parents=True)

    with open(OUTPUT_FILE, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow(["car_type", "dev_id", "time", "cum_mileage", "kWh_per_km", "avg_temp", "avg_speed"])
        
        all_sessions_count = 0
        for filename in target_files:
            path = DATA_ROOT / filename
            if path.exists():
                file_sessions = extract_sessions_optimized(path)
                writer.writerows(file_sessions)
                all_sessions_count += len(file_sessions)
                print(f"[info] Extracted {len(file_sessions)} sessions from {filename}")
            else:
                print(f"[warn] File not found: {path}")
                
    print(f"\n[success] Total {all_sessions_count} sessions saved to {OUTPUT_FILE}")

if __name__ == "__main__":
    main()
