import argparse
import configparser
import csv
import os
from pathlib import Path
from influxdb_client import InfluxDBClient
from datetime import datetime

# 설정 파일 로드
HERE = Path(__file__).resolve().parent
CFG = HERE / "config2.ini"

def load_config():
    cfg = configparser.ConfigParser()
    if not cfg.read(CFG, encoding="utf-8"):
        raise FileNotFoundError(f"config2.ini not found at: {CFG}")
    return (
        cfg["influxdb"]["url"],
        cfg["influxdb"]["token"],
        cfg["influxdb"]["org"],
        cfg["influxdb"]["bucket"]
    )

def extract_bw_data(output_file="bw_aging_sessions.csv"):
    url, token, org, bucket = load_config()
    client = InfluxDBClient(url=url, token=token, org=org, timeout=300000)
    
    # 2023-10-01부터 데이터가 있는 것으로 확인됨
    start_time = "2023-10-01T00:00:00Z"
    end_time = "2025-12-31T23:59:59Z"
    
    # segment_stats_drive에서 필요한 필드 추출
    # accum_mileage: 누적 주행거리 (Absolute Odometer)
    # km_per_kWh: 효율 (전비) -> kWh_per_km로 변환 필요 (1/eff)
    # temp_mean: 평균 온도
    # mileage: 세션 주행 거리 (km) -> 필터링용
    # period: 세션 주행 시간 (sec) -> 평균 속도 계산용
    
    flux = f'''
    from(bucket: "{bucket}")
      |> range(start: {start_time}, stop: {end_time})
      |> filter(fn: (r) => r._measurement == "segment_stats_drive")
      |> filter(fn: (r) => r._field == "accum_mileage" or r._field == "km_per_kWh" or r._field == "temp_mean" or r._field == "mileage" or r._field == "period")
      |> pivot(rowKey: ["_time", "car_id"], columnKey: ["_field"], valueColumn: "_value")
      |> filter(fn: (r) => exists r.accum_mileage and exists r.km_per_kWh and exists r.temp_mean and exists r.mileage and exists r.period)
      |> keep(columns: ["_time", "car_id", "car_type", "accum_mileage", "km_per_kWh", "temp_mean", "mileage", "period"])
    '''
    
    print(f"[info] Extracting BW data from {bucket}...")
    
    with open(output_file, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow(["car_type", "dev_id", "time", "cum_mileage", "kWh_per_km", "avg_temp", "avg_speed"])
        
        count = 0
        try:
            tables = client.query_api().query(flux, org=org)
            for table in tables:
                for record in table.records:
                    car_type = record.values.get("car_type", "UNKNOWN")
                    dev_id = record.values.get("car_id")
                    time_str = record.get_time().strftime("%Y-%m-%d %H:%M:%S")
                    odo = record.values.get("accum_mileage")
                    eff = record.values.get("km_per_kWh")
                    temp = record.values.get("temp_mean")
                    dist = record.values.get("mileage")
                    period = record.values.get("period")
                    
                    # 1. 전비 변환 (km/kWh -> kWh/km)
                    if eff and eff > 0:
                        kwh_km = 1.0 / eff
                    else:
                        continue
                        
                    # 2. 평균 속도 계산 (km / h)
                    if period and period > 0:
                        avg_speed = (dist / period) * 3600
                    else:
                        avg_speed = 0.0
                        
                    # 3. 필터링 (SK 데이터 기준인 10km를 적용하여 초단거리 노이즈 제거)
                    if dist < 10.0:
                        continue
                        
                    # 4. 물리적 한계 필터링 (전비가 비정상적인 경우)
                    if kwh_km < 0.05 or kwh_km > 1.0:
                        continue
                        
                    writer.writerow([car_type, dev_id, time_str, odo, kwh_km, temp, avg_speed])
                    count += 1
                    
            print(f"[info] Successfully extracted {count} sessions to {output_file}")
        except Exception as e:
            print(f"[error] Extraction failed: {e}")
        finally:
            client.close()

if __name__ == "__main__":
    extract_bw_data()

