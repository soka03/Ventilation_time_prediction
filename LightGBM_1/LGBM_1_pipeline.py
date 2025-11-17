import requests
import pandas as pd
import numpy as np
import xml.etree.ElementTree as ET
from datetime import datetime, timedelta
import joblib

# ---------------------------------------------
AIRKOREA_API_KEY = "72634d516d736f6b38306f7a6c4e6d" 
KMA_API_KEY = "seq0BXT_R2GqtAV0__dh7Q" 
SEOUL_TRAFFIC_API_KEY = "5479676464736f6b313035586b706147" 

MODEL_PATHS = {
    'PM25_t_plus_1': './LightGBM_1/model_PM25_t_plus_1.pkl',
    'PM25_t_plus_2': './LightGBM_1/model_PM25_t_plus_2.pkl',
    'PM25_t_plus_3': './LightGBM_1/model_PM25_t_plus_3.pkl'
}

KEY_MAP_AIRKOREA = {
    'SO2': 'SPDX',
    'CO': 'CBMX',
    'O3': 'OZON',
    'NO2': 'NTDX', 
    'PM10': 'PM',
    'PM25': 'FPM'
}

BASE_FEATURES = [
    'SO2', 'CO', 'O3', 'NO2', 'PM10', 'PM25', 
    'WS', 'PS', 'TA', 'HM', 'RN', 'VS', 'Traffic', 
    'WD_sin', 'WD_cos'
]

FINAL_MODEL_COLUMNS = [
    'SO2','CO','O3','NO2','PM10','PM25','WS','PS','TA','HM','RN','VS','Traffic','WD_sin','WD_cos',
    'SO2_t_minus_1','SO2_t_minus_2',
    'CO_t_minus_1','CO_t_minus_2',
    'O3_t_minus_1','O3_t_minus_2',
    'NO2_t_minus_1','NO2_t_minus_2',
    'PM10_t_minus_1','PM10_t_minus_2',
    'PM25_t_minus_1','PM25_t_minus_2',
    'WS_t_minus_1','WS_t_minus_2',
    'PS_t_minus_1','PS_t_minus_2',
    'TA_t_minus_1','TA_t_minus_2',
    'HM_t_minus_1','HM_t_minus_2',
    'RN_t_minus_1','RN_t_minus_2',
    'VS_t_minus_1','VS_t_minus_2',
    'Traffic_t_minus_1','Traffic_t_minus_2',
    'WD_sin_t_minus_1','WD_sin_t_minus_2',
    'WD_cos_t_minus_1','WD_cos_t_minus_2'
]

THRESHOLD_GOOD = 15
THRESHOLD_MODERATE = 35
THRESHOLD_BAD = 75


# ------------------------------------
# 실시간 데이터 API 요청

def safe_float(val):
    """API 응답 값을 float으로 변환"""
    
    try:
        f_val = float(val)
        return np.nan if f_val < -8.0 else f_val
    except (ValueError, TypeError):
        return np.nan

def get_air_data(target_dt):
    """대기질 데이터 API 요청"""
    
    tm_str = target_dt.strftime("%Y%m%d%H00")
    url = f"http://openAPI.seoul.go.kr:8088/{AIRKOREA_API_KEY}/json/TimeAverageAirQuality/1/5/{tm_str}/동대문구"
    
    print(f"[대기질 API] {tm_str} 데이터 요청...", end="")
    
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        data_dict = response.json()

        if 'TimeAverageAirQuality' in data_dict and data_dict['TimeAverageAirQuality']['list_total_count'] > 0:
            api_row = data_dict['TimeAverageAirQuality']['row'][0]
            air_data_final = {}
            for model_key, api_key in KEY_MAP_AIRKOREA.items():
                api_value = api_row.get(api_key) 
                air_data_final[model_key] = safe_float(api_value)
            print("완료")
            
            return air_data_final 
        else:
            if 'RESULT' in data_dict:
                print(f"실패\n [대기질 API 오류] {data_dict['RESULT']['MESSAGE']}")
            else:
                print(f"실패\n [대기질 API 오류] 알 수 없는 응답: {data_dict}")
            return None 
    except requests.exceptions.RequestException as err:
        print(f"실패\n [대기질 API 요청 오류] {err}")
        return None
    except Exception as e:
        print(f"실패\n [대기질 API 파싱 오류] {e}")
        return None
        
def get_weather_data(target_dt):
    """기상청 데이터 API 요청"""
    
    tm_str = target_dt.strftime("%Y%m%d%H00")
    
    domain = "https://apihub.kma.go.kr/api/typ01/url/kma_sfctm2.php?"
    params = f"tm={tm_str}&stn=108&dataType=JSON&authKey={KMA_API_KEY}"
    url = domain + params
    
    print(f"[기상청 API] {tm_str} 데이터 요청...", end="")

    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        response_text = response.text
        
        data_lines = [line for line in response_text.splitlines() if not line.startswith('#')]
        
        if not data_lines or not data_lines[0].strip():
            print(f"실패\n [기상청 API 오류] 해당하는 데이터가 없습니다.")
            return None 
        
        values = data_lines[0].split()

        weather_data = {
            "WS": safe_float(values[3]),
            "PS": safe_float(values[8]),
            "TA": safe_float(values[11]),
            "HM": safe_float(values[13]),
            "RN": safe_float(values[15]),
            "VS": safe_float(values[32]),
            "WD_raw": safe_float(values[2]),
        }
        
        if weather_data.get("RN") == -9.0:
            weather_data["RN"] = 0.0
        
        print("완료")
        
        return weather_data

    except requests.exceptions.RequestException as err:
        print(f"실패\n [기상청 API 요청 오류] {err}")
        return None
    except Exception as e:
        print(f"실패\n [기상청 API 파싱 오류] {e}")
        return None

def get_traffic_data(target_dt):
    """교통량 데이터 API 요청"""
    
    tm_str = target_dt.strftime("%Y%m%d%H00")
    date_str = target_dt.strftime("%Y%m%d")
    hour_str = target_dt.strftime("%H")
    
    url = f"http://openapi.seoul.go.kr:8088/{SEOUL_TRAFFIC_API_KEY}/xml/VolInfo/1/100/F-05/{date_str}/{hour_str}/"
    
    print(f"[교통량 API] {tm_str} 데이터 요청...", end="")
    
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        
        root = ET.fromstring(response.content)
        total_volume = 0
        rows = root.findall('row')
        
        if not rows:
             print(f"실패\n [교통량 API 오류] 해당하는 데이터가 없습니다.")
             return None
        
        for row in rows:
            volume_tag = row.find('vol')
            if volume_tag is not None and volume_tag.text is not None:
                total_volume += int(volume_tag.text)
        
        print("완료")
        
        return {"Traffic": float(total_volume)}

    except requests.exceptions.RequestException as e:
        print(f"실패\n [교통량 API 요청 오류] {e}")
        return None
    except Exception as e:
        print(f"실패\n [교통량 API 파싱 오류] {e}")
        return None



# -------------------------------------------------------
# 실시간 데이터 전처리

def preprocess_data(df_3rows):
    print("\n데이터프레임 전처리 시작...")
    
    df_processed = df_3rows.bfill().ffill()
    
    if 'RN' in df_processed.columns:
        df_processed['RN'] = df_processed['RN'].fillna(0)
    
    if df_processed.isnull().values.any():
        print("[전처리 오류] 3시간치 API 데이터 전체가 비어있거나, 오류로 데이터를 채울 수 없습니다.")
        print("--- 채워지지 않은 결측치 ---")
        print(df_processed.isnull().sum())
        return None
        
    df_processed['WD_sin'] = 0.0
    df_processed['WD_cos'] = 0.0
    
    if 'WD_raw' in df_processed.columns:
        valid_wind_mask = df_processed['WD_raw'].notna() & (df_processed['WD_raw'] > 0)
        
        if valid_wind_mask.any():
            degrees = df_processed.loc[valid_wind_mask, 'WD_raw'] * 10
            degrees = degrees.replace(360, 0)
            radians = np.deg2rad(degrees)
            
            df_processed.loc[valid_wind_mask, 'WD_sin'] = np.sin(radians)
            df_processed.loc[valid_wind_mask, 'WD_cos'] = np.cos(radians)
        
        if 'WD_raw' in df_processed.columns:
            df_processed = df_processed.drop(columns=['WD_raw'])
    
    return df_processed

def flatten_data(processed_df, base_features):
    print("\n모델 입력용 데이터로 펼치는 중...")
    
    if processed_df is None or len(processed_df) != 3:
        print("3행의 전처리된 데이터가 필요합니다.")
        return None

    final_data_row = {}

    try:
        processed_df = processed_df[base_features]
        
        df_t0 = processed_df.iloc[2]
        df_t1 = processed_df.iloc[1]
        df_t2 = processed_df.iloc[0]

        for col in base_features:
            final_data_row[col] = df_t0[col]

        for col in base_features:
            final_data_row[f"{col}_t_minus_1"] = df_t1[col]
            final_data_row[f"{col}_t_minus_2"] = df_t2[col]

    except KeyError as e:
        print(f" 전처리된 DF에 필요한 컬럼이 없습니다: {e}")
        return None
    except IndexError:
        print("processed_df에 3행의 데이터가 없습니다.")
        return None

    final_model_row = pd.DataFrame([final_data_row])
    
    try:
        final_model_row = final_model_row[FINAL_MODEL_COLUMNS]
    except KeyError as e:
        print(f"[오류] 훈련/예측 피처가 일치하지 않습니다: {e}")
        return None
        
    return final_model_row


# ------------------------------------------------------
# 환기 추천 파이프라인

def get_status(value):
    if value <= THRESHOLD_GOOD:
        return "좋음 (Good)"
    if value <= THRESHOLD_MODERATE:
        return "보통 (Moderate)"
    if value <= THRESHOLD_BAD:
        return "나쁨 (Bad)"
    return "매우 나쁨 (Very Bad)"


def get_recommendation(pred_t1, pred_t2, pred_t3, logical_start_time): 
    predictions = [
        {'hour_offset': 1, 'value': pred_t1, 'status': get_status(pred_t1)},
        {'hour_offset': 2, 'value': pred_t2, 'status': get_status(pred_t2)},
        {'hour_offset': 3, 'value': pred_t3, 'status': get_status(pred_t3)},
    ]
    
    # 환기 가능한 시간 (농도 35 이하, '보통' 또는 '좋음') 필터링
    acceptable_hours = [p for p in predictions if p['value'] <= THRESHOLD_MODERATE]
    
    if acceptable_hours:
        best_hour_info = min(acceptable_hours, key=lambda x: x['value'])
        
        best_time_dt = logical_start_time + timedelta(hours=best_hour_info['hour_offset'])
        best_time_str = best_time_dt.strftime("%H시")
        
        message = (
            f"환기 추천: {best_time_str}가 3시간 내 최적의 시간입니다.\n"
            f" (예측 농도: {best_hour_info['value']:.2f} µg/m³ [{best_hour_info['status']}])"
        )
        return message
    else:
        # 환기 보류: 3시간 모두 '나쁨' 이상 (35 초과)
        # 3시간 중 가장 덜 나쁜 시간을 참고용으로 알려줌
        least_bad_hour = min(predictions, key=lambda x: x['value'])
        
        message = (
            f" 환기 보류: 향후 3시간 동안 미세먼지 농도가 '나쁨' 이상('보통' 기준 초과)일 것으로 예측됩니다.\n"
            f"  (참고: 3시간 중 가장 낮은 예측 농도는 {least_bad_hour['value']:.2f} µg/m³ [{least_bad_hour['status']}]입니다.)"
        )
        return message

def run_prediction_pipeline():
    print("훈련된 모델 3개 불러오는 중...")
    MODELS = {} 
    
    try:
        for target, path in MODEL_PATHS.items():
            MODELS[target] = joblib.load(path)
            print(f"[{target}] 모델 '{path}' 로드 성공")
    except FileNotFoundError as e:
        print(f"[오류] 모델 파일을 찾을 수 없습니다: {e}")
        return

    dt_now = datetime.now()
    dt_minus_1 = dt_now - timedelta(hours=1)
    dt_minus_2 = dt_now - timedelta(hours=2)
    
    print("\n현재 시각:", dt_now.strftime('%Y-%m-%d %H:%M'))
    print("\n3시간치 실시간 데이터 수집 시작...")
    
    data_list = []

    for dt in [dt_minus_2, dt_minus_1, dt_now]:
        print(f"\n--- {dt.strftime('%Y-%m-%d %H:00')} 시점 데이터 수집 ---")
        air_data = get_air_data(dt)
        weather_data = get_weather_data(dt)
        traffic_data = get_traffic_data(dt)

        air_data = air_data if air_data is not None else {}
        weather_data = weather_data if weather_data is not None else {}
        traffic_data = traffic_data if traffic_data is not None else {}
        
        merged_data = {**air_data, **weather_data, **traffic_data}
        data_list.append(merged_data)

    df_3rows = pd.DataFrame(data_list, index=['t-2', 't-1', 't'])
    processed_df = preprocess_data(df_3rows)

    if processed_df is None:
        print("[예측 실패] 데이터 전처리에 실패했습니다.")
        return

    final_row = flatten_data(processed_df, BASE_FEATURES)
    
    if final_row is None:
        print("[예측 실패] 모델 입력 데이터 생성에 실패했습니다.")
        return
    
    print("\n--- [최종 예측 결과 ] ---")
    
    try:
        pred_t1 = MODELS['PM25_t_plus_1'].predict(final_row)[0]
        pred_t2 = MODELS['PM25_t_plus_2'].predict(final_row)[0]
        pred_t3 = MODELS['PM25_t_plus_3'].predict(final_row)[0]
        
        print(f"  -> 1시간 뒤 예측 농도: {pred_t1:.2f} µg/m³")
        print(f"  -> 2시간 뒤 예측 농도: {pred_t2:.2f} µg/m³")
        print(f"  -> 3시간 뒤 예측 농도: {pred_t3:.2f} µg/m³")
        
 
        print("\n--- [최종 환기 추천 ] ---")
        recommendation_message = get_recommendation(
            pred_t1, pred_t2, pred_t3, 
            dt_now
        )
        print(recommendation_message)
        
    except Exception as e:
        print(f"[예측/추천 오류] {e}")
        
# --------------------------------------------------------
       
if __name__ == "__main__":
    run_prediction_pipeline()