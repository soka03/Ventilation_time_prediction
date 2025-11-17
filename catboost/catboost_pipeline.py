import requests
import pandas as pd
import numpy as np
import xml.etree.ElementTree as ET
from datetime import datetime, timedelta
import joblib
import os 
import warnings
warnings.filterwarnings('ignore', category=pd.errors.PerformanceWarning)

# ---------------------------------------------

AIRKOREA_API_KEY = "72634d516d736f6b38306f7a6c4e6d" 
KMA_API_KEY = "seq0BXT_R2GqtAV0__dh7Q" 
SEOUL_TRAFFIC_API_KEY = "5479676464736f6b313035586b706147" 

ARTIFACT_DIR = './Ventilation_time_prediction/CatBoost/artifacts/' 
MODELS_PATH = os.path.join(ARTIFACT_DIR, "models_catboost.pkl") 
IMPUTER_PATH = os.path.join(ARTIFACT_DIR, "imputer_median.pkl")
FEATURES_PATH = os.path.join(ARTIFACT_DIR, "feature_cols.pkl")
THRESHOLD_PATH = os.path.join(ARTIFACT_DIR, "threshold.pkl")


# ---------------------------------------------
# API 요청 

KEY_MAP_AIRKOREA = {
    'SO2': 'SPDX',
    'CO': 'CBMX',
    'O3': 'OZON',
    'NO2': 'NTDX', 
    'PM10': 'PM',
    'PM25': 'FPM'
}

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


TARGETS = ["PM25_t_plus_1","PM25_t_plus_2","PM25_t_plus_3"]

def clip_range(s, low=None, high=None):
    s = pd.to_numeric(s, errors="coerce").copy()
    if low is not None:
        s[s < low] = low 
    if high is not None:
        s[s > high] = high
    return s

def create_features(df):
    df = df.copy()

    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df = df.dropna(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)

    if "HM" in df.columns:
        df["HM"] = clip_range(df["HM"], 0, 100)
    for col in ["PM25","PM10","SO2","NO2","O3","CO","WS","RN"]:
        if col in df.columns:
            df[col] = clip_range(df[col], 0, None)
            
    df['WD_sin'] = 0.0
    df['WD_cos'] = 0.0
    if 'WD_raw' in df.columns:
        valid_wind_mask = df['WD_raw'].notna() & (df['WD_raw'] > 0)
        if valid_wind_mask.any():
            degrees = df.loc[valid_wind_mask, 'WD_raw'] * 10
            degrees = degrees.replace(360, 0)
            radians = np.deg2rad(degrees)
            df.loc[valid_wind_mask, 'WD_sin'] = np.sin(radians)
            df.loc[valid_wind_mask, 'WD_cos'] = np.cos(radians)
        df = df.drop(columns=['WD_raw']) 
    
    if 'Traffic' not in df.columns:
        df['Traffic'] = np.nan

    candidates = [c for c in df.columns if c not in ["timestamp"] + TARGETS]
    for c in candidates:
        if df[c].dtype == "O":
            df[c] = pd.to_numeric(df[c], errors="coerce")
    
    num_cols = [c for c in candidates if np.issubdtype(df[c].dtype, np.number)]
    
    df = df.set_index("timestamp")
    if not df.index.is_monotonic_increasing:
        df = df.sort_index()
    
    df[num_cols] = df[num_cols].interpolate(method="time", limit_direction="both")
    
    for c in num_cols:
        df[c] = df[c].fillna(df[c].rolling(3, min_periods=1).median())
        
    if 'RN' in df.columns:
        df['RN'] = df['RN'].fillna(0)

    df = df.reset_index()

    df["hour"] = df["timestamp"].dt.hour
    df["dow"] = df["timestamp"].dt.dayofweek
    df["month"] = df["timestamp"].dt.month
    
    df["is_weekend"] = (df["dow"] > 5).astype(int) 
    df["is_night"] = ((df["hour"] <= 6) | (df["hour"] >= 22)).astype(int)

    base = [c for c in ["PM25","PM10","NO2","O3","SO2","WS","TA","HM"] if c in df.columns]

    df = df.sort_values("timestamp").reset_index(drop=True)

    for col in base:
        df[f"{col}_roll3_mean"] = df[col].rolling(3, min_periods=1).mean()
        df[f"{col}_roll3_std"]  = df[col].rolling(3, min_periods=2).std()
        df[f"{col}_roll6_mean"] = df[col].rolling(6, min_periods=1).mean()
        df[f"{col}_roll6_std"]  = df[col].rolling(6, min_periods=2).std()
        for k in [1,2,3,4,5,6]:
            df[f"{col}_lag{k}"] = df[col].shift(k)

    if set(["WS","WD_sin","WD_cos"]).issubset(df.columns):
        df["WS_sin"] = df["WS"] * df["WD_sin"]
        df["WS_cos"] = df["WS"] * df["WD_cos"]
    if set(["TA","HM"]).issubset(df.columns):
        df["TAxHM"] = df["TA"] * df["HM"]
        
    LEGACY_BASE_FEATURES = [
        'SO2', 'CO', 'O3', 'NO2', 'PM10', 'PM25', 
        'WS', 'PS', 'TA', 'HM', 'RN', 'VS', 'Traffic', 
        'WD_sin', 'WD_cos'
    ]
    
    print("\n[피처 생성] 훈련/예측 호환성을 위해 t-1, t-2 피처 생성 중...")
    
    for col in LEGACY_BASE_FEATURES:
        if col in df.columns:
            df[f"{col}_t_minus_1"] = df[col].shift(1)
            df[f"{col}_t_minus_2"] = df[col].shift(2)
        else:
            print(f"[경고] t-1/t-2 생성에 필요한 원본 컬럼이 없습니다: {col}")
            df[f"{col}_t_minus_1"] = np.nan
            df[f"{col}_t_minus_2"] = np.nan
            
    df = df.copy()
    
    return df

# ------------------------------------------------------
# 환기 추천 파이프라인

def get_status(value, thresh_mod, thresh_good=15, thresh_bad=75):
    if value <= thresh_good:
        return "좋음 (Good)"
    if value <= thresh_mod: 
        return "보통 (Moderate)"
    if value <= thresh_bad:
        return "나쁨 (Bad)"
    return "매우 나쁨 (Very Bad)"


def get_recommendation(pred_t1, pred_t2, pred_t3, logical_start_time, threshold): 
    predictions = [
        {'hour_offset': 1, 'value': pred_t1, 'status': get_status(pred_t1, threshold)},
        {'hour_offset': 2, 'value': pred_t2, 'status': get_status(pred_t2, threshold)},
        {'hour_offset': 3, 'value': pred_t3, 'status': get_status(pred_t3, threshold)},
    ]
    
    acceptable_hours = [p for p in predictions if p['value'] <= threshold]
    
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
        least_bad_hour = min(predictions, key=lambda x: x['value'])
        message = (
            f" 환기 보류: 향후 3시간 동안 미세먼지 농도가 '나쁨' 이상('보통' 기준 초과)일 것으로 예측됩니다.\n"
            f"  (참고: 3시간 중 가장 낮은 예측 농도는 {least_bad_hour['value']:.2f} µg/m³ [{least_bad_hour['status']}]입니다.)"
        )
        return message

def run_prediction_pipeline():
    print("훈련된 아티팩트 4개 불러오는 중...")
    MODELS = {} 
    IMPUTER = None
    FINAL_MODEL_COLUMNS = []
    THRESHOLD_MODERATE = 35.0 
    
    try:
        MODELS = joblib.load(MODELS_PATH) 
        IMPUTER = joblib.load(IMPUTER_PATH) 
        FINAL_MODEL_COLUMNS = joblib.load(FEATURES_PATH) 
        THRESHOLD_MODERATE = joblib.load(THRESHOLD_PATH) 
        
        print(f"[{MODELS_PATH}] 모델 딕셔너리 로드 성공")
        print(f"[{IMPUTER_PATH}] 중앙값 Imputer 로드 성공")
        print(f"[{FEATURES_PATH}] 피처 리스트 로드 성공")
        print(f"[{THRESHOLD_PATH}] 환기 기준값 로드 성공 ({THRESHOLD_MODERATE} µg/m³)")

        assert all(k in MODELS for k in ['PM25_t_plus_1', 'PM25_t_plus_2', 'PM25_t_plus_3'])
        
    except FileNotFoundError as e:
        print(f"[오류] 아티팩트 파일을 찾을 수 없습니다: {e}")
        print(f"확인된 경로: {os.path.abspath(ARTIFACT_DIR)}")
        return
    except Exception as e:
        print(f"[오류] 아티팩트 로드 중 오류 발생: {e}")
        return

    dt_now = datetime.now()
    
    print("\n현재 시각:", dt_now.strftime('%Y-%m-%d %H:%M'))
    print("\n안정적인 결측치 처리를 위해 10시간 치 데이터 수집 시작...")
    
    data_list = []
    
    for i in range(9, -1, -1): 
        dt = dt_now - timedelta(hours=i)
        print(f"\n--- {dt.strftime('%Y-%m-%d %H:00')} 시점 데이터 수집 ---")
        
        air_data = get_air_data(dt)
        weather_data = get_weather_data(dt)
        traffic_data = get_traffic_data(dt)

        air_data = air_data if air_data is not None else {}
        weather_data = weather_data if weather_data is not None else {}
        traffic_data = traffic_data if traffic_data is not None else {}
        
        merged_data = {**air_data, **weather_data, **traffic_data}
        merged_data['timestamp'] = dt.replace(minute=0, second=0, microsecond=0)
        data_list.append(merged_data)

    df_raw = pd.DataFrame(data_list)
    
    if len(df_raw) < 7: 
         print(f"[예측 실패] 피처 엔지니어링에 필요한 최소 데이터 (7시간)를 수집하지 못했습니다. (수집된 데이터: {len(df_raw)}시간)")
         return

    print("\n데이터 전처리 및 피처 엔지니어링 시작...")
    try:
        df_featured = create_features(df_raw)
    except Exception as e:
        print(f"[예측 실패] 피처 엔지니어링 중 오류 발생: {e}")
        return

    try:
        missing_cols = [c for c in FINAL_MODEL_COLUMNS if c not in df_featured.columns]
        if missing_cols:
            print(f"[예측 실패] 피처 엔지니어링 결과에 필수 피처가 누락되었습니다: {missing_cols[:5]}...")
            return
            
        X_predict_raw = df_featured[FINAL_MODEL_COLUMNS].iloc[-1:] 

        if X_predict_raw.empty:
            print("[예측 실패] 피처 엔지니어링 후 예측할 데이터가 없습니다.")
            return
            
        print("중앙값 Imputer 적용 중...")
        X_predict_imputed = IMPUTER.transform(X_predict_raw)
        
        final_row = pd.DataFrame(X_predict_imputed, columns=FINAL_MODEL_COLUMNS, index=X_predict_raw.index)

    except Exception as e:
        print(f"[예측 실패] Imputation 또는 데이터 선별 중 오류: {e}")
        return

    print("\n--- [최종 예측 결과 ] ---")
    
    try:
        pred_t1 = MODELS['PM25_t_plus_1'].predict(final_row)[0]
        pred_t2 = MODELS['PM25_t_plus_2'].predict(final_row)[0]
        pred_t3 = MODELS['PM25_t_plus_3'].predict(final_row)[0]
        
        print(f"  -> 1시간 뒤 예측 농도: {pred_t1:.2f} µg/m³")
        print(f"  -> 2시간 뒤 예측 농도: {pred_t2:.2f} µg/m³")
        print(f"  -> 3시간 뒤 예측 농도: {pred_t3:.2f} µg/m³")
        
    except Exception as e:
        print(f"[예측 오류] 모델 예측 중 오류 발생: {e}")
        return
        
    print("\n--- [최종 환기 추천 ] ---")
    try:
        recommendation_message = get_recommendation(
            pred_t1, pred_t2, pred_t3, 
            dt_now,
            THRESHOLD_MODERATE 
        )
        print(recommendation_message)
        
    except Exception as e:
        print(f"[추천 오류] {e}")
        
# --------------------------------------------------------
        
if __name__ == "__main__":
    run_prediction_pipeline()