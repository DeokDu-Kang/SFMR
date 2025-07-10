import numpy as np
import os
import pandas as pd
from datetime import datetime, timedelta
import math
from geopy.distance import geodesic # 위/경도 간 거리 계산을 위해 추가

# 좌표계 변환을 위한 변환 행렬 및 함수
from rasterio.transform import Affine, xy
from rasterio.warp import calculate_default_transform, reproject, Resampling
from pyproj import Transformer

# Cartopy 데이터 캐시 경로 설정 (이 스크립트에서는 직접 사용되지 않음)
CARTOPY_DATA_DIR = r"D:\Python\cartopy_data"
os.environ["CARTOPY_DIR"] = CARTOPY_DATA_DIR

def process_radar_file(file_path, radar_time_str, nx, ny):
    """
    주어진 레이더 파일에서 강우 강도 데이터를 추출하고 관련 좌표 정보를 반환합니다.

    Args:
        file_path (str): 레이더 .bin 파일의 전체 경로.
        radar_time_str (str): 레이더 데이터의 시간 (YYYYMMDDHHmm 형식).
        nx (int): 레이더 데이터의 x 해상도.
        ny (int): 레이더 데이터의 y 해상도.

    Returns:
        tuple: (rain_array, source_transform, source_crs, radar_datetime_obj)
               - rain_array (np.ndarray): 강우 강도 데이터 (mm/h).
               - source_transform (Affine): LCC 좌표계 변환 행렬.
               - source_crs (str): LCC 좌표계 정의 문자열.
               - radar_datetime_obj (datetime): 레이더 데이터의 datetime 객체.
    """
    with open(file_path, 'rb') as f:
        raw = f.read()

    raw_data = raw[4:] # 앞 4바이트 제거
    data_all = np.frombuffer(raw_data, dtype=np.int16).astype(np.float32)
    data = data_all.reshape((ny, nx))

    null_mask = (data <= -30000) # 관측 반경 밖의 영역 마스킹
    data[null_mask] = np.nan
    data /= 100 # 데이터 값 조정

    # ZR 관계식 변환 상수
    ZRa = 148.0
    ZRb = 1.56

    za = 0.1 / ZRb
    zb = np.log10(ZRa) / ZRb

    def dbz_rain_conv(data, mode=0):
        data = np.asarray(data, dtype=np.float32)
        result = np.full_like(data, np.nan)

        if mode == 0:  # dBZ → mm/h
            mask = ~np.isnan(data)
            intermediate = data[mask] * za - zb
            result[mask] = np.power(10.0, intermediate)
        elif mode == 1:  # mm/h → dBZ
            mask = (data > 0) & ~np.isnan(data)
            result[mask] = 10.0 * np.log10(ZRa * np.power(data[mask], ZRb))
        
        return result

    rain_array = dbz_rain_conv(data, mode=0)

    # 레이더 자료의 너비, 높이, 중심점 좌표, 공간 해상도 정의 (미터 단위)
    source_width = nx
    source_height = ny
    source_center_x = 1121
    source_center_y = 1681
    source_resolution = 500

    # 변환 전 좌표계 (LCC) 정의
    source_crs = "+proj=lcc +lat_1=30 +lat_2=60 +lat_0=38 +lon_0=126 +x_0=0 +y_0=0 +ellps=WGS84 +units=m +no_defs"

    # 이미지(배열)의 행과 열이 LCC 좌표계에서의 좌표로 변환되기 위한 변환 행렬 정의
    source_transform = Affine.scale(source_resolution, source_resolution) * Affine.translation(-source_center_x, -source_center_y)

    radar_datetime_obj = datetime.strptime(radar_time_str, '%Y%m%d%H%M')

    return rain_array, source_transform, source_crs, radar_datetime_obj

# --- 메인 스크립트 시작 ---

dir_SFMR = r"C:\Users\user\Downloads\Result_SW01(only correlation coefficient)"
dir_radar = r"C:\Users\user\Downloads\Radar"

files_SFMR = [f for f in os.listdir(dir_SFMR) if f.endswith('.csv')]
files_radar = [f for f in os.listdir(dir_radar) if f.endswith('.bin')] # or f.startswith('nph-rdr_cmp1_api')]

# 레이더 데이터 처리 및 저장
radar_data_map = {} # key: datetime 객체, value: (rain_array, source_transform, source_crs)
nx, ny = 2305, 2881 # 레이더 해상도 고정

# 위·경도 <-> LCC 좌표계 변환을 위한 Transformer는 하나만 생성하고 재사용합니다.
# source_crs가 모든 레이더 파일에서 동일하다고 가정합니다.
common_source_crs = "+proj=lcc +lat_1=30 +lat_2=60 +lat_0=38 +lon_0=126 +x_0=0 +y_0=0 +ellps=WGS84 +units=m +no_defs"
transformer_4326_to_lcc = Transformer.from_crs('EPSG:4326', common_source_crs, always_xy=True) # (lon, lat) 순서로 입출력
transformer_lcc_to_4326 = Transformer.from_crs(common_source_crs, 'EPSG:4326', always_xy=True) # (lon, lat) 순서로 입출력

print("--- 레이더 파일 처리 중 ---")
for filename in files_radar:
    # 파일명에서 시간 정보 추출 (예: 'rdr_202104231300.bin' 또는 'nph-rdr_cmp1_api')
    if filename.startswith('rdr_') and filename.endswith('.bin'):
        radar_time_str = filename[4:16] # 'rdr_' 뒤 12자리 (YYYYMMDDHHmm)
    elif filename == 'nph-rdr_cmp1_api':
        radar_time_str = '202104231300' # 고정된 파일명일 경우, 코드 상단에 정의된 시간을 사용
    else:
        print(f"경고: 알 수 없는 레이더 파일명 형식: {filename}. 건너뜁니다.")
        continue
    
    full_radar_path = os.path.join(dir_radar, filename)
    
    try:
        rain_array, source_transform, source_crs, radar_datetime_obj = process_radar_file(full_radar_path, radar_time_str, nx, ny)
        radar_data_map[radar_datetime_obj] = (rain_array, source_transform, source_crs)
        print(f"레이더 파일 '{filename}' 처리 완료. 시간: {radar_datetime_obj}")
    except Exception as e:
        print(f"레이더 파일 '{filename}' 처리 중 오류 발생: {e}. 건너뜁니다.")

print("\n--- SFMR 파일 및 매칭 처리 중 ---")

# 위도/경도 매칭을 위한 거리 임계값 (미터)
# 레이더 해상도와 SFMR 위치 정확도를 고려하여 설정
DISTANCE_THRESHOLD_M = 500 # 500미터 이내 (레이더 격자 하나의 대략적인 크기)

for filename_sfmr in files_SFMR:
    full_sfmr_path = os.path.join(dir_SFMR, filename_sfmr)
    
    # 각 SFMR 파일에 대한 매칭된 데이터를 저장할 리스트
    matched_rows_for_sfmr = []    
    
    # # 각 SFMR 파일에 대한 결과 DataFrame 초기화
    # comparison_df_for_sfmr = pd.DataFrame(columns=[
    #     'SFMR_Datetime', 'SFMR_Lat', 'SFMR_Lon', 'SFMR_Rain_Rate',
    #     'Radar_Datetime', 'Radar_Lat', 'Radar_Lon', 'Radar_Rain_Rate'
    # ])

    try:
        df_sfmr = pd.read_csv(full_sfmr_path, header=1, sep='\t')
        df_sfmr = df_sfmr.bfill() #df_sfmr.fillna(method='backfill')  << warning 떠서 변경
        df_sfmr['Datetime'] = df_sfmr['Datetime'].apply(lambda x: datetime.strptime(x, '%Y-%m-%d %H:%M:%S'))
    except Exception as e:
        print(f"SFMR 파일 '{filename_sfmr}' 읽기 오류 또는 형식 불일치: {e}. 건너뜁니다.")
        continue

    print(f"SFMR 파일 '{filename_sfmr}' 처리 중...")

    for index, row in df_sfmr.iterrows():
        sfmr_lat = row['Lat']
        sfmr_lon = row['Lon']
        sfmr_datetime = row['Datetime']
        
        # df_sfmr.columns를 출력하여 정확한 컬럼명을 확인하세요.
        try:
            sfmr_rain_rate = row['Rain']
        except KeyError:
            print(f"경고: SFMR 파일 '{filename_sfmr}'에 'Rain' 컬럼이 없습니다. 확인해주세요.")
            sfmr_rain_rate = np.nan # 컬럼이 없으면 NaN으로 처리

        matched_radar_rain_rate = np.nan
        matched_radar_datetime = None
        matched_radar_lat = np.nan
        matched_radar_lon = np.nan
        actual_distance_km = np.nan # 실제 거리(km) 초기화

        # 1. 시간 매칭: 가장 가까운 레이더 데이터 찾기
        min_time_diff = timedelta(days=9999) # 매우 큰 값으로 초기화
        closest_radar_datetime = None

        for radar_dt in radar_data_map.keys():
            time_diff = abs(sfmr_datetime - radar_dt)
            if time_diff < min_time_diff:
                min_time_diff = time_diff
                closest_radar_datetime = radar_dt
        
        # 2. 시간 매칭 조건 강화 (1초 이내)
        if closest_radar_datetime is not None and min_time_diff <= timedelta(seconds=0):
            matched_rain_array, matched_source_transform, matched_source_crs = radar_data_map[closest_radar_datetime]
            matched_radar_datetime = closest_radar_datetime

        # 3. SFMR 위경도 -> 레이더 LCC 좌표계 -> 레이더 배열 인덱스 변환
            try:
                # transformer_4326_to_lcc는 (lon, lat)을 입력받아 (x, y)를 반환
                # SFMR_Lat, SFMR_Lon 순서가 일반적이므로, (sfmr_lon, sfmr_lat)으로 변환
                lcc_x, lcc_y = transformer_4326_to_lcc.transform(sfmr_lon, sfmr_lat)

                # LCC 좌표계에서 레이더 배열 인덱스로 변환
                # Affine.__invert__().__mul__()은 (x_coord, y_coord)를 받아서 (col_index, row_index)를 반환
                index_col_float, index_row_float = matched_source_transform.__invert__().__mul__((lcc_x, lcc_y))
                
                index_row = math.floor(index_row_float) # 변경
                index_col = math.floor(index_col_float) # 변경

        # 4. 인덱스가 유효 범위 내에 있는지 확인
                if 0 <= index_row < matched_rain_array.shape[0] and \
                   0 <= index_col < matched_rain_array.shape[1]:
                    
                    #matched_radar_rain_rate = matched_rain_array[index_row, index_col]

        # 5. 레이더 격자 중심의 위경도 계산
                    # 해당 격자 인덱스 (index_col, index_row)의 LCC 좌표 중심 계산
                    radar_lcc_x_center, radar_lcc_y_center = matched_source_transform * (index_col + 0.5, index_row + 0.5)
                                        
                    # LCC 좌표를 위경도로 다시 변환 (transformer_lcc_to_4326는 (x, y)를 받아 (lon, lat) 반환)
                    #matched_radar_lon, matched_radar_lat = transformer_lcc_to_4326.transform(radar_lcc_x_center, radar_lcc_y_center)
                    temp_radar_lon, temp_radar_lat = transformer_lcc_to_4326.transform(radar_lcc_x_center, radar_lcc_y_center)

        # 6. 위치 매칭 조건 강화 (거리 임계값 이내)
                    # SFMR 위치와 레이더 격자 중심 간의 거리 계산
                    #distance = geodesic((sfmr_lat, sfmr_lon), (temp_radar_lat, temp_radar_lon)).meters
                    distance_meters = geodesic((sfmr_lat, sfmr_lon), (temp_radar_lat, temp_radar_lon)).meters

                    if distance_meters <= DISTANCE_THRESHOLD_M:
                        matched_radar_rain_rate = matched_rain_array[index_row, index_col]
                        matched_radar_lat = temp_radar_lat
                        matched_radar_lon = temp_radar_lon
                        actual_distance_km = distance_meters / 1000 # km 단위로 변환
                    else:
                        # 거리가 임계값을 초과하면 매칭 실패
                        matched_radar_rain_rate = np.nan
                        actual_distance_km = np.nan # 매칭 실패 시 거리도 NaN
                        # print(f"위치 불일치: SFMR({sfmr_lat:.4f},{sfmr_lon:.4f}) vs Radar({temp_radar_lat:.4f},{temp_radar_lon:.4f}) 거리: {distance_meters:.2f}m")

                else:
                    matched_radar_rain_rate = np.nan # 유효 범위 밖이면 NaN
                    actual_distance_km = np.nan # 유효 범위 밖이면 거리도 NaN            
            except Exception as e:
                # print(f"좌표 변환 또는 인덱싱 오류: {e} for SFMR point ({sfmr_lat}, {sfmr_lon})")
                matched_radar_rain_rate = np.nan
                actual_distance_km = np.nan
        
        # 7. SFMR과 Radar 자료 모두 존재하는 line만 csv에 저장
        if not pd.isna(matched_radar_rain_rate):
            # Radar_Rain_Rate가 매우 작은 값 (예: e-18)일 경우 0으로 처리 (선택 사항)
            if abs(matched_radar_rain_rate) < 1e-10: # 0에 매우 가까운 값은 0으로 간주
                matched_radar_rain_rate = 0.0

            matched_rows_for_sfmr.append({
                'SFMR_Datetime': sfmr_datetime,
                'SFMR_Lat': sfmr_lat,
                'SFMR_Lon': sfmr_lon,
                'SFMR_Rain_Rate': sfmr_rain_rate,
                'Radar_Datetime': matched_radar_datetime,
                'Radar_Lat': matched_radar_lat,
                'Radar_Lon': matched_radar_lon,
                'Radar_Rain_Rate': matched_radar_rain_rate,
                'Distance_km': actual_distance_km # 새롭게 추가된 거리 컬럼
            })
            #comparison_df_for_sfmr = pd.concat([comparison_df_for_sfmr, new_row], ignore_index=True)

    # 각 SFMR 파일에 대한 결과 CSV 저장
#    if not comparison_df_for_sfmr.empty:
    if matched_rows_for_sfmr: # 리스트가 비어있지 않으면 DataFrame 생성 및 저장
        comparison_df_for_sfmr = pd.DataFrame(matched_rows_for_sfmr)
        # 파일명에서 확장자를 제거하고 _Radar_Comparison_StrictMatch.csv 추가
        # 파일명에서 확장자를 제거하고 _Radar_Comparison.csv 추가
        output_filename = os.path.splitext(filename_sfmr)[0] + '_Radar_Comparison_StrictMatch.csv'
        output_csv_path = os.path.join(dir_SFMR, output_filename) # SFMR 디렉토리에 저장
        
        # float_format='%.4f'를 사용하여 소수점 4자리로 제한
        comparison_df_for_sfmr.to_csv(output_csv_path, index=False, encoding='utf-8-sig', float_format='%.4f')
        print(f"SFMR 파일 '{filename_sfmr}'에 대한 엄격 매칭 결과가 다음 위치에 저장되었습니다: {output_csv_path}")
    else:
        print(f"SFMR 파일 '{filename_sfmr}'에 대해 엄격 매칭되는 레이더 데이터가 없어 저장할 파일이 없습니다.")

print("\n--- 모든 SFMR 파일에 대한 매칭 및 저장 완료 ---")