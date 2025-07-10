import numpy as np
import os

import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm, Normalize
from mpl_toolkits.axes_grid1 import make_axes_locatable

# 좌표계 변환을 위한 변환 행렬 및 함수
from rasterio.transform import Affine
from rasterio.warp import calculate_default_transform, reproject, Resampling
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import cartopy.io.shapereader as shpreader
from pyproj import Transformer

# Cartopy 데이터 캐시 경로 설정
CARTOPY_DATA_DIR = r"D:\Python\cartopy_data"
os.environ["CARTOPY_DIR"] = CARTOPY_DATA_DIR

file_path = 'rdr_202105201420.bin' #'nph-rdr_cmp1_api'
time = '202105201420' #'202104231300'
nx, ny = 2305, 2881

with open(file_path, 'rb') as f:
    raw = f.read()

# 앞 4바이트 제거
raw_data = raw[4:]

# 데이터 변환 및 형태 변경
data_all = np.frombuffer(raw_data, dtype=np.int16).astype(np.float32)
data = data_all.reshape((ny, nx))

# 중앙값 출력
cx, cy = nx // 2, ny // 2
print(f"중앙값: {data[cy, cx]}")

# 관측 반경 밖의 영역 마스킹 및 NaN 처리
null_mask = (data <= -30000)
data[null_mask] = np.nan

# 데이터 값 조정
data /= 100

# 배열 형태 확인
print(data.shape)

# ZR 관계식 변환 상수
ZRa = 148.0
ZRb = 1.56

za = 0.1 / ZRb
zb = np.log10(ZRa) / ZRb

def dbz_rain_conv(data, mode=0):
    """
    dBZ와 강수량(mm/h) 간의 ZR 관계 변환을 수행합니다.

    Args:
        data (np.ndarray): 입력 레이더 데이터 (dBZ 또는 mm/h).
        mode (int): 0은 dBZ를 mm/h로, 1은 mm/h를 dBZ로 변환합니다.

    Returns:
        np.ndarray: 변환된 강수량 데이터.
    """
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

##################################################################################################
# 강수량 색상표 정의
colormap_rain = ListedColormap(np.array([
    [250, 250, 250], [0, 200, 255], [0, 155, 245], [0, 74, 245],
    [0, 255, 0], [0, 190, 0], [0, 140, 0], [0, 90, 0],
    [255, 255, 0], [255, 220, 31], [249, 205, 0], [224, 185, 0], [204, 170, 0],
    [255, 102, 0], [255, 50, 0], [210, 0, 0], [180, 0, 0],
    [224, 169, 255], [201, 105, 255], [179, 41, 255], [147, 0, 228],
    [179, 180, 222], [76, 78, 177], [0, 3, 144], [51, 51, 51]
]) / 255)

colormap_rain.set_bad([0, 0, 0, 0]) # NaN 값은 투명하게 설정

# 색상표 경계 정의
bounds = np.array([
    0, 0.1, 0.5, 1,
    2, 3, 4, 5,
    6, 7, 8, 9, 10,
    15, 20, 25, 30,
    40, 50, 60, 70,
    90, 110, 150
])

norm = BoundaryNorm(boundaries=bounds, ncolors=len(colormap_rain.colors))

# 강수량 배열을 색상으로 변환
colored_array = BoundaryNorm(
    boundaries=bounds,
    ncolors=len(colormap_rain.colors)
)(rain_array)
colored_array = Normalize(
    0, len(colormap_rain.colors)
)(colored_array)
colored_array[null_mask] = np.nan
colored_array = (colormap_rain(colored_array) * 255).astype(np.uint8)

ticks = bounds[:]

# # 색상 배열 플롯
# fig, ax = plt.subplots(1, 1, figsize=(5, 5))
# ax.set_title('Colored array')
# ax.set_facecolor('#cccccc')

# divider = make_axes_locatable(ax)
# cax = divider.append_axes('right', size='5%', pad=0)

# im = ax.imshow(colored_array, origin='lower', cmap=colormap_rain, norm=norm)

# cbar = fig.colorbar(im, cax=cax, ticks=ticks)
# cbar.ax.tick_params(labelsize=8)
# cbar.ax.set_title('mm/h', fontsize=8)

###########################################################################################
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

# 변환 행렬을 거친 이미지가 나타내는 경계 정의
source_bounds = {
    'left': -source_center_x * source_resolution,
    'top': (source_height - source_center_y) * source_resolution,
    'right': (source_width - source_center_x) * source_resolution,
    'bottom': -source_center_y * source_resolution
}

# 변환 후 이미지의 변환 행렬, 너비, 높이 계산 (EPSG:3857)
dest_transform, dest_width, dest_height = calculate_default_transform(
    src_crs=source_crs,
    dst_crs='EPSG:3857',
    width=source_width,
    height=source_height,
    **source_bounds,
)

# 변환된 이미지를 담을 빈 배열 정의
converted_array = np.ones((dest_height, dest_width, 4), dtype=np.uint8)

# RGBA 각 채널에 대해 좌표계 변환 수행 (nearest resampling)
for i in range(4):
    reproject(
        source= colored_array[:, :, i],
        destination=converted_array[:, :, i],
        src_transform=source_transform,
        src_crs=source_crs,
        dst_transform=dest_transform,
        dst_crs='EPSG:3857',
        resampling=Resampling.nearest,
    )

# # 변환된 이미지 플롯
# fig, ax = plt.subplots(1, 1, figsize=(5, 5))
# ax.set_title('Converted array')
# ax.set_facecolor('#cccccc')

# divider = make_axes_locatable(ax)
# cax = divider.append_axes('right', size='5%', pad=0)

# im = ax.imshow(converted_array, cmap=colormap_rain, norm=norm)

# cbar = fig.colorbar(im, cax=cax, ticks=ticks)
# cbar.ax.tick_params(labelsize=8)
# cbar.ax.set_title('mm/h', fontsize=8)

###########################################################################################
# 지도 위에 레이더 이미지 표출
fig = plt.figure(figsize=(10, 10))
ax = fig.add_subplot(1, 1, 1, projection=ccrs.epsg(3857))

# 지형 피처 zorder (레이더보다 낮게 설정)
feature_zorder = 0.5
radar_zorder = 2.0

# 1. 해안선 데이터 로드 및 추가
coastline_path = os.path.join(CARTOPY_DATA_DIR, 'shapefiles', 'natural_earth', 'physical', 'ne_50m_coastline', 'ne_50m_coastline.shp')
try:
    coast_shp = shpreader.Reader(coastline_path)
    ax.add_geometries(coast_shp.geometries(), ccrs.PlateCarree(),
                    edgecolor='black', facecolor='none', linewidth=0.8, zorder=feature_zorder)
except Exception as e:
    print(f"해안선 shapefile 로드 오류: {e}")
    print(f"예상 경로: {coastline_path}")

# 2. 국경선 데이터 로드 및 추가
borders_path = os.path.join(CARTOPY_DATA_DIR, 'shapefiles', 'natural_earth', 'cultural', 'ne_50m_admin_0_countries', 'ne_50m_admin_0_countries.shp')
try:
    borders_shp = shpreader.Reader(borders_path)
    ax.add_geometries(borders_shp.geometries(), ccrs.PlateCarree(),
                    edgecolor='black', facecolor='none', linestyle=':', linewidth=0.5, zorder=feature_zorder)
except Exception as e:
    print(f"국경선 shapefile 로드 오류: {e}")
    print(f"예상 경로: {borders_path}")

# 3. 육지 데이터 로드 및 추가
land_path = os.path.join(CARTOPY_DATA_DIR, 'shapefiles', 'natural_earth', 'physical', 'ne_50m_land', 'ne_50m_land.shp')
try:
    land_shp = shpreader.Reader(land_path)
    ax.add_geometries(land_shp.geometries(), ccrs.PlateCarree(),
                    facecolor='lightgray', edgecolor='black', linewidth=0.1, zorder=feature_zorder)
except Exception as e:
    print(f"육지 shapefile 로드 오류: {e}")
    print(f"예상 경로: {land_path}")

# 해양 배경색 설정
ax.set_facecolor('azure')

# EPSG:3857 좌표계로 변환된 데이터의 범위 가져오기
x_min, y_min = dest_transform * (0, dest_height)
x_max, y_max = dest_transform * (dest_width, 0)
extent_3857 = [x_min, x_max, y_min, y_max]

# 지도 범위 설정
ax.set_extent(extent_3857, crs=ccrs.epsg(3857))

# 변환된 RGBA 이미지를 지도에 오버레이
im = ax.imshow(converted_array, origin='upper', extent=extent_3857, transform=ccrs.epsg(3857),
                cmap=colormap_rain, norm=norm, alpha=0.8, zorder=radar_zorder)

plt.title("Radar Overlay on Map", fontsize=14)

# 레이더 데이터에 대한 컬러바 추가
divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="5%", pad=0.1, axes_class=plt.Axes)
cbar = fig.colorbar(im, cax=cax, ticks=ticks)
cbar.ax.tick_params(labelsize=8)
cbar.ax.set_title('mm/h', fontsize=8)

plt.show()

##########################################################################################
# 지도 상 특정 지점의 강수세기 가져오기

# 위·경도 좌표계에서 LCC 좌표계로 변환하는 transformer 정의
lcc_transformer = Transformer.from_crs('EPSG:4326', source_crs)

# 원하는 지점의 위·경도 입력
target_lat = 35.0400
target_lon = 124.4100

# 위·경도 값을 LCC 좌표계로 변환하고 배열 인덱스 계산
index_col, index_row = source_transform.__invert__().__mul__(
    lcc_transformer.transform(target_lat, target_lon)
)

print(index_col, index_row)

# 지점의 강수세기 출력
print(
    f"위도 {target_lat}, 경도 {target_lon} 지점의 {time} 강수 세기는 "
    f"{rain_array[round(index_row), round(index_col)]:.02f}mm/h 입니다."
)
