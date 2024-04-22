#!/usr/bin/env python
# coding: utf-8

import numpy as np
import pygrib
import sys
import datetime
from datetime import timedelta
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import math
import xarray as xr
import metpy.calc as mpcalc
from metpy.units import units
import matplotlib.path as mpath
from scipy.ndimage import maximum_filter, minimum_filter, gaussian_filter
import matplotlib as mpl
import scipy.ndimage as ndimage

file_nm_temp_s = 'anl_surf.{0:4d}{1:02d}{2:02d}{3:02d}'
file_nm_temp_p = 'anl_p_{0}.{1:4d}{2:02d}{3:02d}{4:02d}'
file_nm_temp_l = 'TL479_surf.grib2'
folder_nm_temp = './Data/{0:4d}{1:02d}{2:02d}/'

## 読み込み期間の最初の時刻（UTC）,読み込む時刻の数、時間間隔の指定
# コマンドライン引数から日時を取得
arg_datetime = sys.argv[1]

try:
    # 日時文字列を年月日時分に変換
    year = int(arg_datetime[:4])
    month = int(arg_datetime[4:6])
    day = int(arg_datetime[6:8])
    hour = int(arg_datetime[8:10]) 
    dt = datetime.datetime(year, month, day, hour, 0)
    
except ValueError:
    print("日時の形式が正しくありません。正しい形式は 'yyyymmddhh' です。")
    sys.exit(1)

print("Plotting:",dt)
i_year=dt.year
i_month=dt.month
i_day=dt.day
i_hourZ=dt.hour

## 読み込むGPVの範囲（緯度・経度で東西南北の境界）を指定
(latS, latN, lonW, lonE) = (0, 60, 80, 200)

## 読み込む要素の指定
elem_s_names = ['pt', 'sdwe', 'sp', 'prmsl', '2t', '2ttd', '2sh', '2r', '10u', '10v'] 
elems = ['hgt']

## データサイズを取得するために、GRIB2を読み込む
folder_nm = folder_nm_temp.format(i_year,i_month,i_day)
file_nm = folder_nm + file_nm_temp_s.format(i_year,i_month,i_day,i_hourZ)
grbs = pygrib.open(file_nm)

## 変数の名称や単位の情報を保存
# shortName,parameterName,parameterUnitsをListにsetする
all_elem_names = []
all_elem_s_names = []
all_elem_units = []
n = 1
for g in grbs:
    # 不明だが、不要なデータを扱わないためのif文
    if g['parameterUnits'] != g['parameterName']:
        #print(n,g['shortName'],":",g['parameterName'],g['parameterUnits'])
        all_elem_names.append(g['parameterName'])
        # T-Tdには shortNameがないため、'2ttd'とする
        if g['parameterName'] == "Dewpoint depression (or deficit)":
            all_elem_s_names.append('2ttd')
        else:
            all_elem_s_names.append(g['shortName'])
        all_elem_units.append(g['parameterUnits'])
        n = n + 1
all_e_size = len(all_elem_names)

## 要素のサイズ・情報を取得
e_size = len(elem_s_names)
elem_names = []
elem_units = []

for el in elem_s_names:
    n_ = all_elem_s_names.index(el)
    elem_names.append(all_elem_names[n_])
    elem_units.append(all_elem_units[n_])

## 空間のデータサイズを取得
vals_, lats_, lons_ = grbs[1].data(lat1=latS,lat2=latN,lon1=lonW,lon2=lonE)
(lat_size, lon_size) = vals_.shape
lats = lats_[:,0]
lons = lons_[0,:]

### 指定した全時刻を含む3次元データセットを作成
vals_ = np.zeros([e_size, lat_size, lon_size])

## pygrib open
folder_nm = folder_nm_temp.format(dt.year,dt.month,dt.day)
file_nm = folder_nm + file_nm_temp_s.format(dt.year,dt.month,dt.day,dt.hour)
grbs = pygrib.open(file_nm)
for g in grbs:
    # LongNameのList(elem_names)に、要素があれば代入する
    if g['parameterName'] in elem_names:
        i_elem = elem_names.index(g['parameterName'])
        vals_[i_elem], _, _ = g.data(lat1=latS,lat2=latN,lon1=lonW,lon2=lonE)

## 地表面ジオポテンシャル高度
grbs = pygrib.open(file_nm_temp_l)
surf = grbs[1].data(lat1=latS,lat2=latN,lon1=lonW,lon2=lonE)[0]

## Xarray Dataset 作成
dss = xr.Dataset(
    {
        elem_s_names[0]: (["lat", "lon"], vals_[0]  * units(elem_units[0])),
        elem_s_names[1]: (["lat", "lon"], vals_[1]  * units(elem_units[1])),
        elem_s_names[2]: (["lat", "lon"], vals_[2]  * units(elem_units[2])),
        elem_s_names[3]: (["lat", "lon"], vals_[3]  * units(elem_units[3])),
        elem_s_names[4]: (["lat", "lon"], vals_[4]  * units(elem_units[4])),
        elem_s_names[5]: (["lat", "lon"], vals_[5]  * units(elem_units[5])),
        elem_s_names[6]: (["lat", "lon"], vals_[6]  * units(elem_units[6])),
        elem_s_names[7]: (["lat", "lon"], vals_[7]  * units(elem_units[7])),
        elem_s_names[8]: (["lat", "lon"], vals_[8]  * units(elem_units[8])),
        elem_s_names[9]: (["lat", "lon"], vals_[9]  * units(elem_units[9])),
    },
    coords={
        "lat": lats,
        "lon": lons,
    },
)
dss[elem_s_names[0]].attrs['units'] = elem_units[0]
dss[elem_s_names[1]].attrs['units'] = elem_units[1]
dss[elem_s_names[2]].attrs['units'] = elem_units[2]
dss[elem_s_names[3]].attrs['units'] = elem_units[3]
dss[elem_s_names[4]].attrs['units'] = elem_units[4]
dss[elem_s_names[5]].attrs['units'] = elem_units[5]
dss[elem_s_names[6]].attrs['units'] = elem_units[6]
dss[elem_s_names[7]].attrs['units'] = elem_units[7]
dss[elem_s_names[8]].attrs['units'] = elem_units[8]
dss[elem_s_names[9]].attrs['units'] = elem_units[9]
dss['lat'].attrs['units'] = 'degrees_north'
dss['lon'].attrs['units'] = 'degrees_east'

dss = dss.metpy.parse_cf()

w = gaussian_filter(np.sqrt(dss['10u'].values ** 2 + dss['10v'].values ** 2), sigma=1)
surf = gaussian_filter(surf, sigma=1)

# wが5以下の場所のみフィルタリング
dss['prmsl'] = (["lat", "lon"], np.where(surf >= 8000, gaussian_filter(dss['prmsl'].values, sigma=4), dss['prmsl'].values) * units(elem_units[3]))
dss['prmsl'] = (["lat", "lon"], np.where(w <= 5, gaussian_filter(dss['prmsl'].values, sigma=2), dss['prmsl'].values) * units(elem_units[3]))
dss['prmsl'] = (["lat", "lon"], np.where(w <= 10, gaussian_filter(dss['prmsl'].values, sigma=1), dss['prmsl'].values) * units(elem_units[3]))
#dss['prmsl'] = (["lat", "lon"], np.where(w <= 15, gaussian_filter(dss['prmsl'].values, sigma=1), dss['prmsl'].values) * units(elem_units[3]))
#dss['prmsl'] = (["lat", "lon"], gaussian_filter(dss['prmsl'].values, sigma=1) * units(elem_units[3]))

## 読み込むの高度上限の指定：tagLpより下層の等圧面データをXarray Dataset化する
tagLp = 300

## データサイズを取得するために、GRIB2を読み込む
folder_nm = folder_nm_temp.format(i_year,i_month,i_day)
file_nm = folder_nm + file_nm_temp_p.format(elems[0],i_year,i_month,i_day,i_hourZ)
grbs = pygrib.open(file_nm)

## 要素数
e_size = len(elems)

## 高度のレベル数
grb_tag = grbs(level=lambda l:l >= tagLp)
levels = np.array([g['level'] for g in grb_tag])
l_size = len(levels)

## 空間のデータサイズ
vals_, lats_, lons_ = grb_tag[0].data(lat1=latS,lat2=latN,lon1=lonW,lon2=lonE)
(lat_size, lon_size) = vals_.shape
lats = lats_[:,0]
lons = lons_[0,:]

### 指定した全時刻を含む４次元データセットを作成
val4_ = np.zeros([e_size, l_size, lat_size, lon_size])
## 変数の名称や単位
elem_names = []
elem_units = []

## 要素のループ
folder_nm = folder_nm_temp.format(dt.year,dt.month,dt.day)
for i_elem, elem in enumerate(elems):
## pygrib open
    file_nm = folder_nm + file_nm_temp_p.format('hgt',dt.year,dt.month,dt.day,dt.hour)
    #print(elem," : ",file_nm)
    grbs = pygrib.open(file_nm)
    ## 処理する高度面の選択
    grb_tag = grbs(level=lambda l:l >= tagLp)
    ## 要素名や単位の取得
    elem_names.append(grb_tag[0].parameterName)
    elem_units.append(grb_tag[0].parameterUnits)
    ## レベルのループ
    for i_lev in range(l_size):
        val4_[i_elem][i_lev], _, _ = grb_tag[i_lev].data(lat1=latS,lat2=latN,lon1=lonW,lon2=lonE)

## Xarray Dataset 作成
ds4 = xr.Dataset(
    {
        'hgt': (["level","lat", "lon"], val4_[0]  * units(elem_units[0])),
    },
    coords={
        "level": levels,
        "lat": lats,
        "lon": lons,
    },
)
ds4['hgt'].attrs['units'] = elem_units[0]
ds4['level'].attrs['units'] = 'hPa'
ds4['lat'].attrs['units'] = 'degrees_north'
ds4['lon'].attrs['units'] = 'degrees_east'
ds4 = ds4.metpy.parse_cf()

ds4['hgt'] = (["level", "lat", "lon"], gaussian_filter(ds4['hgt'].values, sigma=1) * units(elem_units[0]))

## 緯度経度で指定したポイントの図上の座標などを取得する関数 transform_lonlat_to_figure() 
# 図法の座標 => pixel座標 => 図の座標　と3回の変換を行う
#  　pixel座標: plt.figureで指定した大きさxDPIに合わせ、左下を原点とするpixelで測った座標   
#  　図の座標: axesで指定した範囲を(0,1)x(0,1)とする座標
# 3つの座標を出力する
#    図の座標, Pixel座標, 図法の座標
def transform_lonlat_to_figure(lonlat, ax, proj):
    # lonlat:経度と緯度  (lon, lat) 
    # ax: Axes図の座標系    ex. fig.add_subplot()の戻り値
    # proj: axで指定した図法 
    #
    # 例 緯度経度をpointで与え、ステレオ図法る場合
    #    point = (140.0,35.0)
    #    proj= ccrs.Stereographic(central_latitude=60, central_longitude=140) 
    #    fig = plt.figure(figsize=(20,16))
    #    ax = fig.add_subplot(1, 1, 1, projection=proj)
    #    ax.set_extent([108, 156, 17, 55], ccrs.PlateCarree())
    #
    ## 図法の変換
    # 参照  https://scitools.org.uk/cartopy/docs/v0.14/crs/index.html                    
    point_proj = proj.transform_point(*lonlat, ccrs.PlateCarree())
    #
    # pixel座標へ変換
    # 参照　https://matplotlib.org/stable/tutorials/advanced/transforms_tutorial.html
    point_pix = ax.transData.transform(point_proj)
    #
    # 図の座標へ変換                                                           
    point_fig = ax.transAxes.inverted().transform(point_pix)
    return point_fig, point_pix, point_proj

## 極大/極小ピーク検出関数                                                             
def detect_peaks(image, filter_size=100, dist_cut=100.0, flag=0):
    # filter_size: この値xこの値 の範囲内の最大値のピークを検出                        
    # dist_cut: この距離内のピークは1つにまとめる                                      
    # flag:  0:maximum検出  0以外:minimum検出                                          
    if flag==0:
      local_max = maximum_filter(image,
            footprint=np.ones((filter_size, filter_size)), mode='constant')
      detected_peaks = np.ma.array(image, mask=~(image == local_max))
    else:
      local_min = minimum_filter(image,
            footprint=np.ones((filter_size, filter_size)), mode='constant')
      detected_peaks = np.ma.array(image, mask=~(image == local_min))
    peaks_index = np.where((detected_peaks.mask != True))
    # peak間の距離行例を求める                                                         
    (x,y) = peaks_index
    size=y.size
    dist=np.full((y.size, y.size), -1.0)
    for i in range(size):
      for j in range(size):
        if i == j:
          dist[i][j]=0.0
        elif i>j:
          d = math.sqrt(((y[i] - y[j])*(y[i] - y[j]))
                        +((x[i] - x[j])*(x[i] - x[j])))
          dist[i][j]= d
          dist[j][i]= d
    # 距離がdist_cut内のpeaksの距離の和と、そのピーク番号を取得する 
    Kinrin=[]
    dSum=[]
    for i in range(size):
      tmpA=[]
      distSum=0.0
      for j in range(size):
        if dist[i][j] < dist_cut and dist[i][j] > 0.0:
          tmpA.append(j)
          distSum=distSum+dist[i][j]
      dSum.append(distSum)
      Kinrin.append(tmpA)
    # Peakから外すPeak番号を求める.  peak間の距離和が最も小さいものを残す              
    cutPoint=[]
    for i in range(size):
      val = dSum[i]
      val_i=image[x[i]][y[i]]
      for k in Kinrin[i]:
        val_k=image[x[k]][y[k]]
        if flag==0 and val_i < val_k:
            cutPoint.append(i)
            break
        if flag!=0 and val_i > val_k:
            cutPoint.append(i)
            break
        if val > dSum[k]:
            cutPoint.append(i)
            break
        if val == dSum[k] and i > k:
            cutPoint.append(i)
            break
    # 戻り値用に外すpeak番号を配列から削除                                             
    newx=[]
    newy=[]
    for i in range(size):
      if (i in cutPoint):
        continue
      newx.append(x[i])
      newy.append(y[i])
    peaks_index=(np.array(newx),np.array(newy))
    return peaks_index

## 地図の描画範囲を指定
i_area = [115,165,15,55]                                                              
# 緯線・経線の指定
dlon,dlat=10,10   # 10度ごとに
## タイトル文字列
dt_str = (dt.strftime("%Y/%m/%d/%HZ")).upper()

## 単位の変更
dss['prmsl'] = dss['prmsl'].metpy.convert_units('hPa')
dss['prmsl'] = dss['prmsl'].metpy.convert_units('hPa')

## 図法指定                                                                             
proj = ccrs.Stereographic(central_latitude=60, central_longitude=140)
latlon_proj = ccrs.PlateCarree()
## 図のSIZE指定inch                                                                        
fig = plt.figure(figsize = (10,8))   
## 余白設定                                                                                
plt.subplots_adjust(left=0, right=1, bottom=0.06, top=0.98)                  
## 作図                                                                                    
ax = fig.add_subplot(1, 1, 1, projection=proj)
ax.set_extent(i_area, latlon_proj)
## 海岸線                                                                                                                               
ax.coastlines(resolution='10m', linewidth=1.6) # 海岸線の解像度を上げる  
## グリッド線を引く                                                               
xticks=np.arange(0,360.1,dlon)
yticks=np.arange(-90,90.1,dlat)
gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=False, linewidth=1, alpha=0.8)
gl.xlocator = mticker.FixedLocator(xticks)
gl.ylocator = mticker.FixedLocator(yticks)

## 等圧線
cn_pre  = ax.contour(dss['lon'], dss['lat'], dss['prmsl'], np.arange(900.0, 1080.0, 4.0), colors='black', linewidths=2.0, linestyles='solid', transform=latlon_proj)
cn_preb = ax.contour(dss['lon'], dss['lat'], dss['prmsl'], np.arange(900.0, 1080.0, 20.0), colors='black', linewidths=3.0, linestyles='solid', transform=latlon_proj)
#ax.clabel(cn_pre, cn_pre.levels, fontsize=11, inline=True, inline_spacing=1, fmt='%i', rightside_up=True)

## H stamp
#maxid = detect_peaks(dss['prmsl'].values, filter_size=6, dist_cut=2.0)
maxid = detect_peaks(dss['prmsl'].values, filter_size=15, dist_cut=5.0)
for i in range(len(maxid[0])):
  wlon = dss['lon'][maxid[1][i]]
  wlat = dss['lat'][maxid[0][i]]
  # 図の範囲内に座標があるか確認                                                                           
  fig_z, _, _ = transform_lonlat_to_figure((wlon,wlat),ax,proj)
  if (fig_z[0] > 0 and fig_z[0] < 1 and fig_z[1] > 0 and fig_z[1] < 1):
    ax.plot(wlon, wlat, marker='x', markersize=10, color="white", transform=latlon_proj)
    ax.text(wlon - 1, wlat + 1, 'H', size=20, color="white", transform=latlon_proj)
    val = dss['prmsl'].values[maxid[0][i]][maxid[1][i]]
    ival = int(val)
    ax.text(fig_z[0], fig_z[1] - 0.01, str(ival), size=20, color="white", transform=ax.transAxes, verticalalignment="top", horizontalalignment="center")

## L stamp
#minid = detect_peaks(dss['prmsl'].values, filter_size=6, dist_cut=2.0, flag=1)
minid = detect_peaks(dss['prmsl'].values, filter_size=15, dist_cut=5.0, flag=1)
for i in range(len(minid[0])):
  wlon = dss['lon'][minid[1][i]]
  wlat = dss['lat'][minid[0][i]]
  # 図の範囲内に座標があるか確認                                                                           
  fig_z, _, _ = transform_lonlat_to_figure((wlon,wlat),ax,proj)
  if (fig_z[0] > 0 and fig_z[0] < 1 and fig_z[1] > 0 and fig_z[1] < 1):
    ax.plot(wlon, wlat, marker='x', markersize=10, color="white", transform=latlon_proj)
    ax.text(wlon - 1, wlat + 1, 'L', size=20, color="white", transform=latlon_proj)
    val = dss['prmsl'].values[minid[0][i]][minid[1][i]]
    ival = int(val)
    ax.text(fig_z[0], fig_z[1] - 0.01, str(ival), size=20, color="white", transform=ax.transAxes, verticalalignment="top", horizontalalignment="center")

# 500hPa 等高度線                                                                                                      
ax.contourf(ds4['lon'], ds4['lat'], ds4['hgt'].sel(level=500.0), levels=np.arange(4980, 6000, 60), cmap='turbo', transform=latlon_proj, extend='both')
                                     
## Title                                                                       
fig.text(0.5, 0.01, dt_str, ha='center', va='bottom', size=18)
## Output
output_fig_nm="{}.png".format(dt.strftime("%Y%m%d%H"))
plt.savefig(output_fig_nm, format="png")
