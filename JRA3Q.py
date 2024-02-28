#!/usr/bin/env python
# coding: utf-8

# JRA3Q GRIB2
#  地上面データ　複数時刻を読み込み、総観場の天気図を作成
#
#    2023/10/05 Ryuta Kurora
#
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
from scipy.ndimage import maximum_filter, minimum_filter
import matplotlib as mpl
import scipy.ndimage as ndimage
from scipy.ndimage import gaussian_filter, median_filter

### データ読み込みのための指定
##! GRIB2の解像度の指定
#  Trueの場合は、 anl_surf/,    fcst_phy2m/
#  Falseの場合は、anal_surf125/,fcst_phy2m125/
f_125 = True

if f_125:   
    file_nm_temp_s = 'anl_surf125.{0:4d}{1:02d}{2:02d}{3:02d}'
    file_nm_temp_p = 'anl_p125_{0}.{1:4d}{2:02d}{3:02d}{4:02d}'
else:
    file_nm_temp_s = 'anl_surf.{0:4d}{1:02d}{2:02d}{3:02d}'
    file_nm_temp_p = 'anl_p.{0:4d}{1:02d}{2:02d}{3:02d}'
        
folder_nm_temp = './Data/{0:4d}{1:02d}{2:02d}/'

##! 読み込み期間の最初の時刻（UTC）,読み込む時刻の数、時間間隔の指定
# コマンドライン引数から日時を取得
arg_datetime = sys.argv[1]

try:
    # 日時文字列を年月日時分に変換
    year = int(arg_datetime[:4])
    month = int(arg_datetime[4:6])
    day = int(arg_datetime[6:8])
    hour = int(arg_datetime[8:10]) 
    dt = datetime.datetime(year, month, day, hour, 0)
    print("指定された日時は:", dt)
    
except ValueError:
    print("日時の形式が正しくありません。正しい形式は 'yyyymmddhh' です。")
    sys.exit(1)

print("Plotting:",dt)
i_year=dt.year
i_month=dt.month
i_day=dt.day
i_hourZ=dt.hour
#
#
##! 読み込むGPVの範囲（緯度・経度で東西南北の境界）を指定
#(latS, latN, lonW, lonE) = (33, 38, 137, 142)  # 東日本付近
#(latS, latN, lonW, lonE) = (18, 45, 115, 145)  # 日本付近
(latS, latN, lonW, lonE) = (-20, 80, 70, 190)  # ASAS領域

## 読み込む要素の指定
elem_s_names = ['tciwv', 'pt', 'sdwe', 'sp', 'prmsl', '2t', '2ttd', '2sh', '2r', '10u', '10v']
elems = ['depr','hgt','rh','tmp','reld', 'relv','spfh','strm','vvel','ugrd','vgrd','vpot',]

## データサイズを取得するために、GRIB2を読み込む
folder_nm = folder_nm_temp.format(i_year,i_month,i_day)
file_nm = folder_nm + file_nm_temp_s.format(i_year,i_month,i_day,i_hourZ)
#print(file_nm)
grbs = pygrib.open(file_nm)
#
## 変数の名称や単位の情報を保存
# shortName,parameterName,parameterUnitsをListにsetする
all_elem_names = []
all_elem_s_names = []
all_elem_units = []
n = 1
for g in grbs:
    #print(g)
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
        elem_s_names[10]: (["lat", "lon"], vals_[10]  * units(elem_units[10])),
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
dss[elem_s_names[10]].attrs['units'] = elem_units[10]
dss['lat'].attrs['units'] = 'degrees_north'
dss['lon'].attrs['units'] = 'degrees_east'

dss = dss.metpy.parse_cf()

## 算出
# 相当温位
dss['ept'] = mpcalc.equivalent_potential_temperature(dss['sp'],dss['2t'],dss['2t']-dss['2ttd'])
# 相対渦度
dss['vort'] = mpcalc.vorticity(dss['10u'],dss['10v'])
# 発散
dss['conv'] = mpcalc.divergence(dss['10u'],dss['10v'])
# シアーパラメーター
dss['shar_para'] = dss['vort'] - dss['conv']






# ガウシアンフィルタを適用
data_msl = dss[elem_s_names[4]].values
sigma = 1.0  # ガウシアンフィルタの標準偏差
smoothed_msl = gaussian_filter(data_msl, sigma=sigma)
dss[elem_s_names[4]] = (["lat", "lon"], smoothed_msl * units(elem_units[4]))












##! 読み込むの高度上限の指定：tagLpより下層の等圧面データをXarray Dataset化する
tagLp = 300

##! 読み込むGPVの範囲（緯度・経度で東西南北の境界）を指定
#(latS, latN, lonW, lonE) = (33, 38, 137, 142)  # 東日本付近
#(latS, latN, lonW, lonE) = (25, 45, 122, 145)  # 日本付近
(latS, latN, lonW, lonE) = (-20, 80, 70, 190)  # ASAS領域

## データサイズを取得するために、GRIB2を読み込む
folder_nm = folder_nm_temp.format(i_year,i_month,i_day)
file_nm = folder_nm + file_nm_temp_p.format(elems[0],i_year,i_month,i_day,i_hourZ)
#print(file_nm)
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
#
## 要素のループ
folder_nm = folder_nm_temp.format(dt.year,dt.month,dt.day)
for i_elem, elem in enumerate(elems):
    ## pygrib open
    file_nm = folder_nm + file_nm_temp_p.format(elem,dt.year,dt.month,dt.day,dt.hour)
    #print(elem," : ",file_nm)
    grbs = pygrib.open(file_nm)
    #
    ## 処理する高度面の選択
    grb_tag = grbs(level=lambda l:l >= tagLp)
    #
    ## 要素名や単位の取得
    elem_names.append(grb_tag[0].parameterName)
    elem_units.append(grb_tag[0].parameterUnits)
        #
    ## レベルのループ
    for i_lev in range(l_size):
        val4_[i_elem][i_lev], _, _ = grb_tag[i_lev].data(lat1=latS,lat2=latN,lon1=lonW,lon2=lonE)

## Xarray Dataset 作成
ds4 = xr.Dataset(
    {
        elems[0]: (["level","lat", "lon"], val4_[0]  * units(elem_units[0])),
        elems[1]: (["level","lat", "lon"], val4_[1]  * units(elem_units[1])),
        elems[2]: (["level","lat", "lon"], val4_[2]  * units(elem_units[2])),
        elems[3]: (["level","lat", "lon"], val4_[3]  * units(elem_units[3])),
        elems[4]: (["level","lat", "lon"], val4_[4]  * units(elem_units[4])),
        elems[5]: (["level","lat", "lon"], val4_[5]  * units(elem_units[5])),
        elems[6]: (["level","lat", "lon"], val4_[6]  * units(elem_units[6])),
        elems[7]: (["level","lat", "lon"], val4_[7]  * units(elem_units[7])),
        elems[8]: (["level","lat", "lon"], val4_[8]  * units(elem_units[8])),
        elems[9]: (["level","lat", "lon"], val4_[9]  * units(elem_units[9])),
        elems[10]: (["level","lat", "lon"], val4_[10]  * units(elem_units[10])),
        elems[11]: (["level","lat", "lon"], val4_[11]  * units(elem_units[11])),
    },
    coords={
        "level": levels,
        "lat": lats,
        "lon": lons,
    },
)
ds4[elems[0]].attrs['units'] = elem_units[0]
ds4[elems[1]].attrs['units'] = elem_units[1]
ds4[elems[2]].attrs['units'] = elem_units[2]
ds4[elems[3]].attrs['units'] = elem_units[3]
ds4[elems[4]].attrs['units'] = elem_units[4]
ds4[elems[5]].attrs['units'] = elem_units[5]
ds4[elems[6]].attrs['units'] = elem_units[6]
ds4[elems[7]].attrs['units'] = elem_units[7]
ds4[elems[8]].attrs['units'] = elem_units[8]
ds4[elems[9]].attrs['units'] = elem_units[9]
ds4[elems[10]].attrs['units'] = elem_units[10]
ds4[elems[11]].attrs['units'] = elem_units[11]
ds4['level'].attrs['units'] = 'hPa'
ds4['lat'].attrs['units'] = 'degrees_north'
ds4['lon'].attrs['units'] = 'degrees_east'
ds4 = ds4.metpy.parse_cf()

# 相当温位の計算
ds4['ttd'] = ds4['tmp'] - ds4['depr']
ds4['ept'] = mpcalc.equivalent_potential_temperature(ds4['level'],ds4['tmp'],ds4['ttd'])


# 相対渦度
ds4['vort'] = mpcalc.vorticity(ds4['ugrd'],ds4['vgrd'])

























# ガウシアンフィルタを適用
data_hgt = ds4['hgt'].values  
data_tmp = ds4['tmp'].values
sigma = 1.0  # ガウシアンフィルタの標準偏差

smoothed_hgt = gaussian_filter(data_hgt, sigma=sigma)
smoothed_tmp = gaussian_filter(data_tmp, sigma=sigma)

# ガウシアンフィルタを適用したデータを元のデータセットに代入します。
ds4['hgt'] = (["level", "lat", "lon"], smoothed_hgt * units(elem_units[1]))
ds4['tmp'] = (["level", "lat", "lon"], smoothed_tmp * units(elem_units[3]))







# 前線客観解析
# 解析に用いる高度
frontlev=925

#ept = ds4['ept'].sel(level=frontlev)
#ept = ds4['ept'].sel(level=frontlev)
ept = ds4['ept'].sel(level=frontlev)
u = ds4['ugrd'].sel(level=frontlev)
v = ds4['vgrd'].sel(level=frontlev)

u5 = ds4['ugrd'].sel(level=500)
v5 = ds4['vgrd'].sel(level=500)

vort = dss['vort'].values


# ガウシアンフィルタを適用
sigma = 4.0  # ガウシアンフィルタの標準偏差
ept = gaussian_filter(ept, sigma=sigma)
u = gaussian_filter(u, sigma=sigma)
v = gaussian_filter(v, sigma=sigma)
u5 = gaussian_filter(u5, sigma=sigma)
v5 = gaussian_filter(v5, sigma=sigma)
vort = gaussian_filter(vort, sigma=sigma)
#ept = median_filter(ept, size=20)
#u = median_filter(u, size=20)
#v = median_filter(v, size=20)

# Front Genesis
dx, dy = mpcalc.lat_lon_grid_deltas(ds4['lon'], ds4['lat'])
grad_ept = np.array(mpcalc.gradient(ept, deltas=(dy, dx)))
mgntd_grad_ept = np.sqrt(grad_ept[0]**2 + grad_ept[1]**2)
grad_u = np.array(mpcalc.gradient(u, deltas=(dy, dx)))
grad_v = np.array(mpcalc.gradient(v, deltas=(dy, dx)))

fg = -(grad_u[1]*grad_ept[1]*grad_ept[1]+grad_v[0]*grad_ept[0]*grad_ept[0]+grad_ept[1]*grad_ept[0]*(grad_u[0]+grad_v[1]))/mgntd_grad_ept*100000*3600
fg = -(grad_u[1]*grad_ept[1]*grad_ept[1]+grad_v[0]*grad_ept[0]*grad_ept[0]+grad_ept[1]*grad_ept[0]*(grad_u[0]+grad_v[1]))/mgntd_grad_ept*100000*3600

#print(ds4['lat'].dims,ds4['vort'].dims)

lat = ds4['lat'].values

# lat を使って計算





# tfpを計算する
#f = np.zeros_like(vort)
#for i in range(vort.shape[0]):
#    for j in range(vort.shape[1]):
#        f[i, j] = vort[i, j] * mgntd_grad_ept[i, j] / math.sin(math.radians(lat[i]))



# ガウシアンフィルタを適用
#sigma = 2.0  # ガウシアンフィルタの標準偏差
#mgntd_grad_ept = gaussian_filter(mgntd_grad_ept, sigma=sigma) 

#NP前線ではTの勾配も勾配方向
#locatefunction = np.zeros_like(ept)
#for i in range(ept.shape[0]):
#    for j in range(ept.shape[1]):
#        locatefunction[i, j] = np.dot(grad_ept[:, i, j], grad_ept[:, i, j] / mgntd_grad_ept[i, j])

# LOCATEFUNCTIONの水平傾度を計算する
grad_mgntd_grad_ept = np.array(mpcalc.gradient(mgntd_grad_ept, deltas=(dy, dx)))
#grad_mgntd_grad_ept = np.array(mpcalc.gradient(locatefunction, deltas=(dy, dx)))

# tfpを計算する
tfp = np.zeros_like(ept)
for i in range(ept.shape[0]):
    for j in range(ept.shape[1]):
        tfp[i, j] = -np.dot(grad_mgntd_grad_ept[:, i, j], grad_ept[:, i, j] / mgntd_grad_ept[i, j]) * 10000000000
        #tfp = (grad_mgntd_grad_ept[0] * v5 - grad_mgntd_grad_ept[1] * u5) / (u5**2 + v5**2)

# ガウシアンフィルタを適用
sigma = 4.0  # ガウシアンフィルタの標準偏差
fg = gaussian_filter(fg, sigma=sigma) 
tfp = gaussian_filter(tfp, sigma=sigma) 

# TFPの水平傾度を計算する
grad_fg = np.array(mpcalc.gradient(fg, deltas=(dy, dx)))
mgntd_grad_fg = np.sqrt(grad_fg[0]**2 + grad_fg[1]**2)

# TFPの極大を抽出する
autofront = np.zeros_like(ept)
for i in range(ept.shape[0]):
    for j in range(ept.shape[1]):
        autofront[i, j] = np.dot(grad_fg[:, i, j], grad_ept[:, i, j] / mgntd_grad_ept[i, j])  
        #autofront = (grad_fg[0] * v5 - grad_fg[1] * u5) / (u5**2 + v5**2)
        
# fの水平傾度を計算する
#grad_f = np.array(mpcalc.gradient(f, deltas=(dy, dx))) 
 
 
 
        
# fの極大を抽出する
#autofront = np.zeros_like(ept)
#for i in range(ept.shape[0]):
    #for j in range(ept.shape[1]):
        #autofront[i, j] = np.dot(grad_f[:, i, j], grad_ept[:, i, j] / mgntd_grad_ept[i, j])  

#grad_fg = np.array(mpcalc.gradient(fg, deltas=(dy, dx)))
#mgntd_grad_fg = np.sqrt(grad_fg[0]**2 + grad_fg[1]**2)

# ガウシアンフィルタを適用
sigma = 4.0  # ガウシアンフィルタの標準偏差
autofront = gaussian_filter(autofront, sigma=sigma) 

print(vort.shape)
print(autofront.shape)

fct=mgntd_grad_ept+6*fg
#autofront[tfp < 0] = np.nan
autofront[u5<-abs(v5)] = np.nan
#autofront[vort < 0] = np.nan
#autofront[fg < 0] = np.nan
#autofront[fct < 0] = np.nan







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


















# カラーマップの設定
#  渦度
cmapVOR = mpl.colors.ListedColormap(['#fff2e5','#fccd9e','#f9b571','#f3a556','#ed953d','#e68524'])
cmapVOR.set_under('white')
cmapVOR.set_over('#cd751d')
boundsVOR = [0,0.00004,0.00008,0.00012,0.00016,0.00020,0.00024]
normVOR = mpl.colors.BoundaryNorm(boundsVOR, cmapVOR.N)
vminVOR, vmaxVOR = min(boundsVOR), max(boundsVOR)



# TFP
cmapTFP = mpl.colors.ListedColormap(['greenyellow', 'yellow', 'gold', 'orange', 'red'])
#cmapTFP = mpl.colors.ListedColormap(['orange'])
cmapTFP.set_over('red')
cmapTFP.set_under('white')
boundsTFP = [327,336,345,348,351,354]
#boundsTFP = [0.0, 100]
normTFP = mpl.colors.BoundaryNorm(boundsTFP, cmapTFP.N)






















### 天気図作図のための指定
# 基準の経度などのデフォルト
set_central_longitude=140
flag_border=False
#
##! 地図の描画範囲を指定
# 0:極東、1:ASAS領域
n_area=0
if n_area == 1:
    i_area = [105,180,0,65]   #ASAS                                                                        
else:
    i_area = [108,156,17,55]  #FEAX 極東                                                                   

# 緯線・経線の指定
dlon,dlat=10,10   # 10度ごとに



## タイトル文字列用
# 初期時刻の文字列
dt_str = (dt.strftime("%Y%m%d%HUTC")).upper()

### 描画の指定
##! 図のSIZE指定inch 
fig_size = (10,8)
#
#! 表示要素指定
flg_spl = True  # 等圧線 True or False
flg_tmp = False  # 気温 True or False
flg_pt  = False  # 温位の描画 True or False
flg_ept = False  # 相当温位の描画  True or False
flg_TTd = False  # 湿り True or False
flg_Sp  = False  # シアーパラメーター  True or False
#
#! 矢羽の表示間隔
if f_125 or n_area==100:
    #wind_slice_n = 1  
    wind_slice_n = 2  
    wind_length = 4.8
else:
    wind_slice_n = 4
    wind_length = 4.8
#
#! 流線の表示
disp_stream_line = False
#
#! 等値線の間隔を指定
levels_tmp0  =np.arange(-60,60,3) # 気温
levels_pt  = np.arange(222, 360, 3.0)  # 温位
levels_ptb = np.arange(240, 360, 15.0) # 温位 太線
levels_ept  = np.arange(222, 360, 3.0)  # 相当温位
levels_eptb = np.arange(240, 360, 15.0) # 相当温位
#
levels_pre  = np.arange(900.0, 1080.0, 4.0) # 気圧
levels_preh = np.arange(998.0, 1022.0, 4.0) # 気圧 点線
levels_preb = np.arange(900.0, 1080.0,20.0) # 気圧 太線 数字

##! 気温　等値線
levels_tmp =np.arange(-60,342,3)
levels_tmp1  =np.arange(-60, 42, 15) # 等値線 太線  
#! 湿数
level_ttd=[0,0.3,1.2,3,18]
cmap_ttd =['green','yellowgreen','0.7','white','yellow']
#
#! シアーパラメーター
level_sp=[1, 3, 5, 10]
cmap_sp=['cyan', 'blue', 'palegreen', 'green']
#
## 単位の変更
dss['2t']  = dss['2t'].metpy.convert_units(units.degC)
dss['10u'] = dss['10u'].metpy.convert_units('knots')
dss['10v'] = dss['10v'].metpy.convert_units('knots')
dss['prmsl'] = dss['prmsl'].metpy.convert_units('hPa')

#
## 図法指定                                                                             
proj = ccrs.Stereographic(central_latitude=60, central_longitude=set_central_longitude)
latlon_proj = ccrs.PlateCarree()
## 図のSIZE指定inch                                                                        
fig = plt.figure(figsize= fig_size)   
## 余白設定                                                                                
plt.subplots_adjust(left=0, right=1, bottom=0.06, top=0.98)                  
## 作図                                                                                    
ax = fig.add_subplot(1, 1, 1, projection=proj)
ax.set_extent(i_area, latlon_proj)
#
## 図に関する設定                                                                
plt.rcParams["contour.negative_linestyle"] = 'dashed'  # 'solid' or dashed
# 
## 海岸線                                                                                                                               
ax.coastlines(resolution='10m', linewidth=1.6) # 海岸線の解像度を上げる  
if (flag_border):
    ax.add_feature(states_provinces, edgecolor='black', linewidth=0.5)
    ax.add_feature(country_borders, edgecolor='black', linewidth=0.5)
#
## グリッド線を引く                                                               
xticks=np.arange(0,360.1,dlon)
yticks=np.arange(-90,90.1,dlat)
gl = ax.gridlines(crs=ccrs.PlateCarree()
         , draw_labels=False
         , linewidth=1, alpha=0.8)
gl.xlocator = mticker.FixedLocator(xticks)
gl.ylocator = mticker.FixedLocator(yticks)
#
# キャプションテキスト
caption_text=""
# 客観前線
ax.contourf(ds4['lon'], ds4['lat'], ept, boundsTFP, cmap=cmapTFP, norm=normTFP, transform=latlon_proj) # 陰影を描く


## 湿数
if flg_TTd :
    caption_text = " T-Td"
    ttd_hatchf = ax.contourf(dss['lon'], dss['lat'],dss['2ttd'], 
                             level_ttd, colors=cmap_ttd,
                             extend='max', alpha=0.15, transform=latlon_proj)
    # colorbarの位置と大きさ指定                                                     
    #  add_axes([左端の距離, 下端からの距離, 横幅, 縦幅])                            
    #ax_reld = fig.add_axes([0.1, 0.0, 0.8, 0.02])  # 図の下
    ax_ttd = fig.add_axes([0.1, 0.1, 0.8, 0.02])  # 図の中
    cb_ttd = fig.colorbar(ttd_hatchf, orientation='horizontal', shrink=0.74,
                           aspect=40, pad=0.01, cax=ax_ttd)
#
## シアーパラメーター
if flg_Sp :
    caption_text = caption_text + " SP"
    sp_hatchf = ax.contourf(dss['lon'], dss['lat'],dss['shar_para'].values * 100000.0, 
                            level_sp, colors=cmap_sp,
                            extend='max', alpha=0.15, transform=latlon_proj)
    # colorbarの位置と大きさ指定                                                     
    ax_sp = fig.add_axes([0.1, 0.1, 0.8, 0.02])  # 図の中
    cb_sp = fig.colorbar(sp_hatchf, orientation='horizontal', shrink=0.74,
                         aspect=40, pad=0.01, cax=ax_sp)
    cb_sp.set_label('shear parameter. (*10$^{5}$ s$^{-1}$)')
#
# 等温度線 実線
if flg_tmp:
    caption_text = caption_text + " Tmp" 
    cn_tmp0 = ax.contour(dss['lon'], dss['lat'], dss['2t'],
                         colors='green', alpha=0.5, linewidths=1.0, levels=levels_tmp0,
                         transform=latlon_proj)
    ax.clabel(cn_tmp0, fontsize=8, inline=True, inline_spacing=1,
              fmt='%i', rightside_up=True)
#
## 温位 pot_temp
if flg_pt :
    caption_text = caption_text + " PT"
    cn_pt   = ax.contour(dss['lon'], dss['lat'],dss['pt'],
                         levels_pt, colors='orange', alpha=0.5,
                         linewidths=1.0, linestyles='solid', transform=latlon_proj)
    cn_ptb  = ax.contour(dss['lon'], dss['lat'],dss['pt'],
                         levels_ptb, colors='orange', alpha=0.5,
                         linewidths=1.5, linestyles='solid', transform=latlon_proj)
    ax.clabel(cn_pt, cn_pt.levels, fontsize=8, inline=True, inline_spacing=1,
              fmt='%i', rightside_up=True)
    ax.clabel(cn_ptb, cn_ptb.levels, fontsize=8, inline=True, inline_spacing=1,
              fmt='%i', rightside_up=True)
#
## 相当温位
if flg_ept :
    caption_text = caption_text + " EPT" 
    cn_ept  = ax.contour(dss['lon'], dss['lat'], dss['ept'], 
                         levels_ept,  colors='red', alpha=0.5,
                         linewidths=0.6, linestyles='solid', transform=latlon_proj)
    cn_eptb = ax.contour(dss['lon'], dss['lat'], dss['ept'],
                         levels_eptb, colors='red', alpha=0.5,
                         linewidths=1.0, linestyles='solid', transform=latlon_proj)
    ax.clabel(cn_ept, cn_ept.levels, fontsize=8, inline=True, inline_spacing=1,
              fmt='%i', rightside_up=True)
    ax.clabel(cn_eptb, cn_eptb.levels, fontsize=10, inline=True, inline_spacing=1,
              fmt='%i', rightside_up=True)
#                                                                                 
## 等圧線
if flg_spl :
    caption_text = caption_text + " Pres(hPa)" 
    cn_pre  = ax.contour(dss['lon'], dss['lat'], dss['prmsl'], levels_pre, 
                         colors='black', linewidths=2.0, linestyles='solid', transform=latlon_proj)
    #cn_preh = ax.contour(dss['lon'], dss['lat'], dss['prmsl'], levels_preh,
    #                     colors='black', linewidths=1.0, linestyles='dashed', transform=latlon_proj)
    cn_preb = ax.contour(dss['lon'], dss['lat'], dss['prmsl'], levels_preb,
                         colors='black', linewidths=3.0, linestyles='solid', transform=latlon_proj)
    ax.clabel(cn_pre, cn_pre.levels, fontsize=11, inline=True, inline_spacing=1, fmt='%i', rightside_up=True)

contour_tfp = ax.contour(ds4['lon'], ds4['lat'], autofront, levels=[0], colors='green', linewidths=2.0, linestyles='solid', transform=latlon_proj) 
    
    
    
    
    

#! 表示する気圧面
disp_pl = 850.0    
# preT hPa面 等温度線
ds4['tmp'] = (ds4['tmp']).metpy.convert_units(units.degC)  # Kelvin => Celsius
cn_tmp = ax.contour(ds4['lon'], ds4['lat'],ds4['tmp'].sel(level=disp_pl),
                    colors='red', linewidths=1.0, linestyles='solid',
                    levels=levels_tmp, transform=latlon_proj )
ax.clabel(cn_tmp, cn_tmp.levels, fontsize=12,
          inline=True, inline_spacing=5, colors='red',
          fmt='%i', rightside_up=True)
cn_tmp1 = ax.contour(ds4['lon'], ds4['lat'],ds4['tmp'].sel(level=disp_pl),
                     colors='red', linewidths=2.0, linestyles='solid', 
                     levels=levels_tmp1, transform=latlon_proj )
ax.clabel(cn_tmp1, cn_tmp1.levels, fontsize=12,
          inline=True, inline_spacing=5,
          fmt='%i', rightside_up=True, colors='red')
#                                                                                 
## 矢羽:データを間引いて描画
wind_slice2 = (slice(None, None, wind_slice_n), slice(None, None, wind_slice_n))
ax.barbs(dss['lon'][wind_slice2[0]],     dss['lat'][wind_slice2[1]], 
         dss['10u'].values[wind_slice2], dss['10v'].values[wind_slice2],
         length=wind_length, pivot='middle', color='black', transform=latlon_proj)
#
## H stamp
#maxid = detect_peaks(dss['prmsl'].values, filter_size=6, dist_cut=2.0)
maxid = detect_peaks(dss['prmsl'].values, filter_size=8, dist_cut=4.0)
for i in range(len(maxid[0])):
  wlon = dss['lon'][maxid[1][i]]
  wlat = dss['lat'][maxid[0][i]]
  # 図の範囲内に座標があるか確認                                                                           
  fig_z, _, _ = transform_lonlat_to_figure((wlon,wlat),ax,proj)
  if ( fig_z[0] > 0.05 and fig_z[0] < 0.95  and fig_z[1] > 0.05 and fig_z[1] < 0.95):
    ax.plot(wlon, wlat, marker='x' , markersize=4, color="blue",transform=latlon_proj)
    ax.text(wlon - 0.5, wlat + 0.5, 'H', size=16, color="blue", transform=latlon_proj)
    val = dss['prmsl'].values[maxid[0][i]][maxid[1][i]]
    ival = int(val)
    ax.text(fig_z[0], fig_z[1] - 0.01, str(ival), size=12, color="blue",
            transform=ax.transAxes,
            verticalalignment="top", horizontalalignment="center")
#
## L stamp
#minid = detect_peaks(dss['prmsl'].values, filter_size=6, dist_cut=2.0, flag=1)
minid = detect_peaks(dss['prmsl'].values, filter_size=8, dist_cut=4.0, flag=1)
for i in range(len(minid[0])):
  wlon = dss['lon'][minid[1][i]]
  wlat = dss['lat'][minid[0][i]]
  # 図の範囲内に座標があるか確認                                                                           
  fig_z, _, _ = transform_lonlat_to_figure((wlon,wlat),ax,proj)
  if ( fig_z[0] > 0.05 and fig_z[0] < 0.95  and fig_z[1] > 0.05 and fig_z[1] < 0.95):
    ax.plot(wlon, wlat, marker='x' , markersize=4, color="red",transform=latlon_proj)
    ax.text(wlon - 0.5, wlat + 0.5, 'L', size=16, color="red", transform=latlon_proj)
    val = dss['prmsl'].values[minid[0][i]][minid[1][i]]
    ival = int(val)
    ax.text(fig_z[0], fig_z[1] - 0.01, str(ival), size=12, color="red",
            transform=ax.transAxes,
            verticalalignment="top", horizontalalignment="center")
#
## Title   
#fig.text(0.5,0.01,"JRA3Q " + dt_str + caption_text,ha='center',va='bottom', size=18)
fig.text(0.5,0.01,"JRA3Q " + dt_str + " Psea,T850",ha='center',va='bottom', size=18)
#
## Output
output_fig_nm="{}Z_surf.jpg".format(dt.strftime("%Y%m%d%H"))
plt.savefig(output_fig_nm, format="jpg")
























##  500hPa高度・渦度天気図

#! 表示する気圧面
disp_pl = 500.0
#
#! 等値線の間隔を指定
levels_ht =np.arange(4800, 36000,  60)  # 高度を60m間隔で実線                       
levels_ht2=np.arange(4800, 36000, 300)  # 高度を300m間隔で太線
levels_vr =np.arange(-0.0002, 0.0002, 0.00004)  # 渦度4e-5毎に等値線
##! 気温　等値線
levels_tmp =np.arange(-60,342,3)
levels_tmp1  =np.arange(-60, 42, 15) # 等値線 太線  
#
#! 渦度のハッチの指定
#levels_h_vr = [0.0, 0.00008, 1.0]    # 0.0以上で 灰色(0.9), 8e-5以上で赤
#colors_h_vr = ['0.9','red']
#alpha_h_vr = 0.3                     # 透過率を指定
#
#! 緯度・経度線の指定
dlon,dlat=10,10   # 10度ごとに
#
## タイトル文字列用
# 初期時刻の文字列
dt_str = (dt.strftime("%Y%m%d%HUTC")).upper()
## 図法指定                                                                             
proj = ccrs.Stereographic(central_latitude=60, central_longitude=set_central_longitude)
latlon_proj = ccrs.PlateCarree()
## 図のSIZE指定inch                                                                        
fig = plt.figure(figsize=(10,8))
## 余白設定                                                                                
plt.subplots_adjust(left=0, right=1, bottom=0.06, top=0.98)                  
## 作図                                                                                    
ax = fig.add_subplot(1, 1, 1, projection=proj)
ax.set_extent(i_area, latlon_proj)
#
# 500hPa 相対渦度のハッチ 0.0以上:着色  0.8*10**-4以上:赤                                  
#cn_relv_hatch2 = ax.contourf(ds4['lon'], ds4['lat'], ds4['relv'].sel(level=disp_pl),
#        levels_h_vr, colors=colors_h_vr,
#        alpha=alpha_h_vr, transform=latlon_proj)
# 5000hPa 相対渦度 実線 0.00004毎 負は破線                                                 
#cn_relv = ax.contour(ds4['lon'], ds4['lat'], ds4['relv'].sel(level=disp_pl),
#        levels_vr, colors='black', linewidths=1.0, transform=latlon_proj)
        
        
# vorを描く(塗り)
ax.contourf(ds4['lon'], ds4['lat'], ds4['relv'].sel(level=disp_pl), boundsVOR, cmap=cmapVOR, norm=normVOR, vmin=vminVOR, vmax=vmaxVOR, transform=latlon_proj) # 陰影を描く
    
# vorを書く(線)  
clevs = np.arange(0, 0.0002, 0.00004)
cs = ax.contour(ds4['lon'], ds4['lat'], ds4['relv'].sel(level=disp_pl), clevs, linewidths=0.4, linestyles='dashed', colors='#cd6600', transform=latlon_proj)
############################ cs.clabel(fontsize=6, fmt="%d")
#clevs = np.arange(0, 0.0002, 0.0002)
#cs = ax.contour(ds4['lon'], ds4['lat'], ds4['relv'].sel(level=disp_pl), clevs, linewidths=0.4, colors='#cd6600', transform=latlon_proj)
#cs.clabel(fontsize=6, fmt="%d")        
        
        
# 500hPa  等高度線 実線 step1:60m毎                                                                                                          
cn_hgt = ax.contour(ds4['lon'], ds4['lat'], ds4['hgt'].sel(level=disp_pl),
                    colors='black',
                    linewidths=2.0, levels=levels_ht, transform=latlon_proj )
ax.clabel(cn_hgt, levels_ht, fontsize=15, inline=True, inline_spacing=5,
          fmt='%i', rightside_up=True)
# 500hPa 等高度線 太線 step1:300m毎                                                        
cn_hgt2= ax.contour(ds4['lon'], ds4['lat'], ds4['hgt'].sel(level=disp_pl),
                    colors='black',
                    linewidths=3.0, levels=levels_ht2, transform=latlon_proj)
ax.clabel(cn_hgt2, fontsize=15, inline=True, inline_spacing=0,
          fmt='%i', rightside_up=True)
          
# preT hPa面 等温度線
ds4['tmp'] = (ds4['tmp']).metpy.convert_units(units.degC)  # Kelvin => Celsius
cn_tmp = ax.contour(ds4['lon'], ds4['lat'],ds4['tmp'].sel(level=disp_pl),
                    colors='red', linewidths=1.0, linestyles='solid',
                    levels=levels_tmp, transform=latlon_proj )
ax.clabel(cn_tmp, cn_tmp.levels, fontsize=12,
          inline=True, inline_spacing=5, colors='red',
          fmt='%i', rightside_up=True)
cn_tmp1 = ax.contour(ds4['lon'], ds4['lat'],ds4['tmp'].sel(level=disp_pl),
                     colors='red', linewidths=2.0, linestyles='solid', 
                     levels=levels_tmp1, transform=latlon_proj )
ax.clabel(cn_tmp1, cn_tmp1.levels, fontsize=12,
          inline=True, inline_spacing=5,
          fmt='%i', rightside_up=True, colors='red')

## 矢羽:データを間引いて描画
wind_slice = (slice(None, None, wind_slice_n), slice(None, None, wind_slice_n))
ax.barbs(ds4['lon'][wind_slice[0]], ds4['lat'][wind_slice[1]],    
         ds4['ugrd'].sel(level=disp_pl).values[wind_slice],
         ds4['vgrd'].sel(level=disp_pl).values[wind_slice],
         length=wind_length, pivot='middle', color='black', transform=latlon_proj)
         
## H stamp                                                                                                 
maxid = detect_peaks(ds4['hgt'].sel(level=disp_pl).values, filter_size=10, dist_cut=8.0)
for i in range(len(maxid[0])):
  wlon = ds4['lon'][maxid[1][i]]
  wlat = ds4['lat'][maxid[0][i]]
  # 図の範囲内に座標があるか確認                                                                           
  fig_z, _, _ = transform_lonlat_to_figure((wlon,wlat),ax,proj)
  if ( fig_z[0] > 0.05 and fig_z[0] < 0.95  and fig_z[1] > 0.05 and fig_z[1] < 0.95):
    ax.text(wlon, wlat, 'H', size=24, color="blue",
            ha='center', va='center', transform=latlon_proj)
## L stamp                                                                                                 
minid = detect_peaks(ds4['hgt'].sel(level=disp_pl).values, filter_size=10, dist_cut=8.0,flag=1)
for i in range(len(minid[0])):
  wlon = ds4['lon'][minid[1][i]]
  wlat = ds4['lat'][minid[0][i]]
  # 図の範囲内に座標があるか確認                                                                           
  fig_z, _, _ = transform_lonlat_to_figure((wlon,wlat),ax,proj)
  if ( fig_z[0] > 0.05 and fig_z[0] < 0.95  and fig_z[1] > 0.05 and fig_z[1] < 0.95):
    ax.text(wlon, wlat, 'L', size=24, color="red",
            ha='center', va='center', transform=latlon_proj)
#
## 海岸線                                                                                                                               
ax.coastlines(resolution='50m', linewidth=1.6) # 海岸線の解像度を上げる  
if (flag_border):
    ax.add_feature(states_provinces, edgecolor='black', linewidth=0.5)
    ax.add_feature(country_borders, edgecolor='black', linewidth=0.5)

## グリッド                                                                   
xticks=np.arange(0,360.1,dlon)
yticks=np.arange(-90,90.1,dlat)
gl = ax.gridlines(crs=ccrs.PlateCarree()
         , draw_labels=False
         , linewidth=1, alpha=0.8)
gl.xlocator = mticker.FixedLocator(xticks)
gl.ylocator = mticker.FixedLocator(yticks)
#                                                                                          
## Title                                                                       
fig.text(0.5,0.01,"JRA3Q " + dt_str + " Z500,VORT",ha='center',va='bottom', size=18)
#fig.text(0.5,0.01,"JRA3Q " + dt_str + caption_text,ha='center',va='bottom', size=18)
## Output
#output_fig_nm="{}Z_surf.jpg".format(dt.strftime("%Y%m%d%H"))
output_fig_nm="{}Z_500hPa.jpg".format(dt.strftime("%Y%m%d%H"))
plt.savefig(output_fig_nm, format="jpg")
