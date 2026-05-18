#!/usr/bin/env python
# coding: utf-8

# In[1]:


from scipy.io import netcdf, loadmat
import numpy as np
import os
from scipy import interpolate
import scipy
import glob
from netCDF4 import Dataset
#from mpl_toolkits.basemap import Basemap,maskoceans
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
from numpy import polyfit
import numpy.matlib
from scipy.stats import pearsonr, linregress
get_ipython().run_line_magic('matplotlib', 'inline')
import copy
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from matplotlib.colors import ListedColormap
import matplotlib.colors as colors
from scipy.signal import detrend
import numpy.ma as ma
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.path as mpath
import matplotlib.ticker as mticker
import warnings
from shutil import copyfile
from scipy.ndimage import convolve1d
import xarray as xr


# In[2]:


def get_array(start,end,lev):
    a=np.arange(start,end,lev)
    
    return list(a[:len(a)//2])+list(a[len(a)//2+1:])


def get_nh_mean(var,lat):
    aa=np.cos(lat/180*np.pi)
    index = np.argmax(lat>=0.)
    aa[:index] = 0.
    if len(np.shape(var))==3:
        bb=var*aa[None,None,:]
        cc=np.sum(bb,2)
        dd=np.sum(aa)
    if len(np.shape(var))==2:
        bb=var*aa[None,:]
        cc=np.sum(bb,1)
        dd=np.sum(aa)
    if len(np.shape(var))==1:
        bb=var*aa
        cc=np.sum(bb)
        dd=np.sum(aa)
    return cc/dd


def get_tropic_mean(var,lat):
    aa=np.cos(lat/180*np.pi)
    index2 = np.argmax(lat>30.)
    index1 = np.argmax(lat>-30.)
    aa[:index1] = 0
    aa[index2:] = 0
    if len(np.shape(var))==3:
        bb=var*aa[None,None,:]
        cc=np.sum(bb,2)
        dd=np.sum(aa)
    if len(np.shape(var))==2:
        bb=var*aa[None,:]
        cc=np.sum(bb,1)
        dd=np.sum(aa)
    if len(np.shape(var))==1:
        bb=var*aa
        cc=np.sum(bb)
        dd=np.sum(aa)
    return cc/dd


def get_regr(ind,var):
    var_=np.reshape(var,[np.size(var,0),-1])
    regressions,_ = np.polyfit(ind, var_, 1)
    regressions = regressions.reshape([np.size(var,1),np.size(var,2)])# 35 year    
    return regressions

def get_corr(ind,var):
    corr = np.zeros((np.size(var,1),np.size(var,2)))
    pval = np.zeros((np.size(var,1),np.size(var,2)))
    for i in range(np.size(var,1)):
        for j in range(np.size(var,2)):
            corr[i,j],pval[i,j] = pearsonr(ind,var[:,i,j])
    return corr,pval

def get_regr_nan(ind,var):
    regressions = np.zeros((np.size(var,1),np.size(var,2)))
    for i in range(np.size(var,1)):
        for j in range(np.size(var,2)):
            aa=var[:,i,j]
            if len(aa[~np.isnan(aa)])>2:
                    regressions[i,j],_ = np.polyfit(ind[~np.isnan(aa)],aa[~np.isnan(aa)],1)
            else:
                regressions[i,j] = np.nan
    return regressions

def get_corr_nan(ind,var):
    corr = np.zeros((np.size(var,1),np.size(var,2)))
    pval = np.zeros((np.size(var,1),np.size(var,2)))
    for i in range(np.size(var,1)):
        for j in range(np.size(var,2)):
            aa=var[:,i,j]
            if len(aa[~np.isnan(aa)])>2:
                corr[i,j],pval[i,j] = pearsonr(ind[~np.isnan(aa)],aa[~np.isnan(aa)])
            else:
                corr[i,j]=np.nan
    return corr,pval

def running_mean(var,window,axis=0,drop=False):
    if len(np.shape(var))==1:
        var_=np.zeros(len(var))
        for i in range(len(var)):
            if i<window//2:
                var_[i]=np.nan
            elif i>len(var)-window//2-1:
                var_[i]=np.nan
            else:
                var_[i]=np.mean(var[i-window//2:i+window//2+1])
    else:
        var_=np.zeros(np.shape(var))
        for i in range(np.size(var,0)):
            if i<window//2:
                var_[i,::]=np.nan
            elif i>len(var)-window//2-1:
                var_[i,::]=np.nan
            else:
                var_[i,::]=np.mean(var[i-window//2:i+window//2+1,::],0)
    if drop:
        return var_[window//2:-window//2]
    else:
        return var_

def eof_norm_mask(data,num,mask):
    if np.size(np.shape(data))==3:
        #data_=np.reshape(data,(np.size(data,0),np.size(data,1)*np.size(data,2)))
        data=data-[np.mean(data,0) for _ in range(np.size(data,0))]
        index=np.where(mask)
        data0=np.zeros((np.size(data,0),len(index[0])))
        for nn in range(len(index[0])):
            data0[:,nn]=data[:,index[0][nn],index[1][nn]]
        [ C, lamda, EOFs ] = np.linalg.svd( data0, full_matrices=0);

        ECs = C * lamda; # Expansion coefficients.
        if num ==1:
            EOF=np.zeros((np.size(data,1),np.size(data,2)))
            for nn in range(len(index[0])):
                EOF[index[0][nn],index[1][nn]]=EOFs[0,nn]*np.std(ECs[:,0])
            PC=ECs[:,0]/np.std(ECs[:,0])
            vc=lamda[0]**2/(sum(lamda**2))
        else:
            PC=np.zeros((np.size(data,0),num))
            vc=np.zeros(num)
            EOF=np.zeros((np.size(data,1),np.size(data,2),num))
            for k in range(num):
                for nn in range(len(index[0])):
                    EOF[index[0][nn],index[1][nn],k]=EOFs[k,nn]*np.std(ECs[:,k])
                PC[:,k]=ECs[:,k]/np.std(ECs[:,k])
                vc[k]=lamda[k]**2/(sum(lamda**2))
    return EOF, PC, vc

    
    
def get_globe_mean(var,lat):
    aa=np.cos(lat/180*np.pi)
    if len(np.shape(var))==3:
        bb=var*aa[None,None,:]
        cc=np.sum(bb,2)
        dd=np.sum(aa)
    if len(np.shape(var))==2:
        bb=var*aa[None,:]
        cc=np.sum(bb,1)
        dd=np.sum(aa)
    if len(np.shape(var))==1:
        bb=var*aa
        cc=np.sum(bb)
        dd=np.sum(aa)
    return cc/dd

def get_arctic_mean(var,lat):
    aa=np.cos(lat/180*np.pi)
    index = np.argmax(lat>=65)
    aa[:index] = 0.
    
    if len(np.shape(var))==3:
        bb=var*aa[None,None,:]
        cc=np.nansum(bb,2)
        dd=np.nansum(aa)
    if len(np.shape(var))==2:
        bb=var*aa[None,:]
        cc=np.nansum(bb,1)
        dd=np.nansum(aa)
    if len(np.shape(var))==1:
        bb=var*aa
        cc=np.nansum(bb)
        dd=np.nansum(aa)
    return cc/dd

def get_trend(var):
    kk = np.size(var,0)
    trend=np.zeros((np.size(var,1),np.size(var,2)))
    for i in range(np.size(var,1)):
        for j in range(np.size(var,2)):
            if np.sum(var[:,i,j]):
                trend[i,j] = polyfit(range(kk),var[:,i,j],1)[0]
            else:
                trend[i,j] = np.nan
    return trend

filename = '/Volumes/Expansion/data/cmip6/historical/ts/MRI-ESM2/r1i1p1/ts_Amon_MRI-ESM2_historical_195001-201412.1deg.nc'
f = Dataset(filename, 'r')
lat = f.variables['lat'][:]
lon = f.variables['lon'][:]

lon_ex = list(lon) + [360.5]
def get_extend(var):
    var1 = np.zeros((np.size(var,0),np.size(var,1)+1))
    var1[:,:-1] = var[::]
    var1[:,-1] = var[:,0]
    return var1
yrs=np.arange(1965,2023,1)


# In[3]:


os.chdir('/Volumes/Xiao/observations/siconc/Had')
filename = glob.glob('siconc_Had_1950_2020.1deg.ann.nc')[0]
f = Dataset(filename, 'r')
aa=f.variables['sic'][20,::]
mask_land=aa.mask


# In[5]:


os.chdir('/Volumes/Xiao/observations/ts/Berkeley')
filename = glob.glob('*1940*1deg.ann.nc')[0]
f = Dataset(filename, 'r')
ts_2d_ann_obs= f.variables['temperature'][10:,:] 


# In[6]:


mask_ao=np.ones(np.shape(mask_land))
mask_ao[:130,:] = 0.
mask_globe=np.ones(np.shape(mask_land))
mask_globe[:20,:] = 0. #mask out some nan values in OBS Ts
coslat = np.cos(lat/180.*np.pi)


# In[7]:


#arctic mode after removing global eof since 1950
eof_globe,pc_globe,vc_globe =eof_norm_mask(ts_2d_ann_obs[:,::]*coslat[None,:,None],1,mask_globe)
tst_2d_ann=ts_2d_ann_obs[:,::]-eof_globe[None,:,:]*pc_globe[:,None,None]/coslat[None,:,None]
eof_ao2,pc_ao2,vc_ao2=eof_norm_mask(tst_2d_ann,2,mask_ao)
ts_reg_ao2 = get_regr(pc_ao2[:,0],tst_2d_ann)


# In[8]:


#arctic mode after removing global eof since 1965
eof_globe,pc_globe,vc_globe =eof_norm_mask(ts_2d_ann_obs[15:,::]*coslat[None,:,None],1,mask_globe)
tst_2d_ann=ts_2d_ann_obs[15:,::]-eof_globe[None,:,:]*pc_globe[:,None,None]/coslat[None,:,None]
eof_ao1,pc_ao1,vc_ao1=eof_norm_mask(tst_2d_ann,2,mask_ao)
ts_reg_ao1 = get_regr(pc_ao1[:,0],tst_2d_ann)


# In[9]:


#arctic mode after removing linear trend since 1965
tst_2d_ann=detrend(ts_2d_ann_obs[15:,::],axis=0)
eof_ao,pc_ao,vc_ao=eof_norm_mask(tst_2d_ann,2,mask_ao)
trend_pattern = get_trend(ts_2d_ann_obs[15:,::])
ts_reg_ao = get_regr(pc_ao[:,0],tst_2d_ann)


# In[10]:


#arctic mode after removing linear trend since 1965, eof over artic with area weighting
mask_ao=np.ones(np.shape(mask_land))
mask_ao[:156,:] = 0.
tst_2d_ann=detrend(ts_2d_ann_obs[15:,::],axis=0)
eof_ao,pc_ao_,vc_ao=eof_norm_mask(tst_2d_ann*coslat[None,:,None],2,mask_ao)
trend_pattern = get_trend(ts_2d_ann_obs[15:,::])
ts_reg_ao_ = get_regr(pc_ao_[:,0],tst_2d_ann)


# In[11]:


#arctic mode after removing global eof since 1965
mask_ao=np.ones(np.shape(mask_land))
mask_ao[:156,:] = 0.
mask_globe=np.ones(np.shape(mask_land))
coslat = np.cos(lat/180.*np.pi)
eof_globe,pc_globe,vc_globe =eof_norm_mask(ts_2d_ann_obs[15:,::]*coslat[None,:,None],1,mask_globe)
tst_2d_ann=ts_2d_ann_obs[15:,::]-eof_globe[None,:,:]*pc_globe[:,None,None]/coslat[None,:,None]
eof_ao1_,pc_ao1_,vc_ao1_=eof_norm_mask(tst_2d_ann*coslat[None,:,None],2,mask_ao)
ts_reg_ao1_ = get_regr(pc_ao1_[:,0],tst_2d_ann)


# In[12]:


arctic_t_2022 = get_arctic_mean(np.mean(ts_2d_ann_obs[15:,::],2),lat)
globe_t_2022 = get_globe_mean(np.mean(ts_2d_ann_obs[15:,::],2),lat)


# In[13]:


os.chdir('/Volumes/Xiao/data/observation/ERA5_1deg')
filename = glob.glob('sfc_1960_2024.yr.1deg.nc')[0]  #1950-2021
f = Dataset(filename, 'r')
psl_era5 = f.variables['msl'][5:-2,:,:]/100.  
psl_trend = get_trend(psl_era5)
psl_era5_detrend = detrend(psl_era5,axis=0)
psl_regr = get_regr(pc_ao[:,0],psl_era5_detrend)
psl_regr_ = get_regr(pc_ao_[:,0],psl_era5_detrend)
psl_regr_globe = get_regr(pc_globe,psl_era5)
psl_regr1 = get_regr(-pc_ao1[:,0],psl_era5-psl_regr_globe[None,:,:]*pc_globe[:,None,None])
psl_regr1_ = get_regr(-pc_ao1_[:,0],psl_era5-psl_regr_globe[None,:,:]*pc_globe[:,None,None])


# In[14]:


def standardize_1d(pc):
    """Return standardized PC (mean 0, std 1)."""
    pc = xr.DataArray(pc, dims=["time"]) if not isinstance(pc, xr.DataArray) else pc
    pc = pc - pc.mean("time")
    return pc / pc.std("time")

def regress_onto_pc(field_da, pc_da):
    """
    Regress field(time,lat,lon) onto pc(time).
    Returns beta(lat,lon): units of field per 1 unit pc.
    """
    # Align time
    field_da, pc_da = xr.align(field_da, pc_da, join="inner")

    # Remove time mean (anomalies)
    field_anom = field_da - field_da.mean("time")
    pc_anom = pc_da - pc_da.mean("time")

    # beta = cov(field, pc) / var(pc)
    cov = (field_anom * pc_anom).mean("time")
    var = (pc_anom ** 2).mean("time")
    beta = cov / var
    return beta

def detrend_xr(da, dim="time"):
    """
    Linear detrend an xarray DataArray along dim using polyfit.
    """
    # polyfit returns coefficients; deg=1 => slope, intercept
    pf = da.polyfit(dim=dim, deg=1, skipna=True)
    fit = xr.polyval(da[dim], pf.polyfit_coefficients)
    return da - fit

def rolling5_center(da):
    """5-pt centered rolling mean with end-trim consistent with your code."""
    return da.rolling(time=5, center=True).mean().isel(time=slice(2, -2))

# -----------------------
# User setup
# -----------------------
models=['CESM2','GISS-E2-1-G']
os.chdir("/Users/zhou272/Downloads/PiControl/")


EOFpi_dict_ = {}
PCpi_dict_  = {}
vcpi_dict_  = {} 
ts_reg_am_ = {}
psl_reg_am_ = {}
# Precompute lat weight for EOF calculation (same as your code)
wlat = np.sqrt(coslat)[None, :, None]   # (1, nlat, 1)

for model in models:
    print(f"\n=== Processing model: {model} ===")
    # -----------------------
    # 1) Read the Atlantic-season TS file used for EOF/PC
    # -----------------------
    ts_file = glob.glob(f"ts_*{model}*.1deg.ann.nc")[0]
    ds_ts = xr.open_dataset(ts_file)
    ts_da = ds_ts["ts"]
    ts_da = ts_da.load()
    ds_ts.close()
    ts_g = detrend(ts_da.values, axis=0, type="linear") 
    ts_w = ts_g* wlat  # (time, lat, lon)

    mask_ao=np.ones(np.shape(mask_land))
    mask_ao[:156,:] = 0.
        
    EOF, PC, vc = eof_norm_mask(ts_w,1,mask_ao)
    
    EOFpi_dict_[model] = EOF
    PCpi_dict_[model]  = PC
    vcpi_dict_[model]  = vc


    filename = glob.glob('psl*'+model+'*.1deg.ann.nc')[0]
    f = Dataset(filename, 'r')
    psl=f.variables['psl'][::]
    
    times=PCpi_dict_[model]
    ts_reg_am_[model] = get_regr(times,ts_g)
    psl_reg_am_[model] = get_regr(times,detrend(psl,axis=0))
    


# In[15]:


fig= plt.figure(figsize=(18,16))

ax0=plt.subplot(331)               
yrs_2022 = np.arange(1965,2023,1)

h1,=plt.plot(yrs_2022,globe_t_2022,'k')
h2,=plt.plot(yrs_2022,arctic_t_2022,'r')
plt.xlim([1964.5,2022.5])
plt.ylim([-0.8,3.8])
plt.tick_params(labelsize=10,direction='out',length=3,width=1.,pad=3)
plt.text(-0.074,1.01,'a',fontsize=13,transform=ax0.transAxes,weight='bold')
plt.ylabel('Temperature Anomaly [K]')
plt.legend(['Global','Arctic'],labelcolor=[h1.get_color(),h2.get_color()],frameon=False,fontsize=9)

ax2=plt.subplot(332)
h1,=plt.plot(yrs,pc_ao[:,0],'-')
h2,=plt.plot(yrs,-pc_ao1[:,0],'--')
h3,=plt.plot(np.arange(1950,2023,1),-pc_ao2[:,0],'-.',color='dimgrey')
h4,=plt.plot(yrs,-pc_ao_[:,0],':',color='r')
plt.xlim([1964.5,2022.5])
plt.plot([1965,2022],[0,0],'k-',linewidth=1)
plt.xlim([1964.5,2022.5])
plt.ylim([-2.75,2.75])
plt.tick_params(labelsize=10,direction='out',length=3,width=1.,pad=3)
plt.text(-0.074,1.01,'b',fontsize=13,transform=ax2.transAxes,weight='bold')
plt.ylabel('Principal Component')
plt.legend(['remove linear trend (1965-2022)','remove leading EOF (1965-2022)','remove leading EOF (1950-2022)','north of 65$\mathrm{^o}$N, area weighting'],fontsize=9,labelcolor=[h1.get_color(),h2.get_color(),h3.get_color(),h4.get_color()],frameon=False)

ax1=plt.subplot(333, projection=ccrs.AzimuthalEquidistant(central_longitude=0.0, central_latitude=90.0))
ax1.set_extent([-180, 180, 50, 90], ccrs.PlateCarree())
CS=ax1.contourf(lon_ex,lat,get_extend(trend_pattern)*10.,get_array(-1.2,1.3,0.2),transform=ccrs.PlateCarree(),cmap='RdBu_r',extend='both')
ax1.contour(lon,lat,psl_trend*10.,get_array(-1,1.1,0.2),transform=ccrs.PlateCarree(),colors='k',linewidths=1.25)
ax1.coastlines(linewidth=0.5)
theta = np.linspace(0, 2*np.pi, 100)
center, radius = [0.5, 0.5], 0.5
verts = np.vstack([np.sin(theta), np.cos(theta)]).T
circle = mpath.Path(verts * radius + center)
ax1.set_boundary(circle, transform=ax1.transAxes)
plt.text(-0.0,0.95,'c',fontsize=13,transform=ax1.transAxes,weight='bold')
plt.title('Long-term trend since 1965')

ax1=plt.subplot(334, projection=ccrs.AzimuthalEquidistant(central_longitude=0.0, central_latitude=90.0))
ax1.set_extent([-180, 180, 50, 90], ccrs.PlateCarree())
CS=ax1.contourf(lon_ex,lat,get_extend(ts_reg_ao),get_array(-1.2,1.3,0.2),transform=ccrs.PlateCarree(),cmap='RdBu_r',extend='both')
ax1.contour(lon_ex,lat,get_extend(psl_regr),get_array(-1,1.1,0.2),transform=ccrs.PlateCarree(),colors='k',linewidths=1.25)
ax1.coastlines(linewidth=0.5)
theta = np.linspace(0, 2*np.pi, 100)
center, radius = [0.5, 0.5], 0.5
verts = np.vstack([np.sin(theta), np.cos(theta)]).T
circle = mpath.Path(verts * radius + center)
ax1.set_boundary(circle, transform=ax1.transAxes)
plt.text(-0.0,0.95,'d',fontsize=13,transform=ax1.transAxes,weight='bold')
plt.title('AM, remove linear trend')


ax1=plt.subplot(335, projection=ccrs.AzimuthalEquidistant(central_longitude=0.0, central_latitude=90.0))
ax1.set_extent([-180, 180, 50, 90], ccrs.PlateCarree())
CS=ax1.contourf(lon_ex,lat,get_extend(-ts_reg_ao1),get_array(-1.2,1.3,0.2),transform=ccrs.PlateCarree(),cmap='RdBu_r',extend='both')
ax1.contour(lon,lat,psl_regr1,get_array(-1,1.1,0.2),transform=ccrs.PlateCarree(),colors='k',linewidths=1.25)
ax1.coastlines(linewidth=0.5)
theta = np.linspace(0, 2*np.pi, 100)
center, radius = [0.5, 0.5], 0.5
verts = np.vstack([np.sin(theta), np.cos(theta)]).T
circle = mpath.Path(verts * radius + center)
ax1.set_boundary(circle, transform=ax1.transAxes)
plt.text(-0.0,0.95,'e',fontsize=13,transform=ax1.transAxes,weight='bold')
plt.title('AM, remove leading EOF')


ax1=plt.subplot(336, projection=ccrs.AzimuthalEquidistant(central_longitude=0.0, central_latitude=90.0))
ax1.set_extent([-180, 180, 50, 90], ccrs.PlateCarree())
CS=ax1.contourf(lon_ex,lat,get_extend(-ts_reg_ao_),get_array(-1.2,1.3,0.2),transform=ccrs.PlateCarree(),cmap='RdBu_r',extend='both')
ax1.contour(lon,lat,-psl_regr_,get_array(-1,1.1,0.2),transform=ccrs.PlateCarree(),colors='k',linewidths=1.25)
ax1.coastlines(linewidth=0.5)
theta = np.linspace(0, 2*np.pi, 100)
center, radius = [0.5, 0.5], 0.5
verts = np.vstack([np.sin(theta), np.cos(theta)]).T
circle = mpath.Path(verts * radius + center)
ax1.set_boundary(circle, transform=ax1.transAxes)
plt.text(-0.0,0.95,'f',fontsize=13,transform=ax1.transAxes,weight='bold')
plt.title('AM, north of 65$\mathrm{^o}$N, area weighting')

model='CESM2'
ax1=plt.subplot(337, projection=ccrs.AzimuthalEquidistant(central_longitude=0.0, central_latitude=90.0))
ax1.set_extent([-180, 180, 50, 90], ccrs.PlateCarree())
CS=ax1.contourf(lon_ex,lat,get_extend(ts_reg_am_[model]),get_array(-1.2,1.3,0.2),transform=ccrs.PlateCarree(),cmap='RdBu_r',extend='both')
ax1.contour(lon,lat,psl_reg_am_[model]/100.,get_array(-1,1.1,0.2),transform=ccrs.PlateCarree(),colors='k',linewidths=1.25)
ax1.coastlines(linewidth=0.5)
theta = np.linspace(0, 2*np.pi, 100)
center, radius = [0.5, 0.5], 0.5
verts = np.vstack([np.sin(theta), np.cos(theta)]).T
circle = mpath.Path(verts * radius + center)
ax1.set_boundary(circle, transform=ax1.transAxes)
#plt.title(model)
plt.text(-0.0,0.95,'g',fontsize=13,transform=ax1.transAxes,weight='bold')
plt.title('AM, CESM2, picontrol')


model='GISS-E2-1-G'
ax1=plt.subplot(338, projection=ccrs.AzimuthalEquidistant(central_longitude=0.0, central_latitude=90.0))
ax1.set_extent([-180, 180, 50, 90], ccrs.PlateCarree())
CS=ax1.contourf(lon_ex,lat,-get_extend(ts_reg_am_[model]),get_array(-1.2,1.3,0.2),transform=ccrs.PlateCarree(),cmap='RdBu_r',extend='both')
ax1.contour(lon,lat,-psl_reg_am_[model]/100.,get_array(-1,1.1,0.2),transform=ccrs.PlateCarree(),colors='k',linewidths=1.25)
ax1.coastlines(linewidth=0.5)
theta = np.linspace(0, 2*np.pi, 100)
center, radius = [0.5, 0.5], 0.5
verts = np.vstack([np.sin(theta), np.cos(theta)]).T
circle = mpath.Path(verts * radius + center)
ax1.set_boundary(circle, transform=ax1.transAxes)
#plt.title(model)
plt.text(-0.0,0.95,'h',fontsize=13,transform=ax1.transAxes,weight='bold')
plt.title('AM, GISS-E2-1-G, picontrol')

ax3=plt.subplot(339)
ind=1200
kernel = np.ones(15)/15
h1,=plt.plot(PCpi_dict_['CESM2'][:ind]+4,linewidth=0.5)
plt.plot(convolve1d(PCpi_dict_['CESM2'][:ind]+4,kernel),linewidth=1,color=h1.get_color())
plt.tick_params(labelsize=10,direction='out',length=3,width=1.,pad=3)
h2,=plt.plot(PCpi_dict_['GISS-E2-1-G'][:ind],linewidth=0.5)
plt.plot(convolve1d(PCpi_dict_['GISS-E2-1-G'][:ind],kernel),linewidth=1,color=h2.get_color())
plt.tick_params(labelsize=10,direction='out',length=3,width=1.,pad=3)
plt.xlim([0,500])
plt.ylim([-2,6])
plt.yticks([-1,0,1,3,4,5],['-1','0','1','-1','0','1'])
plt.plot([0,500],[2,2],'k-')
plt.text(-0.074,1.01,'i',fontsize=13,transform=ax3.transAxes,weight='bold')
plt.text(0.02,0.95,'CESM2',fontsize=10,transform=ax3.transAxes,color=h1.get_color())
plt.text(0.02,0.45,'GISS-E2-1-G',fontsize=10,transform=ax3.transAxes,color=h2.get_color())
plt.ylabel('Principal Component')
plt.xlabel('Yrs')



plt.subplots_adjust(wspace=-0.05,hspace=0.15)

cax = fig.add_axes([0.415, 0.08, 0.2, 0.005])
cb1=plt.colorbar(CS,  cax=cax,orientation='horizontal',pad=0.05,extendfrac=0.1,shrink=0.47,fraction=0.06,aspect=25)
cb1.ax.tick_params(labelsize=8,pad=1.6)
cb1.ax.set_xlabel('[K]',size=8,labelpad=5)
cb1.set_ticks(np.arange(-1.2,1.3,0.4))
box = ax0.get_position()
ax0.set_position([box.x0+0.04, box.y0+0.01, box.width*0.75, box.height*0.9])
box = ax2.get_position()
ax2.set_position([box.x0+0.04, box.y0+0.01, box.width*0.75, box.height*0.9])
box = ax3.get_position()
ax3.set_position([box.x0+0.03, box.y0+0.012, box.width*0.75, box.height*0.9])


plt.savefig('/Users/zhou272/Desktop/Projects/AA1/Fig_AMvsTrend.pdf',bbox_inches='tight', format='pdf', dpi=1600)



