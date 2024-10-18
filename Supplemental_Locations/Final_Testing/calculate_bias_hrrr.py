import xarray as xr
import datetime
from tqdm.auto import tqdm
import numpy as np
from glob import glob
from numba import jit
from pandas import Timestamp
from scipy.stats import gamma
import sys,os
import warnings
warnings.filterwarnings('ignore')


### ------------------------ ###
###    Command line args
### ------------------------ ###

month = int(sys.argv[1])
day_idx = int(sys.argv[2])
num_sls = int(sys.argv[3])

### ------------------------ ###
###    Format files
### ------------------------ ###

sl_file = f'/coeffs_from_bias/blend.supplemental_locations_{month}.co.2p5.nc'

### ------------------------ ###
###    Gamma jit funcs
### ------------------------ ###

script_start = datetime.datetime.now()

def convert_datetime64_to_datetime( usert: np.datetime64 )->datetime.datetime:
    t = np.datetime64( usert, 'us').astype(datetime.datetime)
    return t

print("compiling jit funcs")

@jit
def NUMBA_increment_gamma_components(apcp,nsls,npositive,nzero_thresh, xsum, xsumln,
                                     npositive_nosl, nzero_thresh_nosl, xsum_nosl, xsumln_nosl,
                                     thresh=0.0):

    #ngps, nsls = np.shape(apcp)
    ngps = np.shape(apcp)[0]
    
    for gp in range(ngps):
        if (apcp[gp, 0] <= thresh):
            nzero_thresh_nosl[gp] += 1
        elif (apcp[gp, 0] > thresh) and (apcp[gp, 0] > 0.0):
            npositive_nosl[gp] += 1
            xsum_nosl[gp] += apcp[gp, 0]
            xsumln_nosl[gp] += np.log(apcp[ gp, 0])
        for sl in range(nsls):
            if (apcp[gp, sl] <= thresh):
                nzero_thresh[gp] += 1
            elif (apcp[gp, sl] > thresh) and (apcp[gp, sl] > 0.0):
                npositive[gp] += 1
                xsum[gp] += apcp[gp, sl]
                xsumln[gp] += np.log(apcp[gp, sl])
        
    return npositive, nzero_thresh, xsum, xsumln, npositive_nosl, nzero_thresh_nosl, xsum_nosl, xsumln_nosl

@jit
def compute_gamma_params(npos, nz_thresh,xsum, xsumln):
    DEFAULT_MISSING = -9999.
    
    alpha = np.zeros_like(npos,dtype=np.float32)
    beta = np.zeros_like(npos,dtype=np.float32)
    fz_thresh = np.zeros_like(npos,dtype=np.float32)
    ngps = len(npos)

    for i in range(ngps):
        if (npos[i] >= 1):
            average = xsum[i] / npos[i]
            average_of_log = (1.0 / npos[i]) * xsumln[i]
            dstat = np.log(average) - average_of_log 
            if (dstat != 0.0): #check for dstat != 0
                alpha[i] = (1.0 + (np.sqrt(1.0 + ((4.0 * dstat) / 3.0)))) / (4.0 * dstat)
            else:
                alpha[i] = DEFAULT_MISSING
            if (alpha[i] <= 0.0) or (alpha[i] > 20.):
                alpha[i] = DEFAULT_MISSING
                beta[i] = DEFAULT_MISSING
            else:
                beta[i] = average / alpha[i]
        else:
            alpha[i] = DEFAULT_MISSING
            beta[i] = DEFAULT_MISSING
        
        if ((nz_thresh[i] + npos[i]) <= 0):
            fz_thresh[i] = DEFAULT_MISSING
        else:
            fz_thresh[i] = (nz_thresh[i] / (nz_thresh[i] + npos[i]))
    return alpha, beta, fz_thresh


### ----------------------------------------------------------------------- ###


### To calculate monthy average, need to establish start,end days with respect to end of chosen month
if month == 2:
    this_month = datetime.date(2023,month,28)
else:
    this_month = datetime.date(2023,month,30)

### starting day from which we load the prev 60 days
start =  this_month - datetime.timedelta(days=day_idx) 
print(f"Running for {start.strftime('%Y-%m-%d')}")

#print(day_datetime)

print("Loading supplemental locations data")
sl = xr.open_dataset(f'/scratch2/STI/mdl-sti/Sidney.Lower/supplemental_locations/{sl_file}')
xlocs = sl.xlocations.data
ylocs = sl.ylocations.data
differences = sl.differences.data
sl.close()

### load valid grid points
conus_sample = xr.open_dataset('/scratch2/STI/mdl-sti/Sidney.Lower/supplemental_locations/hrrr_conus_grid.nc')
gp_lat_idx = conus_sample.latitude_idx.data
gp_lon_idx = conus_sample.longitude_idx.data

ngps = len(gp_lat_idx)

### Big ol block of initialization
npos_hrrr = np.zeros((ngps),dtype=np.int32)
nzero_hrrr = np.zeros((ngps),dtype=np.int32)
xsum_hrrr = np.zeros((ngps),dtype=np.float32)
xsumln_hrrr = np.zeros((ngps),dtype=np.float32)

npos_hrrr_nosl = np.zeros((ngps),dtype=np.int32)
nzero_hrrr_nosl = np.zeros((ngps),dtype=np.int32)
xsum_hrrr_nosl = np.zeros((ngps),dtype=np.float32)
xsumln_hrrr_nosl = np.zeros((ngps),dtype=np.float32)

npos_urma = np.zeros((ngps),dtype=np.int32)
nzero_urma = np.zeros((ngps),dtype=np.int32)
xsum_urma = np.zeros((ngps),dtype=np.float32)
xsumln_urma = np.zeros((ngps),dtype=np.float32)

npos_urma_nosl = np.zeros((ngps),dtype=np.int32)
nzero_urma_nosl = np.zeros((ngps),dtype=np.int32)
xsum_urma_nosl = np.zeros((ngps),dtype=np.float32)
xsumln_urma_nosl = np.zeros((ngps),dtype=np.float32)


print("Get supplemental locations at chosen grid points")
### this can be done outside the loop since it's monthly based
x_at_each_station = np.empty((len(gp_lon_idx), num_sls), dtype=int)
y_at_each_station = np.empty((len(gp_lon_idx), num_sls), dtype=int)
for stat in range(len(gp_lat_idx)):
    temp = xlocs[:num_sls,gp_lat_idx[stat], gp_lon_idx[stat]]
    x_at_each_station[stat,:] = temp.astype(int) - 1
    temp = ylocs[:num_sls,gp_lat_idx[stat], gp_lon_idx[stat]]
    y_at_each_station[stat,:] = temp.astype(int) - 1

### some stations have < 50 SLs?
if len(np.where(x_at_each_station > 3000)[0]) > 0:
    print("Adjusting x,y loc indicies for NaN entries")
    count = np.where(x_at_each_station > 3000)[1]
    where_0 = np.where(x_at_each_station > 3000)[0]
    x_at_each_station[where_0, count] = x_at_each_station[where_0, 0]
    y_at_each_station[where_0, count] = y_at_each_station[where_0, 0]


### Some URMA files are missing in Dec/Jan so we need to account for those missing files
### and I really can't be bothered with the sloppiness of a try/except in the 60 day loop
ufiles, hfiles = [],[]
for day in range(60):
    this_day = start - datetime.timedelta(days=day)
    file = f'/scratch2/STI/mdl-sti/Sidney.Lower/data/urma/24h_precip/{this_day.strftime("%Y%m%d")}/urma2p5.{this_day.strftime("%Y%m%d")}.pcp_24h.wexp.grb2'
    tmp = glob(file)
    if len(tmp) == 0:
        continue 
    else:
        ufiles.append(file)
        hfiles.append(f'/scratch2/STI/mdl-sti/Sidney.Lower/data/hrrr/{this_day.strftime("%Y%m%d")}/hrrr.{this_day.strftime("%Y%m%d")}.pcp_24h.t00z.grib2')

valid_urma_days = len(hfiles)

print(f"Loading HRRR & URMA for {this_month.strftime('%B')} through {this_day.strftime('%B')}")
hrrr_data = xr.open_mfdataset(hfiles, engine='grib2io', combine='nested', concat_dim='refDate')
urma_data = xr.open_mfdataset(ufiles, engine='grib2io', combine='nested', concat_dim='refDate')

hrrr_precip = hrrr_data.APCP.data.compute()
urma_precip = urma_data.APCP.data.compute()


monthly_bias_sl = np.zeros((ngps)) #30 days, grid points
monthly_bias_nosl = np.zeros((ngps))

# Calculate Gamma params from 60 days worth of data
print(f"Summing data over previous days from {start}")
for day in tqdm(range(valid_urma_days)):

    sl_hrrr = hrrr_precip[day, y_at_each_station, x_at_each_station] 
    sl_urma = urma_precip[day, y_at_each_station, x_at_each_station]

    sl_hrrr = np.where(sl_hrrr < 0.254, 0., sl_hrrr)
    sl_urma = np.where(sl_urma < 0.254, 0.0, sl_urma)


    npos_hrrr, nzero_hrrr, xsum_hrrr, xsumln_hrrr,npos_hrrr_nosl, nzero_hrrr_nosl, xsum_hrrr_nosl, xsumln_hrrr_nosl = NUMBA_increment_gamma_components(sl_hrrr.astype(np.float32),num_sls,npos_hrrr,nzero_hrrr,xsum_hrrr,xsumln_hrrr,npos_hrrr_nosl,
                                                       nzero_hrrr_nosl,xsum_hrrr_nosl,xsumln_hrrr_nosl,thresh=0.254)

    
    npos_urma, nzero_urma, xsum_urma, xsumln_urma, npos_urma_nosl, nzero_urma_nosl, xsum_urma_nosl, xsumln_urma_nosl = NUMBA_increment_gamma_components(sl_urma.astype(np.float32),num_sls,npos_urma,nzero_urma,xsum_urma,xsumln_urma,npos_urma_nosl,
                                                       nzero_urma_nosl,xsum_urma_nosl,xsumln_urma_nosl,thresh=0.254)


print("Getting gamma params")
alpha_hrrr,beta_hrrr,fraczero_hrrr = compute_gamma_params(npos_hrrr,nzero_hrrr,xsum_hrrr,xsumln_hrrr)
alpha_hrrr_nosl,beta_hrrr_nosl,fraczero_hrrr_nosl = compute_gamma_params(npos_hrrr_nosl,nzero_hrrr_nosl,xsum_hrrr_nosl,xsumln_hrrr_nosl)
alpha_urma,beta_urma,fraczero_urma = compute_gamma_params(npos_urma,nzero_urma,xsum_urma,xsumln_urma)
alpha_urma_nosl,beta_urma_nosl,fraczero_urma_nosl = compute_gamma_params(npos_urma_nosl,nzero_urma_nosl,xsum_urma_nosl,xsumln_urma_nosl)

#print(alpha_hrrr)

#this_day_analysis = urma_precip[0,gp_lat_idx, gp_lon_idx]
this_day_fc = hrrr_precip[0,gp_lat_idx, gp_lon_idx] #this day, first SL == origin grid point; shape is now 2260 grid points
qmd_forecast = np.zeros_like(this_day_fc, dtype=np.float32)
qmd_forecast_nosl = np.zeros_like(this_day_fc, dtype=np.float32)

print(f"QMD for {start.strftime('%Y-%m-%d')}")
### clean bad data ###

fc_cdf = fraczero_hrrr + (1.-fraczero_hrrr)*gamma.cdf(this_day_fc,alpha_hrrr,scale=beta_hrrr)

where_0 = list(np.where(this_day_fc < 0.254)[0]) + list(np.where(fc_cdf < fraczero_urma)[0])
qmd_forecast[where_0] = 0.0
qmd_forecast_nosl[where_0] = 0.0

where_bad_params = np.unique(list(np.where(alpha_urma < 0.0)[0]) + 
                             list(np.where(alpha_hrrr < 0.0)[0]) +
                             list(np.where(beta_hrrr < 0.0)[0]) +
                             list(np.where(beta_urma < 0.0)[0])).astype(np.int32)
qmd_forecast[where_bad_params] = this_day_fc[where_bad_params]
qmd_forecast_nosl[where_bad_params] = this_day_fc[where_bad_params]

frac_above_90 = len(np.where(fc_cdf >= 0.9)[0]) / ngps

all_bad = set(list(np.unique(where_0 + list(where_bad_params))))
gpsss = set(list(np.arange(ngps)))
good_gps = list(gpsss - all_bad)

fc_cdf2 = (fc_cdf[good_gps] - fraczero_urma[good_gps]) / (1.0 - fraczero_urma[good_gps])
qmd_fc = gamma.ppf(fc_cdf2, alpha_urma[good_gps], scale=beta_urma[good_gps])
qmd_forecast[good_gps] = qmd_fc

fc_cdf = fraczero_hrrr_nosl + (1.-fraczero_hrrr_nosl)*gamma.cdf(this_day_fc,alpha_hrrr_nosl,scale=beta_hrrr_nosl)
fc_cdf2 = (fc_cdf[good_gps] - fraczero_urma_nosl[good_gps]) / (1.0 - fraczero_urma_nosl[good_gps])
qmd_fc = gamma.ppf(fc_cdf2, alpha_urma_nosl[good_gps], scale=beta_urma_nosl[good_gps])
qmd_forecast_nosl[good_gps] = qmd_fc


if len(np.where(np.isfinite(qmd_forecast) == False)[0]) > 0:
    print('-------- Warning: ')
    print(f"              NaN: {np.where(np.isfinite(qmd_forecast) == False)}")
#qmd_forecast = np.where(qmd_forecast<0.254, 0., qmd_forecast)
#qmd_forecast_nosl = np.where(qmd_forecast_nosl<0.254, 0., qmd_forecast_nosl)

#print("Calculating Bias")
this_day = start + datetime.timedelta(hours=24)
dt = convert_datetime64_to_datetime(this_day).strftime("%Y%m%d")
a = xr.open_dataset(f'/scratch2/STI/mdl-sti/Sidney.Lower/data/urma/24h_precip/{dt}/urma2p5.{dt}.pcp_24h.wexp.grb2',
               engine='grib2io')
print(f'Loading analysis data (HRRR forecast valid {dt}')
a_at_stations = a.APCP.data[gp_lat_idx, gp_lon_idx]
analysis_precip = np.where(a_at_stations < 0.254, 0.0, a_at_stations)

print("Calculating Bias")
monthly_bias_sl = qmd_forecast - analysis_precip
monthly_bias_nosl = qmd_forecast_nosl - analysis_precip


day_datetime = Timestamp(start.strftime('%Y%m%d'))

print("\nSaving")

bias_sl = xr.Dataset(
    data_vars=dict(
        hrrr_qmd=(["grid_point"], qmd_forecast),
        hrrr_raw=(["grid_point"], this_day_fc),
        urma_analysis=(["grid_point"], analysis_precip)
    ),
    coords=dict(
        grid_point=np.arange(ngps),
        day=day_datetime,
        num_sls=num_sls,
    ),
    attrs=dict(description="URMA analysis, QMD precip, and raw HRRR precip"),
)

bias_nosl = xr.Dataset(
    data_vars=dict(
        hrrr_qmd=(["grid_point"], qmd_forecast_nosl)
    ),
    coords=dict(
        grid_point=np.arange(ngps),
        day=day_datetime,
    ),
    attrs=dict(description="QMD precip without SLs"),
)



#prior to saving, do some directory setup
outdir = f'//scratch2/STI/mdl-sti/Sidney.Lower/supplemental_locations/coeffs_from_bias/test_number_sls/hrrr/'
if not os.path.isdir(outdir):
    os.mkdir(outdir)

#print(f"Saving to {outdir}")
bias_outfile = outdir+f"/bias_nsls{num_sls}_{start.strftime('%Y%m%d')}_CONUS.nc"
bias_sl.to_netcdf(bias_outfile, mode='w',engine='h5netcdf', encoding={"hrrr_qmd": {"zlib": True, "complevel": 4},
                                                                     "hrrr_raw": {"zlib": True, "complevel": 4},
                                                                     "urma_analysis": {"zlib": True, "complevel": 4}})
#bias_outfile = outdir+f"/bias_no_sl_{start.strftime('%Y%m%d')}_CONUS.nc"
#bias_nosl.to_netcdf(bias_outfile, mode='w',engine='h5netcdf', encoding={"hrrr_qmd": {"zlib": True, "complevel": 4}})


script_finish = datetime.datetime.now()
print(f"script took {(script_finish-script_start).total_seconds()}s to run")

