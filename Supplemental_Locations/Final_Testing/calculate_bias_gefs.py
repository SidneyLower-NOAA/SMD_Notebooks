import xarray as xr
import datetime
from tqdm.auto import tqdm
import numpy as np
from glob import glob
import grib2io
from numba import jit
from pandas import Timestamp
from scipy.stats import gamma
import sys,os
import warnings
warnings.filterwarnings('ignore')


from nimbl.metadata import get_metadata
conus_grid_def = get_metadata('grib2_section3', model='blend',region='co' )
conus_grid = grib2io.Grib2GridDef(conus_grid_def[4], conus_grid_def[5:])

### ------------------------ ###
###    Command line args
### ------------------------ ###

month = int(sys.argv[1])
day_idx = int(sys.argv[2])
num_sls = int(sys.argv[3])

### ------------------------ ###
###    Format files
### ------------------------ ###

sl_file = f'/diurnal_cycle/blend.supplemental_locations_{month}.co.2p5.nc'


### ------------------------ ###
###    Gamma jit funcs
### ------------------------ ###

script_start=datetime.datetime.now()

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
def NUMBA_increment_gamma_components_ensemble(apcp,nsls,npositive,nzero_thresh, xsum, xsumln,
                                     npositive_nosl, nzero_thresh_nosl, xsum_nosl, xsumln_nosl,
                                     thresh=0.0):

    nmems, ngps, _ = np.shape(apcp)

    for mem in range(nmems):
        for gp in range(ngps):
            if (apcp[mem, gp, 0] <= thresh):
                    nzero_thresh_nosl[gp] += 1
            elif (apcp[mem, gp, 0] > thresh) and (apcp[mem, gp, 0] > 0.0):
                npositive_nosl[gp] += 1
                xsum_nosl[gp] += apcp[mem, gp, 0]
                xsumln_nosl[gp] += np.log(apcp[mem, gp, 0])
            for sl in range(nsls):
                if (apcp[mem, gp, sl] <= thresh):
                    nzero_thresh[gp] += 1
                elif (apcp[mem, gp, sl] > thresh) and (apcp[mem, gp, sl] > 0.0):
                    npositive[gp] += 1
                    xsum[gp] += apcp[mem, gp, sl]
                    xsumln[gp] += np.log(apcp[mem, gp, sl])
        
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


print("Loading supplemental locations data")
sl = xr.open_dataset(f'/scratch2/STI/mdl-sti/Sidney.Lower/supplemental_locations/{sl_file}')
xlocs = sl.xlocations.data
ylocs = sl.ylocations.data
sl.close()

### load valid grid points
conus_sample = xr.open_dataset('/scratch2/STI/mdl-sti/Sidney.Lower/supplemental_locations/thinned_conus_grid.nc')
gp_lat_idx = conus_sample.latitude_idx.data
gp_lon_idx = conus_sample.longitude_idx.data

ngps = len(gp_lat_idx)

### Big ol block of initialization
npos_gefs = np.zeros((ngps),dtype=np.int32)
nzero_gefs = np.zeros((ngps),dtype=np.int32)
xsum_gefs = np.zeros((ngps),dtype=np.float32)
xsumln_gefs = np.zeros((ngps),dtype=np.float32)

npos_gefs_nosl = np.zeros((ngps),dtype=np.int32)
nzero_gefs_nosl = np.zeros((ngps),dtype=np.int32)
xsum_gefs_nosl = np.zeros((ngps),dtype=np.float32)
xsumln_gefs_nosl = np.zeros((ngps),dtype=np.float32)

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

ufiles, gfiles = [],[]
for day in range(60):
    this_day = start - datetime.timedelta(days=day)
    file = f'/scratch2/STI/mdl-sti/Sidney.Lower/data/urma/6h_precip/{this_day.strftime("%Y%m%d")}/urma2p5.{this_day.strftime("%Y%m%d")}18.pcp_06h.wexp.grb2'
    tmp = glob(file)
    if len(tmp) == 0:
        continue 
    else:
        ufiles.append(file)
        gf = sorted(glob(f'/scratch2/STI/mdl-sti/Sidney.Lower/data/gefs/supplemental_locations_QMD/{this_day.strftime("%Y%m%d")}/gefs*.f48.APCP.6h'))
        gfiles.append(gf)

valid_urma_days = len(ufiles)

print(f"Loading URMA for {this_month.strftime('%B')} through {this_day.strftime('%B')}")
urma_data = xr.open_mfdataset(ufiles, engine='grib2io', combine='nested', concat_dim='refDate')
urma_precip = urma_data.APCP.data.compute()


# Calculate Gamma params from 60 days worth of data
print(f"Summing data over previous days from {start}")
for day in tqdm(range(valid_urma_days)):

    gefs_data = xr.open_mfdataset(gfiles[day], engine='grib2io', combine='nested', concat_dim='perturbationNumber')
    gefs_to_conus = gefs_data.grib2io.interp('bilinear', conus_grid)
    gefs_precip = gefs_to_conus.APCP.data.compute()
    while len(np.where(np.isfinite(gefs_precip) == False)[0]) > 0:
        print('-------- Warning: ')
        print(f"              NaNs in GEFs APCP, reloading")
        gefs_to_conus = gefs_data.grib2io.interp('bilinear', conus_grid)
        gefs_precip = gefs_to_conus.APCP.data.compute()
    

    sl_gefs = gefs_precip[:, y_at_each_station, x_at_each_station] 
    sl_urma = urma_precip[day, y_at_each_station, x_at_each_station]

    sl_gefs = np.where(sl_gefs < 0.254, 0., sl_gefs)
    sl_urma = np.where(sl_urma < 0.254, 0.0, sl_urma)


    npos_gefs, nzero_gefs, xsum_gefs, xsumln_gefs,npos_gefs_nosl, nzero_gefs_nosl, xsum_gefs_nosl, xsumln_gefs_nosl = NUMBA_increment_gamma_components_ensemble(sl_gefs.astype(np.float32),num_sls,npos_gefs,nzero_gefs,xsum_gefs,xsumln_gefs,npos_gefs_nosl,
                                                       nzero_gefs_nosl,xsum_gefs_nosl,xsumln_gefs_nosl,thresh=0.254)

    
    npos_urma, nzero_urma, xsum_urma, xsumln_urma, npos_urma_nosl, nzero_urma_nosl, xsum_urma_nosl, xsumln_urma_nosl = NUMBA_increment_gamma_components(sl_urma.astype(np.float32),num_sls,npos_urma,nzero_urma,xsum_urma,xsumln_urma,npos_urma_nosl,
                                                       nzero_urma_nosl,xsum_urma_nosl,xsumln_urma_nosl,thresh=0.254)


print("Getting gamma params")
alpha_gefs,beta_gefs,fraczero_gefs = compute_gamma_params(npos_gefs,nzero_gefs,xsum_gefs,xsumln_gefs)
alpha_gefs_nosl,beta_gefs_nosl,fraczero_gefs_nosl = compute_gamma_params(npos_gefs_nosl,nzero_gefs_nosl,xsum_gefs_nosl,xsumln_gefs_nosl)
alpha_urma,beta_urma,fraczero_urma = compute_gamma_params(npos_urma,nzero_urma,xsum_urma,xsumln_urma)
alpha_urma_nosl,beta_urma_nosl,fraczero_urma_nosl = compute_gamma_params(npos_urma_nosl,nzero_urma_nosl,xsum_urma_nosl,xsumln_urma_nosl)

#print(alpha_gefs)

gefs_data = xr.open_mfdataset(gfiles[day_idx], engine='grib2io', combine='nested', concat_dim='perturbationNumber')
gefs_to_conus = gefs_data.grib2io.interp('bilinear', conus_grid)
gefs_precip = gefs_to_conus.APCP.data.compute()
this_day_fc =gefs_precip[:,gp_lat_idx, gp_lon_idx] #this day, first SL == origin grid point; shape is now 2260 grid points x 30 members
qmd_forecast = np.zeros_like(this_day_fc, dtype=np.float32)
qmd_forecast_nosl = np.zeros_like(this_day_fc, dtype=np.float32)

print(f"QMD for {start.strftime('%Y-%m-%d')}")
for mem in tqdm(range(30)):
    for gp in range(len(gp_lat_idx)):
        if this_day_fc[mem,gp] == 0.0: #FCST STAYS 0
            qmd_forecast[mem, gp] = 0.0
            qmd_forecast_nosl[mem,gp] = 0.0
        else: #DO QMD
            # WITH SLs
            # CHECK FOR BAD GAMMA PARAMS
            if (alpha_urma[gp] < 0) or (beta_urma[gp] < 0) or (alpha_gefs[gp] < 0) or (beta_gefs[gp] < 0):
                qmd_forecast[mem,gp] = this_day_fc[mem,gp]
            else:
                fc_cdf = fraczero_gefs[gp] + (1.-fraczero_gefs[gp])*gamma.cdf(this_day_fc[mem,gp],alpha_gefs[gp],scale=beta_gefs[gp])
                if (fc_cdf < fraczero_urma[gp]): 
                    # in FORTRAN version :: QMD should be 0 !
                    qmd_forecast[mem,gp] = 0.0 #this_day_fc[mem,gp]
                else:
                    # fc_cdf < 0.9, good
                    # else alternate regression --> slope
                    # count # of cdf >= 0.9
                    fc_cdf2 = (fc_cdf - fraczero_urma[gp]) / (1.0 - fraczero_urma[gp])
                    qmd_fc = gamma.ppf(fc_cdf2, alpha_urma[gp], scale=beta_urma[gp])
                    qmd_forecast[mem,gp] = qmd_fc
            
            # WITHOUT SLs
            # CHECK FOR BAD GAMMA PARAMS
            if (alpha_urma_nosl[gp] < 0) or (beta_urma_nosl[gp] < 0) or (alpha_gefs_nosl[gp] < 0) or (beta_gefs_nosl[gp] < 0):
                qmd_forecast_nosl[mem, gp] = this_day_fc[mem,gp]
            else:
                fc_cdf = fraczero_gefs_nosl[gp] + (1.-fraczero_gefs_nosl[gp]) * gamma.cdf(this_day_fc[mem,gp],
                                                                                          alpha_gefs_nosl[gp],
                                                                                          scale=beta_gefs_nosl[gp])
                if (fc_cdf < fraczero_urma_nosl[gp]):
                    qmd_forecast_nosl[mem, gp] = this_day_fc[mem,gp]
                else:
                    fc_cdf2 = (fc_cdf - fraczero_urma_nosl[gp]) / (1.0 - fraczero_urma_nosl[gp])
                    qmd_fc = gamma.ppf(fc_cdf2, alpha_urma_nosl[gp], scale=beta_urma_nosl[gp])
                    qmd_forecast_nosl[mem,gp] = qmd_fc


if len(np.where(np.isfinite(qmd_forecast) == False)[0]) > 0:
    print('-------- Warning: ')
    print(f"              NaN: {np.where(np.isfinite(qmd_forecast) == False)}")
#qmd_forecast = np.where(qmd_forecast<0.254, 0., qmd_forecast)
#qmd_forecast_nosl = np.where(qmd_forecast_nosl<0.254, 0., qmd_forecast_nosl)

#print("Calculating Bias")
this_day = start + datetime.timedelta(hours=48)
dt = convert_datetime64_to_datetime(this_day).strftime("%Y%m%d")
a = xr.open_dataset(f'/scratch2/STI/mdl-sti/Sidney.Lower/data/urma/6h_precip/{dt}/urma2p5.{dt}18.pcp_06h.wexp.grb2',
               engine='grib2io')
print(f'Loading analysis data (gefs forecast valid {dt}')
a_at_stations = a.APCP.data[gp_lat_idx, gp_lon_idx]
analysis_precip = np.where(a_at_stations < 0.254, 0.0, a_at_stations)

day_datetime = Timestamp(start.strftime('%Y%m%d'))
print("\nSaving")
bias_sl = xr.Dataset(
    data_vars=dict(
        gefs_qmd=(["member","grid_point"], qmd_forecast),
        gefs_raw=(["member","grid_point",], this_day_fc),
        urma_analysis=(["grid_point"], analysis_precip)
    ),
    coords=dict(
        member=np.arange(30),
        grid_point=np.arange(ngps),
        day=day_datetime,
        num_sls=num_sls,
    ),
    attrs=dict(description="URMA analysis, QMD precip, and raw gefs precip"),
)

bias_nosl = xr.Dataset(
    data_vars=dict(
        gefs_qmd=(["member","grid_point"], qmd_forecast_nosl)
    ),
    coords=dict(
        member=np.arange(30),
        grid_point=np.arange(ngps),
        day=day_datetime,
    ),
    attrs=dict(description="QMD precip without SLs"),
)



#prior to saving, do some directory setup
outdir = f'//scratch2/STI/mdl-sti/Sidney.Lower/supplemental_locations/diurnal_cycle/gefs/'
if not os.path.isdir(outdir):
    os.mkdir(outdir)

#print(f"Saving to {outdir}")
bias_outfile = outdir+f"/bias_nsls{num_sls}_{start.strftime('%Y%m%d')}_CONUS.nc"
bias_sl.to_netcdf(bias_outfile, mode='w',engine='h5netcdf', encoding={"gefs_qmd": {"zlib": True, "complevel": 4},
                                                                     "gefs_raw": {"zlib": True, "complevel": 4},
                                                                     "urma_analysis": {"zlib": True, "complevel": 4}})
bias_outfile = outdir+f"/bias_nosl_{start.strftime('%Y%m%d')}_CONUS.nc"
bias_nosl.to_netcdf(bias_outfile, mode='w',engine='h5netcdf', encoding={"gefs_qmd": {"zlib": True, "complevel": 4}})

script_finish=datetime.datetime.now()

print(f"script took {(script_finish-script_start).total_seconds()}s to run")

