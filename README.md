# Sidney's Notebooks for NBM Post-Processing

- how2_grib2io: introduction to grib2io for processing GRIB2 files, navigating and extracting messages, and handling data with xarray and metpy

- QMD: demonstrates quantile mapping, first in 1D then extending to GEFS maximum temperature forecast mapped to URMA, using grib2io backend support in xarray for grid interpolation and scipy stats for CDF construction.

- Data_Assimilation: introduction to data assimilation, briefly touching on 3DVar and introducing coalescence

- Coalescence_Notebooks
  - Coalescence_Demo: more in depth look at using coelescence to determine a more physically meaningful deterministic value from the GEFS ensemble forecasts for 6h precipitation
  - Coalescence_Testing: handful of test cases to explore objective function space, free parameter tunings, and minimization/interpolation options
 
- Supplemental Locations
  - Analyze_Supplemental_Locations: exploring supplemental locations (SL) results, analyzing efficacy of correlations between selected points and their SLs and testing various parameter/model variations
  - Precip_Gamma_Params: developing python-based method to calculate gamma CDF parameters for MSWEP archive
  - Diurnal_Cycle: exploring methods to incorporate diurnal cycle information into supplemental locations algorithm
  - Forecast_Scoring_and_Verification  & Final Testing: wrapping up verification of supplemental locations baseline method
