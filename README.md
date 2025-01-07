# Generative machine learning methods for multivariate ensemble post-processing

This repository provides python codes for the conditional generative model to multivariate post-processing accompanying the paper

> Chen, J., Janke, T., Steinke, F. and Lerch, S., 2024. Generative machine learning methods for multivariate ensemble post-processing. *Ann. Appl. Stat. 18 (1) 159 - 183*. https://doi.org/10.1214/23-AOAS1784
> Chen, J., Janke, T., Steinke, F. and Lerch, S., 2022. Generative machine learning methods for multivariate ensemble post-processing. *arXiv preprint arXiv:2211.01345*.

**The version of the Tensorflow library used for the code is Tensorflow 2.3, a newer version of Tensorflow may cause the code to fail. :)**

## Data

The data needed for reproducing the results is publicly available:

> Chen, Jieyu; Lerch, Sebastian (2022): Full data for wind speed post-processing. figshare. Dataset. https://doi.org/10.6084/m9.figshare.19453622 

> Chen, Jieyu; Lerch, Sebastian (2022): Full data for temperature post-processing. figshare. Dataset. https://doi.org/10.6084/m9.figshare.19453580 

**ECMWF forecasts from TIGGE dataset**

https://software.ecmwf.int/wiki/display/TIGGE/Models

- Forecast data: 2-days ahead 50-member ensemble forecasts
- Time range: Daily forecasts from 2007-01-03 to 2016-12-31
- Meteorological variables (**t2m** and **ws** are the target variables for post-processing):

|Variable| Description|
|-------------|---------------|
|**t2m**| 2-m temperature|
|d2m| 2-m dewpoint temperature|
|cape| Convective available potential energy|
|sp| Surface pressure|
|tcc| Total cloud cover|
|q_pl850| Specific humidity at 850 hPa|
|u_pl850| U component of wind at 850 hPa|
|v_pl850| V component of wind at 850 hPa|
|ws_pl850| Wind speed at 850 hPa|
|u_pl500| U component of wind at 500 hPa|
|v_pl500| V component of wind at 500 hPa|
|gh_pl500| Geopotential height at 500 hPa|
|ws_pl500| Wind speed at 500 hPa|
|u10| 10-m U component of wind|
|v10| 10-m V component of wind|
|**ws**| 10-m wind speed|
|sshf| Sensible heat flux|
|slhf| Latent heat flux|
|ssr| Shortwave radiation flux|
|str| Longwave radiation flux|

**Observations at weather stations operated by DWD**

- Observation data: Daily observations of the target variable (2-m temperature and 10-m wind speed)
- Metadata:

|Variable| Description|
|-------------|---------------|
|lon| Longitude of station|
|lat| Latitude of station|
|alt| Altitude of station|
|orog| Altitude of model grid point|
|doy| Sine-transformed value of the day of the year|

## Explanation of the code files

- For reproducing the multivariate forecasts presented in the main paper (*please download the two datasets above from figshare first*):

|File name| Explanation |
|-------------|---------------|
|**`CGM_mvpp_t2m.py`**| Python script to implement CGM for post-processing of multivariate 2-m temperature forecasts at multiple stations. |
|**`CGM_mvpp_ws.py`**| Python script to implement CGM for post-processing of multivariate 10-m wind speed forecasts at multiple stations. |
|**`DRN_unipp_t2m.ipynb`**| Jupyter notebook to implement DRN for univariate post-processing of 2-m temperature forecasts. |
|**`DRN_unipp_ws.ipynb`**| Jupyter notebook to implement DRN for univariate post-processing of 10-m wind speed forecasts. |
|**`DRNcopula_mvpp_t2m.R`**| R script to implement DRN + ECC/GCA for post-processing of multivariate 2-m temperature forecasts at multiple stations. |
|**`DRNcopula_mvpp_ws.R`**| R script to implement DRN + ECC/GCA for post-processing of multivariate 10-m wind speed forecasts at multiple stations. |
|**`EMOScopula_mvpp_t2m.R`**| R script to implement EMOS + ECC/GCA for post-processing of multivariate 2-m temperature forecasts at multiple stations. |
|**`EMOScopula_mvpp_ws.R`**| R script to implement EMOS + ECC/GCA for post-processing of multivariate 10-m wind speed forecasts at multiple stations. |
|`cgm_models_linear.py`| Class of the conditional generative models for multivariate post-processing (the version used in the main paper), build with Keras. |
|`scoring_rules_supp.py`| Codes for computing several proper scoring rules in python. |

- Others:

|File name| Explanation |
|-------------|---------------|
|`cgm_models_nonlinear.py`| Codes for the class of the conditional generative models for multivariate post-processing (the version considered in the ablation study of the supplement), build with Keras |
