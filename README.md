# Generative machine learning methods for multivariate ensemble post-processing

This repository provides python codes for the conditional generative model to multivariate post-processing accompanying the paper

> Chen, Jieyu; Janke, Tim; Lerch, Sebastian (2022):

## Data

The data needed for reproducing the results is publicly available:

> Chen, Jieyu; Lerch, Sebastian (2022): Full data for wind speed post-processing. figshare. Dataset. https://doi.org/10.6084/m9.figshare.19453622 

> Chen, Jieyu; Lerch, Sebastian (2022): Full data for temperature post-processing. figshare. Dataset. https://doi.org/10.6084/m9.figshare.19453580 

**ECMWF forecasts from TIGGE dataset**

https://software.ecmwf.int/wiki/display/TIGGE/Models

- Variables: 50-member ensemble forecasts of the target variable (2 m temperature or 10 m wind speed), and some additional predictors
- Time range: Daily forecasts from 2007-01-03 to 2016-12-31

**Observations at weather stations operated by DWD?**

- Variables: Daily observations of the target variable (2 m temperature or 10 m wind speed), and the location information (longitude, latitude, altitude) of each station

## 

cgm_models.py include a class of conditional generative models for
wind speed and temperature forecast separately.

mvpp_cgm_ws.py and mvpp_cgm_t2m.py provide codes to use the conditional generative models for multivariate
post-processing of wind speed or temperature, experimented on a test of 100 repetitions.

generating_test_stations.py shows how to select a list of test sets, with each set contains several weather 
stations that are located close to each other.

