# Generative machine learning methods for multivariate ensemble post-processing

This repository provides python and R codes accompanying the paper

> to be added

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

## Multivariate post-processing

We focus on preserving the spatial dependence structure in the multivariate forecasts during ensemble post-processing. 

The raw data in feather format could be downloaded from

The generative models for multivariate post-processing are performed using Python language, while the standard
copula-based multivariate approaches are applied using R language.

cgm_models.py include a class of conditional generative models for
wind speed and temperature forecast separately.

mvpp_cgm_ws.py and mvpp_cgm_t2m.py provide codes to use the conditional generative models for multivariate
post-processing of wind speed or temperature, experimented on a test of 100 repetitions.

generating_test_stations.py shows how to select a list of test sets, with each set contains several weather 
stations that are located close to each other.

model_hyperparameter_tuning_wind.py and model_hyperparameter_tuning_tem.py provide a way to find the optimal
hyperparameters for the generative models using the 'hyperopt' library.

multipp_t2m_10dim.R shows an example of using standard copula-based approaches for post-processing 
10-dimensional temperature forecasts.

mvpp_drn_copula_ws.R and mvpp_drn_copula_t2m.R provide codes to perform the two-step copula-based multivariate post-processing
with DRN (distributional regression network) for the univariate post-processing and ECC (ensemble copula coupling) or GCA
(Gaussian copula approach) for multivariate extention, i.e., DRN + ECC and DRN + GCA, for wind speed and temperature forecast respectively.

