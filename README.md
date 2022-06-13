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

igep_models_all_wind_noplot.py and igep_models_all_tem_noplot.py include a class of generative models for
wind speed and temperature data separately.

generative_model_wind.py and generative_model_t2m.py provide codes to use the generative models for multivariate
post-processing, experimented on a list of test sets.

generating_test_stations.py shows how to select a list of test sets, with each set contains several weather 
stations that are located close to each other.

model_hyperparameter_tuning_wind.py and model_hyperparameter_tuning_tem.py provide a way to find the optimal
hyperparameters for the generative models using the 'hyperopt' library.

multipp_t2m_10dim.R shows an example of using standard copula-based approaches for post-processing 
10-dimensional temperature forecasts.

multipp_nncopula_t2m_10dim.R provides a way to perform 'NN + ECC' and 'NN + SSh' for multivariate ensemble
post-processing of 10-dimensional temperature forecasts.

multipp_nngca_t2m_10dim.R provides a way to perform 'NN + GCA' for multivariate ensemble post-processing 
of 10-dimensional temperature forecasts.

