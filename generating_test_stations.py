
import numpy as np
import pandas as pd
import random as rd


# read data
path = 'C:/Users/gm2154/Documents/0-TIGGE_ECMWF_Germany/temperature_ensemble_forecasts.feather'
t2m_ens_complete = pd.read_feather(path)

# read stations information
path = 'C:/Users/gm2154/Documents/0-TIGGE_ECMWF_Germany/station_info_temperature.feather'
station_info = pd.read_feather(path)

station_all = station_info['station_id']
station_count = t2m_ens_complete.groupby(['station'])['obs'].count()

# select stations that have enough days (more than 2,500 days) of forecast & observation data
stations_enough_obs = station_count[station_count >= 3500]
stationid_enough_obs = stations_enough_obs.index

# select stations with altitude lower than 1,000 meters.
station_low = station_info[station_info['station_alt'] < 1000]['station_id']
station_used = station_all[station_all.isin(station_low) & 
                           station_all.isin(stationid_enough_obs)]

# station IDs that are considered in the study
station_used_tem = station_used

# import distance matrices (produced in R using "geosphere" package)
geo_distance = pd.read_csv("C:/Users/gm2154/Documents/0-TIGGE_ECMWF_Germany/geo_distance.csv", index_col=0)
geo_stations_all = pd.Series(geo_distance.index.values)
geo_stations_used = station_used.apply(lambda val: int(val))
labels_stations = geo_stations_all.isin(geo_stations_used)

geo_distance_matrix = geo_distance.loc[labels_stations.values,labels_stations.values]
geo_distance_matrix = geo_distance_matrix.rename(columns=lambda val: float(val), 
                                                 index=lambda val: float(val))


# produce new subsets of stations for testing different post-processing models
# first pick a random station, then select several nearest stations
x = 100     # number of repeated tests
dim = 10     # dimension of the multivariate forecast considered in the tests

dist_samples = np.empty((x,dim))

for test in range(x):
    center_station = rd.sample(list(station_used), 1)
    
    distances = geo_distance_matrix[center_station]
    
    sort_dist = distances.sort_values()
    
    dist_station_sample = sort_dist.index[:dim].values
    
    dist_samples[test,] = dist_station_sample


# save the test sets of stations
np.savetxt("ws_dist_10samples.csv", dist_samples, delimiter=",")



