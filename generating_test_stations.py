"""
@author: Jieyu Chen

"""

import numpy as np
import pandas as pd
import random as rd


# specify target variable
var = "ws" # "ws" (wind speed) or "t2m" (temperature)

# read data
path = '/windspeed_data_cgm_std.feather'
# or temperature_data_cgm_std.feather
ens_complete = pd.read_feather(path)

station_all = ens_complete.groupby(['station'])['station'].unique().index
station_count = ens_complete.groupby(['station'])['obs'].count()
station_alt = ens_complete.groupby(['station'])['alt'].unique()

# select stations that have enough days (more than 2,500 days) of forecast & observation data
station_enough_obs = station_count[station_count >= 3500].index

# select stations with altitude lower than 1,000 meters.
station_low = station_alt[station_alt < 1000].index

# the list of station IDs that are used in the study
station_used = station_all[station_all.isin(station_low) & 
                           station_all.isin(station_enough_obs)]
station_used

### generate 100 repetitions of different subset of multiple stations for the experiments
# import distance matrices (produced in R using "geosphere" package)
geo_distance = pd.read_csv("/" + var + "_geo_distance.csv", index_col=0)
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
np.savetxt(var + "_dist_"+ str(dim) +"samples.csv", dist_samples, delimiter=",")



