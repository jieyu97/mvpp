# library(scoringRules)
library(lubridate)
# library(crch)
library(tidyverse)
# library(reshape2)
# library(MASS)
library(feather)

##################################################
####### helper function for reading and splitting raw ensemble forecast data:
read_ndim_data = function(subset_station_lists, n_sample, raw_ens_fcst){
  stations_sample = subset_station_lists[n_sample,]
  
  ens_fcst_subset = raw_ens_fcst[which(raw_ens_fcst$station %in% stations_sample),]
  
  # Fixed training data 2007-2015, test data 2016
  eval_start <- as.Date("2016-01-01 00:00 UTC")
  eval_end <- as.Date("2016-12-31 00:00 UTC")
  eval_dates <- seq(eval_start, eval_end, by = "1 day")
  data_eval_all <- subset(ens_fcst_subset, 
                          date >= eval_start & date <= eval_end)
  
  raw_ens_eval = as.data.frame(data_eval_all[,c(1,3,4:53,2)])
  
  return(raw_ens_eval)
}

#####################################
######## temperature:
path_ens_complete <- "/home/chen_jieyu/IGEP/ens_fc_t2m_complete.feather"
ens_fc_t2m_complete <- read_feather(path_ens_complete)

dist_5samples = 
  read.csv("/home/chen_jieyu/IGEP/dist_5samples.csv",
           header = FALSE)
dist_10samples = 
  read.csv("/home/chen_jieyu/IGEP/dist_10samples.csv",
           header = FALSE)
dist_20samples = 
  read.csv("/home/chen_jieyu/IGEP/dist_20samples.csv",
           header = FALSE)

######## loop over 100 tests of subset stations:
an = 100

for (n in 1:an) {
  print(n)

  ens_fcst_5dim = read_ndim_data(dist_5samples, n, ens_fc_t2m_complete)
  
  ens_fcst_10dim = read_ndim_data(dist_10samples, n, ens_fc_t2m_complete)
  
  ens_fcst_20dim = read_ndim_data(dist_20samples, n, ens_fc_t2m_complete)

  write.csv(ens_fcst_5dim,
          file = paste0("/Data/Jieyu_data/raw_ens_fcst/t2m_5dim_sa",n,".csv"))
  
  write.csv(ens_fcst_10dim,
            file = paste0("/Data/Jieyu_data/raw_ens_fcst/t2m_10dim_sa",n,".csv"))
  
  write.csv(ens_fcst_20dim,
            file = paste0("/Data/Jieyu_data/raw_ens_fcst/t2m_20dim_sa",n,".csv"))

}


#####################################
######## wind speed:
load("/home/chen_jieyu/IGEP/ecmwf_ens_wind_data.RData")

ws_dist_5samples = 
  read.csv("/home/chen_jieyu/IGEP/ws_dist_5samples.csv",
           header = FALSE)
ws_dist_10samples = 
  read.csv("/home/chen_jieyu/IGEP/ws_dist_10samples.csv",
           header = FALSE)
ws_dist_20samples = 
  read.csv("/home/chen_jieyu/IGEP/ws_dist_20samples.csv",
           header = FALSE)

######## loop over 100 tests of subset stations:
an = 100

for (n in 1:an) {
  print(n)
  
  ens_fcst_5dim = read_ndim_data(ws_dist_5samples, n, ens_fc_ws_complete)
  
  ens_fcst_10dim = read_ndim_data(ws_dist_10samples, n, ens_fc_ws_complete)
  
  ens_fcst_20dim = read_ndim_data(ws_dist_20samples, n, ens_fc_ws_complete)
  
  write.csv(ens_fcst_5dim,
            file = paste0("/Data/Jieyu_data/raw_ens_fcst/ws_5dim_sa",n,".csv"))
  
  write.csv(ens_fcst_10dim,
            file = paste0("/Data/Jieyu_data/raw_ens_fcst/ws_10dim_sa",n,".csv"))
  
  write.csv(ens_fcst_20dim,
            file = paste0("/Data/Jieyu_data/raw_ens_fcst/ws_20dim_sa",n,".csv"))
  
}
