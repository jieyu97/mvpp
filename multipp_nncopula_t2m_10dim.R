library(scoringRules)
library(lubridate)
library(crch)
library(tidyverse)
library(reshape2)
library(MASS)
library(feather)

parameters_nn = 
  read.csv('/home/chen_jieyu/IGEP/nn_t2m_test_para.csv',
           header = TRUE)
parameters_nn$date = as.Date(parameters_nn$date)


path_ens_complete <- "/home/chen_jieyu/IGEP/ens_fc_t2m_complete.feather"
ens_fc_t2m_complete <- read_feather(path_ens_complete)

dist_samples = 
  read.csv("/home/chen_jieyu/IGEP/dist_10samples.csv",
           header = FALSE)


tests = 100

for (n in 1:tests) {
  print(n)
  stations_list = dist_samples[n,]
  
  parameters_nn_subset = parameters_nn[which(parameters_nn$station_id %in% stations_list),]
  ens_fc_t2m_subset = ens_fc_t2m_complete[which(ens_fc_t2m_complete$station %in% stations_list),]
  
  # add mean and variance to the ensemble data
  ens_fc_t2m_mv_subset = ens_fc_t2m_subset %>%
    mutate(t2m_mean = 
             apply(ens_fc_t2m_subset[,4:53], 1, FUN = function(x) 
             {mean(x, na.rm = TRUE)} )) %>%
    mutate(t2m_var = 
             apply(ens_fc_t2m_subset[,4:53], 1, FUN = function(x) 
             {var(x, na.rm = TRUE)} ))
  ens_fc_t2m_mv_subset = ens_fc_t2m_mv_subset[,c(1,2,3,54,55,4:53)]
  
  # training data 2007-2015
  eval_start <- as.Date("2016-01-01 00:00 UTC")
  eval_end <- as.Date("2016-12-31 00:00 UTC")
  eval_dates <- seq(eval_start, eval_end, by = "1 day")
  data_eval_all <- subset(ens_fc_t2m_mv_subset, 
                          date >= eval_start & date <= eval_end)

  ##### 1. ECC-Q
  qlevels <- 1:50/51
  ecc_eval_all = as.data.frame(data_eval_all[,c(1,3,6:55,2)])
  emos_eval_all = as.data.frame(data_eval_all[,c(1,3,6:55,2)])
  
  for(day_id in 1:length(eval_dates)){
    
    today <- eval_dates[day_id]
    data_today <- subset(data_eval_all, date == today)
    parameters_nn_today = subset(parameters_nn_subset, date == today)
    
    for(this_station in stations_list){
      ind_st <- which(stations_list == this_station)
      data_eval <- subset(data_today, station == this_station)
      parameters_eval = subset(parameters_nn_today, station_id == this_station)
      
      if(nrow(data_eval) == 0){next}
      if(!is.finite(data_eval$obs)){next}
      
      loc_st = parameters_eval$mean
      sc_st = parameters_eval$std
      
      ensfc_tmp = data_eval[6:55]
      
      # generate 50 samples from ensemble PP marginal distribution
      EMOS_sample = qnorm(qlevels, mean = loc_st, sd = sc_st) # -Q
      
      reorder = rank(ensfc_tmp, ties.method = "random")
      mvpp_thisstation = EMOS_sample[reorder]
      
      nrow = which(ecc_eval_all$date == today & 
                     ecc_eval_all$station == this_station)
      
      ecc_eval_all[nrow,3:52] = as.numeric(mvpp_thisstation)
      emos_eval_all[nrow,3:52] = as.numeric(EMOS_sample)
    }
  }
  print('ECC done')
  
  ### 2. SSh-Q
  
  train_obs_start <- ymd_hm("2007-01-03 00:00 UTC")
  train_obs_end <- ymd_hm("2015-12-31 00:00 UTC")
  train_obs_dates <- seq.POSIXt(train_obs_start, train_obs_end, by = "day")
  train_obs_all_long = ens_fc_t2m_subset %>%
    dplyr::select(date, station, obs) %>%
    subset(date >= train_obs_start & date <= train_obs_end)
  train_obs_all = dcast(train_obs_all_long,date~station)
  train_obs_all = train_obs_all[,c('date',as.character(stations_list))]
  
  qlevels <- 1:50/51 # -Q
  ssh_eval_all = as.data.frame(data_eval_all[,c(1,3,6:55,2)])
  
  for(day_id in 1:length(eval_dates)){
    
    today <- eval_dates[day_id]
    data_today <- subset(data_eval_all, date == today)
    parameters_nn_today = subset(parameters_nn_subset, date == today)
    
    obs_rows = sample(x = 1:length(train_obs_dates), size = 50, replace = FALSE)
    
    for(this_station in stations_list){
      ind_st <- which(stations_list == this_station)
      data_eval <- subset(data_today, station == this_station)
      parameters_eval = subset(parameters_nn_today, station_id == this_station)
      
      if(nrow(data_eval) == 0){next}
      if(!is.finite(data_eval$obs)){next}

      loc_st = parameters_eval$mean
      sc_st = parameters_eval$std
      
      obs_tmp = train_obs_all[obs_rows, 1+ind_st]
      
      # generate 50 samples from ensemble PP marginal distribution
      EMOS_sample = qnorm(qlevels, mean = loc_st, sd = sc_st) # -Q
      
      reorder = rank(obs_tmp, ties.method = "random")
      mvpp_thisstation = EMOS_sample[reorder]
      
      nrow = which(ssh_eval_all$date == today & 
                     ssh_eval_all$station == this_station)
      
      ssh_eval_all[nrow,3:52] = as.numeric(mvpp_thisstation)
    }
  }
  print('SSh done')
  
  write.csv(emos_eval_all,
            file = paste0("/home/chen_jieyu/IGEP/nn_copula_t2m_10dim/nn_10nbhd_sa",n,".csv"))
  write.csv(ecc_eval_all,
            file = paste0("/home/chen_jieyu/IGEP/nn_copula_t2m_10dim/nnecc_10nbhd_sa",n,".csv"))
  write.csv(ssh_eval_all,
            file = paste0("/home/chen_jieyu/IGEP/nn_copula_t2m_10dim/nnssh_10nbhd_sa",n,".csv"))
  
  print('saved.')
}




