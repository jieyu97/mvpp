library(scoringRules)
library(lubridate)
library(crch)
library(tidyverse)
library(reshape2)
library(MASS)
library(feather)

path_ens_complete <- "/home/chen_jieyu/IGEP/ens_fc_t2m_complete.feather"
ens_fc_t2m_complete <- read_feather(path_ens_complete)

dist_samples = 
  read.csv("/home/chen_jieyu/IGEP/dist_samples.csv",
           header = FALSE)


tests = 150

for (n in 1:tests) {
  print(n)
  stations_list = dist_samples[n,]
  
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
  
  # Univariate PP using EMOS local with fixed 9-year training period
  vdate = date("2016-01-01")
  train_length = as.numeric(vdate - date("2007-01-01") - 1)
  par_output = postproc_local(vdate = vdate, 
                              train_length = train_length, 
                              data = ens_fc_t2m_mv_subset)
  
  ##### 1. ECC-Q
  qlevels <- 1:50/51
  ecc_eval_all = as.data.frame(data_eval_all[,c(1,3,6:55,2)])
  emos_eval_all = as.data.frame(data_eval_all[,c(1,3,6:55,2)])
  
  for(day_id in 1:length(eval_dates)){
    
    today <- eval_dates[day_id]
    data_today <- subset(data_eval_all, date == today)
    
    for(this_station in stations_list){
      ind_st <- which(stations_list == this_station)
      data_eval <- subset(data_today, station == this_station)
      if(nrow(data_eval) == 0){next}
      if(!is.finite(data_eval$obs)){next}
      loc_st <- c(cbind(1, data_eval$t2m_mean) %*% par_output[ind_st,1:2])
      scsquared_tmp <- c(cbind(1, data_eval$t2m_var) %*% par_output[ind_st,3:4])
      if(is.na(scsquared_tmp)){
        next
      }
      if(scsquared_tmp <= 0){
        # print("negative scale, taking absolute value")
        sc_st <- sqrt(abs(scsquared_tmp))
      } else{
        sc_st <- sqrt(scsquared_tmp)
      }
      
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
  
  ##### 2. SSh-Q
  
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
    
    obs_rows = sample(x = 1:length(train_obs_dates), size = 50, replace = FALSE)
    
    for(this_station in stations_list){
      ind_st <- which(stations_list == this_station)
      data_eval <- subset(data_today, station == this_station)
      if(nrow(data_eval) == 0){next}
      if(!is.finite(data_eval$obs)){next}
      loc_st <- c(cbind(1, data_eval$t2m_mean) %*% par_output[ind_st,1:2])
      scsquared_tmp <- c(cbind(1, data_eval$t2m_var) %*% par_output[ind_st,3:4])
      if(is.na(scsquared_tmp)){
        next
      }
      if(scsquared_tmp <= 0){
        # print("negative scale, taking absolute value")
        sc_st <- sqrt(abs(scsquared_tmp))
      } else{
        sc_st <- sqrt(scsquared_tmp)
      }
      
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
  
  ##### 3. GCA-Q/R/QO/S/T
  
  #----- Step 1. in the paper:
  train_obs_latent = train_obs_all
  train_obs_latent[,2:(length(stations_list)+1)] = NA
  for(day_id in 1:length(train_obs_dates)){
    
    today <- train_obs_dates[day_id]
    data_today <- subset(ens_fc_t2m_mv_subset, date == today)
    
    for(this_station in stations_list){
      ind_st <- which(stations_list == this_station)
      data_eval <- subset(data_today, station == this_station)
      
      if(nrow(data_eval) == 0){next}
      
      if(!is.finite(data_eval$obs)){next}
      loc_st <- c(cbind(1, data_eval$t2m_mean) %*% par_output[ind_st,1:2])
      scsquared_tmp <- c(cbind(1, data_eval$t2m_var) %*% par_output[ind_st,3:4])
      if(is.na(scsquared_tmp)){next}
      if(scsquared_tmp <= 0){
        # print("negative scale, taking absolute value")
        sc_st <- sqrt(abs(scsquared_tmp))
      } else {
        sc_st <- sqrt(scsquared_tmp)
      }
      
      # generate latent past observations
      latent_obs = qnorm(pnorm(data_eval$obs, mean = loc_st, sd = sc_st))
      
      nrow = which(train_obs_latent$date == today)
      train_obs_latent[nrow, (ind_st + 1)] = as.numeric(latent_obs)
    }
  }
  
  #----- Step 2. in the paper
  # get correlation matrix of the latent observation variables distribution
  corr_obs <- cor(train_obs_latent[,2:(length(stations_list)+1)], 
                  use = "complete.obs",
                  method = "pearson") # or "pairwise.complete.obs"
  
  #----- Step 3. & 4. in the paper
  gca_eval_all = as.data.frame(data_eval_all[,c(1,3,6:55,2)])
  
  for(day_id in 1:length(eval_dates)){
    
    today <- eval_dates[day_id]
    data_today <- subset(data_eval_all, date == today)
    
    #----- Step 3. in the paper
    mvsample <- mvrnorm(n = 50, mu = rep(0,length(stations_list)), Sigma = corr_obs)
    
    for(this_station in stations_list){
      ind_st <- which(stations_list == this_station)
      data_eval <- subset(data_today, station == this_station)
      if(nrow(data_eval) == 0){next}
      if(!is.finite(data_eval$obs)){next}
      loc_st <- c(cbind(1, data_eval$t2m_mean) %*% par_output[ind_st,1:2])
      scsquared_tmp <- c(cbind(1, data_eval$t2m_var) %*% par_output[ind_st,3:4])
      if(is.na(scsquared_tmp)){
        next
      }
      if(scsquared_tmp <= 0){
        # print("negative scale, taking absolute value")
        sc_st <- sqrt(abs(scsquared_tmp))
      } else{
        sc_st <- sqrt(scsquared_tmp)
      }
      
      #----- Step 4. in the paper
      mvpp_thisstation = 
        qnorm(pnorm(mvsample[,ind_st]), mean = loc_st, sd = sc_st)
  
      nrow = which(gca_eval_all$date == today & 
                     gca_eval_all$station == this_station)
      
      gca_eval_all[nrow,3:52] = as.numeric(mvpp_thisstation)
    }
  }
  print('GCA done')
  
  write.csv(emos_eval_all,
            file = paste0("/home/chen_jieyu/IGEP/r_ens_output/emos_10nbhd_sa",n,".csv"))
  write.csv(ecc_eval_all,
            file = paste0("/home/chen_jieyu/IGEP/r_ens_output/ecc_10nbhd_sa",n,".csv"))
  write.csv(ssh_eval_all,
            file = paste0("/home/chen_jieyu/IGEP/r_ens_output/ssh_10nbhd_sa",n,".csv"))
  write.csv(gca_eval_all,
            file = paste0("/home/chen_jieyu/IGEP/r_ens_output/gca_10nbhd_sa",n,".csv"))
  print('saved.')
}


