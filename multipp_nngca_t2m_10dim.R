library(scoringRules)
library(lubridate)
library(crch)
library(dplyr)
library(reshape2)
library(MASS)
library(feather)

para_nn_train = 
  read.csv('/home/chen_jieyu/IGEP/nn_t2m_train_para.csv',
           header = TRUE)
para_nn_train$date = as.Date(para_nn_train$date)

para_nn_test = 
  read.csv('/home/chen_jieyu/IGEP/nn_t2m_test_para.csv',
           header = TRUE)
para_nn_test$date = as.Date(para_nn_test$date)

path_ens_complete <- "/home/chen_jieyu/IGEP/ens_fc_t2m_complete.feather"
ens_fc_t2m_complete <- read_feather(path_ens_complete)

dist_samples = 
  read.csv("/home/chen_jieyu/IGEP/dist_10samples.csv",
           header = FALSE)

tests = 100

for (n in 1:tests) {
  print(n)
  stations_list = dist_samples[n,]
  
  para_nn_train_subset = para_nn_train[which(para_nn_train$station_id %in% stations_list),]
  para_nn_test_subset = para_nn_test[which(para_nn_test$station_id %in% stations_list),]
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
  
  ##### GCA-Q
  train_obs_start <- ymd_hm("2007-01-03 00:00 UTC")
  train_obs_end <- ymd_hm("2015-12-31 00:00 UTC")
  train_obs_dates <- seq.POSIXt(train_obs_start, train_obs_end, by = "day")
  train_obs_all_long = ens_fc_t2m_subset %>%
    dplyr::select(date, station, obs) %>%
    subset(date >= train_obs_start & date <= train_obs_end)
  train_obs_all = dcast(train_obs_all_long, date~station)
  train_obs_all = train_obs_all[,c('date',as.character(stations_list))]
  
  #----- Step 1. in the paper:
  train_obs_latent = train_obs_all
  train_obs_latent[,2:(length(stations_list)+1)] = NA
  
  for(day_id in 1:length(train_obs_dates)){
    
    today <- train_obs_dates[day_id]
    data_today <- subset(ens_fc_t2m_mv_subset, date == today)
    parameters_nn_today = subset(para_nn_train_subset, date == today)
    
    for(this_station in stations_list){
      ind_st <- which(stations_list == this_station)
      data_eval <- subset(data_today, station == this_station)
      parameters_eval = subset(parameters_nn_today, station_id == this_station)
      
      if(nrow(data_eval) == 0){next}
      if(!is.finite(data_eval$obs)){next}
      
      loc_st = parameters_eval$mean
      sc_st = parameters_eval$std
      
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
    parameters_nn_today = subset(para_nn_test_subset, date == today)
    
    #----- Step 3. in the paper
    mvsample <- mvrnorm(n = 50, mu = rep(0,length(stations_list)), Sigma = corr_obs)
    
    for(this_station in stations_list){
      ind_st <- which(stations_list == this_station)
      data_eval <- subset(data_today, station == this_station)
      parameters_eval = subset(parameters_nn_today, station_id == this_station)
      
      if(nrow(data_eval) == 0){next}
      if(!is.finite(data_eval$obs)){next}
      
      loc_st = parameters_eval$mean
      sc_st = parameters_eval$std
      
      #----- Step 4. in the paper
      mvpp_thisstation = 
        qnorm(pnorm(mvsample[,ind_st]), mean = loc_st, sd = sc_st)
      
      nrow = which(gca_eval_all$date == today & 
                     gca_eval_all$station == this_station)
      
      gca_eval_all[nrow,3:52] = as.numeric(mvpp_thisstation)
    }
  }
  print('GCA done')
  write.csv(gca_eval_all,
            file = paste0("/home/chen_jieyu/IGEP/nn_copula_t2m_10dim/nngca_10nbhd_sa",n,".csv"))
  print('saved.')
}