library(scoringRules)
library(lubridate)
library(crch)
library(tidyverse)
library(reshape2)
library(MASS)
library(feather)
library(truncnorm)

var = "ws" # ws or t2m
dim = 20 # 5 or 10 or 20

# read post-processed parameters for training period
para_nn_train = 
  read.csv(paste0('/home/chen_jieyu/IGEP/nn_', var, '_train_std.csv'),
           header = TRUE)
para_nn_train$date = as.Date(para_nn_train$date)
# read post-processed parameters for test period
para_nn_test = 
  read.csv(paste0('/home/chen_jieyu/IGEP/nn_', var, '_test_std.csv'),
           header = TRUE)
para_nn_test$date = as.Date(para_nn_test$date)

# read data
path = "/home/chen_jieyu/IGEP/windspeed_data_cgm_std.feather"
# path = "~/0-project1/windspeed_data_cgm_std.feather"
data_complete = read_feather(path)

# select useful columns
data_used = data_complete[, 1:55]

# read test subsets of stations
path_stations = paste0("/home/chen_jieyu/IGEP/", var, "_dist_", dim, "samples.csv")
dist_samples = read.csv(path_stations, header = FALSE)


eval_start <- as.Date("2016-01-01 00:00 UTC")
eval_end <- as.Date("2016-12-31 00:00 UTC")
eval_dates <- seq(eval_start, eval_end, by = "1 day")

train_obs_start <- as.Date("2007-01-03 00:00 UTC")
train_obs_end <- as.Date("2015-12-31 00:00 UTC")
train_obs_dates <- seq(train_obs_start, train_obs_end, by = "1 day")

# repeat the procedure for 100 times
n_rep = 100

for (n in 1:n_rep) {
  print(n)
  
  stations_list = dist_samples[n,]
  
  para_subset = para_nn_test[which(para_nn_test$station %in% stations_list),]
  data_subset = data_used[which(data_used$station %in% stations_list),]
  
  # training data 2007-2015
  data_eval_all <- subset(data_subset, 
                          date >= eval_start & date <= eval_end)
  
  ##### 1. ECC-Q
  qlevels <- 1:50/51
  ecc_eval_all = as.data.frame(data_eval_all[, 1:53])
  
  for(day_id in 1:length(eval_dates)){
    
    today <- eval_dates[day_id]
    data_today <- subset(data_eval_all, date == today)
    para_today = subset(para_subset, date == today)
    
    for(this_station in stations_list){
      
      ind_st <- which(stations_list == this_station)
      data_eval <- subset(data_today, station == this_station)
      para_eval = subset(para_today, station == this_station)
      
      if(nrow(data_eval) == 0){next}
      if(!is.finite(data_eval$obs)){next}
      
      loc_st = para_eval$X0
      sc_st = para_eval$X1
      
      enc_fcst = data_eval[4:53]
      
      # generate 50 samples from ensemble PP marginal distribution
      EMOS_sample = qtruncnorm(qlevels, a = 0, mean = loc_st, sd = sc_st) # -Q
      
      reorder = rank(enc_fcst, ties.method = "random")
      mvpp_thisstation = EMOS_sample[reorder]
      
      nrow = which(ecc_eval_all$date == today & 
                     ecc_eval_all$station == this_station)
      
      ecc_eval_all[nrow, 4:53] = as.numeric(mvpp_thisstation)
    }
  }
  print('ECC done')
  
  
  ##### 2. GCA
  para_train_subset = para_nn_train[which(para_nn_train$station %in% stations_list),]
  para_test_subset = para_nn_test[which(para_nn_test$station %in% stations_list),]
  
  #----- Step 1. in the paper:
  train_obs_all_long = data_subset %>%
    dplyr::select(date, station, obs) %>%
    subset(date >= train_obs_start & date <= train_obs_end)
  
  train_obs_all = dcast(train_obs_all_long, date ~ station)
  train_obs_all = train_obs_all[, c('date', as.character(stations_list))]
  train_obs_latent = train_obs_all
  
  train_obs_latent[, 2:(length(stations_list)+1)] = NA
  
  for(day_id in 1:length(train_obs_dates)){
    
    today <- train_obs_dates[day_id]
    data_today <- subset(data_subset, date == today)
    para_today = subset(para_train_subset, date == today)
    
    for(this_station in stations_list){
      
      ind_st <- which(stations_list == this_station)
      data_eval <- subset(data_today, station == this_station)
      para_eval = subset(para_today, station == this_station)
      
      if(nrow(data_eval) == 0){next}
      if(!is.finite(data_eval$obs)){next}
      
      loc_st = para_eval$X0
      sc_st = para_eval$X1
      
      # generate latent past observations
      latent_obs = qnorm(ptruncnorm((data_eval$obs + 0.01), a = 0, 
                                    mean = loc_st, sd = sc_st))
      
      if(latent_obs == Inf) {latent_obs = 8.209536}
      
      nrow = which(train_obs_latent$date == today)
      
      train_obs_latent[nrow, (ind_st + 1)] = as.numeric(latent_obs)
    }
  }
  
  #----- Step 2. in the paper
  # get correlation matrix of the latent observation variables distribution
  corr_obs <- cor(train_obs_latent[, -1], use = "complete.obs",
                  method = "pearson") # or "pairwise.complete.obs"
  
  #----- Step 3. & 4. in the paper
  gca_eval_all = as.data.frame(data_eval_all[, 1:53])
  
  for(day_id in 1:length(eval_dates)){
    
    today <- eval_dates[day_id]
    data_today <- subset(data_eval_all, date == today)
    para_today = subset(para_test_subset, date == today)
    
    #----- Step 3. in the paper
    mvsample <- mvrnorm(n = 50, mu = rep(0, length(stations_list)), 
                        Sigma = corr_obs)
    
    for(this_station in stations_list){
      
      ind_st <- which(stations_list == this_station)
      data_eval <- subset(data_today, station == this_station)
      para_eval = subset(para_today, station == this_station)
      
      if(nrow(data_eval) == 0){next}
      if(!is.finite(data_eval$obs)){next}
      
      loc_st = para_eval$X0
      sc_st = para_eval$X1
      
      #----- Step 4. in the paper
      mvpp_thisstation = 
        qtruncnorm(pnorm(mvsample[, ind_st]), a = 0, 
                   mean = loc_st, sd = sc_st)
  
      nrow = which(gca_eval_all$date == today & 
                     gca_eval_all$station == this_station)
      
      gca_eval_all[nrow, 4:53] = as.numeric(mvpp_thisstation)
    }
  }
  print('GCA done')
  
  write.csv(ecc_eval_all,
            file = paste0("/Data/Jieyu_data/mvpp_drn_ecc/", var,
                          "_", dim, "dim_sa", n, ".csv"))
  write.csv(gca_eval_all,
            file = paste0("/Data/Jieyu_data/mvpp_drn_gca/", var,
                          "_", dim, "dim_sa", n, ".csv"))
  print('saved.')
}


