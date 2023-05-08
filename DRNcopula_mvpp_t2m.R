# The following multivariate post-processing codes are adapted from the paper
# 'Simulation-based comparison of multivariate ensemble post-processing methods' 
# by Lerch et al, together with the codes on https://github.com/slerch/multiv_pp

# Before running this script, we should run the Jupyter notebook `DRN_unipp_t2m.ipynb` first.

library(scoringRules)
library(lubridate)
library(crch)
library(dplyr)
library(reshape2)
library(MASS)
library(feather)

var = 't2m'
dim = 20       # 5 or 10 or 20

# Please change PATH accordingly
PATH = './'

# Read DRN univariate post-processed forecast distribution parameters for training period
para_nn_train = read.csv(paste0(PATH, 'nn_', var, '_train.csv'), header = TRUE)
para_nn_train$date = as.Date(para_nn_train$date)

# Read DRN univariate post-processed forecast distribution parameters for test period
para_nn_test = read.csv(paste0(PATH, 'nn_', var, '_test.csv'), header = TRUE)
para_nn_test$date = as.Date(para_nn_test$date)

# Read data
path_data = paste0(PATH, 'temperature_data_cgm_std.feather')
data_complete = read_feather(path_data)

# Select useful columns
data_used = data_complete[, 1:55]

# Read test subsets of stations
path_stations = paste0(PATH, var, '_dist_', dim, 'samples.csv')
dist_samples = read.csv(path_stations, header = FALSE)

# Define training period and testing period
eval_start = as.Date("2016-01-01 00:00 UTC")
eval_end = as.Date("2016-12-31 00:00 UTC")
eval_dates = seq(eval_start, eval_end, by = "1 day")
train_obs_start = as.Date("2007-01-03 00:00 UTC")
train_obs_end = as.Date("2015-12-31 00:00 UTC")
train_obs_dates = seq(train_obs_start, train_obs_end, by = "1 day")

# Repeat multivariate post-processing on 100 different subsets of stations in the test data
n_rep = 100

for (n in 1:n_rep) {
  print(n)
  
  stations_list = dist_samples[n,]
  
  para_subset = para_nn_test[which(para_nn_test$station %in% stations_list),]
  data_subset = data_used[which(data_used$station %in% stations_list),]
  
  # Training data 2007-2015
  data_eval_all = subset(data_subset, date >= eval_start & date <= eval_end)
  
  ##############################################################################
  ############## DRN + ECC-Q (Ensemble copula coupling)
  ##############################################################################
  
  qlevels = 1:50/51
  ecc_eval_all = as.data.frame(data_eval_all[, 1:53])
  
  for(day_id in 1:length(eval_dates)){
    
    today = eval_dates[day_id]
    data_today = subset(data_eval_all, date == today)
    para_today = subset(para_subset, date == today)
    
    for(this_station in stations_list){
      
      ind_st = which(stations_list == this_station)
      data_eval = subset(data_today, station == this_station)
      para_eval = subset(para_today, station == this_station)
      
      if(nrow(data_eval) == 0){next}
      if(!is.finite(data_eval$obs)){next}
      
      loc_st = para_eval$X0
      sc_st = para_eval$X1
      
      enc_fcst = data_eval[4:53]
      
      # Generate 50 samples from DRN univariate post-processed marginal distribution
      DRN_sample = qnorm(qlevels, mean = loc_st, sd = sc_st)
      
      reorder = rank(enc_fcst, ties.method = "random")
      mvpp_thisstation = DRN_sample[reorder]
      
      nrow = which(ecc_eval_all$date == today & 
                     ecc_eval_all$station == this_station)
      
      ecc_eval_all[nrow, 4:53] = as.numeric(mvpp_thisstation)
    }
  }
  print('ECC done')
  
  ##############################################################################
  ############## DRN + GCA (Gaussian copula approach)
  ##############################################################################  

  para_train_subset = para_nn_train[which(para_nn_train$station %in% stations_list),]
  para_test_subset = para_nn_test[which(para_nn_test$station %in% stations_list),]
  
  #----- Step 1. in the paper by Lerch et al:
  train_obs_all_long = data_subset %>%
    dplyr::select(date, station, obs) %>%
    subset(date >= train_obs_start & date <= train_obs_end)
  
  train_obs_all = dcast(train_obs_all_long, date ~ station)
  train_obs_all = train_obs_all[, c('date', as.character(stations_list))]
  train_obs_latent = train_obs_all
  
  train_obs_latent[, 2:(length(stations_list)+1)] = NA
  
  for(day_id in 1:length(train_obs_dates)){
    
    today = train_obs_dates[day_id]
    data_today = subset(data_subset, date == today)
    para_today = subset(para_train_subset, date == today)
    
    for(this_station in stations_list){
      
      ind_st = which(stations_list == this_station)
      data_eval = subset(data_today, station == this_station)
      para_eval = subset(para_today, station == this_station)
      
      if(nrow(data_eval) == 0){next}
      if(!is.finite(data_eval$obs)){next}
      
      loc_st = para_eval$X0
      sc_st = para_eval$X1
      
      # Generate latent past observations
      latent_obs = qnorm(pnorm(data_eval$obs, mean = loc_st, sd = sc_st))
      
      nrow = which(train_obs_latent$date == today)
      
      train_obs_latent[nrow, (ind_st + 1)] = as.numeric(latent_obs)
    }
  }
  
  #----- Step 2. in the paper by Lerch et al:
  # Get correlation matrix of the latent observation variables distribution
  corr_obs = cor(train_obs_latent[, -1], use = "complete.obs",
                  method = "pearson") # or "pairwise.complete.obs"
  
  #----- Step 3. & 4. in the paper by Lerch et al:
  gca_eval_all = as.data.frame(data_eval_all[, 1:53])
  
  for(day_id in 1:length(eval_dates)){
    
    today = eval_dates[day_id]
    data_today = subset(data_eval_all, date == today)
    para_today = subset(para_test_subset, date == today)
    
    #----- Step 3. in the paper by Lerch et al:
    mvsample = mvrnorm(n = 50, mu = rep(0, length(stations_list)), Sigma = corr_obs)
    
    for(this_station in stations_list){
      
      ind_st = which(stations_list == this_station)
      data_eval = subset(data_today, station == this_station)
      para_eval = subset(para_today, station == this_station)
      
      if(nrow(data_eval) == 0){next}
      if(!is.finite(data_eval$obs)){next}
      
      loc_st = para_eval$X0
      sc_st = para_eval$X1
      
      #----- Step 4. in the paper by Lerch et al:
      mvpp_thisstation = 
        qnorm(pnorm(mvsample[, ind_st]), mean = loc_st, sd = sc_st)
  
      nrow = which(gca_eval_all$date == today & 
                     gca_eval_all$station == this_station)
      
      gca_eval_all[nrow, 4:53] = as.numeric(mvpp_thisstation)
    }
  }
  print('GCA done')
  
  # Save multivariate post-processed forecasts
  write.csv(ecc_eval_all,
            file = paste0(PATH, var, "_", dim, "dim_drnecc_sa", n, ".csv"))
  write.csv(gca_eval_all,
            file = paste0(PATH, var, "_", dim, "dim_drngca_sa", n, ".csv"))
  print('saved.')
}


