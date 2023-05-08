# The following multivariate post-processing codes are adapted from the paper
# 'Simulation-based comparison of multivariate ensemble post-processing methods' 
# by Lerch et al, together with the codes on https://github.com/slerch/multiv_pp

library(scoringRules)
library(lubridate)
library(crch)
library(dplyr)
library(reshape2)
library(MASS)
library(feather)

var = 't2m'
dim = 10       # 5 or 10 or 20

# Please change PATH accordingly
PATH = './'

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

##############################################################################
############## EMOS functions (from https://github.com/slerch/multiv_pp)
##############################################################################

# get CRPS from input
objective_fun = function(par, ensmean, ensvar, obs){
  m = cbind(1, ensmean) %*% par[1:2]
  ssq_tmp = cbind(1, ensvar) %*% par[3:4]
  if(any(ssq_tmp < 0)){
    return(999999)
  } else{
    s = sqrt(ssq_tmp)
    return(sum(crps_norm(y = obs, location = m, scale = s)))
  }
}

# get gradient of CRPS w.r.t. parameters
gradfun_wrapper = function(par, obs, ensmean, ensvar){
  loc = cbind(1, ensmean) %*% par[1:2]
  sc = sqrt(cbind(1, ensvar) %*% par[3:4])
  dcrps_dtheta = gradcrps_norm(y = obs, location = loc, scale = sc) 
  out1 = dcrps_dtheta[,1] %*% cbind(1, ensmean)
  out2 = dcrps_dtheta[,2] %*% 
    cbind(1/(2*sqrt(par[3]+par[4]*ensvar)), 
          ensvar/(2*sqrt(par[3]+par[4]*ensvar)))
  return(as.numeric(cbind(out1,out2)))
}

# post-processing function - EMOS local
postproc_local = function(vdate, train_length, data){
  
  # determine training set
  train_end = vdate - days(2)
  train_start = train_end - days(train_length - 1)
  
  par_out = matrix(NA, ncol = 4, nrow = length(stations_list))
  
  data_train_dates = subset(data, date >= train_start & date <= train_end)
  
  # loop over stations
  for(this_station in stations_list){
    
    data_train = subset(data_train_dates, station == this_station)
    
    # remove incomplete cases (= NA obs or fc)
    data_train = data_train[complete.cases(data_train), ]
    
    # skip station if there are too few forecast cases
    if(nrow(data_train) < 10){next}
    
    # determine optimal EMOS coefficients a,b,c,d using minimum CRPS estimation
    optim_out = optim(par = c(1,1,1,1), 
                       fn = objective_fun,
                       gr = gradfun_wrapper,
                       ensmean = data_train$t2m_mean, 
                       ensvar = data_train$t2m_var, 
                       obs = data_train$obs,
                       method = "BFGS")
    
    # check convergence of the numerical optimization
    if(optim_out$convergence != 0){
      message("numerical optimization did not converge")
    }
    
    par_out[which(stations_list == this_station), ] = optim_out$par
  }
  
  # return optimal parameters
  return(par_out)
}
##############################################################################

# Repeat multivariate post-processing on 100 different subsets of stations in the test data
n_rep = 100

for (n in 1:an) {
  print(n)
  
  stations_list = dist_samples[n,]
  
  data_subset = data_used[which(data_used$station %in% stations_list),]
  
  # Training data 2007-2015
  data_eval_all = subset(data_subset, date >= eval_start & date <= eval_end)
  
  # Univariate post-processing using EMOS local with fixed 9-year training period
  vdate = date("2016-01-01")
  train_length = as.numeric(vdate - date("2007-01-01") - 1)
  par_output = postproc_local(vdate = vdate, 
                              train_length = train_length, 
                              data = data_subset)
  
  ##############################################################################
  ############## EMOS + ECC-Q (Ensemble copula coupling)
  ##############################################################################

  qlevels = 1:50/51
  ecc_eval_all = as.data.frame(data_eval_all[, 1:53])
  
  for(day_id in 1:length(eval_dates)){
    
    today = eval_dates[day_id]
    data_today = subset(data_eval_all, date == today)
    
    for(this_station in stations_list){
      
      ind_st = which(stations_list == this_station)
      data_eval = subset(data_today, station == this_station)
      
      if(nrow(data_eval) == 0){next}
      if(!is.finite(data_eval$obs)){next}
      
      loc_st = c(cbind(1, data_eval$t2m_mean) %*% par_output[ind_st,1:2])
      scsquared = c(cbind(1, data_eval$t2m_var) %*% par_output[ind_st,3:4])
      
      if(is.na(scsquared)){next}
      if(scsquared <= 0){
        # print("negative scale, taking absolute value")
        sc_st = sqrt(abs(scsquared))
      } else{
        sc_st = sqrt(scsquared)
      }
      
      enc_fcst = data_eval[4:53]
      
      # Generate 50 samples from EMOS univariate post-processed marginal distribution
      EMOS_sample = qnorm(qlevels, mean = loc_st, sd = sc_st)
      
      reorder = rank(enc_fcst, ties.method = "random")
      mvpp_thisstation = EMOS_sample[reorder]
      
      nrow = which(ecc_eval_all$date == today & 
                     ecc_eval_all$station == this_station)
      
      ecc_eval_all[nrow, 4:53] = as.numeric(mvpp_thisstation)
    }
  }
  print('ECC done')
  
  ##############################################################################
  ############## EMOS + GCA (Gaussian copula approach)
  ############################################################################## 
  
  #----- Step 1. in the paper by Lerch et al:
  train_obs_all_long = data_subset %>%
    dplyr::select(date, station, obs) %>%
    subset(date >= train_obs_start & date <= train_obs_end)
  
  train_obs_all = dcast(train_obs_all_long,date~station)
  train_obs_all = train_obs_all[,c('date',as.character(stations_list))]
  train_obs_latent = train_obs_all
  
  train_obs_latent[, 2:(length(stations_list)+1)] = NA
  
  for(day_id in 1:length(train_obs_dates)){
    
    today = train_obs_dates[day_id]
    data_today = subset(data_subset, date == today)
    
    for(this_station in stations_list){
      
      ind_st = which(stations_list == this_station)
      data_eval = subset(data_today, station == this_station)
      
      if(nrow(data_eval) == 0){next}
      if(!is.finite(data_eval$obs)){next}
      
      loc_st = c(cbind(1, data_eval$t2m_mean) %*% par_output[ind_st,1:2])
      scsquared = c(cbind(1, data_eval$t2m_var) %*% par_output[ind_st,3:4])
      
      if(is.na(scsquared)){next}
      if(scsquared <= 0){
        # print("negative scale, taking absolute value")
        sc_st = sqrt(abs(scsquared))
      } else {
        sc_st = sqrt(scsquared)
      }
      
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
    
    #----- Step 3. in the paper by Lerch et al:
    mvsample = mvrnorm(n = 50, mu = rep(0, length(stations_list)), Sigma = corr_obs)
    
    for(this_station in stations_list){
      
      ind_st = which(stations_list == this_station)
      data_eval = subset(data_today, station == this_station)
      
      if(nrow(data_eval) == 0){next}
      if(!is.finite(data_eval$obs)){next}
      
      loc_st = c(cbind(1, data_eval$t2m_mean) %*% par_output[ind_st,1:2])
      scsquared = c(cbind(1, data_eval$t2m_var) %*% par_output[ind_st,3:4])
      
      if(is.na(scsquared)){next}
      if(scsquared <= 0){
        # print("negative scale, taking absolute value")
        sc_st = sqrt(abs(scsquared))
      } else{
        sc_st = sqrt(scsquared)
      }
      
      #----- Step 4. in the paper by Lerch et al:
      mvpp_thisstation = 
        qnorm(pnorm(mvsample[,ind_st]), mean = loc_st, sd = sc_st)
  
      nrow = which(gca_eval_all$date == today & 
                     gca_eval_all$station == this_station)
      
      gca_eval_all[nrow, 4:53] = as.numeric(mvpp_thisstation)
    }
  }
  print('GCA done')
  
  # Save multivariate post-processed forecasts
  write.csv(ecc_eval_all,
            file = paste0(PATH, var, "_", dim, "dim_emosecc_sa", n, ".csv"))
  write.csv(gca_eval_all,
            file = paste0(PATH, var, "_", dim, "dim_emosgca_sa", n, ".csv"))
  print('saved.')
}


