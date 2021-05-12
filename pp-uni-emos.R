Sys.setlocale(category = "LC_ALL", locale = "english")

library(scoringRules)
library(lubridate)
library(crch)
library(tidyverse)
library(ggplot2)

load("E:/0-TIGGE_ECMWF_Germany/ecmwf_ensemble_subset_data.RData")

# univariate postprocessing:
#   EMOS local, EMOS global.
# -- with output of 4 parameters for Gaussian dist.: 
#    emos-loc, emos-loc-rolling, emos-gl.
# multivariate postprocessing:
#   EMOS local + ECC-R/S/Q, EMOS local + Schaake shuffle,
#   EMOS local + GCA;
#   IGEP (implicit generative models).

# copy from https://github.com/slerch/ppnn/blob/master/benchmark_postprocessing_models/standard_postprocessing/cluster_experiments/emos_local.R

############
##### Univariate Postprocessing
############

# data
stations_list = unique(ens_fc_t2m_subset$station)
# add mean and variance to the ensemble data
ens_fc_t2m_mv_subset = ens_fc_t2m_subset %>%
  mutate(t2m_mean = 
           apply(ens_fc_t2m_subset[,4:53], 1, FUN = function(x) 
             {mean(x, na.rm = TRUE)} )) %>%
  mutate(t2m_var = 
           apply(ens_fc_t2m_subset[,4:53], 1, FUN = function(x) 
           {var(x, na.rm = TRUE)} ))
ens_fc_t2m_mv_subset = ens_fc_t2m_mv_subset[,c(1,2,3,54,55,4:53)]
View(head(ens_fc_t2m_mv_subset))

range(ens_fc_t2m_subset$date)

# get CRPS from input
objective_fun <- function(par, ensmean, ensvar, obs){
  m <- cbind(1, ensmean) %*% par[1:2]
  ssq_tmp <- cbind(1, ensvar) %*% par[3:4]
  if(any(ssq_tmp < 0)){
    return(999999)
  } else{
    s <- sqrt(ssq_tmp)
    return(sum(crps_norm(y = obs, location = m, scale = s)))
  }
}

# get gradient of CRPS w.r.t. parameters
gradfun_wrapper <- function(par, obs, ensmean, ensvar){
  loc <- cbind(1, ensmean) %*% par[1:2]
  sc <- sqrt(cbind(1, ensvar) %*% par[3:4])
  dcrps_dtheta <- gradcrps_norm(y = obs, location = loc, scale = sc) 
  out1 <- dcrps_dtheta[,1] %*% cbind(1, ensmean)
  out2 <- dcrps_dtheta[,2] %*% 
    cbind(1/(2*sqrt(par[3]+par[4]*ensvar)), 
          ensvar/(2*sqrt(par[3]+par[4]*ensvar)))
  return(as.numeric(cbind(out1,out2)))
}

#####################################################################
# post-processing function
#####################################################################
# 1. EMOS local
postproc_local <- function(vdate, train_length, data){
  
  # determine training set
  train_end <- vdate - days(2)
  train_start <- train_end - days(train_length - 1)
  
  par_out <- matrix(NA, ncol = 4, nrow = length(stations_list))
  
  data_train_dates <- subset(data, date >= train_start & date <= train_end)
  
  # loop over stations
  for(this_station in stations_list){
    
    # data_train <- subset(data, date >= train_start & date <= train_end & station == this_station)
    data_train <- subset(data_train_dates, station == this_station)
    
    # remove incomplete cases (= NA obs or fc)
    data_train <- data_train[complete.cases(data_train), ]
    
    # skip station if there are too few forecast cases
    if(nrow(data_train) < 10){
      next
    }
    
    # determine optimal EMOS coefficients a,b,c,d using minimum CRPS estimation
    optim_out <- optim(par = c(1,1,1,1), 
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
    
    par_out[which(stations_list == this_station), ] <- optim_out$par
  }
  
  # return optimal parameters
  return(par_out)
}
# 2. EMOS global
postproc_global <- function(vdate, train_length, data){
  
  # determine training set
  train_end <- vdate - days(2)
  train_start <- train_end - days(train_length - 1)
  
  data_train <- subset(data, date >= train_start & date <= train_end)
  
  # remove incomplete cases (= NA obs or fc)
  data_train <- data_train[complete.cases(data_train), ]
  
  # determine optimal EMOS coefficients a,b,c,d using minimum CRPS estimation
  optim_out <- optim(par = c(1,1,1,1), 
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
  
  # return optimal parameters
  return(optim_out$par)
}

#################################################################
save.image("E:/0-TIGGE_ECMWF_Germany/ecmwf_ensemble_subset_data.RData")

# test date
eval_start <- as.Date("2016-01-01 00:00 UTC")
eval_end <- as.Date("2016-12-31 00:00 UTC")
eval_dates <- seq(eval_start, eval_end, by = "1 day")
data_eval_all <- subset(ens_fc_t2m_mv_subset, 
                        date >= eval_start & date <= eval_end)

#------------------------------ 1. EMOS local
#--------------------------- a. fixed training data 2007-2015
vdate = date("2016-01-01")
train_length = as.numeric(vdate - date("2007-01-01") - 1)
par_output = postproc_local(vdate = vdate, 
                            train_length = train_length, 
                            data = ens_fc_t2m_mv_subset)
crps_pp <- NULL
pit_pp <- NULL

for(day_id in 1:length(eval_dates)){
  
  today <- eval_dates[day_id]
  data_today <- subset(data_eval_all, date == today)
  #==# ens_today = today - days(2)
  #==# ens_data_today = subset(ens_fc_t2m_mv_subset, date == ens_today)
  
  # progress indicator
  if(day(as.Date(today)) == 1){
    cat("Starting at", paste(Sys.time()), ":", 
        as.character(today), "\n"); flush(stdout())
  }
  
  # out of sample distribution parameters for today
  loc <- rep(NA, length(stations_list))
  sc <- rep(NA, length(stations_list))
  for(this_station in stations_list){
    ind_st <- which(stations_list == this_station)
    # print(ind_st)
    data_eval <- subset(data_today, station == this_station)
    #==# ens_data_eval = subset(ens_data_today, station == this_station)
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
    loc[ind_st] <- loc_st
    sc[ind_st] <- sc_st
  }
  
  # CRPS
  crps_today <- crps_norm(y = data_today$obs, mean = loc, sd = sc)
  crps_pp[which(data_eval_all$date == today)] <- crps_today
  
  # PIT
  pit_today = pnorm(data_today$obs, mean = loc, sd = sc)
  pit_pp[which(data_eval_all$date == today)] <- pit_today
}

crps_emosloc_fix9y = crps_pp
pit_emosloc_fix9y = pit_pp

#------------------------ b. rolling window (28 days) training data
crps_pp <- NULL
pit_pp <- NULL

m = 28

for(day_id in 1:length(eval_dates)){
  
  today <- eval_dates[day_id]
  data_today <- subset(data_eval_all, date == today)
  #==# ens_today = today - days(2)
  #==# ens_data_today = subset(ens_fc_t2m_mv_subset, date == ens_today)
  
  # progress indicator
  if(day(as.Date(today)) == 1){
    cat("Starting at", paste(Sys.time()), ":", 
        as.character(today), "\n"); flush(stdout())
  }
  
  # post-processing
  par_out <- postproc_local(vdate = today, train_length = m, 
                            data = ens_fc_t2m_mv_subset)
  
  # out of sample distribution parameters for today
  loc <- rep(NA, length(stations_list))
  sc <- rep(NA, length(stations_list))
  for(this_station in stations_list){
    ind_st <- which(stations_list == this_station)
    # print(ind_st)
    data_eval <- subset(data_today, station == this_station)
    #==# ens_data_eval = subset(ens_data_today, station == this_station)
    if(!is.finite(data_eval$obs)){next}
    loc_st <- c(cbind(1, data_eval$t2m_mean) %*% par_out[ind_st,1:2])
    scsquared_tmp <- c(cbind(1, data_eval$t2m_var) %*% par_out[ind_st,3:4])
    if(is.na(scsquared_tmp)){
      next
    }
    if(scsquared_tmp <= 0){
      # print("negative scale, taking absolute value")
      sc_st <- sqrt(abs(scsquared_tmp))
    } else{
      sc_st <- sqrt(scsquared_tmp)
    }
    loc[ind_st] <- loc_st
    sc[ind_st] <- sc_st
  }
  
  # CRPS
  crps_today <- crps_norm(y = data_today$obs, mean = loc, sd = sc)
  crps_pp[which(data_eval_all$date == today)] <- crps_today
  
  # PIT
  pit_today = pnorm(data_today$obs, mean = loc, sd = sc)
  pit_pp[which(data_eval_all$date == today)] <- pit_today
}

crps_emosloc_rolling28d = crps_pp
pit_emosloc_rolling28d = pit_pp

#----------------------------------- 2. EMOS global
vdate = date("2016-01-01")
train_length = as.numeric(vdate - date("2007-01-01") - 1)
par_output_gl = postproc_global(vdate = vdate, 
                                train_length = train_length, 
                                data = ens_fc_t2m_mv_subset)
crps_pp <- NULL
pit_pp <- NULL

for(day_id in 1:length(eval_dates)){
  
  today <- eval_dates[day_id]
  #==# ens_today = today - days(2)
  
  data_eval <- subset(ens_fc_t2m_mv_subset, date == today)
  #==# ens_data_eval = subset(ens_fc_t2m_mv_subset, date == ens_today)
  
  # progress indicator
  if(day(as.Date(today)) == 1){
    cat("Starting at", paste(Sys.time()), ":", 
        as.character(today), "\n"); flush(stdout())
  }
  
  # out of sample distribution parameters for today
  loc <- c(cbind(1, data_eval$t2m_mean) %*% par_output_gl[1:2])
  scsquared_tmp <- c(cbind(1, data_eval$t2m_var) %*% par_output_gl[3:4])
  if(any(scsquared_tmp <= 0)){
    print("negative scale, taking absolute value")
    sc <- sqrt(abs(scsquared_tmp))
  } else{
    sc <- sqrt(scsquared_tmp)
  }
  
  # CRPS
  crps_today <- crps_norm(y = data_eval$obs, mean = loc, sd = sc)
  crps_pp[which(data_eval_all$date == today)] <- crps_today
  
  # PIT
  pit_today = pnorm(data_eval$obs, mean = loc, sd = sc)
  pit_pp[which(data_eval_all$date == today)] <- pit_today
}

crps_emosglob_fix9y = crps_pp
pit_emosglob_fix9y = pit_pp

#############################################
############ comparison and visualization
#############################################

crps_ens <- NULL
rank_ens <- NULL

for(day_id in 1:length(eval_dates)){
  
  today <- eval_dates[day_id]
  #==# ens_today = today - days(2)
  
  data_eval <- subset(ens_fc_t2m_mv_subset, date == today)
  #==# ens_data_eval = subset(ens_fc_t2m_mv_subset, date == ens_today)
  
  # progress indicator
  if(day(as.Date(today)) == 1){
    cat("Starting at", paste(Sys.time()), ":", 
        as.character(today), "\n"); flush(stdout())
  }
  
  ens_dat = apply(data_eval[,6:55], 2, function(x){
    as.numeric(x)
  })
  
  # CRPS
  crps_today <- crps_sample(y = data_eval$obs, 
                            dat = ens_dat, 
                            method = "edf")
  crps_ens[which(data_eval_all$date == today)] <- crps_today
  
  # ranks
  rank_today = apply(cbind(data_eval$obs, ens_dat), 1, function(x) 
    rank(x, ties = "random")[1])
  rank_ens[which(data_eval_all$date == today)] <- rank_today
}

summary(crps_ens)
summary(crps_emosloc_fix9y)
summary(crps_emosloc_rolling28d)
summary(crps_emosglob_fix9y)

hist(rank_ens)
hist(pit_emosloc_fix9y)
hist(pit_emosloc_rolling28d)
hist(pit_emosglob_fix9y)

crps_test_all = tibble(date = data_eval_all$date,
                       station = data_eval_all$station,
                       obs = data_eval_all$obs,
                       emosloc_fix9y = crps_emosloc_fix9y,
                       emosloc_rolling28d = crps_emosloc_rolling28d,
                       emosglob_fix9y = crps_emosglob_fix9y)

save.image("E:/0-TIGGE_ECMWF_Germany/ecmwf_ens_subset_unipp.RData")

crps_test_each_station = crps_test_all %>%
  group_by(station) %>%
  group_split()

ggplot(crps_test_all) +
  geom_histogram(aes(emosloc_fix9y), binwidth = 0.1, 
                 fill = 'steelblue', alpha = 0.6) +
  geom_histogram(aes(emosloc_rolling28d), binwidth = 0.1, 
                 fill = 'orange', alpha = 0.6) +
  geom_histogram(aes(emosglob_fix9y), binwidth = 0.1, 
                 fill = 'forestgreen', alpha = 0.6)





