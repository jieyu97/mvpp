library(scoringRules)
library(lubridate)
library(crch)
library(tidyverse)
library(ggplot2)

load("E:/0-TIGGE_ECMWF_Germany/ecmwf_ensemble_subset_data.RData")

########## Fixed training data 2007-2015
eval_start <- as.Date("2016-01-03 00:00 UTC")
eval_end <- as.Date("2016-12-31 00:00 UTC")
eval_dates <- seq(eval_start, eval_end, by = "1 day")
data_eval_all <- subset(ens_fc_t2m_mv_subset, 
                        date >= eval_start & date <= eval_end)

# Univariate PP using EMOS local with fixed 9-year training period
vdate = date("2016-01-03")
train_length = as.numeric(vdate - date("2007-01-03") - 1)
par_output = postproc_local(vdate = vdate, 
                            train_length = train_length, 
                            data = ens_fc_t2m_mv_subset)

##########################################
################# 1. ECC-Q/R/QO/S/T
##########################################

########## Different sampling/quantization method:
qlevels <- 1:50/51 # -Q
# qlevels <- (1:50-0.5)/50 # -QO
# breakpoints <- 0:50/50 # -S
# qlevels <- runif(50, min = breakpoints[1:50], max = breakpoints[2:51]) # -S

ecc_eval_all = as.data.frame(data_eval_all)
emos_eval_all = as.data.frame(data_eval_all)

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
    
    ensfc_tmp = data_eval[6:55]
    
    # generate 50 samples from ensemble PP marginal distribution
    
    # -T
    # qlevels <- pnorm(ensfc_tmp, mean = mean(ensfc_tmp), sd = sd(ensfc_tmp))
    EMOS_sample = 
      qnorm(qlevels, mean = loc_st, sd = sc_st) # -Q/QO/S/T
      # rnorm(m, mean = par[1], sd = par[2]) # -R
      
    reorder = rank(ensfc_tmp, ties.method = "random")
    mvpp_thisstation = EMOS_sample[reorder]
    
    nrow = which(ecc_eval_all$date == today & 
                   ecc_eval_all$station == this_station)
    ecc_eval_all[nrow,6:55] = as.numeric(mvpp_thisstation)
    
    emos_eval_all[nrow,6:55] = as.numeric(EMOS_sample)
  }
}

ecc_eval_all$t2m_mean = 
  apply(ecc_eval_all[,6:55], 1, FUN = function(x) {mean(x, na.rm = TRUE)} )
ecc_eval_all$t2m_var = 
  apply(ecc_eval_all[,6:55], 1, FUN = function(x) {var(x, na.rm = TRUE)} )
emos_eval_all$t2m_mean = 
  apply(emos_eval_all[,6:55], 1, FUN = function(x) {mean(x, na.rm = TRUE)} )
emos_eval_all$t2m_var = 
  apply(emos_eval_all[,6:55], 1, FUN = function(x) {var(x, na.rm = TRUE)} )

View(ecc_eval_all)
View(emos_eval_all)

##########################################
################# 2. SSh-Q/R/QO/S/T (Schaake Shuffle)
##########################################

train_obs_start <- ymd_hm("2007-01-03 00:00 UTC")
train_obs_end <- ymd_hm("2015-12-31 00:00 UTC")
train_obs_dates <- seq.POSIXt(train_obs_start, train_obs_end, by = "day")
train_obs_all_long = ens_fc_t2m_subset %>%
  dplyr::select(date, station, obs) %>%
  subset(date >= train_obs_start & date <= train_obs_end)
train_obs_all = as.data.frame(matrix(NA, length(train_obs_dates), 
                                     (1 + length(stations_list)) ))
colnames(train_obs_all) = c("date", stations_list)
train_obs_all$date = train_obs_dates
for (d in 1:length(train_obs_dates)) {
  for (s in 1:length(stations_list)) {
    n = which(train_obs_all_long$date == train_obs_dates[d] &
                train_obs_all_long$station == stations_list[s])
    train_obs_all[d,(1+s)] = ifelse(length(n) == 1, 
                                    train_obs_all_long$obs[n], NA)
  }
}

View(train_obs_all)

########## Different sampling/quantization method:
qlevels <- 1:50/51 # -Q
# qlevels <- (1:50-0.5)/50 # -QO
# breakpoints <- 0:50/50 # -S
# qlevels <- runif(50, min = breakpoints[1:50], max = breakpoints[2:51]) # -S

ssh_eval_all = as.data.frame(data_eval_all)
emos_eval_all_n = as.data.frame(data_eval_all)

for(day_id in 1:length(eval_dates)){
  
  today <- eval_dates[day_id]
  data_today <- subset(data_eval_all, date == today)
  #==# ens_today = today - days(2)
  #==# ens_data_today = subset(ens_fc_t2m_mv_subset, date == ens_today)
  
  obs_rows = sample(x = 1:length(train_obs_dates), size = 50, replace = FALSE)
  
  # progress indicator
  if(day(as.Date(today)) == 1){
    cat("Starting at", paste(Sys.time()), ":", 
        as.character(today), "\n"); flush(stdout())
  }
  
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

    obs_tmp = train_obs_all[obs_rows, 1+ind_st]
    
    # generate 50 samples from ensemble PP marginal distribution
    
    # -T
    # qlevels <- pnorm(ensfc_tmp, mean = mean(ensfc_tmp), sd = sd(ensfc_tmp))
    EMOS_sample = 
      qnorm(qlevels, mean = loc_st, sd = sc_st) # -Q/QO/S/T
    # rnorm(m, mean = par[1], sd = par[2]) # -R
    
    reorder = rank(obs_tmp, ties.method = "random")
    mvpp_thisstation = EMOS_sample[reorder]
    
    nrow = which(ssh_eval_all$date == today & 
                   ssh_eval_all$station == this_station)
    ssh_eval_all[nrow,6:55] = as.numeric(mvpp_thisstation)
    
    emos_eval_all_n[nrow,6:55] = as.numeric(EMOS_sample)
  }
}

ssh_eval_all$t2m_mean = 
  apply(ssh_eval_all[,6:55], 1, FUN = function(x) {mean(x, na.rm = TRUE)} )
ssh_eval_all$t2m_var = 
  apply(ssh_eval_all[,6:55], 1, FUN = function(x) {var(x, na.rm = TRUE)} )
emos_eval_all_n$t2m_mean = 
  apply(emos_eval_all_n[,6:55], 1, FUN = function(x) {mean(x, na.rm = TRUE)} )
emos_eval_all_n$t2m_var = 
  apply(emos_eval_all_n[,6:55], 1, FUN = function(x) {var(x, na.rm = TRUE)} )

# check
emos_eval_all_n == emos_eval_all

##########################################
################# 3. GCA-Q/R/QO/S/T
##########################################

library(MASS)
cov_obs <- cov(train_obs_all[,2:8], use = "complete.obs",
               method = "pearson") # or "pairwise.complete.obs"
# mvsample <- mvrnorm(n = 50, mu = rep(0,7), Sigma = cov_obs)

gca_eval_all = as.data.frame(data_eval_all)

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
  
  mvsample <- mvrnorm(n = 50, mu = rep(0,7), Sigma = cov_obs)
  
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
    
    # generate 50 samples from multivariate Gaussian latent distribution

    mvpp_thisstation = 
      qnorm(pnorm(mvsample[,ind_st]), mean = loc_st, sd = sc_st)

    # If return Inf in the 'qnorm' function, then replace Inf with:
    inf_label = which(mvpp_thisstation == Inf)
    mvpp_thisstation[inf_label] = qnorm(0.9999999999999999, mean = loc_st, sd = sc_st)
    
    nrow = which(gca_eval_all$date == today & 
                   gca_eval_all$station == this_station)
    gca_eval_all[nrow,6:55] = as.numeric(mvpp_thisstation)
  }
}

gca_eval_all$t2m_mean = 
  apply(ecc_eval_all[,6:55], 1, FUN = function(x) {mean(x, na.rm = TRUE)} )
gca_eval_all$t2m_var = 
  apply(ecc_eval_all[,6:55], 1, FUN = function(x) {var(x, na.rm = TRUE)} )

View(gca_eval_all)
which(gca_eval_all == Inf)

# scores
es_ecc = rep(NA, length(eval_dates))
vs_ecc = rep(NA, length(eval_dates))
es_ssh = rep(NA, length(eval_dates))
vs_ssh = rep(NA, length(eval_dates))
es_gca = rep(NA, length(eval_dates))
vs_gca = rep(NA, length(eval_dates))
es_emos = rep(NA, length(eval_dates))
vs_emos = rep(NA, length(eval_dates))
es_ens = rep(NA, length(eval_dates))
vs_ens = rep(NA, length(eval_dates))

for(day_id in 1:length(eval_dates)){
  # print(day_id)
  today <- eval_dates[day_id]
  
  data_today <- subset(data_eval_all, date == today )
  
  ecc_data_today <- subset(ecc_eval_all, date == today)
  ssh_data_today <- subset(ssh_eval_all, date == today)
  gca_data_today <- subset(gca_eval_all, date == today)
  emos_data_today <- subset(emos_eval_all, date == today)
  
  ecc_es_today = es_sample(y = ecc_data_today$obs, 
                           dat = as.matrix(ecc_data_today[,6:55]))
  ecc_vs_today = vs_sample(y = ecc_data_today$obs, 
                           dat = as.matrix(ecc_data_today[,6:55]))
  ssh_es_today = es_sample(y = ssh_data_today$obs, 
                           dat = as.matrix(ssh_data_today[,6:55]))
  ssh_vs_today = vs_sample(y = ssh_data_today$obs, 
                           dat = as.matrix(ssh_data_today[,6:55]))
  gca_es_today = es_sample(y = gca_data_today$obs, 
                           dat = as.matrix(gca_data_today[,6:55]))
  gca_vs_today = vs_sample(y = gca_data_today$obs, 
                           dat = as.matrix(gca_data_today[,6:55]))
  
  emos_es_today = es_sample(y = emos_data_today$obs, 
                           dat = as.matrix(emos_data_today[,6:55]))
  emos_vs_today = vs_sample(y = emos_data_today$obs, 
                           dat = as.matrix(emos_data_today[,6:55]))
  ens_es_today = es_sample(y = emos_data_today$obs,
                           dat = as.matrix(data_today[,6:55]))
  ens_vs_today = vs_sample(y = emos_data_today$obs,
                           dat = as.matrix(data_today[,6:55]))
  
  es_ecc[day_id] = ecc_es_today
  vs_ecc[day_id] = ecc_vs_today
  es_ssh[day_id] = ssh_es_today
  vs_ssh[day_id] = ssh_vs_today
  es_gca[day_id] = gca_es_today
  vs_gca[day_id] = gca_vs_today
  
  es_emos[day_id] = emos_es_today
  vs_emos[day_id] = emos_vs_today
  es_ens[day_id] = ens_es_today
  vs_ens[day_id] = ens_vs_today
}

summary(es_ecc)
summary(es_ssh)
summary(es_gca)
summary(es_emos)
summary(es_ens)

summary(vs_ecc)
summary(vs_ssh)
summary(vs_gca)
summary(vs_emos)
summary(vs_ens)

save.image("E:/0-TIGGE_ECMWF_Germany/ecmwf_ens_subset_mvpp.RData")



library(ensembleBMA)

# rank histograms:
# raw ensemble members
verifRankHist(ens_fc_t2m_mv_subset[,6:55],ens_fc_t2m_mv_subset[,3])
# ensemble members after univariate post-processing
verifRankHist(emos_eval_all[,6:55],emos_eval_all[,3])
# ensemble members generated from GCA method
verifRankHist(gca_eval_all[,6:55],gca_eval_all[,3])

for (this_station in stations_list) {
  this_station_label = which(emos_eval_all$station == this_station)
  verifRankHist(emos_eval_all[this_station_label,6:55],
                emos_eval_all[this_station_label,3])
  verifRankHist(gca_eval_all[this_station_label,6:55],
                emos_eval_all[this_station_label,3])
}

# average multivariate rank histogram
multi_rank = rep(NA, length(eval_dates))
for(day_id in 1:length(eval_dates)){
  # print(day_id)
  today <- eval_dates[day_id]
  
  ecc_data_today <- subset(ecc_eval_all, date == today)
  # ssh_data_today <- subset(ssh_eval_all, date == today)
  # gca_data_today <- subset(gca_eval_all, date == today)
  # emos_data_today <- subset(emos_eval_all, date == today)
  
  observations = ecc_data_today[,3]
  forecasts = ecc_data_today[,6:55]
  
  trajectory_rank <- apply(cbind(observations, forecasts), 1, function(x) 
    rank(x, ties = "random"))
  average_rank = apply(trajectory_rank, 1, mean)
  obs_average_rank = rank(average_rank, ties = "random")[1]

  multi_rank[day_id] = obs_average_rank
}

hist(multi_rank)











