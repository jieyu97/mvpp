library(forecast)
library(scoringRules)
library(lubridate)
library(parallel)
library(dplyr)

DIM = 10
var = 'ws'

n_mc_cores = 44
n_rep = 100

# test date
eval_start = as.Date("2016-01-01 00:00 UTC")
eval_end = as.Date("2016-12-31 00:00 UTC")
eval_dates0 = seq(eval_start, eval_end, by = "1 day")
eval_dates0_rep = rep(eval_dates0, each = DIM)

crps_cgm_all = as.data.frame(matrix(NA, nrow = length(eval_dates0_rep), 
                                        ncol = n_rep))
crps_nnecc_all = crps_cgm_all
crps_nngca_all = crps_cgm_all
crps_ecc_all = crps_cgm_all
crps_gca_all = crps_cgm_all


for (n in 1:n_rep) {
  print(n)
  
  # for 'ws' with 5 dim and 20 dim, add _ after sa
  ecc_ens = read.csv(paste0("/Data/Jieyu_data/rmvpp_", var, "_", 
                            DIM, "dim/ecc_", DIM, "nbhd_sa", n, ".csv"))
  ecc_ens = ecc_ens[,-1]
  ecc_ens$date = as.Date(ecc_ens$date)
  
  gca_ens = read.csv(paste0("/Data/Jieyu_data/rmvpp_", var, "_", 
                            DIM, "dim/gca_", DIM, "nbhd_sa", n, ".csv"))
  gca_ens = gca_ens[,-1]
  gca_ens$date = as.Date(gca_ens$date)
  
  cgm_ens = read.csv(paste0("/Data/Jieyu_data/mvpp_cgm/",
                            var, "_", DIM, "dim_sa", n, ".csv"))
  cgm_ens$date = as.Date(cgm_ens$date)
  
  nnecc_ens = read.csv(paste0("/Data/Jieyu_data/mvpp_drn_ecc/",
                              var, "_", DIM, "dim_sa", n, ".csv"))
  nnecc_ens = nnecc_ens[,-1]
  nnecc_ens$date = as.Date(nnecc_ens$date)
  
  nngca_ens = read.csv(paste0("/Data/Jieyu_data/mvpp_drn_gca/",
                              var, "_", DIM, "dim_sa", n, ".csv"))
  nngca_ens = nngca_ens[,-1]
  nngca_ens$date = as.Date(nngca_ens$date)
  
  eval_dates = unique(cgm_ens$date)
  
  x = which(table(cgm_ens$date) != DIM)

  if(length(x) > 0) {eval_dates = eval_dates[-x]}
  
  labels = which(eval_dates0_rep %in% eval_dates)
  
  test_labels = which(cgm_ens$date %in% eval_dates)

  
  allscores = mclapply(1:length(test_labels), mc.cores = n_mc_cores, 
                       function(label){
    test_label = test_labels[label]

    cgm_crps = crps_sample(y = cgm_ens$obs[test_label],
                          dat = as.matrix(cgm_ens[test_label,3:52]),
                          method = "edf")
         
    nnecc_crps = crps_sample(y = nnecc_ens$obs[test_label],
                            dat = as.matrix(nnecc_ens[test_label,4:53]),
                            method = "edf")                           
                          
    nngca_crps = crps_sample(y = nngca_ens$obs[test_label],
                            dat = as.matrix(nngca_ens[test_label,4:53]),
                            method = "edf")                         
                          
    ecc_crps = crps_sample(y = ecc_ens$obs[test_label],
                          dat = as.matrix(ecc_ens[test_label,3:52]),
                          method = "edf")                  
                          
    gca_crps = crps_sample(y = gca_ens$obs[test_label],
                          dat = as.matrix(gca_ens[test_label,3:52]),
                          method = "edf")                           
                              
    all = c(cgm_crps, nnecc_crps, nngca_crps, ecc_crps, gca_crps)
            
    return(all)
  })
  
  allscores_df = data.frame(matrix(unlist(allscores), nrow=length(allscores), byrow=TRUE))
  
  crps_cgm_all[labels,n] = allscores_df[,1]
  crps_nnecc_all[labels,n] = allscores_df[,2]
  crps_nngca_all[labels,n] = allscores_df[,3]
  crps_ecc_all[labels,n] = allscores_df[,4]
  crps_gca_all[labels,n] = allscores_df[,5]
  
}

save.image(paste0("/home/chen_jieyu/IGEP/crps_all_", 
                  var, "_", DIM, "dim", ".RData"))



