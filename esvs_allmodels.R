library(forecast)
library(scoringRules)
library(lubridate)
library(parallel)
library(dplyr)

DIM = 5
var = 'ws'

n_mc_cores = 44
n_rep = 100

eval_start <- as.Date("2016-01-01 00:00 UTC")
eval_end <- as.Date("2016-12-31 00:00 UTC")
eval_dates0 <- seq(eval_start, eval_end, by = "1 day")

es_cgm_all = as.data.frame(matrix(NA, nrow = length(eval_dates0), ncol = n_rep))
vs1_cgm_all = es_cgm_all
vs2_cgm_all = es_cgm_all
  
es_nnecc_all = es_cgm_all
vs1_nnecc_all = es_cgm_all
vs2_nnecc_all = es_cgm_all

es_nngca_all = es_cgm_all
vs1_nngca_all = es_cgm_all
vs2_nngca_all = es_cgm_all

es_ecc_all = es_cgm_all
vs1_ecc_all = es_cgm_all
vs2_ecc_all = es_cgm_all

es_gca_all = es_cgm_all
vs1_gca_all = es_cgm_all
vs2_gca_all = es_cgm_all


for (n in 1:n_rep) {
  print(n)

  # for 'ws' with 5 dim and 20 dim, add _ after sa
  ecc_ens = read.csv(paste0("/Data/Jieyu_data/rmvpp_", var, "_", 
                            DIM, "dim/ecc_", DIM, "nbhd_sa_", n, ".csv"))
  ecc_ens = ecc_ens[,-1]
  ecc_ens$date = as.Date(ecc_ens$date)
  
  gca_ens = read.csv(paste0("/Data/Jieyu_data/rmvpp_", var, "_", 
                            DIM, "dim/gca_", DIM, "nbhd_sa_", n, ".csv"))
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
  
  labels = which(eval_dates0 %in% eval_dates)
  
  
  allscores = mclapply(1:length(eval_dates), mc.cores = n_mc_cores, 
                       function(day_id){
    today <- eval_dates[day_id]

    cgm_today = subset(cgm_ens, date == today)
    cgm_es_today = es_sample(y = cgm_today$obs,
                               dat = as.matrix(cgm_today[,3:52]))
    cgm_vs1_today = vs_sample(y = cgm_today$obs,
                               dat = as.matrix(cgm_today[,3:52]),
                               p = 1)
    cgm_vs2_today = vs_sample(y = cgm_today$obs,
                               dat = as.matrix(cgm_today[,3:52]),
                               p = 0.5)
                       
    nnecc_today = subset(nnecc_ens, date == today)
    nnecc_es_today = es_sample(y = nnecc_today$obs,
                               dat = as.matrix(nnecc_today[,4:53]))
    nnecc_vs1_today = vs_sample(y = nnecc_today$obs,
                               dat = as.matrix(nnecc_today[,4:53]),
                               p = 1)
    nnecc_vs2_today = vs_sample(y = nnecc_today$obs,
                               dat = as.matrix(nnecc_today[,4:53]),
                               p = 0.5)                           
                           
    nngca_today = subset(nngca_ens, date == today)
    nngca_es_today = es_sample(y = nngca_today$obs,
                               dat = as.matrix(nngca_today[,4:53]))
    nngca_vs1_today = vs_sample(y = nngca_today$obs,
                               dat = as.matrix(nngca_today[,4:53]),
                               p = 1)
    nngca_vs2_today = vs_sample(y = nngca_today$obs,
                               dat = as.matrix(nngca_today[,4:53]),
                               p = 0.5)                                                                         
                               
    ecc_today = subset(ecc_ens, date == today)
    ecc_es_today = es_sample(y = ecc_today$obs,
                               dat = as.matrix(ecc_today[,3:52]))
    ecc_vs1_today = vs_sample(y = ecc_today$obs,
                               dat = as.matrix(ecc_today[,3:52]),
                               p = 1)
    ecc_vs2_today = vs_sample(y = ecc_today$obs,
                               dat = as.matrix(ecc_today[,3:52]),
                               p = 0.5)                           
                           
    gca_today = subset(gca_ens, date == today)
    gca_es_today = es_sample(y = gca_today$obs,
                               dat = as.matrix(gca_today[,3:52]))
    gca_vs1_today = vs_sample(y = gca_today$obs,
                               dat = as.matrix(gca_today[,3:52]),
                               p = 1)
    gca_vs2_today = vs_sample(y = gca_today$obs,
                               dat = as.matrix(gca_today[,3:52]),
                               p = 0.5)   
                                                      
    all = c(cgm_es_today, cgm_vs1_today, cgm_vs2_today,
            nnecc_es_today, nnecc_vs1_today, nnecc_vs2_today,
            nngca_es_today, nngca_vs1_today, nngca_vs2_today,
            ecc_es_today, ecc_vs1_today, ecc_vs2_today,
            gca_es_today, gca_vs1_today, gca_vs2_today)
    return(all)
  })
  
  allscores_df = data.frame(matrix(unlist(allscores), nrow=length(allscores), byrow=TRUE))
  
  es_cgm_all[labels,n] = allscores_df[,1]
  vs1_cgm_all[labels,n] = allscores_df[,2]
  vs2_cgm_all[labels,n] = allscores_df[,3]
    
  es_nnecc_all[labels,n] = allscores_df[,4]
  vs1_nnecc_all[labels,n] = allscores_df[,5]
  vs2_nnecc_all[labels,n] = allscores_df[,6]
  
  es_nngca_all[labels,n] = allscores_df[,7]
  vs1_nngca_all[labels,n] = allscores_df[,8]
  vs2_nngca_all[labels,n] = allscores_df[,9]
  
  es_ecc_all[labels,n] = allscores_df[,10]
  vs1_ecc_all[labels,n] = allscores_df[,11]
  vs2_ecc_all[labels,n] = allscores_df[,12]
  
  es_gca_all[labels,n] = allscores_df[,13]
  vs1_gca_all[labels,n] = allscores_df[,14]
  vs2_gca_all[labels,n] = allscores_df[,15]
}


save.image(paste0("/home/chen_jieyu/IGEP/esvs_all_", 
                  var, "_", DIM, "dim", ".RData"))

