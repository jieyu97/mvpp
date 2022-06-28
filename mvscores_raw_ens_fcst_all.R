library(forecast)
library(scoringRules)
library(lubridate)
library(parallel)
library(tidyverse)

an = 100

eval_start <- as.Date("2016-01-01 00:00 UTC")
eval_end <- as.Date("2016-12-31 00:00 UTC")
eval_dates0 <- seq(eval_start, eval_end, by = "1 day")

es_t2m_5d_all = as.data.frame(matrix(NA, nrow = length(eval_dates0), ncol = an))
es_t2m_10d_all = es_t2m_5d_all
es_t2m_20d_all = es_t2m_5d_all
es_ws_5d_all = es_t2m_5d_all
es_ws_10d_all = es_t2m_5d_all
es_ws_20d_all = es_t2m_5d_all

vs2_t2m_5d_all = es_t2m_5d_all
vs2_t2m_10d_all = es_t2m_5d_all
vs2_t2m_20d_all = es_t2m_5d_all
vs2_ws_5d_all = es_t2m_5d_all
vs2_ws_10d_all = es_t2m_5d_all
vs2_ws_20d_all = es_t2m_5d_all

vs1_t2m_5d_all = es_t2m_5d_all
vs1_t2m_10d_all = es_t2m_5d_all
vs1_t2m_20d_all = es_t2m_5d_all
vs1_ws_5d_all = es_t2m_5d_all
vs1_ws_10d_all = es_t2m_5d_all
vs1_ws_20d_all = es_t2m_5d_all

compute_es_vs_lists = function(data_ens_fcst, n_cores_use = 44){
  
  eval_dates = unique(data_ens_fcst$date)
  
  allscores = mclapply(1:length(eval_dates), mc.cores = n_cores_use, 
                       function(day_id){
    
    today <- eval_dates[day_id]
    
    ens_data_today <- subset(data_ens_fcst, date == today)
    
    ens_es_today = es_sample(y = ens_data_today$obs, 
                             dat = as.matrix(ens_data_today[,3:52]))
    
    ens_vs2_today = vs_sample(y = ens_data_today$obs, 
                              dat = as.matrix(ens_data_today[,3:52]),
                              p = 0.5)
    
    ens_vs1_today = vs_sample(y = ens_data_today$obs, 
                              dat = as.matrix(ens_data_today[,3:52]),
                              p = 1)
    
    all = c(ens_es_today, ens_vs2_today, ens_vs1_today)
    return(all)
  })
  
  allscores_df = data.frame(matrix(unlist(allscores), 
                                   nrow=length(allscores), byrow=TRUE))
  
  colnames(allscores_df) = c('es', 'vs2', 'vs1')
  
  return(allscores_df)
}


path_raw_ens = '/Data/Jieyu_data/raw_ens_fcst/'

for (n in 1:an) {
  print(n)
  
  t2m_5dim_all = read.csv(paste0(path_raw_ens,"t2m_5dim_sa",n,".csv"))
  t2m_5dim_all = t2m_5dim_all[,-1]
  t2m_10dim_all = read.csv(paste0(path_raw_ens,"t2m_10dim_sa",n,".csv"))
  t2m_10dim_all = t2m_10dim_all[,-1]
  t2m_20dim_all = read.csv(paste0(path_raw_ens,"t2m_20dim_sa",n,".csv"))
  t2m_20dim_all = t2m_20dim_all[,-1]
  
  ws_5dim_all = read.csv(paste0(path_raw_ens,"ws_5dim_sa",n,".csv"))
  ws_5dim_all = ws_5dim_all[,-1]
  ws_10dim_all = read.csv(paste0(path_raw_ens,"ws_10dim_sa",n,".csv"))
  ws_10dim_all = ws_10dim_all[,-1]
  ws_20dim_all = read.csv(paste0(path_raw_ens,"ws_20dim_sa",n,".csv"))
  ws_20dim_all = ws_20dim_all[,-1]
  
  mvs_raw_t2m_5dim = compute_es_vs_lists(t2m_5dim_all)
  mvs_raw_t2m_10dim = compute_es_vs_lists(t2m_10dim_all)
  mvs_raw_t2m_20dim = compute_es_vs_lists(t2m_20dim_all)
  
  mvs_raw_ws_5dim = compute_es_vs_lists(ws_5dim_all)
  mvs_raw_ws_10dim = compute_es_vs_lists(ws_10dim_all)
  mvs_raw_ws_20dim = compute_es_vs_lists(ws_20dim_all)

  es_t2m_5d_all[,n] = mvs_raw_t2m_5dim$es
  es_t2m_10d_all[,n] = mvs_raw_t2m_10dim$es
  es_t2m_20d_all[,n] = mvs_raw_t2m_20dim$es
  es_ws_5d_all[,n] = mvs_raw_ws_5dim$es
  es_ws_10d_all[,n] = mvs_raw_ws_10dim$es
  es_ws_20d_all[,n] = mvs_raw_ws_20dim$es
  
  vs2_t2m_5d_all[,n] = mvs_raw_t2m_5dim$vs2
  vs2_t2m_10d_all[,n] = mvs_raw_t2m_10dim$vs2
  vs2_t2m_20d_all[,n] = mvs_raw_t2m_20dim$vs2
  vs2_ws_5d_all[,n] = mvs_raw_ws_5dim$vs2
  vs2_ws_10d_all[,n] = mvs_raw_ws_10dim$vs2
  vs2_ws_20d_all[,n] = mvs_raw_ws_20dim$vs2
  
  vs1_t2m_5d_all[,n] = mvs_raw_t2m_5dim$vs1
  vs1_t2m_10d_all[,n] = mvs_raw_t2m_10dim$vs1
  vs1_t2m_20d_all[,n] = mvs_raw_t2m_20dim$vs1
  vs1_ws_5d_all[,n] = mvs_raw_ws_5dim$vs1
  vs1_ws_10d_all[,n] = mvs_raw_ws_10dim$vs1
  vs1_ws_20d_all[,n] = mvs_raw_ws_20dim$vs1
}

save.image("/home/chen_jieyu/IGEP/mvscores_raw_ens_fcst_all.RData")
