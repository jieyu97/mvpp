library(forecast)
library(scoringRules)
library(lubridate)
library(parallel)
library(dplyr)

an = 100
DIM = 5

path_igep1 = '/home/chen_jieyu/IGEP/ig325_wind_fixhp/'

# test date
eval_start = as.Date("2016-01-01 00:00 UTC")
eval_end = as.Date("2016-12-31 00:00 UTC")
eval_dates0 = seq(eval_start, eval_end, by = "1 day")
eval_dates0_rep = rep(eval_dates0, each = DIM)

crps_igfinal_all = as.data.frame(matrix(NA, nrow = length(eval_dates0_rep), ncol = an))
rank_igfinal_all = crps_igfinal_all

for (n in 1:an) {
  print(n)
  pn = n-1

  ig322_eval_all = read.csv(paste0(path_igep1,"ws_5dim_ig325_",pn,".csv"))
  ig322_eval_all$date = as.Date(ig322_eval_all$date)
  
  eval_dates = unique(ig322_eval_all$date)
  labels = which(eval_dates0_rep %in% eval_dates)
  
  test_labels = 1:nrow(ig322_eval_all)

  allscores = mclapply(test_labels, mc.cores = 44, function(label){

    ig322_crps_label = crps_sample(y = ig322_eval_all$obs[label],
                                dat = as.matrix(ig322_eval_all[label,3:52]),
                                method = "edf")
    ig322_rank_label = rank(as.numeric(cbind(ig322_eval_all$obs[label],
                                               ig322_eval_all[label,3:52])),
                          ties = "random")[1]
                                                 
    all = c(ig322_crps_label, ig322_rank_label)
    return(all)
  })

  allscores_df = data.frame(matrix(unlist(allscores), nrow=length(allscores), byrow=TRUE))
  
  crps_igfinal_all[labels,n] = allscores_df[,1]
  rank_igfinal_all[labels,n] = allscores_df[,2]  
}

save.image("/home/chen_jieyu/IGEP/igfinal_ws_5dim_uni.RData")

# summary(crps_ens)

# hist(rank_ens)



