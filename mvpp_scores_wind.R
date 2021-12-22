library("forecast") # , lib.loc = '/home/chen_jieyu/R/x86_64-pc-linux-gnu-library/4.0/')
library("scoringRules") # , lib.loc = '/home/chen_jieyu/R/x86_64-pc-linux-gnu-library/4.0/')
library("lubridate") # , lib.loc = '/home/chen_jieyu/R/x86_64-pc-linux-gnu-library/4.0/')
library("parallel") # , lib.loc = '/home/chen_jieyu/R/x86_64-pc-linux-gnu-library/4.0/')
# library(tidyverse)
# library(devtools)
# install_github("FK83/scoringRules")

an = 100

eval_start <- as.Date("2016-01-01 00:00 UTC")
eval_end <- as.Date("2016-12-31 00:00 UTC")
eval_dates0 <- seq(eval_start, eval_end, by = "1 day")

es_igfinal_all = as.data.frame(matrix(NA, nrow = length(eval_dates0), ncol = an))
vs_igfinal_all = es_igfinal_all

path_igep1 = '/home/chen_jieyu/IGEP/ig325_wind_fixhp/'

for (n in 1:an) {
  print(n)
  pn = n-1
  
  ig322_eval_all = read.csv(paste0(path_igep1,"ws_20dim_ig325_",pn,".csv"))
  ig322_eval_all$date = as.Date(ig322_eval_all$date)
  
  eval_dates = unique(ig322_eval_all$date)
  labels = which(eval_dates0 %in% eval_dates)
  
  allscores = mclapply(1:length(eval_dates), mc.cores = 44, function(day_id){
    today <- eval_dates[day_id]

    ig322_data_today = subset(ig322_eval_all, date == today)
    
    ig322_es_today = es_sample(y = ig322_data_today$obs,
                               dat = as.matrix(ig322_data_today[,3:52]))
    ig322_vs_today = vs_sample(y = ig322_data_today$obs,
                               dat = as.matrix(ig322_data_today[,3:52]),
                               p = 1)
    
    all = c(ig322_es_today, ig322_vs_today)
    return(all)
  })
  
  allscores_df = data.frame(matrix(unlist(allscores), nrow=length(allscores), byrow=TRUE))
    
  es_igfinal_all[labels,n] = allscores_df[,1]
  vs_igfinal_all[labels,n] = allscores_df[,2]
}

save.image("/home/chen_jieyu/IGEP/igfinal_ws_20dim_multi.RData")


# vs_ig2add_all_new = vs_ig2add_all

