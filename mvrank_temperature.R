library(lubridate)
library(parallel)

load("/home/chen_jieyu/IGEP/igfinal_t2m_20dim_multi.RData")

an = 100

eval_start <- as.Date("2016-01-01 00:00 UTC")
eval_end <- as.Date("2016-12-31 00:00 UTC")
eval_dates0 <- seq(eval_start, eval_end, by = "1 day")

igfinal_avr_all = as.data.frame(matrix(NA, nrow = length(eval_dates0), ncol = an))

igfinal_bdr_all = igfinal_avr_all
igfinal_mvr_all = igfinal_avr_all

# Average rank
compute_avr <- function(obs, fc){
  x <- cbind(obs, fc)
  x.ranks <- apply(x,1,rank)
  x.preranks <- apply(x.ranks,1,mean)
  x.rank <- rank(x.preranks,ties="random")
  return(x.rank[1])
}
# Band depth rank
compute_bdr <- function(obs, fc){
  x <- cbind(obs, fc)
  d <- dim(x)
  x.prerank <- array(NA,dim=d)
  for(i in 1:d[1]) {
    z <- x[i,]
    tmp.ranks <- rank(z)
    x.prerank[i,] <- tmp.ranks * {d[2] - tmp.ranks} + {tmp.ranks - 1} *
      sapply(z, function(x, z) sum(x == z), z = z)
  }
  x.rank <- apply(x.prerank, 2, mean)
  x.rank <- rank(x.rank, ties = "random")[1]
  return(x.rank)
}
# Multivariate rank
compute_mvr <- function(obs, fc){
  x <- cbind(obs, fc)
  d <- dim(x)
  x.prerank <- numeric(d[2])
  for(i in 1:d[2]) {
    x.prerank[i] <- sum(apply(x<=x[,i],2,all))
  }
  x.rank <- rank(x.prerank,ties="random")
  return(x.rank[1])
}

path_igep = '/home/chen_jieyu/IGEP/ig326_tem_fixhp/'

for (n in 1:an) {
  print(n)
  pn = n-1
  ig2aux_eval_all = read.csv(paste0(path_igep,"t2m_20dim_ig326_",pn,".csv"))
  
  ig2aux_eval_all$date = as.Date(ig2aux_eval_all$date)
  
  eval_dates = unique(ig2aux_eval_all$date)
  labels = which(eval_dates0 %in% eval_dates)
  
  #------ multivariate rank histograms
  
  allranks = mclapply(1:length(eval_dates), mc.cores=44, function(day_id){
    today <- eval_dates[day_id]
    
    ig2aux_data_today = subset(ig2aux_eval_all, date == today)
    
    ig2aux_avr = compute_avr(obs = ig2aux_data_today$obs, 
                                      fc = ig2aux_data_today[,3:52])
    ig2aux_bdr = compute_bdr(obs = ig2aux_data_today$obs, 
                                      fc = ig2aux_data_today[,3:52])
    ig2aux_mvr = compute_mvr(obs = ig2aux_data_today$obs, 
                                      fc = ig2aux_data_today[,3:52])
    
    all = c(ig2aux_avr, ig2aux_bdr, ig2aux_mvr)
    return(all)
  })
  
  allranks_df = data.frame(matrix(unlist(allranks), nrow=length(allranks), byrow=TRUE))
  
  igfinal_avr_all[labels,n] = allranks_df[,1]
  igfinal_bdr_all[labels,n] = allranks_df[,2]
  igfinal_mvr_all[labels,n] = allranks_df[,3]
  
}

save.image("/home/chen_jieyu/IGEP/igfinal_t2m_20dim_multi.RData")
