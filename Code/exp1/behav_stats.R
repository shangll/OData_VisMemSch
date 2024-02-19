library(tidyverse)
library(ggpubr)
library(rstatix)
library(pwr)


options(scipen=0,digits=5)
cond_list <- c("within","between")



# ################ EXPERIMENT 1a ################# #


# ##### #
# BEHAV #
# ##### #

setwd("U:\\Documents\\DCC\\exp3\\AllExpRes")
df_mean_all <- read.csv("data_allExp_mean.csv")

df_exp1a <- subset(df_mean_all,df_mean_all$exp=="exp1a")
df_exp1a$setsize <- factor(df_exp1a$setsize,levels=c(1,2,4,8,16))

# df_exp1a %>%
#   group_by(exp,setsize,cond) %>%
#   identify_outliers(rt)

# # test#########################
# mss_16w <- subset(df_exp1a,(df_exp1a$cond=="within")&(df_exp1a$setsize==16))
# ggbarplot(
#   mss_16w,y="rt",add="mean_se")




# ANOVA
# #####

# ACC: 2-way anova
res.aov <- anova_test(
  data=df_exp1a,dv=acc,wid=subj,
  within=c(setsize,cond),
  type=3,effect.size="pes",detailed=TRUE)
get_anova_table(res.aov)
#
simpeff <- df_exp1a %>%
  group_by(setsize) %>%
  anova_test(dv=acc,wid=subj,within=cond,
             type=3,effect.size="pes",detailed=TRUE) %>%
  get_anova_table()
simpeff
pwc <- df_exp1a %>%
  group_by(setsize) %>%
  pairwise_t_test(acc~cond,paired=TRUE,
                  p.adjust.method="bonferroni")
pwc
#
simpeff <- df_exp1a %>%
  group_by(cond) %>%
  anova_test(dv=acc,wid=subj,within=setsize,
             type=3,effect.size="pes",detailed=TRUE) %>%
  get_anova_table()
simpeff
pwc <- df_exp1a %>%
  group_by(cond) %>%
  pairwise_t_test(acc~setsize,paired=TRUE,
                  p.adjust.method="bonferroni")
pwc


# RT: 2-way anova
res.aov <- anova_test(
  data=df_exp1a,dv=rt,wid=subj,
  within=c(setsize,cond),
  type=3,effect.size="pes",detailed=TRUE)
get_anova_table(res.aov)
#
simpeff <- df_exp1a %>%
  group_by(setsize) %>%
  anova_test(dv=rt,wid=subj,within=cond,
             type=3,effect.size="pes",detailed=TRUE) %>%
  get_anova_table()
# fix(simpeff)
pwc <- df_exp1a %>%
  group_by(setsize) %>%
  pairwise_t_test(rt~cond,paired=TRUE,
                  p.adjust.method="bonferroni")
# fix(pwc)
#
simpeff <- df_exp1a %>%
  group_by(cond) %>%
  anova_test(dv=rt,wid=subj,within=setsize,
             type=3,effect.size="pes",detailed=TRUE) %>%
  get_anova_table()
fix(simpeff)
pwc <- df_exp1a %>%
  group_by(cond) %>%
  pairwise_t_test(rt~setsize,paired=TRUE,
                  p.adjust.method="bonferroni")
# fix(pwc)


# predicting
# ##########

w_obs_rt <- subset(df_exp1a,(setsize==16)&(cond=="within"))$rt
b_obs_rt <- subset(df_exp1a,(setsize==16)&(cond=="between"))$rt
w_16_lm <- subset(df_exp1a,(setsize==16)&(cond=="within"))$lm
w_16_log <- subset(df_exp1a,(setsize==16)&(cond=="within"))$log
b_16_lm <- subset(df_exp1a,(setsize==16)&(cond=="between"))$lm
b_16_log <- subset(df_exp1a,(setsize==16)&(cond=="between"))$log

print('within: lm')
t_val <- t.test(w_16_lm,w_obs_rt,paired=TRUE)
# t_val
dat <- data.frame(mss16=c(w_16_lm,w_obs_rt),
                  pred=rep(c("lm","rt"),each=length(subj_list)))
d <- dat %>% cohens_d(mss16~pred,paired=TRUE)
# d
sprintf('t = %0.3f, df = %0.3f, p = %0.3f, d = %0.3f',
        t_val$statistic,t_val$parameter,t_val$p.value,d[1,4])

# ----------------------------
print('within: log2')
t_val <- t.test(w_16_log,w_obs_rt,paired=TRUE)
# t_val
dat <- data.frame(mss16=c(w_16_log,w_obs_rt),
                  pred=rep(c("log","rt"),each=length(subj_list)))
d <- dat %>% cohens_d(mss16~pred,paired=TRUE)
# d
sprintf('t = %0.3f, df = %0.3f, p = %0.3f, d = %0.3f',
        t_val$statistic,t_val$parameter,t_val$p.value,d[1,4])

# ----------------------------
print('bertween: lm')
t_val <- t.test(b_16_lm,b_obs_rt,paired=TRUE)
# t_val
dat <- data.frame(mss16=c(b_16_lm,b_obs_rt),
                  pred=rep(c("lm","rt"),each=length(subj_list)))
d <- dat %>% cohens_d(mss16~pred,paired=TRUE)
# d
sprintf('t = %0.3f, df = %0.3f, p = %0.3f, d = %0.3f',
        t_val$statistic,t_val$parameter,t_val$p.value,d[1,4])

# ----------------------------
print('between: log2')
t_val <- t.test(b_16_log,b_obs_rt,paired=TRUE)
# t_val
dat <- data.frame(mss16=c(b_16_log,b_obs_rt),
                  pred=rep(c("log","rt"),each=length(subj_list)))
d <- dat %>% cohens_d(mss16~pred,paired=TRUE)
# d
sprintf('t = %0.3f, df = %0.3f, p = %0.3f, d = %0.3f',
        t_val$statistic,t_val$parameter,t_val$p.value,d[1,4])
#


# Modelling
# #########
subj_list <- unique(df_exp1a$subj)
df_exp1a$setsize <- as.numeric(as.character(df_exp1a$setsize))

coeff_lm_list <- c()
coeff_log_list <- c()
r2_lm_list <- c()
r2_log_list <- c()

for (n in subj_list){
  for (k in cond_list){
    
    df_subj <- subset(df_exp1a,(subj==n)&(cond==k))
    lm_model <- lm(formula=rt~setsize,df_subj)
    log_model <- lm(formula=rt~log(setsize,2),df_subj)
    res_lm <- summary(lm_model)
    res_log <- summary(log_model)
    coeff_lm <- coef(res_lm)
    coeff_log <- coef(res_log)
    coeff_lm_list <- append(coeff_lm_list,coeff_lm[2,1])
    coeff_log_list <- append(coeff_log_list,coeff_log[2,1])
    
    r2_lm_list <- append(r2_lm_list,res_lm$r.squared)
    r2_log_list <- append(r2_log_list,res_log$r.squared)
    
    
  }
}


# compare the coeffects between w & b
conds <- rep(cond_list,times=length(subj_list))
subjs <- rep(subj_list,each=2)
df_coeff <- data.frame(
  subj=subjs,cond=conds,lm=coeff_lm_list,
  log=coeff_log_list,r2_lm=r2_lm_list,
  r2_log=r2_log_list)
df_coeff$r_lm <- with(df_coeff,sqrt(r2_lm))
df_coeff$r_log <- with(df_coeff,sqrt(r2_log))
coeff_w <- subset(df_coeff,cond=="within")
coeff_b <- subset(df_coeff,cond=="between")
t_val <- t.test(coeff_w$log,coeff_b$log,paired=TRUE)
t_val
d <- df_coeff %>% cohens_d(log~cond,paired=TRUE)
d
print("Compare the slope coefficients between w&b")
sprintf('t = %0.3f, p = %0.3f, d = %0.3f',
        t_val$statistic,t_val$p.value,d[1,4])

# compare 2 models

# r1 <- cor(datAnd$CZ, datAnd$CRAT)
# r2 <- pcor(c(1,2,3),cov(pc))
# zf1=1/2*log((1+r1)/(1-r1))
# zf2=1/2*log((1+r2)/(1-r2))
# zr <- (zf1-zf2)/sqrt(1/(44-3)+1/(44-3))
# pnorm(zr)

df_cond <- aggregate(df_exp1a$rt,
                     by=list(df_exp1a$setsize,
                             df_exp1a$cond),mean)
df_cond <- plyr::rename(df_cond,
                        c("Group.1"="setsize",
                          "Group.2"="cond","x"="rt"))
mean_lm <- summary(lm(formula=rt~setsize+cond+setsize:cond,df_cond))
mean_log <- summary(lm(formula=rt~log(setsize,2)+cond+log(setsize,2):cond,df_cond))

subjN <- length(subj_list)
r_lm <- sqrt(mean_lm$r.squared)
r_log <- sqrt(mean_log$r.squared)
zf_lm <- 1/2*log((1+r_lm)/(1-r_lm))
zf_log <- 1/2*log((1+r_log)/(1-r_log))
s <- sqrt(1/(subjN-3)+1/(subjN-3))
z <- (zf_lm-zf_log)/s
pnorm(z)



# for (k in cond_list){
#   df_cond <- subset(df_coeff,(cond==k))
#   df_r2_cond <- data.frame(r_val=c(df_cond$r_lm,df_cond$r_log),
#                            pred=rep(c("lm","log"),each=length(subj_list)))
#   t_val <- t.test(df_cond$r_lm,df_cond$r_log)
#   print("--------------")
#   print(k)
#   print(t_val)
#   d <- df_r2_cond %>% cohens_d(r_val~pred)
#   print(d)
#   sprintf('t = %0.3f, p = %0.3f, d = %0.3f',
#           t_val$statistic,t_val$p.value,d[1,4])
# }


# for (k in cond_list){
#   df_cond_allSubj <- subset(df_exp1a,cond==k)
#   df_cond <- aggregate(df_cond_allSubj$rt,
#                        by=list(df_cond_allSubj$setsize),mean)
#   df_cond <- plyr::rename(df_cond,
#                           c("Group.1"="setsize",
#                             "x"="rt"))
#   mean_lm = lm(formula=rt~setsize,df_cond)
#   mean_log = lm(formula=rt~log(setsize,2),df_cond)
#   
#   res <- anova(mean_lm,mean_log,test="F")
#   print("--------------")
#   print(k)
#   print(summary(mean_lm))
#   print(summary(mean_log))
#   print(res)
#   print(AIC(mean_lm,mean_log))
# }
# 



# ################ EXPERIMENT 3 ################# #

# ##### #
# BEHAV #
# ##### #

df_exp3 <- subset(df_mean_all,df_mean_all$exp=="exp3")
df_exp3$setsize <- factor(df_exp3$setsize,levels=c(1,2,4,8))

df_exp3_size <- aggregate(df_exp3$rt,
                          by=list(df_exp3$subj,
                                  df_exp3$setsize),mean)
df_exp3_size <- plyr::rename(df_exp3_size,
                             c("Group.1"="subj",
                               "Group.2"="setsize","x"="rt"))

# df_exp3 %>%
#   group_by(exp,setsize,cond) %>%
#   identify_outliers(rt)


# ANOVA
# #####

# ACC: 2-way anova
res.aov <- anova_test(
  data=df_exp3,dv=acc,wid=subj,
  within=c(setsize,cond),
  type=3,effect.size="pes",detailed=TRUE)
get_anova_table(res.aov)

# RT: 2-way anova
res.aov <- anova_test(
  data=df_exp3,dv=rt,wid=subj,
  within=c(setsize,cond),
  type=3,effect.size="pes",detailed=TRUE)
get_anova_table(res.aov)
#
simpeff <- df_exp3 %>%
  group_by(setsize) %>%
  anova_test(dv=rt,wid=subj,within=cond,
             type=3,effect.size="pes",detailed=TRUE) %>%
  get_anova_table()
simpeff
pwc <- df_exp3 %>%
  group_by(setsize) %>%
  pairwise_t_test(rt~cond,paired=TRUE,
                  p.adjust.method="bonferroni")
pwc
#
simpeff <- df_exp3 %>%
  group_by(cond) %>%
  anova_test(dv=rt,wid=subj,within=setsize,
             type=3,effect.size="pes",detailed=TRUE) %>%
  get_anova_table()
simpeff
pwc <- df_exp3 %>%
  group_by(cond) %>%
  pairwise_t_test(rt~setsize,paired=TRUE,
                  p.adjust.method="bonferroni")
pwc


# predicting
# ##########
subj_list <- unique(df_exp3$subj)
w_obs_rt <- subset(df_exp3,(setsize==8)&(cond=="within"))$rt
b_obs_rt <- subset(df_exp3,(setsize==8)&(cond=="between"))$rt
w_8_lm <- subset(df_exp3,(setsize==8)&(cond=="within"))$lm
w_8_log <- subset(df_exp3,(setsize==8)&(cond=="within"))$log
b_8_lm <- subset(df_exp3,(setsize==8)&(cond=="between"))$lm
b_8_log <- subset(df_exp3,(setsize==8)&(cond=="between"))$log

# t_val <- t.test(w_8_lm,w_obs_rt,paired=TRUE,alternative="greater")
t_val <- t.test(w_8_lm,w_obs_rt,paired=TRUE)
# t_val
dat <- data.frame(mss8=c(w_8_lm,w_obs_rt),
                  pred=rep(c("lm","rt"),each=length(subj_list)))
d <- dat %>% cohens_d(mss8~pred,paired=TRUE)
# d
print('within: lm')
sprintf('t = %0.3f, df = %0.3f, p = %0.3f, d = %0.3f',
        t_val$statistic,t_val$parameter,t_val$p.value,d[1,4])

# ----------------------------
print('within: log2')
t_val <- t.test(w_8_log,w_obs_rt,paired=TRUE)
# t_val
dat <- data.frame(mss8=c(w_8_log,w_obs_rt),
                  pred=rep(c("log","rt"),each=length(subj_list)))
d <- dat %>% cohens_d(mss8~pred,paired=TRUE)
# d
sprintf('t = %0.3f, df = %0.3f, p = %0.3f, d = %0.3f',
        t_val$statistic,t_val$parameter,t_val$p.value,d[1,4])
# ----------------------------
print('bertween: lm')
# t_val <- t.test(w_8_lm,w_obs_rt,paired=TRUE,alternative="greater")
t_val <- t.test(b_8_lm,b_obs_rt,paired=TRUE)
# t_val
dat <- data.frame(mss8=c(b_8_lm,b_obs_rt),
                  pred=rep(c("lm","rt"),each=length(subj_list)))
d <- dat %>% cohens_d(mss8~pred,paired=TRUE)
# d
sprintf('t = %0.3f, df = %0.3f, p = %0.3f, d = %0.3f',
        t_val$statistic,t_val$parameter,t_val$p.value,d[1,4])

# ----------------------------
print('between: log2')
sprintf('t = %0.3f, p = %0.3f, d = %0.3f',
        t_val$statistic,t_val$p.value,d[1,4])

t_val <- t.test(b_8_log,b_obs_rt,paired=TRUE)
# t_val
dat <- data.frame(mss8=c(b_8_log,b_obs_rt),
                  pred=rep(c("log","rt"),each=length(subj_list)))
d <- dat %>% cohens_d(mss8~pred,paired=TRUE)
# d
print('between: log2')
sprintf('t = %0.3f, df = %0.3f, p = %0.3f, d = %0.3f',
        t_val$statistic,t_val$parameter,t_val$p.value,d[1,4])



# Modelling
# #########

coeff_lm_list <- c()
coeff_log_list <- c()
r2_lm_list <- c()
r2_log_list <- c()
coeff_lm_cond_list <- c()
coeff_log_cond_list <- c()

for (n in subj_list){
  subj_exp3_size <- subset(df_exp3_size,(subj==n))
  lm_model_cond <- lm(formula=rt~setsize,subj_exp3_size)
  log_model_cond <- lm(formula=rt~log(setsize,2),subj_exp3_size)
  res_lm_cond <- summary(lm_model_cond)
  res_log_cond <- summary(log_model_cond)
  coeff_lm_cond <- coef(res_lm_cond)
  coeff_log_cond <- coef(res_log_cond)
  coeff_lm_cond_list <- append(coeff_lm_cond_list,coeff_lm_cond[2,1])
  coeff_log_cond_list <- append(coeff_log_cond_list,coeff_log_cond[2,1])
  for (k in cond_list){
    df_subj <- subset(df_exp3,(subj==n)&(cond==k))
    lm_model <- lm(formula=rt~setsize,df_subj)
    log_model <- lm(formula=rt~log(setsize,2),df_subj)
    res_lm <- summary(lm_model)
    res_log <- summary(log_model)
    coeff_lm = coef(res_lm)
    coeff_log = coef(res_log)
    coeff_lm_list <- append(coeff_lm_list,coeff_lm[2,1])
    coeff_log_list <- append(coeff_log_list,coeff_log[2,1])
    
    r2_lm_list <- append(r2_lm_list,res_lm$r.squared)
    r2_log_list <- append(r2_log_list,res_log$r.squared)
  }
}

df_coeff_mean <- data.frame(
  subj=subj_list,lm=coeff_lm_cond_list,
  log=coeff_log_cond_list)

# compare the coeffects between w & b
conds <- rep(cond_list,times=length(subj_list))
subjs <- rep(subj_list,each=2)
df_coeff <- data.frame(
  subj=subjs,cond=conds,lm=coeff_lm_list,
  log=coeff_log_list,r2_lm=r2_lm_list,
  r2_log=r2_log_list)
df_coeff$r_lm <- with(df_coeff,sqrt(r2_lm))
df_coeff$r_log <- with(df_coeff,sqrt(r2_log))
coeff_w <- subset(df_coeff,cond=="within")
coeff_b <- subset(df_coeff,cond=="between")
t_val <- t.test(coeff_w$log,coeff_b$log,paired=TRUE)
t_val
d <- df_coeff %>% cohens_d(log~cond,paired=TRUE)
d
print("Compare the slope coefficients between w&b")
sprintf('t = %0.3f, p = %0.3f, d = %0.3f',
        t_val$statistic,t_val$p.value,d[1,4])

# compare 2 models

# r1 <- cor(datAnd$CZ, datAnd$CRAT)
# r2 <- pcor(c(1,2,3),cov(pc))
# zf1=1/2*log((1+r1)/(1-r1))
# zf2=1/2*log((1+r2)/(1-r2))
# zr <- (zf1-zf2)/sqrt(1/(44-3)+1/(44-3))
# pnorm(zr)

df_cond <- aggregate(df_exp3$rt,
                     by=list(df_exp3$setsize,
                             df_exp3$cond),mean)
df_cond <- plyr::rename(df_cond,
                        c("Group.1"="setsize",
                          "Group.2"="cond","x"="rt"))
mean_lm <- summary(lm(formula=rt~setsize+cond+setsize:cond,df_cond))
mean_log <- summary(lm(formula=rt~log(setsize,2)+cond+log(setsize,2):cond,df_cond))

subjN <- length(subj_list)
r_lm <- sqrt(mean_lm$r.squared)
r_log <- sqrt(mean_log$r.squared)
zf_lm <- 1/2*log((1+r_lm)/(1-r_lm))
zf_log <- 1/2*log((1+r_log)/(1-r_log))
s <- sqrt(1/(subjN-3)+1/(subjN-3))
z <- (zf_lm-zf_log)/s
pnorm(z)



# ######## #
# DEOCDING #
# ######## #

setwd("U:\\Documents\\DCC\\exp3\\Results")

# 1. decoding
recog_label_list <- c("w1","w2","w4","w8","b1","b2","b4","b8")

df_deco_raw <- read.csv("deco_data_subj.csv")
df_deco <- subset(df_deco_raw,(type %in% recog_label_list)&(pred=="o2r"))

for (tag in c("p1","n1","p2")){
  if (tag=="p1"){
    t0 <- 0.1
    t1 <- 0.16}
  else if (tag=="n1"){
    t0 <- 0.16
    t1 <- 0.2}
  else{
    t0 <- 0.2
    t1 <- 0.3
  }
  print("--------------")
  print(tag)
  deco_all <- subset(df_deco,
                     (df_deco$time>=t0)&
                       (df_deco$time<t1)&
                       (df_deco$type %in% recog_label_list))
  deco <- aggregate(x=deco_all$acc,by=list(deco_all$type,deco_all$subj),mean)
  deco <- plyr::rename(deco,c("Group.1"="type","Group.2"="subj","x"="acc"))
  deco$cond <- str_split_fixed(deco$type, "", 2)[,1]
  deco$setsize <- str_split_fixed(deco$type, "", 2)[,2]
  
  deco_t <- aggregate(x=deco$acc,by=list(deco$cond,deco$subj),mean)
  deco_t <- plyr::rename(deco_t,c("Group.1"="cond","Group.2"="subj","x"="acc"))
  deco_t_w <- subset(deco_t,cond=="w")
  deco_t_b <- subset(deco_t,cond=="b")
  t_val <- t.test(deco_t_w$acc,deco_t_b$acc,paired=TRUE,alternative='greater')
  print(get_anova_table(t_val))
  d <- deco_t %>% cohens_d(acc~cond)
  d
  print(sprintf('t = %0.3f, p = %0.3f, d = %0.3f',
                t_val$statistic,t_val$p.value,d[1,4]))
  print("--------------")
  
  # anova
  res.aov <- anova_test(
    data=deco,dv=acc,wid=subj,within=c(setsize,cond),
    type=3,effect.size="pes",detailed=TRUE)
  print(get_anova_table(res.aov))
  
  cond.effect <- deco %>%
    group_by(cond) %>%
    anova_test(dv=acc,wid=subj,within=setsize) %>%
    get_anova_table()
  fix(cond.effect)
  
  setsize.effect <- deco %>%
    group_by(setsize) %>%
    anova_test(dv=acc,wid=subj,within=cond) %>%
    get_anova_table()
  fix(setsize.effect)
  
  # # ttest
  # deco_cate <- aggregate(x=deco$acc,by=list(deco$cond,deco$subj),mean)
  # deco_cate <- plyr::rename(deco_cate,
  #                           c("Group.1"="cond","Group.2"="subj","x"="acc"))
  # deco_w <- subset(deco_cate,deco_cate$cond=="w")
  # deco_b <- subset(deco_cate,deco_cate$cond=="b")
  # t.test(deco_w$acc,deco_b$acc,paired=TRUE)
}



# #####
# EEG #
# #####



df_erp <- read.csv("erp_eeg.csv")

recog_labels <- c("w/1","w/2","w/4","w/8","b/1","b/2","b/4","b/8")
picks <- c("P5","P6","P7","P8","PO3","PO4","PO7","PO8","PO9","PO10","O1","O2")

erp <- subset(df_erp,(df_erp$type %in% recog_labels))
erp <- subset(erp,select=c(c("subj","time","type"),picks))
erp$cond <- str_split_fixed(erp$type, "/", 2)[,1]
erp$setsize <- str_split_fixed(erp$type, "/", 2)[,2]
erp_simi <- subset(erp,select=picks)
erp$simi <- apply(erp_simi,1,mean)
cond_list <- c("w","b")

for (tag in c("p1","n1","p2")){
  if (tag=="p1"){
    t0 <- 0.1
    t1 <- 0.16}
  else if (tag=="n1"){
    t0 <- 0.16
    t1 <- 0.2}
  else{
    t0 <- 0.2
    t1 <- 0.3
  }
  
  erp_stat <- subset(erp,(erp$time>=t0)&(erp$time<t1))
  erp_stat <- aggregate(x=erp_stat$simi,
                        by=list(erp_stat$cond,erp_stat$setsize,erp_stat$subj),
                        mean)
  erp_stat <- plyr::rename(erp_stat,
                           c("Group.1"="cond","Group.2"="setsize",
                             "Group.3"="subj","x"="simi"))
  df_erp_size <- aggregate(erp_stat$simi,by=list(erp_stat$subj,
                                                 erp_stat$setsize),mean)
  df_erp_size <- plyr::rename(df_erp_size,
                              c("Group.1"="subj",
                                "Group.2"="setsize","x"="simi"))
  
  # anova
  res.aov <- anova_test(
    data=erp_stat,dv=simi,wid=subj,within=c(setsize,cond),
    type=3,effect.size="pes",detailed=TRUE)
  print("--------------")
  print(tag)
  print(get_anova_table(res.aov))
  
  cond.effect <- erp_stat %>%
    group_by(cond) %>%
    anova_test(dv=simi,wid=subj,within=setsize) %>%
    get_anova_table()
  fix(cond.effect)
  
  setsize.effect <- erp_stat %>%
    group_by(setsize) %>%
    anova_test(dv=simi,wid=subj,within=cond) %>%
    get_anova_table()
  fix(setsize.effect)
  
  
  
  # predicting
  # ##########
  erp_stat$setsize <- as.numeric(as.character(erp_stat$setsize))
  df_erp_size$setsize <- as.numeric(as.character(df_erp_size$setsize))
  w_8_lm <- c()
  b_8_lm <- c()
  w_8_log <- c()
  b_8_log <- c()
  
  for (n in subj_list){
    for (k in cond_list){
      trainData <- subset(erp_stat,(subj==n)&(setsize!=8)&(cond==k))
      testData <- data.frame(setsize=c(8))
      lm_pred <- lm(formula=simi~setsize,trainData)
      log_pred <- lm(formula=simi~log(setsize,2),trainData)
      lm_8 <- predict(lm_pred,newdata=testData)
      log_8 <- predict(log_pred,newdata=testData)
      if(k=="w"){
        w_8_lm <- append(w_8_lm,lm_8[[1]])
        w_8_log <- append(w_8_log,log_8[[1]])
      }else{
        b_8_lm <- append(b_8_lm,lm_8[[1]])
        b_8_log <- append(b_8_log,log_8[[1]])
      }
    }
  }
  w_obs <- subset(erp_stat,(setsize==8)&(cond=="w"))
  b_obs <- subset(erp_stat,(setsize==8)&(cond=="b"))
  w_obs_rt <- w_obs$simi
  b_obs_rt <- b_obs$simi
  
  t_val <- t.test(w_8_lm,w_obs_rt,alternative="greater")
  t_val
  dat <- data.frame(mss8=c(w_8_lm,w_obs_rt),
                    pred=rep(c("lm","simi"),each=length(subj_list)))
  d <- dat %>% cohens_d(mss8~pred)
  d
  
  print('predicting')
  print('#########')
  print('within: lm')
  print(sprintf('t = %0.3f, p = %0.3f, d = %0.3f',
                t_val$statistic,t_val$p.value,d[1,4]))
  
  # ----------------------------
  print('within: log2')
  t_val <- t.test(w_8_log,w_obs_rt)
  t_val
  dat <- data.frame(mss8=c(w_8_log,w_obs_rt),
                    pred=rep(c("log","simi"),each=length(subj_list)))
  d <- dat %>% cohens_d(mss8~pred)
  d
  print(sprintf('t = %0.3f, p = %0.3f, d = %0.3f',
                t_val$statistic,t_val$p.value,d[1,4]))
  
  t_val <- t.test(b_8_lm,b_obs_rt,alternative="greater")
  t_val
  dat <- data.frame(mss8=c(b_8_lm,b_obs_rt),
                    pred=rep(c("lm","simi"),each=length(subj_list)))
  d <- dat %>% cohens_d(mss8~pred)
  d
  # ----------------------------
  print('bertween: lm')
  print(sprintf('t = %0.3f, p = %0.3f, d = %0.3f',
                t_val$statistic,t_val$p.value,d[1,4]))
  
  t_val <- t.test(b_8_log,b_obs_rt)
  t_val
  dat <- data.frame(mss8=c(b_8_log,b_obs_rt),
                    pred=rep(c("log","simi"),each=length(subj_list)))
  d <- dat %>% cohens_d(mss8~pred)
  d
  # ----------------------------
  print('between: log2')
  print(sprintf('t = %0.3f, p = %0.3f, d = %0.3f',
                t_val$statistic,t_val$p.value,d[1,4]))
  print('#########')
  
  
  
  # Modelling
  # #########
  
  coeff_lm_list <- c()
  coeff_log_list <- c()
  r2_lm_list <- c()
  r2_log_list <- c()
  
  coeff_lm_cond_list <- c()
  coeff_log_cond_list <- c()
  
  for (n in subj_list){
    subj_erp_size <- subset(df_erp_size,(subj==n))
    lm_model_cond <- lm(formula=simi~setsize,subj_erp_size)
    log_model_cond <- lm(formula=simi~log(setsize,2),subj_erp_size)
    res_lm_cond <- summary(lm_model_cond)
    res_log_cond <- summary(log_model_cond)
    coeff_lm_cond <- coef(res_lm_cond)
    coeff_log_cond <- coef(res_log_cond)
    coeff_lm_cond_list <- append(coeff_lm_cond_list,coeff_lm_cond[2,1])
    coeff_log_cond_list <- append(coeff_log_cond_list,coeff_log_cond[2,1])
    for (k in cond_list){
      df_subj <- subset(erp_stat,(subj==n)&(cond==k))
      lm_model <- lm(formula=simi~setsize,df_subj)
      log_model <- lm(formula=simi~log(setsize,2),df_subj)
      res_lm <- summary(lm_model)
      res_log <- summary(log_model)
      coeff_lm <- coef(res_lm)
      coeff_log <- coef(res_log)
      coeff_lm_list <- append(coeff_lm_list,coeff_lm[2,1])
      coeff_log_list <- append(coeff_log_list,coeff_log[2,1])
      
      r2_lm_list <- append(r2_lm_list,res_lm$r.squared)
      r2_log_list <- append(r2_log_list,res_log$r.squared)
      
    }
  }
  
  coeff_erp_mean <- data.frame(
    subj=subj_list,lm=coeff_lm_cond_list,
    log=coeff_log_cond_list)
  
  # compare the coeffects between w & b
  conds <- rep(cond_list,times=length(subj_list))
  subjs <- rep(subj_list,each=2)
  coeff_erp <- data.frame(
    subj=subjs,cond=conds,lm=coeff_lm_list,
    log=coeff_log_list,r2_lm=r2_lm_list,
    r2_log=r2_log_list)
  coeff_erp$r_lm <- with(coeff_erp,sqrt(r2_lm))
  coeff_erp$r_log <- with(coeff_erp,sqrt(r2_log))
  coeff_w <- subset(coeff_erp,cond=="w")
  coeff_b <- subset(coeff_erp,cond=="b")
  t_val <- t.test(coeff_w$log,coeff_b$log,paired=TRUE)
  t_val
  d <- coeff_erp %>% cohens_d(log~cond,paired=TRUE)
  d
  print("Compare the slope coefficients between w&b")
  print(sprintf('t = %0.3f, p = %0.3f, d = %0.3f',
                t_val$statistic,t_val$p.value,d[1,4]))
  
  # correlation between behav & erp
  for (k in cond_list){
    if (k=="w"){kk = "within"}
    else{kk = "between"}
    erp_cond <- subset(coeff_erp,cond==k)
    behav_cond <- subset(df_coeff,cond==kk)
    cp_r_lm <- cor.test(erp_cond$lm,behav_cond$lm,method=c("pearson"))
    cp_r_log <- cor.test(erp_cond$log,behav_cond$log,method=c("pearson"))
    print(k)
    print("linear")
    print(cp_r_lm)
    print("log 2")
    print(cp_r_log)
    # ----------------------------
  }
  cp_r_lm <- cor.test(coeff_erp_mean$lm,df_coeff_mean$lm,method=c("pearson"))
  cp_r_log <- cor.test(coeff_erp_mean$log,df_coeff_mean$log,method=c("pearson"))
  print("linear")
  print(cp_r_lm)
  print("log 2")
  print(cp_r_log)
  # ----------------------------
  
  
  # compare 2 models
  erp_cond <- aggregate(erp_stat$simi,
                        by=list(erp_stat$setsize,
                                erp_stat$cond),mean)
  erp_cond <- plyr::rename(erp_cond,
                           c("Group.1"="setsize",
                             "Group.2"="cond","x"="simi"))
  mean_lm <- summary(lm(formula=simi~setsize+cond+setsize:cond,erp_cond))
  mean_log <- summary(lm(formula=simi~log(setsize,2)+cond+log(setsize,2):cond,erp_cond))
  
  subjN <- length(subj_list)
  r_lm <- sqrt(mean_lm$r.squared)
  r_log <- sqrt(mean_log$r.squared)
  zf_lm <- 1/2*log((1+r_lm)/(1-r_lm))
  zf_log <- 1/2*log((1+r_log)/(1-r_log))
  s <- sqrt(1/(subjN-3)+1/(subjN-3))
  z <- (zf_lm-zf_log)/s
  print(pnorm(z))
} 









