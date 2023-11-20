library(tidyverse)
library(ggpubr)
library(ggplot2)
library(rstatix)

setwd("U:\\Documents\\DCC\\exp4\\Results")


# ################ 1. BEHAV ################# #

df_behav <- read.csv("exp4_mean.csv")
df_behav$setsize <- factor(df_behav$setsize,levels=c(1,2,4,8))

df_targ <- subset(df_behav,(cond=="wt")|(cond=="bt"))
df_wb <- subset(df_behav,(cond=="wb"))
df_distr <- subset(df_behav,(cond=="ww")|(cond=="bb"))
df_ww <- subset(df_behav,(cond=="ww"))
df_bb <- subset(df_behav,(cond=="bb"))
write.csv(df_targ,file="plt_targ.csv",row.names=F)
write.csv(df_wb,file="plt_wb.csv",row.names=F)

# ACC: 2-way anova
res.aov <- anova_test(
  data=df_targ,dv=Correct,wid=Subject,
  within=c(setsize,cond),
  type=3,effect.size="pes",detailed=TRUE)
get_anova_table(res.aov)

# plot
windows()
ggbarplot(df_targ,x="setsize",y="Correct",add="mean_se",
          color="gray15",fill="cond",palette=NULL,
          position=position_dodge(0.75),ylim=c(0.5,1),
          ylab="ACC",xlab="MSS",title="Target-trials",
          legend.title="Category")+
  scale_fill_manual(name="Category",values=c("#4393C3","#D6604D"),
                    breaks=c("bt","wt"),
                    labels=c("between-target","within-target"))

# RT: 2-way anova
res.aov <- anova_test(
  data=df_targ,dv=RT,wid=Subject,
  within=c(setsize,cond),
  type=3,effect.size="pes",detailed=TRUE)
get_anova_table(res.aov)
#
simpeff <- df_targ %>%
  group_by(setsize) %>%
  anova_test(dv=RT,wid=Subject,within=cond,
             type=3,effect.size="pes",detailed=TRUE) %>%
  get_anova_table()
simpeff
pwc <- df_targ %>%
  group_by(setsize) %>%
  pairwise_t_test(RT~cond,paired=TRUE,
                  p.adjust.method="bonferroni")
pwc
#
simpeff <- df_targ %>%
  group_by(cond) %>%
  anova_test(dv=RT,wid=Subject,within=setsize,
             type=3,effect.size="pes",detailed=TRUE) %>%
  get_anova_table()
simpeff
pwc <- df_targ %>%
  group_by(cond) %>%
  pairwise_t_test(RT~setsize,paired=TRUE,
                  p.adjust.method="bonferroni")
pwc

#plot
windows()
ggbarplot(df_targ,x="setsize",y="RT",add="mean_se",
          color="gray15",fill="cond",palette=NULL,
          position=position_dodge(0.75),ylim=c(0.2,0.7),
          ylab="RT",xlab="MSS",title="Target-trials",
          legend.title="Category")+
  scale_fill_manual(name="Category",values=c("#4393C3","#D6604D"),
                    breaks=c("bt","wt"),
                    labels=c("between-target","within-target"))

# wb: 1-way anova
res.aov <- anova_test(
  data=df_wb,dv=RT,wid=Subject,
  within=setsize,
  type=3,effect.size="pes",detailed=TRUE)
get_anova_table(res.aov)
pwc <- df_wb %>%
  pairwise_t_test(RT~setsize,paired=TRUE,
                  p.adjust.method="bonferroni")
pwc
# ww: 1-way anova
res.aov <- anova_test(
  data=df_ww,dv=RT,wid=Subject,
  within=setsize,
  type=3,effect.size="pes",detailed=TRUE)
get_anova_table(res.aov)
pwc <- df_ww %>%
  pairwise_t_test(RT~setsize,paired=TRUE,
                  p.adjust.method="bonferroni")
pwc
# bb: 1-way anova
res.aov <- anova_test(
  data=df_bb,dv=RT,wid=Subject,
  within=setsize,
  type=3,effect.size="pes",detailed=TRUE)
get_anova_table(res.aov)
pwc <- df_bb %>%
  pairwise_t_test(RT~setsize,paired=TRUE,
                  p.adjust.method="bonferroni")
pwc

# ww & bb
res.aov <- anova_test(
  data=df_distr,dv=RT,wid=Subject,
  within=c(setsize,cond),
  type=3,effect.size="pes",detailed=TRUE)
get_anova_table(res.aov)
simpeff <- df_distr %>%
  group_by(cond) %>%
  anova_test(dv=RT,wid=Subject,within=setsize,
             type=3,effect.size="pes",detailed=TRUE) %>%
  get_anova_table()
simpeff
simpeff <- df_distr %>%
  group_by(setsize) %>%
  anova_test(dv=RT,wid=Subject,within=cond,
             type=3,effect.size="pes",detailed=TRUE) %>%
  get_anova_table()
simpeff

# wb & ww & bb
df_distrAll <- rbind(df_wb,df_distr)
res.aov <- anova_test(
  data=df_distrAll,dv=RT,wid=Subject,
  within=c(setsize,cond),
  type=3,effect.size="pes",detailed=TRUE)
get_anova_table(res.aov)

# plot
windows()
ggbarplot(df_wb,x="setsize",y="RT",add="mean_se",
          color="gray80",fill="cond",palette=NULL,
          position=position_dodge(0.75),ylim=c(0.2,0.7),
          ylab="RT",xlab="MSS",title="Distractor-trials",
          legend.title="Category")+
  scale_fill_manual(name="Category",values=c("gray80"),
                    breaks=c("wt"),
                    labels=c("within-between"))



# ################ 2. ERP ################# #

df_erp <- read.csv("exp4_n2pc.csv")
df_erp$setsize <- factor(df_erp$setsize,levels=c(1,2,4,8))
cond_targ <- c("wt","bt")

t0 <- 0.2
t1 <- 0.3
df_n2pc <- subset(df_erp,(df_erp$time>=t0)&(df_erp$time<t1))
df_n2pc <- aggregate(cbind(n2pc,contr,ipsi)~subj+cond+setsize,data=df_n2pc,mean)
df_lr <- data.frame(amp=c(df_n2pc$contr,df_n2pc$ipsi),
                    pos=rep(c("contr","ipsi"),each=nrow(df_n2pc)),
                    subj=rep(df_n2pc$subj,times=2),
                    cond=rep(df_n2pc$cond,times=2),
                    setsize=rep(df_n2pc$setsize,times=2))

erp_targ <- subset(df_n2pc,(cond=="wt")|(cond=="bt"))
erp_distr <- subset(df_n2pc,(cond=="wb"))

# test n2pc
df_lr_targ <- subset(df_lr,(cond=="wt")|(cond=="bt"))
df_lr_distr <- subset(df_lr,(cond=="wb"))
# wt vs bt: 3-way anova
res.aov <- anova_test(
  data=df_lr_targ,dv=amp,wid=subj,
  within=c(setsize,cond,pos),
  type=3,effect.size="pes",detailed=TRUE)
get_anova_table(res.aov)
#
simpeff <- df_lr_targ %>%
  group_by(setsize,cond) %>%
  anova_test(dv=amp,wid=subj,within=pos,
             type=3,effect.size="pes",detailed=TRUE) %>%
  get_anova_table()
simpeff
#
pwc <- df_lr_targ %>%
  group_by(setsize,cond) %>%
  pairwise_t_test(amp~pos,paired=TRUE,
                  p.adjust.method="bonferroni")
pwc

# wb: 2-way anova
# 2way to compare ipsi and contr
wb_df <- data.frame(subj=rep(erp_distr$subj,time=2),
                    setsize=rep(erp_distr$setsize,time=2),
                    cond=rep(c('contr','ipsi'),each=length(erp_distr$cond)),
                    amp=c(erp_distr$contr,erp_distr$ipsi))
res.aov <- anova_test(
  data=wb_df,dv=amp,wid=subj,
  within=c(setsize,cond),
  type=3,effect.size="pes",detailed=TRUE)
get_anova_table(res.aov)
pwc <- wb_df %>%
  pairwise_t_test(amp~cond,paired=TRUE,
                  p.adjust.method="bonferroni")
pwc

windows()
ggbarplot(wb_df,x="setsize",y="amp",add="mean_se",
          color="gray15",fill="cond",palette=NULL,
          position=position_dodge(0.75),
          ylab="Amplitude (μV)",xlab="MSS",title="Distractor-trials",
          legend.title="Electrode")+
  scale_fill_manual(name="Electrode",values=c("#D6604D","#4393C3"),
                    breaks=c("contr","ipsi"),
                    labels=c("contr","ipsi"))

# # ttest
# n2pc_wt <- subset(erp_targ,cond=="wt")
# n2pc_bt <- subset(erp_targ,cond=="bt")
# t_val <- t.test(n2pc_wt$n2pc,n2pc_bt$n2pc)
# t_val
# d <- erp_targ %>% cohens_d(n2pc~cond)
# d
# sprintf('t = %0.3f,p = %0.3f,d = %0.3f',
#         t_val$statistic,t_val$p.value,d[1,4])

# wt vs bt: 2-way anova
res.aov <- anova_test(
  data=erp_targ,dv=n2pc,wid=subj,
  within=c(setsize,cond),
  type=3,effect.size="pes",detailed=TRUE)
get_anova_table(res.aov)
#
simpeff <- erp_targ %>%
  group_by(setsize) %>%
  anova_test(dv=n2pc,wid=subj,within=cond,
             type=3,effect.size="pes",detailed=TRUE) %>%
  get_anova_table()
simpeff
pwc <- erp_targ %>%
  group_by(setsize) %>%
  pairwise_t_test(n2pc~cond,paired=TRUE,
                  p.adjust.method="bonferroni")
pwc
#
simpeff <- erp_targ %>%
  group_by(cond) %>%
  anova_test(dv=n2pc,wid=subj,within=setsize,
             type=3,effect.size="pes",detailed=TRUE) %>%
  get_anova_table()
simpeff
pwc <- erp_targ %>%
  group_by(cond) %>%
  pairwise_t_test(n2pc~setsize,paired=TRUE,
                  p.adjust.method="bonferroni")
pwc

windows()
ggbarplot(erp_targ,x="setsize",y="n2pc",add="mean_se",
          color="gray15",fill="cond",palette=NULL,
          position=position_dodge(0.75),
          ylab="N2pc (μV)",xlab="MSS",title="Target-trials",
          legend.title="Category")+
  scale_fill_manual(name="Category",values=c("#4393C3","#D6604D"),
                    breaks=c("bt","wt"),
                    labels=c("between-target","within-target"))

# ################ 3. Decoding ################# #

df_deco <- read.csv("deco_data_all.csv")
df_deco$cond <- paste(str_split_fixed(df_deco$type, "", 3)[,1],
                      str_split_fixed(df_deco$type, "", 3)[,2],sep="")
df_deco$setsize <- str_split_fixed(df_deco$type, "", 3)[,3]
df_deco$setsize <- factor(df_deco$setsize,levels=c(1,2,4,8))

deco_targ <- subset(df_deco,(cond=="wt")|(cond=="bt"))
deco_distr <- subset(df_deco,(cond=="wb"))

t0 <- 0.2
t1 <- 0.3
deco_n2pc_raw <- subset(df_deco,(df_deco$time>=t0)&(df_deco$time<t1))
deco_n2pc <- aggregate(x=deco_n2pc_raw$acc,
                       by=list(deco_n2pc_raw$cond,
                               deco_n2pc_raw$setsize,
                               deco_n2pc_raw$subj),mean)
deco_n2pc <- plyr::rename(deco_n2pc,
                          c("Group.1"="cond","Group.2"="setsize",
                            "Group.3"="subj","x"="acc"))
deco_n2pc_targ <- subset(deco_n2pc,(cond=="wt")|(cond=="bt"))
deco_n2pc_distr <- subset(deco_n2pc,(cond=="wb"))

# wt vs bt vs wb: 2-way anova
res.aov <- anova_test(
  data=deco_n2pc,dv=acc,wid=subj,
  within=c(setsize,cond),
  type=3,effect.size="pes",detailed=TRUE)
get_anova_table(res.aov)
#
simpeff <- deco_n2pc %>%
  group_by(setsize) %>%
  anova_test(dv=acc,wid=subj,within=cond,
             type=3,effect.size="pes",detailed=TRUE) %>%
  get_anova_table()
simpeff
pwc <- deco_n2pc %>%
  group_by(setsize) %>%
  pairwise_t_test(acc~cond,paired=TRUE,
                  p.adjust.method="bonferroni")
pwc
#
simpeff <- deco_n2pc %>%
  group_by(cond) %>%
  anova_test(dv=acc,wid=subj,within=setsize,
             type=3,effect.size="pes",detailed=TRUE) %>%
  get_anova_table()
simpeff
pwc <- deco_n2pc %>%
  group_by(cond) %>%
  pairwise_t_test(acc~setsize,paired=TRUE,
                  p.adjust.method="bonferroni")
pwc


# wt vs bt: 2-way anova
res.aov <- anova_test(
  data=deco_n2pc_targ,dv=acc,wid=subj,
  within=c(setsize,cond),
  type=3,effect.size="pes",detailed=TRUE)
get_anova_table(res.aov)
#
simpeff <- deco_n2pc_targ %>%
  group_by(setsize) %>%
  anova_test(dv=acc,wid=subj,within=cond,
             type=3,effect.size="pes",detailed=TRUE) %>%
  get_anova_table()
simpeff
pwc <- deco_n2pc_targ %>%
  group_by(setsize) %>%
  pairwise_t_test(acc~cond,paired=TRUE,
                  p.adjust.method="bonferroni")
pwc
#
simpeff <- deco_n2pc_targ %>%
  group_by(cond) %>%
  anova_test(dv=acc,wid=subj,within=setsize,
             type=3,effect.size="pes",detailed=TRUE) %>%
  get_anova_table()
simpeff
pwc <- deco_n2pc_targ %>%
  group_by(cond) %>%
  pairwise_t_test(acc~setsize,paired=TRUE,
                  p.adjust.method="bonferroni")
pwc

windows()
ggbarplot(deco_n2pc_targ,x="setsize",y="acc",add="mean_se",
          color="gray15",fill="cond",palette=NULL,ylim=c(0.5,0.7),
          position=position_dodge(0.75),
          ylab="ACC",xlab="MSS",title="decoding wt vs bt",
          legend.title="Category")+
  scale_fill_manual(name="Category",values=c("#4393C3","#D6604D"),
                    breaks=c("bt","wt"),
                    labels=c("between-target","within-target"))























