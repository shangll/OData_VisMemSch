#!/usr/bin/env python
#-*-coding:utf-8 -*-

# EXP. 4 (Behavioural):
# N2pc
# 2023.Mar.9
# linlin.shang@donders.ru.nl

from exp4_config import subjAllN,subjList,filePath,outliers,\
    resPath,behavFigPath,sizeList,cateList,condList,cond_list,\
    p_crit,crit_rt,crit_sd,crit_acc,trialN_all,\
    set_filepath,tag_savefile,save_fig

import os
from math import log
import random
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.pyplot import MultipleLocator
import seaborn as sns
import statsmodels.api as sm
from sklearn.linear_model import LinearRegression
model = LinearRegression()
import pingouin as pg
import statannot



# --- --- --- Set Global Parameters --- --- --- #

def AIC(y_test, y_pred, k, n):
    resid = y_test - y_pred
    SSR = sum(resid ** 2)
    AICValue = 2*k+n*log(float(SSR)/n)
    return AICValue



# --- --- --- 1. Read Files --- --- ---

dataPath = set_filepath(filePath,'Data','DataBehav')
df_test = pd.read_csv(os.path.join(dataPath,'test.csv'),sep=',')
df_sch_raw = pd.read_csv(os.path.join(dataPath,'sch.csv'),sep=',')
figPath = behavFigPath
df_sch_raw.rename(columns={'cond':'cond_raw'},inplace=True)
df_sch_raw['cond'] = df_sch_raw['cond_raw']

df_sch_raw.loc[(df_sch_raw['cond_raw']=='tb')|
               (df_sch_raw['cond_raw']=='bt'),'cond'] = 'bt'
df_sch_raw.loc[(df_sch_raw['cond_raw']=='tw')|
               (df_sch_raw['cond_raw']=='wt'),'cond'] = 'wt'
df_sch_raw.loc[(df_sch_raw['cond_raw']=='wb')|
               (df_sch_raw['cond_raw']=='bw'),'cond'] = 'wb'
# pd.set_option('display.max_rows',None)
# pd.set_option('display.max_columns',None)

subj_info = pd.DataFrame()
age_list,gender_list,hand_list = [],[],[]
for n in subjList+outliers:
    df_subj = df_sch_raw[
        df_sch_raw['Subject']==n].copy(deep=True).reset_index(drop=True)
    age_list.append(df_subj.loc[0,'age'])
    gender_list.append(df_subj.loc[0,'sex'])

print('*** *** *** *** *** ***')
print('averge age: %0.2f, sd: %0.2f, range: %d ~ %d'
      %(np.mean(age_list),np.std(age_list),
        min(age_list),max(age_list)))
male_list = list(filter(lambda x:x=='M',gender_list))
print('male: %d, female: %d'%(
    len(male_list),len(gender_list)-len(male_list)))
print('*** *** *** *** *** ***')

# --- --- --- 2. Exclusion --- --- ---
df_sch_raw = df_sch_raw[df_sch_raw['Subject'].isin(subjList)]
incorr_points = len(df_sch_raw[df_sch_raw['Correct']==0])
df_sch_clean = df_sch_raw.copy(deep=True)
df_sch_clean = df_sch_clean[df_sch_clean['Subject'].isin(subjList)].reset_index(drop=True)

# 2.1 <0.2 sec
df_sch_clean.loc[df_sch_clean['Correct']==1,'Correct'] = \
    np.where((df_sch_clean.loc[(df_sch_clean['Correct']==1),'RT']<crit_rt),
             0,1)
# 2.2 ±3 sd
outRTs = df_sch_clean[df_sch_clean['Correct']==1].copy(deep=True).groupby(
    ['Subject','cond','setsize'])['RT'].transform(
    lambda x:stats.zscore(x))
df_sch_clean.loc[np.where(np.abs(outRTs)>crit_sd)[0],'Correct'] = 0
del_points = len(df_sch_clean[df_sch_clean['Correct']==0])-incorr_points
print('RT: delete %0.3f%% data points'%(del_points/len(df_sch_raw)*100))

del_indx = pd.DataFrame()
del_indx_list = df_sch_clean[df_sch_clean['Correct']==0].index.tolist()
df_sch_clean['indx'] = list(range(0,trialN_all))*subjAllN
del_indx[['subj','del_indx']] = df_sch_clean.loc[del_indx_list,['Subject','indx']]
del df_sch_clean['indx']

if tag_savefile==1:
    del_indx.to_csv(os.path.join(resPath,'del_indx.csv'),
                    mode='w',header=True,index=False)

# --- --- ---
df_sch_del = df_sch_clean.copy(deep=True)
df_sch_del = df_sch_del[df_sch_del['Correct']==1]

df_sch_mean = df_sch_del.groupby(
    ['Subject','setsize','cond'])[
    ['RT','Correct']].agg(np.mean).reset_index()
df_sch_mean_clean = df_sch_clean.groupby(
    ['Subject','setsize','cond'])[
    'Correct'].agg(np.mean).reset_index()
df_sch_mean['Correct'] = df_sch_mean_clean['Correct']
if tag_savefile==1:
    df_sch_mean.to_csv(os.path.join(resPath,'exp4_mean.csv'),
                       mode='w',header=True,index=False)

# 2.3 subj outliers
out_subjs = []
# acc
acc_check = df_sch_mean.groupby(
    ['Subject','setsize'])['Correct'].agg(np.mean).reset_index()
out_subjs_acc = list(set(
    acc_check.loc[acc_check['Correct']<crit_acc,'Subject'].tolist()))
print('ourliers (ACC):',out_subjs_acc)
acc_check_cond = df_sch_mean.groupby(
    ['Subject','setsize','cond'])['Correct'].agg(np.mean).reset_index()
print(list(set(
    acc_check_cond.loc[acc_check_cond['Correct']<crit_acc,
    ['Subject','cond','setsize','Correct']].tolist())))
# # ±3 sd
# outRTs_mean = df_sch_mean.groupby(
#     ['cond','setsize'])['RT'].transform(
#     lambda x:stats.zscore(x))
# out_subjs_rt = list(set(
#     df_sch_mean.loc[np.where(np.abs(outRTs_mean)>crit_sd)[0],
#                     'Subject'].values.tolist()))
# print('ourliers (RT):',out_subjs_rt)
# out_subjs += out_subjs_acc+out_subjs_rt
# out_subjs = list(set(out_subjs))
# print('ourliers (ACC+RT):',out_subjs)




# --- --- --- 3. Descriptive Statistic --- --- ---

# ACC & RT
for y in ['Correct','RT']:
    fig = plt.figure(figsize=(12,8))
    # sns.set_style("whitegrid")
    sns.violinplot(x='setsize',y=y,data=df_sch_mean,
                   hue='cond',palette='Set2',inner='quartile',
                   hue_order=cond_list,saturation=0.75)
    sns.stripplot(x='setsize',y=y,data=df_sch_mean,
                  hue='cond',dodge=True,palette='Set2',
                  hue_order=cond_list)
    plt.grid(linestyle=':')
    if y=='Correct':
        ymin,ymax = 0.5,1.2
    else:
        ymin,ymax = 0.2,1.5
    plt.ylim(ymin,ymax)
    plt.legend(loc='best',ncol=4,fontsize=10)
    figName = os.path.join(figPath,'descr_%s')%(y)
    save_fig(fig,figName)

#
fig = plt.figure(figsize=(12,8))
sns.lineplot(data=df_sch_mean,x='setsize',y='RT',hue='cond',
             style='cond',markers=True,
             errorbar=('se',0),palette='Set2')
plt.grid(linestyle=':')
figName = os.path.join(figPath,'descr_rt_line')
save_fig(fig,figName)

# mean data
mean_data = df_sch_del.groupby(
    ['setsize','cond'])[['RT','Correct']].agg(np.mean).reset_index()
mean_data_clean = df_sch_clean.groupby(
    ['setsize','cond'])[['Correct']].agg(np.mean).reset_index()
mean_data['Correct'] = mean_data_clean['Correct']
mean_data['RT'] = mean_data['RT'].apply(lambda x:'%.3f'%x)
mean_data['Correct'] = mean_data['Correct'].apply(lambda x:'%.3f'%x)
print('*** *** *** *** *** ***')
print(mean_data)
print('')



#
# barplot for mean RT
mpl.rcParams.update({'font.size':26})
plt_size = 26
fig,ax = plt.subplots(1,2,figsize=(18,9),sharex=True,sharey=True)
ax = ax.ravel()

x = -0.75
y = 0.82
sub_title = 'Target-Present Trials'
fig_lab = '(A)'
sns.barplot(data=df_sch_del[df_sch_del['cond'].isin(['wt','bt'])],
            x='setsize',y='RT',hue='cond',hue_order=['wt','bt'],
            palette=['tomato','deepskyblue'],saturation=0.75,width=0.55,
            errorbar='se',capsize=0.15,errcolor='grey',
            errwidth=1.5,ax=ax[0])
ax[0].set_xlabel('Memory Set Size')
ax[0].set_ylabel('RT (sec)')
ax[0].set_ylim(0.0,0.75)
y_major_locator = MultipleLocator(0.25)
ax[0].yaxis.set_major_locator(y_major_locator)
ax[0].set_title(sub_title,fontsize=plt_size,pad=20)
ax[0].text(x,y,fig_lab,ha='center',
           va='top',color='k',fontsize=plt_size)
h,_ = ax[0].get_legend_handles_labels()
ax[0].legend(h,['T-Dw','T-Db'],loc='upper left',ncol=1,fontsize=20,
             frameon=False).set_title(None)

sub_title = 'Target-Absent Trials'
fig_lab = '(B)'
sns.barplot(data=df_sch_del[df_sch_del['cond'].isin(['wb','ww','bb'])],
            x='setsize',y='RT',hue='cond',hue_order=['wb','ww','bb'],
            palette=['lightgrey','gold','mediumseagreen'],
            saturation=0.75,
            errorbar='se',capsize=0.15,errcolor='grey',
            errwidth=1.5,ax=ax[1])
ax[1].set_xlabel('Memory Set Size')
ax[1].set_ylabel(None)
ax[1].set_title(sub_title,fontsize=plt_size,pad=20)
ax[1].text(x,y,fig_lab,ha='center',
           va='top',color='k',fontsize=plt_size)
h,_ = ax[1].get_legend_handles_labels()
ax[1].legend(h,['Dw-Db','Dw-Dw','Db-Db'],loc='upper left',
             ncol=1,fontsize=20,
             frameon=False).set_title(None)
# plt.suptitle('Experiment 2')
# fig.text(0.5,0,'Memory Set Size',ha='center')
sns.despine(offset=15,trim=True)
plt.tight_layout()

figName = os.path.join(figPath,'behav_descr_bar')
save_fig(fig,figName)

# #
# # barplot for mean RT
# mpl.rcParams.update({'font.size':18})
# fig,ax = plt.subplots(1,1,figsize=(12,9))
# sns.barplot(data=df_sch_del[df_sch_del['cond'].isin(['wt','bt','wb'])],
#             x='setsize',y='RT',hue='cond',hue_order=['wt','bt','wb'],
#             palette=['tomato','deepskyblue','lightgrey'],
#             saturation=0.75,errorbar='se',capsize=0.15,errcolor='grey',
#             errwidth=1.5,ax=ax)
# ax.set_xlabel('Memory Set Size')
# ax.set_ylabel('RT (sec)')
# ax.set_ylim(0.2,0.7)
# ax.set_title('Experiment 2',fontsize=20)
# h,_ = ax.get_legend_handles_labels()
# ax.legend(h,['WT','BT','WB'],loc='upper left',ncol=1,fontsize=13,
#              frameon=False).set_title(None)
#
# sns.despine(offset=15,trim=True)
# plt.tight_layout()
# figName = os.path.join(figPath,'behav_descr_bar')
# save_fig(fig,figName)


# # --- --- --- 4. Statistics --- --- ---
#
# # ANOVA: interaction
# for y in ['Correct','RT']:
#     print('*** *** ***')
#     print(y)
#     aov = pg.rm_anova(dv=y,within=['setsize','cond'],subject='Subject',
#                       data=df_sch_mean,detailed=True,effsize="np2")
#     print('*** *** *** *** *** ***')
#     print('ANOVA: interaction')
#     print(y)
#     pg.print_table(aov)
#
#     if aov.loc[
#         aov['Source']=='setsize * cond',
#         'p-GG-corr'].values<p_crit:
#         mean_table = df_sch_mean.groupby(
#             ['cond','setsize'])['RT'].agg(
#             ['mean','std']).round(3)
#         print(mean_table)
#     print('')

    # print('*** *** *** *** *** ***')
    # pwc1 = pg.pairwise_tests(dv=y,within=['setsize','cond'],
    #                          subject='subj',data=df_sch_mean_del)
    # pwc2 = pg.pairwise_tests(dv=y,within=['cond','setsize'],
    #                          subject='subj',data=df_sch_mean_del)
    # pg.print_table(pwc1)
    # print('*** *** *** *** *** ***')
    # pg.print_table(pwc2)

