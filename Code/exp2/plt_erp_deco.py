#!/usr/bin/env python
#-*-coding:utf-8 -*-

# EXP. 4 (Behavioural):
# N2pc
# 2023.Mar.9
# linlin.shang@donders.ru.nl

from exp4_config import subjAllN,subjList,filePath,outliers,wb_names,\
    resPath,behavFigPath,sizeList,cateList,condList,cond_list,\
    p_crit,crit_rt,crit_sd,crit_acc,trialN_all,decoDataPath_all,\
    decoFigPath_all,set_filepath,tag_savefile,save_fig

import os
from math import log
import random
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from sklearn.linear_model import LinearRegression
model = LinearRegression()
import pingouin as pg
import statannot




# --- --- --- 1. Read Files --- --- ---

dataPath = set_filepath(filePath,'Data','DataBehav')
df_erp = pd.read_csv(os.path.join(resPath,'exp4_n2pc.csv'),sep=',')

#!/usr/bin/env python
# -*-coding:utf-8 -*-

# EXP.4 (EEG): configure
# 2023.Mar.13
# linlin.shang@donders.ru.nl



#%% --- --- --- --- --- --- libraries --- --- --- --- --- ---
from exp4_config import epoDataPath,evkDataPath,resPath,grpFigPath,\
    subjList,subjAllN,sizeList,condList,cond_list,tmin,tmax,chance_crit,\
    show_flg,tag_savefile,save_fig,p_crit,n_permutations,targ_names
import mne
import numpy as np
import pandas as pd
from scipy.stats import sem
import random
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.pyplot import MultipleLocator
import seaborn as sns
import os

from mne.stats import permutation_cluster_1samp_test,\
    permutation_cluster_test
from mne.stats import f_threshold_mway_rm,f_mway_rm,fdr_correction

scale = (1e+6)

def find_sig(clu,clu_p):
    acc_sig,grp_sig = t_points*[0],t_points*[0]
    grp_label = 0

    for c,p in zip(clu,clu_p):
        if p<p_crit:
            grp_label += 1
            acc_sig[(t0_indx+c[0][0]):(t0_indx+c[0][-1]+1)] =\
                [1]*len(c[0])
            grp_sig[(t0_indx+c[0][0]):(t0_indx+c[0][-1]+1)] =\
                [grp_label]*len(c[0])
    return acc_sig,grp_sig
def check_sig(time_list):
    sig_start,sig_end = [],[]
    count = 1
    for n in range(len(time_list)):
        if ((n==0) and (time_list[n]>0)) or\
                ((n>0) and (time_list[n]>0) and
                 (round(time_list[n]-time_list[n-1],3)>
                  0.004)):
            sig_start.append(time_list[n])
            if count>1:
                sig_end.append(time_list[n-1])
            count += 1
    if len(time_list)!=0:
        sig_end.append(time_list[len(time_list)-1])
    return sig_start,sig_end


df_sch_n2pc = pd.read_csv(os.path.join(resPath,'exp4_n2pc.csv'),sep=',')
df_sch_n2pc['contr'] = df_sch_n2pc['contr']*scale
df_sch_n2pc['ipsi'] = df_sch_n2pc['ipsi']*scale
df_sch_n2pc['n2pc'] = df_sch_n2pc['n2pc']*scale

deco_n2pc_mean = pd.read_csv(os.path.join(resPath,'deco_data_mean.csv'),sep=',')
df_n2pc = df_sch_n2pc[(df_sch_n2pc['time']>=0.2)&
                      (df_sch_n2pc['time']<0.3)].reset_index(drop=True)

df_contr = df_sch_n2pc.groupby(
    ['subj','time','setsize','cond'])['contr'].agg(np.mean).reset_index()
df_contr['loc'] = ['contr']*len(df_contr)
df_contr.rename(columns={'contr':'amp'},inplace=True)
df_ipsi = df_sch_n2pc.groupby(
    ['subj','time','setsize','cond'])['ipsi'].agg(np.mean).reset_index()
df_ipsi['loc'] = ['ipsi']*len(df_ipsi)
df_ipsi.rename(columns={'ipsi':'amp'},inplace=True)
df_contr_ipsi = df_sch_n2pc.groupby(
    ['subj','time','setsize','cond'])['n2pc'].agg(np.mean).reset_index()
df_contr_ipsi['loc'] = ['n2pc']*len(df_ipsi)
df_contr_ipsi.rename(columns={'n2pc':'amp'},inplace=True)
df_contr_ipsi = pd.concat([df_contr_ipsi,df_contr,df_ipsi],
                           axis=0,ignore_index=True)
times = df_contr['time'].values
# plot
mpl.rcParams.update({'font.size':24})
fig,ax = plt.subplots(1,2,figsize=(20,9))
ax[0].axhline(0.0,color='k',linestyle=':')
ax[0].axvline(0.0,color='k',linestyle=':')
ax[0].axvline(0.2,ls='--',color='grey')
ax[0].axvline(0.3,ls='--',color='grey')
ax[0].text(0.225,2,'N2pc',color='grey',fontsize=16,fontweight='bold')
sns.lineplot(data=df_contr_ipsi,x='time',y='amp',
             hue='loc',hue_order=['contr','ipsi','n2pc'],
             palette=['tomato','deepskyblue','orange'],
             lw=2,ax=ax[0])
ymin,ymax = ax[0].get_ylim()
# ax.fill_between(times,ymin,ymax,
#                 where=(times>=0.2)&(times<0.3),
#                 color='whitesmoke',alpha=0.01)
h,_ = ax[0].get_legend_handles_labels()
ax[0].legend(h,['Contra','Ipsi','Contra-Ipsi'],
             loc='lower left',ncol=1,fontsize=18,
             frameon=True).set_title(None)
ax[0].set_title(None)
ax[0].set_xlabel('Time (sec)')
ax[0].set_ylabel('μV')
ax[0].set_ylim(ymin=-3,ymax=10.5)
y_major_locator = MultipleLocator(3)
ax[0].yaxis.set_major_locator(y_major_locator)
ax[0].text(-0.275,10.5,'(A)',ha='center',
           va='top',color='k',fontsize=22)
# barplot
plt_bar_data = df_contr_ipsi[
    (df_contr_ipsi['cond'].isin(['wt','bt']))&
    (df_contr_ipsi['time']<0.3)&(df_contr_ipsi['time']>=0.2)].groupby(
    ['subj','setsize','cond','loc'])['amp'].agg(np.mean).reset_index()
sns.barplot(data=plt_bar_data,
            x='setsize',y='amp',hue='loc',hue_order=['contr','ipsi'],
            palette=['tomato','deepskyblue'],saturation=0.75,
            errorbar='se',capsize=0.15,errcolor='grey',
            errwidth=1.5,ax=ax[1])
ax[1].set_title(None)
ax[1].set_xlabel(xlabel='Memory Set Size')
ax[1].set_ylabel(ylabel=None)
# y_major_locator = MultipleLocator(1)
# ax[1].yaxis.set_major_locator(y_major_locator)
h,_ = ax[1].get_legend_handles_labels()
ax[1].legend(h,['Contra','Ipsi'],
             loc='lower left',ncol=1,fontsize=18,
             frameon=True).set_title(None)
ax[1].text(-0.7,9.4,'(B)',ha='center',
           va='top',color='k',fontsize=22)
plt.suptitle('Target-Present Trials')
sns.despine(offset=15,trim=True)
fig.tight_layout()
# figName = os.path.join(grpFigPath,'contr_ipsi_targ.png')
figName = os.path.join(grpFigPath,'contr_ipsi_targ.tif')
save_fig(fig,figName)
plt.close('all')


t = df_sch_n2pc.loc[(df_sch_n2pc['subj']==1)&
                    (df_sch_n2pc['setsize']==1)&
                    (df_sch_n2pc['cond']=='wt'),'time'].values
t0_clu,t1_clu = 0.1,0.4
df_sch_n2pc_t = df_sch_n2pc[(df_sch_n2pc['time']>=t0_clu)&
                            (df_sch_n2pc['time']<t1_clu)].reset_index(drop=True)
t_crop = df_sch_n2pc_t.loc[(df_sch_n2pc_t['subj']==1)&
                           (df_sch_n2pc_t['setsize']==1)&
                           (df_sch_n2pc_t['cond']=='wt'),'time'].values
erp_arr = np.zeros(
    [subjAllN,8,len(t_crop)])
for indx,n in enumerate(subjList):
    count = 0
    for h in sizeList:
        for k,cond in enumerate(['wt','bt']):
            erp_arr[indx,count,:] = df_sch_n2pc_t.loc[
                (df_sch_n2pc_t['subj']==n)&
                (df_sch_n2pc_t['setsize']==h)&
                (df_sch_n2pc_t['cond']==cond),'n2pc'].values
            count += 1
tail = 0
factor_levels = [4,2]
effects = 'A:B'
def stat_fun(*args):
    factor_levels = [4,2]
    effects = 'A:B'
    return f_mway_rm(
        np.array(args).transpose(1,0,2),
        factor_levels=factor_levels,
        effects=effects,return_pvals=False)[0]
f_thresh = f_threshold_mway_rm(
    subjAllN,factor_levels,effects,p_crit)
f_obs,clu,clu_p,h0 = permutation_cluster_test(
    erp_arr.transpose(1,0,2),stat_fun=stat_fun,threshold=f_thresh,
    tail=tail,n_jobs=None,n_permutations=n_permutations,
    buffer_size=None,out_type='mask')
print(clu)
print(clu_p)
grp_start,grp_end = [],[]
for c,p in zip(clu,clu_p):
    if p < p_crit:
        grp_start.append(t_crop.tolist()[c[0]][0])
        grp_end.append(t_crop.tolist()[c[0]][-2])
print(grp_start,grp_end)
# setsize
def stat_fun(*args):
    factor_levels = [4,2]
    effects = 'A'
    return f_mway_rm(
        np.array(args).transpose(1,0,2),
        factor_levels=factor_levels,
        effects=effects,return_pvals=False)[0]
effects = 'A'
f_thresh = f_threshold_mway_rm(
    subjAllN,factor_levels,effects,p_crit)
f_obs,clu,clu_p,h0 = permutation_cluster_test(
    erp_arr.transpose(1,0,2),stat_fun=stat_fun,threshold=f_thresh,
    tail=tail,n_jobs=None,n_permutations=n_permutations,
    buffer_size=None,out_type='mask')
print(clu)
print(clu_p)
grp_start_size,grp_end_size = [],[]
for c,p in zip(clu,clu_p):
    if p < p_crit:
        grp_start_size.append(t_crop.tolist()[c[0]][0])
        grp_end_size.append(t_crop.tolist()[c[0]][-2])
print(grp_start_size,grp_end_size)
# category
def stat_fun(*args):
    factor_levels = [4,2]
    effects = 'B'
    return f_mway_rm(
        np.array(args).transpose(1,0,2),
        factor_levels=factor_levels,
        effects=effects,return_pvals=False)[0]
effects = 'B'
f_thresh = f_threshold_mway_rm(
    subjAllN,factor_levels,effects,p_crit)
f_obs,clu,clu_p,h0 = permutation_cluster_test(
    erp_arr.transpose(1,0,2),stat_fun=stat_fun,threshold=f_thresh,
    tail=tail,n_jobs=None,n_permutations=n_permutations,
    buffer_size=None,out_type='mask')
print(clu)
print(clu_p)
grp_start_cate,grp_end_cate = [],[]
for c,p in zip(clu,clu_p):
    if p < p_crit:
        grp_start_cate.append(t_crop.tolist()[c[0]][0])
        grp_end_cate.append(t_crop.tolist()[c[0]][-2])
print(grp_start_cate,grp_end_cate)


# plot
lw_wid = 1.5
leg_font = 16
title_font = 24
mpl.rcParams.update({'font.size':24})
fig,ax = plt.subplots(2,2,figsize=(20,20))
ax = ax.ravel()
# size
clrs_all_b = sns.color_palette('Blues',n_colors=35)
clrs_all = sns.color_palette('GnBu_d')
clrs = [clrs_all_b[28],clrs_all_b[20],clrs_all[2],clrs_all[0]]
plt_erp_mean = df_sch_n2pc[df_sch_n2pc['cond'].isin(['bt','wt'])].groupby(
    ['subj','time','cond','setsize'])['n2pc'].agg(np.mean).reset_index()
plt_erp_mean['type'] = plt_erp_mean['cond'].values+\
                       plt_erp_mean['setsize'].apply(str).values
sns.lineplot(data=plt_erp_mean,x='time',y='n2pc',
             hue='setsize',palette=clrs,lw=lw_wid,ax=ax[1])
ymin,ymax = ax[1].get_ylim()
if grp_start_size!=[]:
    count = 0
    for h in range(len(grp_start_size)):
        ax[1].fill_between(
            t,ymin,ymax,
            where=(t>=grp_start_size[h])&(t<grp_end_size[h]),
            color='grey',alpha=0.1)
        ax[1].text(0.05,-3+count,'%.3f-%.3f sec'%(
            grp_start_size[h],grp_end_size[h]),color='grey',
                   fontsize=leg_font)
        count += -0.4
ax[1].set_xlim(xmin=tmin,xmax=tmax)
ax[1].set_xticks(np.arange(tmin,tmax,0.2),labels=[])
ax[1].set_title('Main Effect of Memory Set Size',fontsize=title_font)
ax[1].set_xlabel(xlabel=None)
ax[1].set_ylim(ymin=-5.1,ymax=1.1)
ax[1].set_yticks(np.arange(ymin,ymax,2),labels=[])
ax[1].set_ylabel(ylabel=None)
y_major_locator = MultipleLocator(2)
ax[1].yaxis.set_major_locator(y_major_locator)
x_major_locator = MultipleLocator(0.2)
ax[1].xaxis.set_major_locator(x_major_locator)
ax[1].axvline(0,ls='--',color='k')
ax[1].axhline(0,ls='--',color='k')
h,_ = ax[1].get_legend_handles_labels()
ax[1].legend(h,['MSS 1','MSS 2','MSS 4','MSS 8'],
             loc='lower left',ncol=1,fontsize=leg_font,
             frameon=False).set_title(None)

# category
sns.lineplot(data=plt_erp_mean,x='time',y='n2pc',
             hue='cond',hue_order=['wt','bt'],
             palette=['tomato','deepskyblue'],
             lw=lw_wid,ax=ax[2])
ymin,ymax = ax[2].get_ylim()
if grp_start_cate!=[]:
    ax[2].fill_between(
        t,ymin,ymax,
        where=(t>=grp_start_cate[0])&(t<grp_end_cate[0]),
        color='grey',alpha=0.1)
    ax[2].text(0.015,-3,'%.3f-%.3f sec'%(
        grp_start_cate[0],grp_end_cate[0]),color='grey',
               fontsize=leg_font)
ax[2].set_xlim(xmin=tmin,xmax=tmax)
ax[2].set_title('Main Effect of Category',fontsize=title_font)
ax[2].set_xlabel(xlabel='Time (sec)')
ax[2].set_ylabel(ylabel='μV')
ax[2].set_ylim(ymin=-5.1,ymax=1.1)
y_major_locator = MultipleLocator(2)
ax[2].yaxis.set_major_locator(y_major_locator)
ax[2].xaxis.set_major_locator(x_major_locator)
ax[2].axvline(0,ls='--',color='k')
ax[2].axhline(0,ls='--',color='k')
h,_ = ax[2].get_legend_handles_labels()
ax[2].legend(h,['T-Dw','T-Db'],
             loc='lower left',ncol=1,fontsize=leg_font,
             frameon=False).set_title(None)
# interaction
wt_clr = [sns.color_palette('Reds')[4],
          sns.color_palette('Reds')[3],
          sns.color_palette('Reds')[2],
          sns.color_palette('Reds')[1]]
bt_clr = [sns.color_palette('Blues')[4],
          sns.color_palette('Blues')[3],
          sns.color_palette('Blues')[2],
          sns.color_palette('Blues')[1]]
sns.lineplot(data=plt_erp_mean,x='time',y='n2pc',
             hue='type',hue_order=['wt1','wt2','wt4','wt8',
                                   'bt1','bt2','bt4','bt8'],
             palette=wt_clr+bt_clr,
             errorbar=('ci',0),lw=lw_wid,ax=ax[3])
ymin,ymax = ax[3].get_ylim()
if grp_start!=[]:
    ax[3].fill_between(
        t,ymin,ymax,
        where=(t>=grp_start[0])&(t<grp_end[0]),
        color='grey',alpha=0.1)
    ax[3].text(0.1,-5,'%.3f-%.3f sec'%(
        grp_start[0],grp_end[0]),color='grey',
               fontsize=leg_font)
# ax.grid(True)
ax[3].set_xlim(xmin=tmin,xmax=tmax)
ax[3].set_title('Interaction Effect',fontsize=title_font)
ax[3].set_xlabel(xlabel='Time (sec)')
ax[3].set_ylabel(ylabel=None)
ax[3].set_ylim(ymin=-5.1,ymax=1.1)
y_major_locator = MultipleLocator(2)
ax[3].set_yticks(np.arange(ymin,ymax,2),labels=[])
ax[3].yaxis.set_major_locator(y_major_locator)
ax[3].xaxis.set_major_locator(x_major_locator)
ax[3].axvline(0,ls='--',color='k')
ax[3].axhline(0,ls='--',color='k')
h,_ = ax[3].get_legend_handles_labels()
ax[3].legend(h,['T-Dw/1','T-Dw/2','T-Dw/4','T-Dw/8',
                'T-Db/1','T-Db/2','T-Db/4','T-Db/8'],
             loc='lower left',ncol=2,fontsize=leg_font,
             frameon=False).set_title(None)
# barplot
n2pc_plt = df_sch_n2pc_t[df_sch_n2pc_t['cond'].isin(['wt','bt'])].groupby(
    ['subj','cond','setsize'])['n2pc'].agg(np.mean).reset_index()
sns.barplot(data=n2pc_plt,
            x='setsize',y='n2pc',hue='cond',hue_order=['wt','bt'],
            palette=['tomato','deepskyblue'],saturation=0.75,
            errorbar='se',capsize=0.15,errcolor='grey',
            errwidth=1,ax=ax[0])
ax[0].set_title('Comparison',fontsize=title_font,pad=10)
ax[0].set_xlabel(xlabel='Memory Set Size')
ax[0].set_ylabel(ylabel='μV')
y_major_locator = MultipleLocator(1.5)
ax[0].yaxis.set_major_locator(y_major_locator)
h,_ = ax[0].get_legend_handles_labels()
ax[0].legend(h,['T-Dw','T-Db'],
             loc='upper left',ncol=1,fontsize=leg_font,
             frameon=True).set_title(None)
# fig.text(0,0.5,'μV',va='center',rotation='vertical')
plt.suptitle('N2pc',y=0.93)
ax[1].text(-0.23,1.75,'(B)',ha='center',
           va='top',color='k',fontsize=22)
ax[2].text(-0.23,1.75,'(C)',ha='center',
           va='top',color='k',fontsize=22)
ax[3].text(-0.23,1.75,'(D)',ha='center',
           va='top',color='k',fontsize=22)
ax[0].text(-0.7,0.45,'(A)',ha='center',
           va='top',color='k',fontsize=22)

sns.despine(offset=15,trim=True)
fig.tight_layout()
# plt.margins(0,0)
plt.subplots_adjust(hspace=0.65,wspace=0.1)
# figName = os.path.join(grpFigPath,'clst_n2pc.png')
figName = os.path.join(grpFigPath,'clst_n2pc.tif')
save_fig(fig,figName)
#

#
# interaction
fig,ax = plt.subplots(1,1,figsize=(12,9))
ax = sns.lineplot(data=df_n2pc[(df_n2pc['cond'].isin(['wt','bt']))&
                         (df_n2pc['time']>=grp_start[0])&
                         (df_n2pc['time']<grp_end[0])],
            x='setsize',y='n2pc',hue='cond',
            palette=['tomato','deepskyblue'],
                  err_style="bars",errorbar ='se')
ax.set_title('Interaction',fontsize=title_font)
ax.set_xlabel(xlabel='Memory Set Size')
ax.set_ylabel(ylabel=None)
y_major_locator = MultipleLocator(1)
ax.yaxis.set_major_locator(y_major_locator)
h,_ = ax.get_legend_handles_labels()
ax.legend(h,['T-Dw','T-Db'],
          loc='upper left',ncol=1,fontsize=leg_font,
          frameon=True).set_title(None)

sns.despine(offset=15,trim=True)
fig.tight_layout()
plt.subplots_adjust(hspace=0.45,wspace=0.15)
# figName = os.path.join(grpFigPath,'clst_inter_barplt.png')
figName = os.path.join(grpFigPath,'clst_inter_barplt.tif')
save_fig(fig,figName)

# WB
df_sch_n2pc_wb_all = df_sch_n2pc[df_sch_n2pc['cond']=='wb']
df_sch_n2pc_contr = df_sch_n2pc_wb_all.loc[(df_sch_n2pc_wb_all['cond']=='wb'),
['contr','time','subj','setsize','cond']]
df_sch_n2pc_contr.rename(columns={'contr':'amp'},inplace=True)
df_sch_n2pc_contr['cond'] = ['contr']*len(df_sch_n2pc_contr)
df_sch_n2pc_ipsi = df_sch_n2pc_wb_all.loc[df_sch_n2pc_wb_all['cond']=='wb',
['ipsi','time','subj','setsize','cond']]
df_sch_n2pc_ipsi.rename(columns={'ipsi':'amp'},inplace=True)
df_sch_n2pc_ipsi['cond'] = ['ipsi']*len(df_sch_n2pc_ipsi)
df_sch_n2pc_wb = pd.concat([df_sch_n2pc_contr,df_sch_n2pc_ipsi],
                           axis=0,ignore_index=True)

df_n2pc = df_sch_n2pc_wb[(df_sch_n2pc_wb['time']>=0.2)&
                         (df_sch_n2pc_wb['time']<0.3)].reset_index(drop=True)
df_sch_n2pc_t = df_sch_n2pc_wb[(df_sch_n2pc_wb['time']>=t0_clu)&
                               (df_sch_n2pc_wb['time']<t1_clu)].reset_index(drop=True)
erp_arr = np.zeros(
    [subjAllN,8,len(t_crop)])
for indx,n in enumerate(subjList):
    count = 0
    for h in sizeList:
        for k,cond in enumerate(['contr','ipsi']):
            erp_arr[indx,count,:] = df_sch_n2pc_t.loc[
                (df_sch_n2pc_t['subj']==n)&
                (df_sch_n2pc_t['setsize']==h)&
                (df_sch_n2pc_t['cond']==cond),'amp'].values
            count += 1
tail = 0
factor_levels = [4,2]
effects = 'A:B'
def stat_fun(*args):
    factor_levels = [4,2]
    effects = 'A:B'
    return f_mway_rm(
        np.array(args).transpose(1,0,2),
        factor_levels=factor_levels,
        effects=effects,return_pvals=False)[0]
f_thresh = f_threshold_mway_rm(
    subjAllN,factor_levels,effects,p_crit)
f_obs,clu,clu_p,h0 = permutation_cluster_test(
    erp_arr.transpose(1,0,2),stat_fun=stat_fun,threshold=f_thresh,
    tail=tail,n_jobs=None,n_permutations=n_permutations,
    buffer_size=None,out_type='mask')
print(clu)
print(clu_p)
grp_start,grp_end = [],[]
for c,p in zip(clu,clu_p):
    if p < p_crit:
        grp_start.append(t_crop.tolist()[c[0]][0])
        grp_end.append(t_crop.tolist()[c[0]][-2])
print(grp_start,grp_end)
# setsize
def stat_fun(*args):
    factor_levels = [4,2]
    effects = 'A'
    return f_mway_rm(
        np.array(args).transpose(1,0,2),
        factor_levels=factor_levels,
        effects=effects,return_pvals=False)[0]
effects = 'A'
f_thresh = f_threshold_mway_rm(
    subjAllN,factor_levels,effects,p_crit)
f_obs,clu,clu_p,h0 = permutation_cluster_test(
    erp_arr.transpose(1,0,2),stat_fun=stat_fun,threshold=f_thresh,
    tail=tail,n_jobs=None,n_permutations=n_permutations,
    buffer_size=None,out_type='mask')
print(clu)
print(clu_p)
grp_start_size,grp_end_size = [],[]
for c,p in zip(clu,clu_p):
    if p < p_crit:
        grp_start_size.append(t_crop.tolist()[c[0]][0])
        grp_end_size.append(t_crop.tolist()[c[0]][-2])
print(grp_start_size,grp_end_size)
# category
def stat_fun(*args):
    factor_levels = [4,2]
    effects = 'B'
    return f_mway_rm(
        np.array(args).transpose(1,0,2),
        factor_levels=factor_levels,
        effects=effects,return_pvals=False)[0]
effects = 'B'
f_thresh = f_threshold_mway_rm(
    subjAllN,factor_levels,effects,p_crit)
f_obs,clu,clu_p,h0 = permutation_cluster_test(
    erp_arr.transpose(1,0,2),stat_fun=stat_fun,threshold=f_thresh,
    tail=tail,n_jobs=None,n_permutations=n_permutations,
    buffer_size=None,out_type='mask')
print(clu)
print(clu_p)
grp_start_cate,grp_end_cate = [],[]
for c,p in zip(clu,clu_p):
    if p < p_crit:
        grp_start_cate.append(t_crop.tolist()[c[0]][0])
        grp_end_cate.append(t_crop.tolist()[c[0]][-2])
print(grp_start_cate,grp_end_cate)

# plot
lw_wid = 1.5
leg_font = 15
mpl.rcParams.update({'font.size':24})
fig,ax = plt.subplots(1,2,figsize=(20,9))
ax = ax.ravel()
sns.lineplot(data=df_sch_n2pc_wb,x='time',y='amp',
             hue='cond',hue_order=['contr','ipsi'],
             palette=['tomato','deepskyblue'],
             lw=lw_wid,ax=ax[0])
ymin,ymax = ax[0].get_ylim()
if grp_start_cate!=[]:
    ax[0].fill_between(
        t,ymin,ymax,
        where=(t>=grp_start_cate[0])&(t<grp_end_cate[0]),
        color='grey',alpha=0.1)
    ax[0].text(0.05,6,'%.3f-%.3f sec'%(
        grp_start_cate[0],grp_end_cate[0]),color='grey',
               fontsize=leg_font)
ax[0].set_xlim(xmin=tmin,xmax=tmax)
ax[0].set_title('')
ax[0].set_xlabel(xlabel='Time (sec)')
# x_major_locator = MultipleLocator(0.2)
# ax[0].xaxis.set_major_locator(x_major_locator)
ax[0].set_ylabel(ylabel='μV')
ax[0].set_ylim(ymin=-3,ymax=11.9)
y_major_locator = MultipleLocator(3)
ax[0].yaxis.set_major_locator(y_major_locator)
ax[0].axvline(0,ls='--',color='k')
ax[0].axhline(0,ls='--',color='k')
ax[0].spines['top'].set_visible(False)
ax[0].spines['right'].set_visible(False)
h,_ = ax[0].get_legend_handles_labels()
ax[0].legend(h,['Contra','Ipsi'],
             loc='lower left',ncol=1,fontsize=leg_font,
             frameon=False).set_title(None)
ax[0].text(-0.24,12.2,'(A)',ha='center',
           va='top',color='k',fontsize=22)
# barplot
plt_bar_mean = df_sch_n2pc_t.groupby(
        ['subj','cond','setsize'])['amp'].agg(np.mean).reset_index()

sns.barplot(data=plt_bar_mean,
            x='setsize',y='amp',hue='cond',hue_order=['contr','ipsi'],
            palette=['tomato','deepskyblue'],saturation=0.75,
            errorbar='se',capsize=0.15,errcolor='grey',
            errwidth=1.5,ax=ax[1])
ax[1].set_title('')
ax[1].set_xlabel(xlabel='Memory Set Size')
ax[1].set_ylabel(ylabel=None)

y_major_locator = MultipleLocator(2)
ax[1].yaxis.set_major_locator(y_major_locator)
h,_ = ax[1].get_legend_handles_labels()
ax[1].legend(h,['Contra','Ipsi'],
             loc='lower left',ncol=1,fontsize=leg_font,
             frameon=True).set_title(None)
ax[1].text(-0.75,6.5,'(B)',ha='center',
           va='top',color='k',fontsize=22)
plt.suptitle('Target-Absent Trials')

sns.despine(offset=15,trim=True)
fig.tight_layout()
# figName = os.path.join(grpFigPath,'clst_n2pc_wb.png')
figName = os.path.join(grpFigPath,'clst_n2pc_wb.tif')
save_fig(fig,figName)



# # ####################################################
# # Decoding
# t_list = np.load(file=os.path.join(
#     decoDataPath_all,'t_list.npy'),allow_pickle=True)
# t_points = len(t_list)
# t0_indx,t1_indx = np.where(t_list>=0.1)[0][0],\
#                   np.where(t_list<=0.4)[0][-1]
# t_indx = t_list[t0_indx:t1_indx]
#
# # barplot
# acc_df = pd.read_csv(
#     os.path.join(resPath,'deco_data_all.csv'),sep=',')
# acc_df_bar = acc_df[acc_df['pred']=='each']
# acc_df_bar['cond'] = acc_df_bar['type'].str[0:2]
# acc_df_bar['setsize'] = acc_df_bar['type'].str[2]
# #
# mpl.rcParams.update({'font.size':24})
# fig,ax = plt.subplots(1,figsize=(12,9))
# sns.barplot(data=acc_df_bar[acc_df_bar['cond'].isin(['wt','bt'])],
#             x='setsize',y='acc',hue='cond',hue_order=['wt','bt'],
#             palette=['crimson','dodgerblue'],saturation=0.75,
#             errorbar='se',capsize=0.15,errcolor='grey',
#             errwidth=1.5,ax=ax)
# ax.set_title('Target-Present Trial Decoding for N2pc')
# ax.set_ylim(ymin=0.5,ymax=0.56)
# ax.set_xlabel(xlabel='Memory Set Size')
# ax.set_ylabel(ylabel='AUC')
# y_major_locator = MultipleLocator(0.02)
# ax.yaxis.set_major_locator(y_major_locator)
# h,_ = ax.get_legend_handles_labels()
# ax.legend(h,['T-Dw','T-Db'],
#              loc='upper right',ncol=1,fontsize=18,
#              frameon=True).set_title(None)
# sns.despine(offset=15,trim=True)
# fig.tight_layout()
# # figName = os.path.join(decoFigPath_all,'deco_bar_n2pc.png')
# figName = os.path.join(decoFigPath_all,'deco_bar_n2pc.tif')
# save_fig(fig,figName)
#
#
# # permutation
# acc_subjAll_cond = np.load(
#     file=os.path.join(resPath,'deco_acc_all.npy'),
#     allow_pickle=True)
# acc_subjAll_cond = acc_subjAll_cond.astype(np.float64)
# acc_mean,acc_sem = dict(),dict()
# for n,label in enumerate(targ_names+wb_names):
#     acc_mean[label] = np.mean(acc_subjAll_cond[n],axis=0)
#     acc_sem[label] = sem(acc_subjAll_cond[n])
#
# def clu_permu_cond(acc_data_all,stat_fun):
#     acc_data = acc_data_all[:,t0_indx:t1_indx]
#     threshold = None
#     f_obs,clu,clu_p,H0 = permutation_cluster_test(
#         acc_data,
#         n_permutations=n_permutations,
#         threshold=threshold,tail=0,
#         stat_fun=stat_fun,n_jobs=None,
#         out_type='indices')
#     print(clu)
#     print(clu_p)
#     acc_sig,grp_sig = find_sig(clu,clu_p)
#     return acc_sig,grp_sig
# def clu_permu_1samp_t(acc_data_all):
#     acc_data = acc_data_all[:,t0_indx:t1_indx]
#     threshold = None
#     tail = 0
#     # degrees_of_freedom = len(acc_data) - 1
#     # threshold = scipy.stats.t.ppf(
#     #     1-p_crit/2,df=degrees_of_freedom)
#     t_obs,clu,clu_p,H0 = permutation_cluster_1samp_test(
#         acc_data-chance_crit,n_permutations=n_permutations,
#         threshold=threshold,tail=tail,
#         out_type='indices',verbose=True)
#     print(clu)
#     print(clu_p)
#     acc_sig,grp_sig = find_sig(clu,clu_p)
#     return acc_sig,grp_sig
# def stat_fun(*args):
#     factor_levels = [2,4]
#     effects = 'A:B'
#     return f_mway_rm(
#         np.array(args).transpose(1,0,2),
#         factor_levels=factor_levels,
#         effects=effects,return_pvals=False)[0]
# acc_sig,sig_grp = clu_permu_cond(
#     acc_subjAll_cond[0:8],
#     stat_fun)
# # category
# def stat_fun_maineff(*args):
#     factor_levels = [2,4]
#     effects = 'A'
#     return f_mway_rm(
#         np.array(args).transpose(1,0,2),
#         factor_levels=factor_levels,
#         effects=effects,return_pvals=False)[0]
# tail = 0
# factor_levels = [2,4]
# effects = 'A'
# f_thresh = f_threshold_mway_rm(
#     subjAllN,factor_levels,effects,p_crit)
# f_obs,clu,clu_p,H0 = permutation_cluster_test(
#     acc_subjAll_cond[0:8],stat_fun=stat_fun_maineff,
#     threshold=f_thresh,
#     tail=tail,n_jobs=None,n_permutations=n_permutations,
#     buffer_size=None,out_type='indices')
# acc_sig,grp_sig = find_sig(clu,clu_p)
# print(clu)
# print(clu_p)
# grp_start_cate,grp_end_cate = [],[]
# for c,p in zip(clu,clu_p):
#     if p<p_crit:
#         grp_start_cate.append(t_list[c[0]][0])
#         grp_end_cate.append(t_list[c[0]][-2])
# print(grp_start_cate,grp_end_cate)
# # setsize
# def stat_fun_maineff(*args):
#     factor_levels = [2,4]
#     effects = 'B'
#     return f_mway_rm(
#         np.array(args).transpose(1,0,2),
#         factor_levels=factor_levels,
#         effects=effects,return_pvals=False)[0]
# tail = 0
# factor_levels = [2,4]
# effects = 'B'
# f_thresh = f_threshold_mway_rm(
#     subjAllN,factor_levels,effects,p_crit)
# f_obs,clu,clu_p,H0 = permutation_cluster_test(
#     acc_subjAll_cond[0:8,:,t0_indx:t1_indx],
#     stat_fun=stat_fun_maineff,
#     threshold=f_thresh,
#     tail=tail,n_jobs=None,n_permutations=n_permutations,
#     buffer_size=None,out_type='indices')
# acc_sig,grp_sig = find_sig(clu,clu_p)
# print(clu)
# print(clu_p)
# grp_start_size,grp_end_size = [],[]
# for c,p in zip(clu,clu_p):
#     if p<p_crit:
#         grp_start_size.append(t_indx[c[0]][0])
#         grp_end_size.append(t_indx[c[0]][-2])
# print(grp_start_size,grp_end_size)
#
# #
# deco_n2pc_plt = deco_n2pc_mean[(deco_n2pc_mean['type'].isin(
#     targ_names))&(deco_n2pc_mean['pred']=='each')]
# # plot each condition
# mpl.rcParams.update({'font.size':20})
# clr = ['crimson','deepskyblue']
# fig, ax = plt.subplots(
#     2,4,sharex=True,sharey=True,figsize=(20,16))
# ax = ax.ravel()
# plt_labels = ['within/1','within/2','within/4','within/8',
#               'between/1','between/2','between/4','between/8']
# for indx,label in enumerate(['wt1','wt2','wt4','wt8',
#                              'bt1','bt2','bt4','bt8']):
#     y = deco_n2pc_plt.loc[deco_n2pc_plt['type']==label,'acc']
#     ax[indx].axhline(0.5,color='k',linestyle='--')
#     ax[indx].axvline(0.0,color='k',linestyle=':')
#     ax[indx].plot(t_list,y,
#                   color='crimson',linestyle='-',
#                   linewidth=lw_wid,label=plt_labels[indx])
#     ymax = 0.75
#     ymin_show = 0.48
#     sig_df = deco_n2pc_plt[deco_n2pc_plt['type']==label]
#     for k in set(sig_df['grp_label']):
#         if k > 0:
#             sig_times = sig_df.loc[
#                 (sig_df['grp_label']==k) &
#                 (sig_df['type']==label),'time']
#             ax[indx].plot(sig_times,[ymin_show]*len(sig_times),
#                           color='crimson',linewidth=lw_wid)
#             print(sig_times)
#
#     ax[indx].fill_between(t_list,
#                           acc_mean[label]-acc_sem[label],
#                           acc_mean[label]+acc_sem[label],
#                           color='crimson',alpha=0.1,
#                           edgecolor='none')
#     ax[indx].set_yticks(np.arange(0.45,0.8,0.1))
#     ax[indx].set_title(plt_labels[indx])
# fig.text(0.5,-0.04,'Time (sec)',ha='center')
# fig.text(0,0.5,'AUC',va='center',rotation='vertical')
#
# sns.despine(offset=15,trim=True)
# plt.tight_layout()
# # title = 'deco_each_cond.png'
# title = 'deco_each_cond.tif'
# save_fig(fig,os.path.join(decoFigPath_all,title))
#
# #
# # plot main eff
# mpl.rcParams.update({'font.size':20})
# clr = ['crimson','deepskyblue']
# fig, ax = plt.subplots(
#     1,2,sharex=True,sharey=True,figsize=(20,16))
# ax = ax.ravel()
# # category
# deco_n2pc_plt = acc_df[(acc_df['type'].isin(
#     ['wt','bt']))&(acc_df['pred']=='cate')]
# ax[0].axhline(0.5,color='k',linestyle='--')
# ax[0].axvline(0.0,color='k',linestyle=':')
# sns.lineplot(data=deco_n2pc_plt,x='time',y='acc',
#              hue='type',hue_order=['wt','bt'],
#              palette=['crimson','dodgerblue'],
#              lw=lw_wid,ax=ax[0])
# ymin,ymax = ax[0].get_ylim()
# # for label in targ_names:
# #     ax[0].fill_between(t_list,
# #                        acc_mean[label]-acc_sem[label],
# #                        acc_mean[label]+acc_sem[label],
# #                        color='crimson',alpha=0.1,
# #                        edgecolor='none')
# if grp_start_cate!=[]:
#     ax[0].fill_between(
#         t,ymin,ymax,
#         where=(t>=grp_start_cate[0])&(t<grp_end_cate[0]),
#         color='grey',alpha=0.1)
#     ax[0].text(0.05,0.6,'%.3f-%.3f sec'%(
#         grp_start_cate[0],grp_end_cate[0]),color='grey',
#                fontsize=leg_font)
# ax[0].set_title('Main Effect of Category')
# ax[0].set_xlabel(xlabel=None)
# ax[0].set_ylabel(ylabel='AUC')
# h,_ = ax[0].get_legend_handles_labels()
# ax[0].legend(h,['WT','BT'],
#              loc='upper right',ncol=1,fontsize=18,
#              frameon=True).set_title(None)
# # setsize
# deco_n2pc_plt = acc_df[(acc_df['pred']=='size')]
# clrs_all_b = sns.color_palette('Blues',n_colors=35)
# clrs_all = sns.color_palette('GnBu_d')
# clrs = [clrs_all_b[28],clrs_all_b[20],clrs_all[2],clrs_all[0]]
# ax[1].axhline(0.5,color='k',linestyle='--')
# ax[1].axvline(0.0,color='k',linestyle=':')
# sns.lineplot(data=deco_n2pc_plt,x='time',y='acc',
#              hue='type',hue_order=['1','2','4','8'],
#              palette=clrs,
#              lw=lw_wid,ax=ax[1])
# ymin,ymax = ax[1].get_ylim()
# if grp_start_size!=[]:
#     ax[1].fill_between(
#         t,ymin,ymax,
#         where=(t>=grp_start_size[0])&(t<grp_end_size[0]),
#         color='grey',alpha=0.1)
#     ax[1].text(0.05,0.6,'%.3f-%.3f sec'%(
#         grp_start_size[0],grp_end_size[0]),color='grey',
#                fontsize=leg_font)
# ax[1].set_title('Main Effect of Memory Set Size')
# ax[1].set_xlabel(xlabel=None)
# ax[1].set_ylabel(ylabel='AUC')
# h,_ = ax[1].get_legend_handles_labels()
# ax[1].legend(h,['MSS 1','MSS 2','MSS 4','MSS 8'],
#              loc='upper right',ncol=1,fontsize=18,
#              frameon=True).set_title(None)
# sns.despine(offset=15,trim=True)
# plt.tight_layout()
# # title = 'deco_n2pc_main_eff.png'
# title = 'deco_n2pc_main_eff.tif'
# save_fig(fig,os.path.join(decoFigPath_all,title))