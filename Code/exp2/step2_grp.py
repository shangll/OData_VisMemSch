#!/usr/bin/env python
# -*-coding:utf-8 -*-

# EXP.4 (EEG): configure
# 2023.Mar.13
# linlin.shang@donders.ru.nl



#%% --- --- --- --- --- --- libraries --- --- --- --- --- ---
from exp4_config import epoDataPath,evkDataPath,resPath,grpFigPath,\
    subjList,subjAllN,sizeList,condList,cond_list,tmin,tmax,\
    show_flg,tag_savefile,save_fig,p_crit,n_permutations,targ_names
import mne
import numpy as np
import pandas as pd
import pingouin as pg
import random
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.pyplot import MultipleLocator
import seaborn as sns
import os

from mne.stats import permutation_cluster_1samp_test,\
    permutation_cluster_test
from mne.stats import f_threshold_mway_rm,f_mway_rm,fdr_correction

from matplotlib.backends.backend_pdf import PdfPages

# pp = PdfPages('exp4_res.pdf')
# def savePDF(df,df_name):
#     fig,ax = plt.subplots(figsize=(12,4))
#     ax.axis('tight')
#     ax.axis('off')
#     ax.table(cellText=df.values,colLabels=df.columns,loc='center')
#     plt.title(df_name)
#     pp.savefig(fig,bbox_inches='tight')
def find_sig(clu,clu_p,t_points):
    acc_sig, grp_sig = t_points*[0],t_points*[0]
    grp_label = 0

    for c,p in zip(clu,clu_p):
        if p < p_crit:
            grp_label += 1
            acc_sig[c[0][0]:(c[0][-1]+1)] = \
                [1]*len(c[0])
            grp_sig[c[0][0]:(c[0][-1]+1)] = \
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
df_n2pc = df_sch_n2pc[(df_sch_n2pc['time']>=0.2)&
                      (df_sch_n2pc['time']<0.3)].reset_index(drop=True)
df_n2pc_avg = df_n2pc.groupby(
    ['subj','cond','setsize'])[['contr','ipsi','n2pc']].agg(np.mean).reset_index()
df_contr_ipsi = df_n2pc.groupby(
    ['subj'])[['contr','ipsi','n2pc']].agg(np.mean).reset_index()

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
# # setsize
# erp_arr = np.zeros(
#         [subjAllN,4,len(t)])
# for indx,n in enumerate(subjList):
#     for k,cond in enumerate(sizeList):
#         erp_arr[indx,k,:] = df_sch_n2pc[
#             (df_sch_n2pc['subj']==n)&
#             (df_sch_n2pc['setsize']==cond)
#         ].groupby(['time'])['n2pc'].agg(np.mean).values
# def stat_fun_1way(*args):
#     factor_levels = [4,1]
#     effects = 'A'
#     return f_mway_rm(
#         np.array(args).transpose(1,0,2),
#         factor_levels=factor_levels,
#         effects=effects,return_pvals=False)[0]
# factor_levels = [4,1]
# effects = 'A'
# f_thresh = f_threshold_mway_rm(
#     subjAllN,factor_levels,effects,p_crit)
# f_obs,clu,clu_p,h0 = permutation_cluster_test(
#     erp_arr.transpose(1,0,2),stat_fun=stat_fun_1way,threshold=f_thresh,
#     tail=tail,n_jobs=None,n_permutations=n_permutations,
#     buffer_size=None,out_type='mask')
# print(clu)
# print(clu_p)
# grp_start_size,grp_end_size = [],[]
# for c,p in zip(clu,clu_p):
#     if p < p_crit:
#         grp_start_size.append(t.tolist()[c[0]][0])
#         grp_end_size.append(t.tolist()[c[0]][-2])
# print(grp_start_size,grp_end_size)
# # category
# erp_arr = np.zeros(
#         [subjAllN,2,len(t)])
# for indx,n in enumerate(subjList):
#     for k,cond in enumerate(['wt','bt']):
#         erp_arr[indx,k,:] = df_sch_n2pc[
#             (df_sch_n2pc['subj']==n)&
#             (df_sch_n2pc['cond']==cond)
#         ].groupby(['time'])['n2pc'].agg(np.mean).values
# t_thresh = None
# tail = -1
# degrees_of_freedom = len(erp_arr)-1
# t_obs,clu,clu_p,H0 = permutation_cluster_1samp_test(
#     erp_arr[:,0,:]-erp_arr[:,1,:],
#     n_permutations=n_permutations,threshold=t_thresh,
#     tail=tail,out_type='mask',verbose=True)
# print(clu)
# print(clu_p)
# grp_start_cate,grp_end_cate = [],[]
# for c,p in zip(clu,clu_p):
#     if p < p_crit:
#         grp_start_cate.append(t.tolist()[c[0]][0])
#         grp_end_cate.append(t.tolist()[c[0]][-2])
# print(grp_start_cate,grp_end_cate)
# plot
plt_erp_mean = df_sch_n2pc[df_sch_n2pc['cond'].isin(['bt','wt'])].groupby(
    ['time','cond','setsize'])['n2pc'].agg(np.mean).reset_index()
mpl.rcParams.update({'font.size':26})
fig,ax = plt.subplots(1,figsize=(20,12))
sns.lineplot(data=plt_erp_mean,x='time',y='n2pc',
                 hue='cond',hue_order=['bt','wt'],
                 style='setsize',
                 palette=['crimson','dodgerblue'],lw=3,ax=ax)
ymin,ymax = ax.get_ylim()
if grp_start!=[]:
    ax.fill_between(
        t,ymin,ymax,
        where=(t>=grp_start[0])&(t<grp_end[0]),
        color='grey',alpha=0.1)
    plt.text(0.05,-3*1e-6,'%.3f-%.3f sec'%(
        grp_start[0],grp_end[0]),color='grey')
# ax.grid(True)
ax.set_xlim(xmin=tmin,xmax=tmax)
ax.set_title('')
ax.set_xlabel(xlabel='Time (sec)')
ax.set_ylabel(ylabel='μV')
x_major_locator = MultipleLocator(0.1)
ax.xaxis.set_major_locator(x_major_locator)
ax.axvline(0,ls='--',color='k')
ax.axhline(0,ls='--',color='k')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
h,_ = ax.get_legend_handles_labels()
# ax.legend(h,eff_label,loc='best',ncol=1).set_title(eff_tag)
# plt.grid(linestyle=':')
fig.tight_layout()
figName = os.path.join(grpFigPath,'clst_n2pc.png')
save_fig(fig,figName)
# size
clrs_all_b = sns.color_palette('Blues',n_colors=35)
clrs_all = sns.color_palette('GnBu_d')
clrs = [clrs_all_b[28],clrs_all_b[20],clrs_all[2],clrs_all[0]]
plt_erp_mean = df_sch_n2pc[df_sch_n2pc['cond'].isin(['bt','wt'])].groupby(
    ['subj','time','setsize'])['n2pc'].agg(np.mean).reset_index()
mpl.rcParams.update({'font.size':26})
fig,ax = plt.subplots(1,figsize=(20,12))
sns.lineplot(data=plt_erp_mean,x='time',y='n2pc',
                 hue='setsize',
                 palette=clrs,lw=3,ax=ax)
ymin,ymax = ax.get_ylim()
if grp_start_size!=[]:
    count = 0
    for h in range(len(grp_start_size)):
        ax.fill_between(
            t,ymin,ymax,
            where=(t>=grp_start_size[h])&(t<grp_end_size[h]),
            color='grey',alpha=0.1)
        plt.text(0.05,-3*1e-6+count,'%.3f-%.3f sec'%(
            grp_start_size[h],grp_end_size[h]),color='grey')
        count += -0.25*1e-6
# ax.grid(True)
ax.set_xlim(xmin=tmin,xmax=tmax)
ax.set_title('')
ax.set_xlabel(xlabel='Time (sec)')
ax.set_ylabel(ylabel='μV')
x_major_locator = MultipleLocator(0.1)
ax.xaxis.set_major_locator(x_major_locator)
ax.axvline(0,ls='--',color='k')
ax.axhline(0,ls='--',color='k')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
h,_ = ax.get_legend_handles_labels()
# ax.legend(h,eff_label,loc='best',ncol=1).set_title(eff_tag)
# plt.grid(linestyle=':')
fig.tight_layout()
figName = os.path.join(grpFigPath,'clst_n2pc_size.png')
save_fig(fig,figName)
# category
plt_erp_mean = df_sch_n2pc[df_sch_n2pc['cond'].isin(['bt','wt'])].groupby(
    ['subj','time','cond'])['n2pc'].agg(np.mean).reset_index()
mpl.rcParams.update({'font.size':26})
fig,ax = plt.subplots(1,figsize=(20,12))
sns.lineplot(data=plt_erp_mean,x='time',y='n2pc',
                 hue='cond',hue_order=['wt','bt'],
                 palette=['crimson','dodgerblue'],lw=3,ax=ax)
ymin,ymax = ax.get_ylim()
if grp_start_cate!=[]:
    ax.fill_between(
        t,ymin,ymax,
        where=(t>=grp_start_cate[0])&(t<grp_end_cate[0]),
        color='grey',alpha=0.1)
    plt.text(0.05,-3*1e-6,'%.3f-%.3f sec'%(
        grp_start_cate[0],grp_end_cate[0]),color='grey')
# ax.grid(True)
ax.set_xlim(xmin=tmin,xmax=tmax)
ax.set_title('')
ax.set_xlabel(xlabel='Time (sec)')
ax.set_ylabel(ylabel='μV')
x_major_locator = MultipleLocator(0.1)
ax.xaxis.set_major_locator(x_major_locator)
ax.axvline(0,ls='--',color='k')
ax.axhline(0,ls='--',color='k')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
h,_ = ax.get_legend_handles_labels()
# ax.legend(h,eff_label,loc='best',ncol=1).set_title(eff_tag)
# plt.grid(linestyle=':')
fig.tight_layout()
figName = os.path.join(grpFigPath,'clst_n2pc_cate.png')
save_fig(fig,figName)

# ------------------------------------------------------------
# wb
df_n2pc_avgAll = df_sch_n2pc[(df_sch_n2pc['cond']=='wb')].groupby(
    ['time'])[['contr','ipsi','n2pc']].agg(np.mean).reset_index()
df_n2pc_wbAll = df_sch_n2pc[(df_sch_n2pc['cond']=='wb')].groupby(
    ['time'])[['contr','ipsi','n2pc']].agg(np.mean).reset_index()
# PO7/8
scale = 1
chan_loc = ['contr','ipsi','n2pc']
lineStys = ['-','-','--']
mpl.rcParams.update({'font.size':16})
clrs = sns.color_palette('Paired',n_colors=40)
fig,ax = plt.subplots(1,1,figsize=(20,12))
count = 0
for chan,lineSty in zip(chan_loc,lineStys):
    x = df_n2pc_avgAll['time'].values
    y = df_n2pc_avgAll[chan].values
    ax.plot(x,y*scale,linestyle=lineSty,color=clrs[count],
            linewidth=2.5,label=chan,alpha=1)
    count += 1

ax.grid(False)
ax.set_xlim(xmin=tmin,xmax=tmax)
ax.set_ylim(ymin=-3e-6,ymax=12e-6)
ax.set_title('WB PO7/8')
ax.set_xlabel(xlabel='Time (sec)')
ax.set_ylabel(ylabel='μV')
ax.axvline(0,ls='--',color='k')
ax.axhline(0,ls='--',color='k')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.legend(loc='best',ncol=2,fontsize=10)
# plt.grid(linestyle=':')
fig.tight_layout()
figName = os.path.join(grpFigPath,'grp_PO78_wb_contr_ipsi.png')
save_fig(fig,figName)

#
df_n2pc_avgAll = df_sch_n2pc[(df_sch_n2pc['cond']!='ww')|
                             (df_sch_n2pc['cond']!='bb')|
                             (df_sch_n2pc['cond']!='wb')].groupby(
    ['time'])[['contr','ipsi','n2pc']].agg(np.mean).reset_index()
df_n2pc_wbAll = df_sch_n2pc[(df_sch_n2pc['cond']=='wb')].groupby(
    ['time'])[['contr','ipsi','n2pc']].agg(np.mean).reset_index()
# PO7/8
scale = 1
chan_loc = ['contr','ipsi','n2pc']
lineStys = ['-','-','--']
mpl.rcParams.update({'font.size':16})
clrs = sns.color_palette('Paired',n_colors=40)
fig,ax = plt.subplots(1,1,figsize=(20,12))
count = 0
for chan,lineSty in zip(chan_loc,lineStys):
    x = df_n2pc_avgAll['time'].values
    y = df_n2pc_avgAll[chan].values
    ax.plot(x,y*scale,linestyle=lineSty,color=clrs[count],
            linewidth=2.5,label=chan,alpha=1)
    count += 1

ax.grid(False)
ax.set_xlim(xmin=tmin,xmax=tmax)
ax.set_ylim(ymin=-3e-6,ymax=12e-6)
ax.set_title('Target trials PO7/8')
ax.set_xlabel(xlabel='Time (sec)')
ax.set_ylabel(ylabel='μV')
ax.axvline(0,ls='--',color='k')
ax.axhline(0,ls='--',color='k')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.legend(loc='best',ncol=2,fontsize=10)
# plt.grid(linestyle=':')
fig.tight_layout()
figName = os.path.join(grpFigPath,'grp_PO78_targ_contr_ipsi.png')
save_fig(fig,figName)


#
df_n2pc_cate = df_n2pc.groupby(
    ['subj','cond'])[['contr','ipsi','n2pc']].agg(np.mean).reset_index()
df_n2pc_size = df_n2pc.groupby(
    ['subj','setsize'])[['contr','ipsi','n2pc']].agg(np.mean).reset_index()


df_n2pc_mean = df_sch_n2pc.groupby(
        ['time','cond','setsize'])[['contr','ipsi','n2pc']].agg(np.mean).reset_index()
# category
# PO7/8
scale = 1
chan_loc = ['contr','ipsi']
lineStys = ['-','--']
mpl.rcParams.update({'font.size':16})
clrs = sns.color_palette('Paired',n_colors=40)
fig,ax = plt.subplots(2,2,figsize=(20,12),sharex=True,sharey=True)
ax = ax.ravel()
for indx,setsize in enumerate(sizeList):
    count = 0
    for cond in cond_list:
        for chan,lineSty in zip(chan_loc,lineStys):
            x = df_n2pc_mean.loc[(df_n2pc_mean['setsize']==setsize)&
                                 (df_n2pc_mean['cond']==cond),'time'].values
            y = df_n2pc_mean.loc[(df_n2pc_mean['setsize']==setsize)&
                                 (df_n2pc_mean['cond']==cond),chan].values
            ax[indx].plot(x,y*scale,linestyle=lineSty,color=clrs[count],
                          linewidth=1,label=str(cond)+'/'+chan,alpha=1)
            count += 1
    ax[indx].grid(True)
    ax[indx].set_xlim(xmin=tmin,xmax=tmax)
    ax[indx].set_title('PO7/PO8')
    ax[indx].set_xlabel(xlabel='Time (sec)')
    ax[indx].set_ylabel(ylabel='μV')
    ax[indx].axvline(0,ls='--',color='k')
    ax[indx].axhline(0,ls='--',color='k')
    ax[indx].spines['top'].set_visible(False)
    ax[indx].spines['right'].set_visible(False)
plt.legend(loc='best',ncol=2,fontsize=10)
plt.grid(linestyle=':')
fig.tight_layout()
figName = os.path.join(grpFigPath,'grp_PO78_cate_subplt.png')
save_fig(fig,figName)
# pp.savefig(fig,bbox_inches='tight')
#
# n2pc
clrs = sns.color_palette('Set1',n_colors=5)
scale = 1
mpl.rcParams.update({'font.size':16})
fig,ax = plt.subplots(2,2,figsize=(20,12),sharex=True,sharey=True)
ax = ax.ravel()
for indx,setsize in enumerate(sizeList):
    count = 0
    for cond in cond_list:
        if cond in ['ww','bb']:
            lineSty = '--'
        else:
            lineSty = '-'
        x = df_n2pc_mean.loc[(df_n2pc_mean['setsize']==setsize)&
                             (df_n2pc_mean['cond']==cond),'time'].values
        y = df_n2pc_mean.loc[(df_n2pc_mean['setsize']==setsize)&
                             (df_n2pc_mean['cond']==cond),'n2pc'].values
        ax[indx].plot(x,y*scale,linestyle=lineSty,color=clrs[count],
                      linewidth=1,label=cond,alpha=1)
        count += 1
    ax[indx].grid(True)
    ax[indx].set_xlim(xmin=tmin,xmax=tmax)
    ax[indx].set_title('N2pc MSS %d'%setsize)
    ax[indx].set_xlabel(xlabel='Time (sec)')
    ax[indx].set_ylabel(ylabel='μV')
    ax[indx].axvline(0,ls='--',color='k')
    ax[indx].axhline(0,ls='--',color='k')
    ax[indx].spines['top'].set_visible(False)
    ax[indx].spines['right'].set_visible(False)
plt.legend(loc='best',ncol=2,fontsize=10)
plt.grid(linestyle=':')
fig.tight_layout()
figName = os.path.join(grpFigPath,'grp_n2pc_cate_subplt.png')
save_fig(fig,figName)
# pp.savefig(fig,bbox_inches='tight')

# set size
# PO7/8
clrs = sns.color_palette('Paired',n_colors=40)
scale = 1
chan_loc = ['contr','ipsi']
lineStys = ['-','--']
mpl.rcParams.update({'font.size':16})
clrs = sns.color_palette('Paired',n_colors=40)
fig,ax = plt.subplots(2,3,figsize=(20,12),sharex=True,sharey=True)
ax = ax.ravel()
for indx,cond in enumerate(cond_list):
    count = 0
    for setsize in sizeList:
        for chan,lineSty in zip(chan_loc,lineStys):
            x = df_n2pc_mean.loc[(df_n2pc_mean['setsize']==setsize)&
                                 (df_n2pc_mean['cond']==cond),'time'].values
            y = df_n2pc_mean.loc[(df_n2pc_mean['setsize']==setsize)&
                                 (df_n2pc_mean['cond']==cond),chan].values
            ax[indx].plot(x,y*scale,linestyle=lineSty,color=clrs[count],
                          linewidth=1,label='MSS %d/'%setsize+chan,alpha=1)
            count += 1
    ax[indx].grid(True)
    ax[indx].set_xlim(xmin=tmin,xmax=tmax)
    ax[indx].set_title('MSS %d PO7/PO8'%setsize)
    ax[indx].set_xlabel(xlabel='Time (sec)')
    ax[indx].set_ylabel(ylabel='μV')
    ax[indx].axvline(0,ls='--',color='k')
    ax[indx].axhline(0,ls='--',color='k')
    ax[indx].spines['top'].set_visible(False)
    ax[indx].spines['right'].set_visible(False)
ax[indx].legend(loc='best',ncol=2,fontsize=10)
plt.grid(linestyle=':')
fig.tight_layout()
figName = os.path.join(grpFigPath,'grp_PO78_size_subplt.png')
save_fig(fig,figName)
# pp.savefig(fig,bbox_inches='tight')

#
# n2pc
clrs = sns.color_palette('Set1',n_colors=4)
scale = 1
mpl.rcParams.update({'font.size':16})
fig,ax = plt.subplots(2,3,figsize=(20,9),sharex=True,sharey=True)
ax = ax.ravel()
for indx,cond in enumerate(cond_list):
    count = 0
    for setsize in sizeList:
        if cond in ['ww','bb']:
            lineSty = '--'
        else:
            lineSty = '-'
        x = df_n2pc_mean.loc[(df_n2pc_mean['setsize']==setsize)&
                             (df_n2pc_mean['cond']==cond),'time'].values
        y = df_n2pc_mean.loc[(df_n2pc_mean['setsize']==setsize)&
                             (df_n2pc_mean['cond']==cond),'n2pc'].values
        ax[indx].plot(x,y*scale,linestyle=lineSty,color=clrs[count],
                      linewidth=1,label='MSS %d'%setsize,alpha=1)
        count += 1
    ax[indx].grid(True)
    ax[indx].set_xlim(xmin=tmin,xmax=tmax)
    ax[indx].set_title('N2pc %s Category'%cond.upper())
    ax[indx].set_xlabel(xlabel='Time (sec)')
    ax[indx].set_ylabel(ylabel='μV')
    ax[indx].axvline(0,ls='--',color='k')
    ax[indx].axhline(0,ls='--',color='k')
    ax[indx].spines['top'].set_visible(False)
    ax[indx].spines['right'].set_visible(False)
ax[indx].legend(loc='best',ncol=2,fontsize=10)
plt.grid(linestyle=':')
fig.tight_layout()
figName = os.path.join(grpFigPath,'grp_n2pc_size_subplt.png')
save_fig(fig,figName)
# pp.savefig(fig,bbox_inches='tight')

# average all conditions
for mean_tag in ['cond','setsize','each']:
    if mean_tag=='each':
        df_n2pc_mean = df_sch_n2pc.groupby(
            ['time','cond','setsize'])[['contr','ipsi','n2pc']].agg(np.mean).reset_index()
    elif mean_tag=='cond':
        df_n2pc_mean = df_sch_n2pc.groupby(
            ['time','cond'])[['contr','ipsi','n2pc']].agg(np.mean).reset_index()
    else:
        df_n2pc_mean = df_sch_n2pc.groupby(
            ['time','setsize'])[['contr','ipsi','n2pc']].agg(np.mean).reset_index()
    # PO7/8
    scale = 1
    chan_loc = ['contr','ipsi']
    lineStys = ['-','--']
    mpl.rcParams.update({'font.size':16})
    clrs = sns.color_palette('Paired',n_colors=40)
    fig,ax = plt.subplots(1,figsize=(20,12))
    if mean_tag=='each':
        count = 0
        for cond in ['wt','bt','wb','ww','bb']:
            for setsize in sizeList:
                for chan,lineSty in zip(chan_loc,lineStys):
                    x = df_n2pc_mean.loc[(df_n2pc_mean['setsize']==setsize)&
                                         (df_n2pc_mean['cond']==cond),'time'].values
                    y = df_n2pc_mean.loc[(df_n2pc_mean['cond']==cond)&
                                         (df_n2pc_mean['setsize']==setsize),chan].values
                    ax.plot(x,y*scale,linestyle=lineSty,color=clrs[count],
                            linewidth=2,label=cond+'/'+str(setsize)+'/'+chan,alpha=1)
                    count+=1
    else:
        if mean_tag=='cond':
            plt_conds = ['wt','bt','wb','ww','bb']
        else:
            plt_conds = sizeList
        count = 0
        for cond in plt_conds:
            for chan,lineSty in zip(chan_loc,lineStys):
                x = df_n2pc_mean.loc[(df_n2pc_mean[mean_tag]==cond),'time'].values
                y = df_n2pc_mean.loc[(df_n2pc_mean[mean_tag]==cond),chan].values
                ax.plot(x,y*scale,linestyle=lineSty,color=clrs[count],
                        linewidth=2,label=str(cond)+'/'+chan,alpha=1)
                count += 1
    ax.grid(True)
    ax.set_xlim(xmin=tmin,xmax=tmax)
    ax.set_title('PO7/PO8')
    ax.set_xlabel(xlabel='Time (sec)')
    ax.set_ylabel(ylabel='μV')
    ax.axvline(0,ls='--',color='k')
    ax.axhline(0,ls='--',color='k')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.legend(loc='best',ncol=5,fontsize=10)
    plt.grid(linestyle=':')
    fig.tight_layout()
    figName = os.path.join(grpFigPath,'grp_PO78_%s.png'%mean_tag)
    save_fig(fig,figName)
    # pp.savefig(fig,bbox_inches='tight')
    #
    # n2pc
    scale = 1
    mpl.rcParams.update({'font.size':16})
    fig,ax = plt.subplots(1,figsize=(20,12))
    if mean_tag=='each':
        for cond in ['wt','bt','wb','ww','bb']:
            for setsize in sizeList:
                if cond in ['ww','bb']:
                    lineSty = '--'
                else:
                    lineSty = '-'
                x = df_n2pc_mean.loc[(df_n2pc_mean['setsize']==setsize)&
                                     (df_n2pc_mean['cond']==cond),'time'].values
                y = df_n2pc_mean.loc[(df_n2pc_mean['cond']==cond)&
                                     (df_n2pc_mean['setsize']==setsize),'n2pc'].values
                ax.plot(x,y*scale,label=cond+'/'+str(setsize)+'/n2pc',linestyle=lineSty,
                        linewidth=2,alpha=1)
    else:
        if mean_tag=='cond':
            plt_conds = ['wt','bt','wb','ww','bb']
        else:
            plt_conds = sizeList
        count = 0
        for cond in plt_conds:
            if cond in ['ww','bb']:
                lineSty = '--'
            else:
                lineSty = '-'
            x = df_n2pc_mean.loc[(df_n2pc_mean[mean_tag]==cond),'time'].values
            y = df_n2pc_mean.loc[(df_n2pc_mean[mean_tag]==cond),'n2pc'].values
            ax.plot(x,y*scale,linestyle=lineSty,color=clrs[count],
                    linewidth=2,label=str(cond),alpha=1)
            count += 1
    ax.grid(True)
    ax.set_xlim(xmin=tmin,xmax=tmax)
    ax.set_title('N2pc')
    ax.set_xlabel(xlabel='Time (sec)')
    ax.set_ylabel(ylabel='μV')
    ax.axvline(0,ls='--',color='k')
    ax.axhline(0,ls='--',color='k')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.legend(loc='best',ncol=5,fontsize=10)
    plt.grid(linestyle=':')
    fig.tight_layout()
    figName = os.path.join(grpFigPath,'grp_n2pc_%s.png'%mean_tag)
    save_fig(fig,figName)
    # pp.savefig(fig,bbox_inches='tight')

# boxplot
fig = plt.figure(figsize=(12,8))
# sns.set_style("whitegrid")
# sns.violinplot(x='setsize',y='n2pc',data=df_n2pc,
#                hue='cond',palette='Set2',inner='quartile',
#                hue_order=cond_list,saturation=0.75)
# sns.stripplot(x='setsize',y='n2pc',data=df_n2pc,
#               hue='cond',dodge=True,palette='Set2',
#               hue_order=cond_list)
sns.boxplot(x='setsize',y='n2pc',data=df_n2pc,
            hue='cond',notch=True,palette='Set2')
plt.grid(linestyle=':')
plt.legend(loc='best',ncol=2,fontsize=10)
figName = os.path.join(grpFigPath,'grp_box_n2pc.png')
save_fig(fig,figName)
# pp.savefig(fig,bbox_inches='tight')

# lineplot
fig = plt.figure(figsize=(12,8))
# sns.set_style("whitegrid")
sns.lineplot(x='setsize',y='n2pc',data=df_n2pc,
             hue='cond',style='cond',markers=True,
             errorbar=('se',2),err_style='bars',palette='Set2')
plt.grid(linestyle=':')
plt.legend(loc='best',ncol=2,fontsize=10)
figName = os.path.join(grpFigPath,'grp_line_n2pc.png')
save_fig(fig,figName)
# pp.savefig(fig,bbox_inches='tight')
# pp.close()

