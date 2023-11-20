#!/usr/bin/env python
# -*-coding:utf-8 -*-

# EXP. 3-3.1 (EEG): evoke
# Temporal dynamics of non-target processing vary with
# memory set size in different category conditions
# 2021.Nov.24
# linlin.shang@donders.ru.nl



#%% --- --- --- --- --- --- libraries --- --- --- --- --- ---

from eeg_config import resPath,allResFigPath,\
    subjList_final,subjAllN_final,subjList,subjAllN,\
    sizeList,comp_list,condList,\
    cond_label_list,recog_label_list,recog_labels,\
    postChans,frontChans,p_crit,t_space,n_permutations,\
    save_fig,set_filepath
import mne
from mne.stats import permutation_cluster_test
from mne.stats import f_threshold_mway_rm,f_mway_rm
import os
import numpy as np
from scipy.signal import argrelextrema
import pandas as pd
import pingouin as pg

from sklearn.linear_model import LinearRegression
import statsmodels.api as sm
import statannot

import seaborn as sns
import matplotlib as mpl
import matplotlib.pyplot as plt

from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = "all"
model = LinearRegression()



pick_tag,pred = 'simi','o2r'
decoFigPath = set_filepath(allResFigPath,'%s'%pick_tag)

deco_subj_all = pd.read_csv(
    os.path.join(resPath,'deco_data_subj.csv'),sep=',')
deco_subj = deco_subj_all[
    (deco_subj_all['chans']==pick_tag)&
    (deco_subj_all['pred']==pred)&
    (deco_subj_all['type'].isin(recog_label_list))].reset_index(drop=True)
deco_subj['cond'] = deco_subj['type'].str.split('',expand=True)[1]
deco_subj['setsize'] = deco_subj['type'].str.split('',expand=True)[2]
for t0,t1,cp_tag in (
        (0.1,0.3,'all3'),(0.1,0.16,'p1'),(0.16,0.2,'n1'),(0.2,0.3,'p2')):
    data = deco_subj[(deco_subj['time']>=t0)&
                     (deco_subj['time']<t1)].groupby(
        ['subj','cond','setsize'])['acc'].agg(np.mean).reset_index()
    aov = pg.rm_anova(
        dv='acc',within=['cond','setsize'],subject='subj',
        data=data,detailed=True,effsize='np2')
    print(cp_tag)
    pg.print_table(aov)

    # bar plot
    mpl.rcParams.update({'font.size':18})
    fig,ax = plt.subplots(figsize=(12,9))
    # sns.barplot(data=cp_df,x='cp',y='simi',palette=sns.xkcd_palette(clrs))
    cp_bar = sns.barplot(data=data,x='setsize',y='acc',
                         hue='cond',hue_order=['w','b'],
                         palette='Blues',errorbar='se',
                         capsize=0.1,errcolor='grey')
    plt.title('%s'%cp_tag.upper())
    plt.xlabel('Memory Set Size')
    plt.ylabel('Mean AUC')
    # plt.grid(linestyle='--')
    ax.set_ylim(0.5,0.6)
    ax.set_yticks(np.arange(0.5,0.65,0.05))
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    if cp_tag=='p1':
        h,_ = ax.get_legend_handles_labels()
        ax.legend(h,['within','between'],
                  loc='best',ncol=1).set_title('Category')
    else:
        ax.get_legend().remove()
    figName = os.path.join(decoFigPath,
                           'aov_deco_bar_%s_%s')%(pick_tag,cp_tag)
    save_fig(fig,figName)

    # point plot
    mpl.rcParams.update({'font.size':18})
    clrs_all_b = sns.color_palette('Blues',n_colors=35)
    clrs_all = sns.color_palette('GnBu_d')
    clrs = [clrs_all_b[28],clrs_all_b[20],clrs_all[2],clrs_all[0]]
    fig,ax = plt.subplots(figsize=(12,9))
    # sns.barplot(data=cp_df,x='cp',y='simi',palette=sns.xkcd_palette(clrs))
    cp_point = sns.pointplot(data=data,x='cond',y='acc',
                             hue='setsize',dodge=True,
                             palette=clrs,errorbar='se',
                             capsize=0.1)
    plt.title('%s'%cp_tag.upper())
    plt.xlabel('Category')
    plt.ylabel('Mean AUC')
    # plt.grid(linestyle='--')
    ax.set_yticks(np.arange(0.5,0.65,0.05))
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    if cp_tag=='p1':
        h,_ = ax.get_legend_handles_labels()
        ax.legend(h,['size 1','size 2','size 4','size 8'],
                  loc='best',ncol=2).set_title('Memory Set Size')
    else:
        ax.get_legend().remove()
    figName = os.path.join(decoFigPath,
                           'aov_deco_point_%s_%s')%(pick_tag,cp_tag)
    save_fig(fig,figName)

    # # plot
    # colStyle = ['sandybrown','darkseagreen']
    # dotStyle = ['^','o']
    # fig,ax = plt.subplots(1,figsize=(9,6))
    # for k,cond in enumerate(cond_label_list):
    #     data_mean = data[data['cond']==cond].groupby(
    #         ['setsize'])['acc'].agg(np.mean).reset_index()
    #     ax.plot(data_mean['setsize'],data_mean['acc'],
    #             color=colStyle[k],marker=dotStyle[k],
    #             linewidth=2,alpha=1,label=cond)
    # plt.xlabel('Memory Set Size')
    # plt.ylabel('Mean AUC')
    # plt.grid(linestyle='--')
    # # plt.title('%s'%title)
    # ax.spines['right'].set_visible(False)
    # ax.spines['top'].set_visible(False)
    # ax.legend(loc='best',ncol=2)
    # figName = os.path.join(decoFigPath,
    #                        'aov_deco_%s_%s_%s')%(pick_tag,pred,cp_tag)
    # save_fig(fig,figName)
    # # bar plot
    # colStyle = ['sandybrown','darkseagreen']
    # fig,ax = plt.subplots(figsize=(9,6))
    # coeff_box = sns.boxplot(
    #     x='setsize',y='acc',data=data,
    #     hue='cond',hue_order=cond_label_list,
    #     palette=colStyle)
    # statannot.add_stat_annotation(
    #     coeff_box,data=data,x='setsize',y='acc',hue='cond',
    #     box_pairs=[(('1','w'),('1','b')),
    #                (('2','w'),('2','b')),
    #                (('4','w'),('4','b')),
    #                (('8','w'),('8','b'))],
    #     test='t-test_paired',text_format='star',
    #     loc='inside',verbose=2)
    # plt.xlabel('Memory Set Size')
    # plt.ylabel('Mean AUC')
    # plt.grid(linestyle='--')
    # ax.spines['right'].set_visible(False)
    # ax.spines['top'].set_visible(False)
    # ax.legend(loc='best',ncol=1)
    # figName = os.path.join(
    #     decoFigPath,
    #     'aov_deco_bar_%s_%s_%s')%(pick_tag,pred,cp_tag)
    # save_fig(fig,figName)

    acc_w = deco_subj[
        (deco_subj['time']>=t0)&(deco_subj['time']<t1)&
        (deco_subj['cond']=='w')].groupby(
        ['subj','cond'])['acc'].agg(np.mean).reset_index()
    acc_b = deco_subj[
        (deco_subj['time']>=t0)&(deco_subj['time']<t1)&
        (deco_subj['cond']=='b')].groupby(
        ['subj','cond'])['acc'].agg(np.mean).reset_index()
    acc_cate = pd.concat([acc_w,acc_b],axis=0,ignore_index=True)

    t_val = pg.ttest(acc_b['acc'].values,acc_w['acc'].values,
                     paired=True,alternative='two-sided')
    pg.print_table(t_val)
    print('*** *** *** *** *** ***')

    # # bar plot
    # colStyle = ['sandybrown','darkseagreen']
    # fig,ax = plt.subplots(figsize=(9,9))
    # coeff_box = sns.boxplot(
    #     x='cond',y='acc',data=acc_cate,
    #     order=cond_label_list,
    #     palette=colStyle)
    # statannot.add_stat_annotation(
    #     coeff_box,data=acc_cate,x='cond',y='acc',
    #     box_pairs=[('w','b')],
    #     test='t-test_paired',text_format='star',
    #     loc='inside',verbose=2)
    # plt.xlabel('Category Condition')
    # plt.ylabel('Mean AUC')
    # plt.grid(linestyle='--')
    # ax.spines['right'].set_visible(False)
    # ax.spines['top'].set_visible(False)
    # ax.legend(loc='best',ncol=1)
    # figName = os.path.join(
    #     decoFigPath,
    #     'aov_deco_bar_cate_%s_%s_%s')%(pick_tag,pred,cp_tag)
    # save_fig(fig,figName)

# ----------------------------------------------------------

deco_all = pd.read_csv(
    os.path.join(resPath,'deco_data.csv'),sep=',')
deco = deco_all[
    (deco_all['chans']==pick_tag)&
    (deco_all['pred']==pred)].reset_index(drop=True)
deco['cond'] = deco['type'].str.split('',expand=True)[1]
deco['setsize'] = deco['type'].str.split('',expand=True)[2]

# cluster-based anova
t0_clu,t1_clu = 0.1,0.3
erp_recog_t = deco_subj[
    (deco_subj['time']>=t0_clu)&
    (deco_subj['time']<t1_clu)].reset_index(drop=True)
# erp_recog_t = erp_recog
t = erp_recog_t.loc[(erp_recog_t['subj']==0)&
                    (erp_recog_t['type']=='w1'),
                    'time'].values
erp_arr = np.zeros(
    [subjAllN,len(recog_label_list),len(t)])
for n in range(subjAllN):
    for k,cond in enumerate(recog_label_list):
        erp_arr[n,k,:] = erp_recog_t.loc[
            (erp_recog_t['subj']==n)&
            (erp_recog_t['type']==cond),
            'acc'].values
# permutation
def stat_fun(*args):
    factor_levels = [2,4]
    effects = 'A:B'
    return f_mway_rm(
        np.array(args).transpose(1,0,2),
        factor_levels=factor_levels,
        effects=effects,return_pvals=False)[0]
tail = 0
# pthresh = 0.001
factor_levels = [2,4]
effects = 'A:B'
# f_thresh = f_threshold_mway_rm(
#     subjAllN_final,factor_levels,effects,p_crit)
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
        grp_start.append(t.tolist()[c[0]][0])
        grp_end.append(t.tolist()[c[0]][-2])
print(grp_start,grp_end)

#
def stat_fun_main(*args):
    factor_levels = [2,4]
    effects = 'B'
    return f_mway_rm(
        np.array(args).transpose(1,0,2),
        factor_levels=factor_levels,
        effects=effects,return_pvals=False)[0]
tail = 0
# pthresh = 0.001
factor_levels = [2,4]
effects = 'B'
# f_thresh = f_threshold_mway_rm(
#     subjAllN_final,factor_levels,effects,p_crit)
f_thresh = f_threshold_mway_rm(
    subjAllN,factor_levels,effects,p_crit)
f_obs,clu,clu_p,h0 = permutation_cluster_test(
    erp_arr.transpose(1,0,2),stat_fun=stat_fun_main,threshold=f_thresh,
    tail=tail,n_jobs=None,n_permutations=n_permutations,
    buffer_size=None,out_type='mask')
print(clu)
print(clu_p)
grp_start,grp_end = [],[]
for c,p in zip(clu,clu_p):
    if p < p_crit:
        grp_start.append(t.tolist()[c[0]][0])
        grp_end.append(t.tolist()[c[0]][-2])
print(grp_start,grp_end)


# --- --- ---
# anova
if grp_start!=[]:
    t0 = grp_start[0]
    t1 = grp_end[0]
    data = deco_subj[(deco_subj['time']>=t0)&
                     (deco_subj['time']<=t1)].groupby(
        ['subj','cond','setsize'])['acc'].agg(np.mean).reset_index()
    aov = pg.rm_anova(
        dv='acc',within=['cond','setsize'],subject='subj',
        data=data,detailed=True,effsize='np2')
    pg.print_table(aov)

    # plot
    colStyle = ['sandybrown','darkseagreen']
    dotStyle = ['^','o']
    fig,ax = plt.subplots(1,figsize=(9,6))
    for k,cond in enumerate(cond_label_list):
        data_mean = data[data['cond']==cond].groupby(
            ['setsize'])['acc'].agg(np.mean).reset_index()
        ax.plot(data_mean['setsize'],data_mean['acc'],
                color=colStyle[k],marker=dotStyle[k],
                linewidth=2,alpha=1,label=cond)
    plt.xlabel('Memory Set Size')
    plt.ylabel('Mean AUC')
    plt.grid(linestyle='--')
    # plt.title('%s'%title)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.legend(loc='best',ncol=2)
    figName = os.path.join(decoFigPath,
        'aov_deco_%s_%s')%(pick_tag,pred)
    save_fig(fig,figName)
    # bar plot
    colStyle = ['sandybrown','darkseagreen']
    fig,ax = plt.subplots(figsize=(9,6))
    coeff_box = sns.boxplot(
        x='setsize',y='acc',data=data,
        hue='cond',hue_order=cond_label_list,
        palette=colStyle)
    statannot.add_stat_annotation(
        coeff_box,data=data,x='setsize',y='acc',hue='cond',
        box_pairs=[(('1','w'),('1','b')),
                   (('2','w'),('2','b')),
                   (('4','w'),('4','b')),
                   (('8','w'),('8','b'))],
        test='t-test_paired',text_format='star',
        loc='inside',verbose=2)
    plt.xlabel('Memory Set Size')
    plt.ylabel('Mean AUC')
    plt.grid(linestyle='--')
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.legend(loc='best',ncol=1)
    figName = os.path.join(decoFigPath,
        'aov_deco_bar_%s_%s')%(pick_tag,pred)
    save_fig(fig,figName)

# data.to_csv(os.path.join(resPath,'deco_inter.csv'))

# Peak Latency
for cond in recog_label_list:
    deco_sig = deco[
        (deco['type']==cond)&
        (deco['grp_label']>=1)].reset_index(drop=True)
    sig_start,sig_end = deco_sig['time'].values[0],\
                        deco_sig['time'].values[-1]

    peak_indx = argrelextrema(
        deco_sig['acc'].values,np.greater)[0].tolist()[0]
    print(cond)
    print(deco_sig.loc[peak_indx,['time','acc']])
    print(sig_start,sig_end)


# --------------------------------------------------------------------
pred = 'o2r_cate'
deco_subj_all = pd.read_csv(
    os.path.join(resPath,'deco_data_subj.csv'),sep=',')
deco_subj = deco_subj_all[
    (deco_subj_all['chans']==pick_tag)&
    (deco_subj_all['pred']==pred)].reset_index(drop=True)
deco_subj['cond'] = deco_subj['type'].str.split('',expand=True)[1]
deco_subj['setsize'] = deco_subj['type'].str.split('',expand=True)[2]

deco_all = pd.read_csv(
    os.path.join(resPath,'deco_data.csv'),sep=',')
deco = deco_all[
    (deco_all['chans']==pick_tag)&
    (deco_all['pred']==pred)].reset_index(drop=True)
deco['cond'] = deco['type'].str.split('',expand=True)[1]
deco['setsize'] = deco['type'].str.split('',expand=True)[2]

# permutation
def stat_fun(*args):
    factor_levels = [2,1]
    effects = 'A'
    return f_mway_rm(
        np.array(args).transpose(1,0,2),
        factor_levels=factor_levels,
        effects=effects,return_pvals=False)[0]

# cluster-based anova
t0_clu,t1_clu = deco.loc[
    (deco['sig_label']==1)&
    (deco['cond']=='b'),'time'].values[0],deco.loc[
    (deco['sig_label']==1)&
    (deco['cond']=='w'),'time'].values[-1]
erp_recog_t = deco_subj[
    (deco_subj['time']>=t0_clu)&
    (deco_subj['time']<=t1_clu)].reset_index(drop=True)
# erp_recog_t = erp_recog
t = erp_recog_t.loc[(erp_recog_t['subj']==0)&
                    (erp_recog_t['cond']=='w'),
                    'time'].values
erp_arr = np.zeros(
    [subjAllN,len(cond_label_list),len(t)])
for n in range(subjAllN):
    for k,cond in enumerate(cond_label_list):
        erp_arr[n,k,:] = erp_recog_t.loc[
            (erp_recog_t['subj']==n)&
            (erp_recog_t['type']==cond),
            'acc'].values
tail = 0
# pthresh = 0.001
factor_levels = [2,1]
effects = 'A'
# f_thresh = f_threshold_mway_rm(
#     subjAllN_final,factor_levels,effects,p_crit)
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
        grp_start.append(t.tolist()[c[0]][0])
        grp_end.append(t.tolist()[c[0]][-2])
print(grp_start,grp_end)


t0,t1 = deco.loc[
    (deco['sig_label']==1)&
    (deco['cond']=='b'),'time'].values[0],deco.loc[
    (deco['sig_label']==1)&
    (deco['cond']=='w'),'time'].values[0]
t0_end,t1_end = deco.loc[
    (deco['sig_label']==1)&
    (deco['cond']=='b'),'time'].values[-1],deco.loc[
    (deco['sig_label']==1)&
    (deco['cond']=='w'),'time'].values[-1]

acc_w = deco_subj[
    (deco_subj['time']>=t0)&(deco_subj['time']<=t1)&
    (deco_subj['cond']=='w')].groupby(
        ['subj','cond'])['acc'].agg(np.mean).reset_index()
acc_b = deco_subj[
    (deco_subj['time']>=t0)&(deco_subj['time']<=t1)&
    (deco_subj['cond']=='b')].groupby(
        ['subj','cond'])['acc'].agg(np.mean).reset_index()

t_val = pg.ttest(acc_b['acc'].values,acc_w['acc'].values,
                 paired=True,alternative='greater')
print(t0,t1,t0_end,t1_end)
print('b > w')
pg.print_table(t_val)



pred = 'o2r_size'
deco_subj_all = pd.read_csv(
    os.path.join(resPath,'deco_data_subj.csv'),sep=',')
deco_subj = deco_subj_all[
    (deco_subj_all['chans']==pick_tag)&
    (deco_subj_all['pred']==pred)].reset_index(drop=True)
deco_subj['cond'] = deco_subj['type'].str.split('',expand=True)[1]
deco_subj['setsize'] = deco_subj['type'].str.split('',expand=True)[2]

deco_all = pd.read_csv(
    os.path.join(resPath,'deco_data.csv'),sep=',')
deco = deco_all[
    (deco_all['chans']==pick_tag)&
    (deco_all['pred']==pred)].reset_index(drop=True)
deco['cond'] = deco['type'].str.split('',expand=True)[1]
deco['setsize'] = deco['type'].str.split('',expand=True)[2]

# permutation
def stat_fun(*args):
    factor_levels = [4,1]
    effects = 'A'
    return f_mway_rm(
        np.array(args).transpose(1,0,2),
        factor_levels=factor_levels,
        effects=effects,return_pvals=False)[0]

# cluster-based anova
t1_clu,t2_clu,t4_clu,t8_clu = deco.loc[
    (deco['sig_label']==1)&
    (deco['cond']=='1'),'time'].values[0],deco.loc[
    (deco['sig_label']==1)&
    (deco['cond']=='2'),'time'].values[0],deco.loc[
    (deco['sig_label']==1)&
    (deco['cond']=='4'),'time'].values[-1],deco.loc[
    (deco['sig_label']==1)&
    (deco['cond']=='8'),'time'].values[-1]

if t1_clu>=t2_clu:
    t0_clu = t2_clu
else:
    t0_clu = t1_clu
erp_recog_t = deco_subj[
    (deco_subj['time']>=t0_clu)&
    (deco_subj['time']<=t8_clu)].reset_index(drop=True)
# erp_recog_t = erp_recog
t = erp_recog_t.loc[(erp_recog_t['subj']==0)&
                    (erp_recog_t['cond']=='1'),
                    'time'].values
erp_arr = np.zeros(
    [subjAllN,len(sizeList),len(t)])
for n in range(subjAllN):
    for k,cond in enumerate(sizeList):
        erp_arr[n,k,:] = erp_recog_t.loc[
            (erp_recog_t['subj']==n)&
            (erp_recog_t['type']==str(cond)),
            'acc'].values
tail = 0
# pthresh = 0.001
factor_levels = [4,1]
effects = 'A'
# f_thresh = f_threshold_mway_rm(
#     subjAllN_final,factor_levels,effects,p_crit)
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
        grp_start.append(t.tolist()[c[0]][0])
        grp_end.append(t.tolist()[c[0]][-2])
print(grp_start,grp_end)

t0_clu,t1_clu = deco.loc[
    (deco['sig_label']==1)&
    (deco['cond']=='1'),'time'].values[-1],deco.loc[
    (deco['sig_label']==1)&
    (deco['cond']=='8'),'time'].values[-1]

df_size = deco_subj[
    (deco_subj['time']>=t0)&(deco_subj['time']<=t1)]
aov = pg.rm_anova(
        dv='acc',within=['cond'],subject='subj',
        data=df_size,detailed=True,effsize='np2')
pg.print_table(aov)

for t0,t1,cp_tag in (
        (0.1,0.16,'p1'),(0.16,0.2,'n1'),(0.2,0.3,'p2')):
    df_size = deco_subj[
        (deco_subj['time']>=t0)&(deco_subj['time']<t1)].groupby(
        ['subj','cond'])['acc'].agg(np.mean).reset_index()

    aov = pg.rm_anova(
        dv='acc',within=['cond'],subject='subj',
        data=df_size,detailed=True,effsize='np2')

    print(cp_tag)
    pg.print_table(aov)


t1_clu,t2_clu,t4_clu,t8_clu = deco.loc[
    (deco['sig_label']==1)&
    (deco['cond']=='1'),'time'].values[0],deco.loc[
    (deco['sig_label']==1)&
    (deco['cond']=='2'),'time'].values[0],deco.loc[
    (deco['sig_label']==1)&
    (deco['cond']=='4'),'time'].values[0],deco.loc[
    (deco['sig_label']==1)&
    (deco['cond']=='8'),'time'].values[0]
print(t1_clu,t2_clu,t4_clu,t8_clu)
t1_clu,t2_clu,t4_clu,t8_clu = deco.loc[
    (deco['sig_label']==1)&
    (deco['cond']=='1'),'time'].values[-1],deco.loc[
    (deco['sig_label']==1)&
    (deco['cond']=='2'),'time'].values[-1],deco.loc[
    (deco['sig_label']==1)&
    (deco['cond']=='4'),'time'].values[-1],deco.loc[
    (deco['sig_label']==1)&
    (deco['cond']=='8'),'time'].values[-1]
print(t1_clu,t2_clu,t4_clu,t8_clu)