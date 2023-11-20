#!/usr/bin/env python
# -*-coding:utf-8 -*-

# EXP. 3-3.1 (EEG): evoke
# Temporal dynamics of non-target processing vary with
# memory set size in different category conditions
# 2021.Nov.24
# linlin.shang@donders.ru.nl



#%% --- --- --- --- --- --- libraries --- --- --- --- --- ---

from eeg_config import epoDataPath,resPath,glmFigPath,\
    subjList_final,subjAllN_final,subjList,subjAllN,\
    comp_list,condList,\
    cond_label_list,recog_label_list,sizeList,recog_labels,\
    postChans,frontChans,n_permutations,chance_crit,p_crit,\
    save_fig,set_filepath
import mne
from mne.stats import permutation_cluster_1samp_test, \
    permutation_cluster_test,f_threshold_mway_rm,f_mway_rm
import os
import numpy as np
import pandas as pd
import pingouin as pg
import scipy.stats
import statsmodels.api as sm
from statsmodels.formula.api import glm
from sklearn import preprocessing
import seaborn as sns
import statannot

import matplotlib.pyplot as plt


def find_sig(clu,clu_p,t_points):
    acc_sig, grp_sig = t_points*[0],t_points*[0]
    grp_label = 0

    for c,p in zip(clu, clu_p):
        if p <= p_crit:
            grp_label += 1
            acc_sig[c[0][0]:(c[0][-1]+1)] = \
                [1]*len(c[0])
            grp_sig[c[0][0]:(c[0][-1]+1)] = \
                [grp_label]*len(c[0])
    return acc_sig, grp_sig

def stat_fun(*args):
    factor_levels = [4,2]
    effects = 'A:B'
    return f_mway_rm(
        np.array(args),factor_levels=factor_levels,
        effects=effects,return_pvals=False)[0]

def clu_permu_cond(erp_data):
    tail = 0
    # pthresh = 0.001
    factor_levels = [4,2]
    effects = 'A:B'
    # f_thresh = f_threshold_mway_rm(
    #     subjAllN_final,factor_levels,effects,p_crit)
    f_thresh = f_threshold_mway_rm(
        subjAllN,factor_levels,effects,p_crit)

    f_obs,clu,clu_p,h0 = permutation_cluster_test(
        erp_data,stat_fun=stat_fun,threshold=f_thresh,
        tail=tail,n_jobs=None,
        n_permutations=n_permutations,
        buffer_size=None,out_type='indices')
    print(clu)
    print(clu_p)

    acc_sig, grp_sig = find_sig(clu,clu_p,erp_data.shape[-1])
    return acc_sig, grp_sig

def clu_permu_1samp_t(acc_data):
    threshold = None
    tail = 0
    degrees_of_freedom = len(acc_data) - 1
    t_thresh = scipy.stats.t.ppf(
        1 - p_crit / 2, df=degrees_of_freedom)

    t_obs, clu, clu_p, H0 = permutation_cluster_1samp_test(
        acc_data-chance_crit,n_permutations=n_permutations,
        threshold=t_thresh, tail=tail,
        out_type='indices', verbose=True)
    print(clu)
    print(clu_p)

    acc_sig = find_sig(clu,clu_p,len(acc_data))
    return acc_sig

#%%  --- --- --- --- --- --- Main Function --- --- --- --- --- --- #

# --- --- ---
# # post
# pick_tag = 'post'
# picks = postChans
# --- --- ---
# # PO7/8
# pick_tag = 'po78'
# picks = ['PO7','PO8']
# --- --- ---
# # cluster
# pick_tag = 'clu_post'
# clu1 = ['P5','P7','P8',
#         'PO3','PO7','PO9',
#         'O1']
# clu2 = ['AF3','AF4','AF7','AF8',
#         'F1','F3','F5','F7',
#         'FC5']
# clu3 = ['P6','P8',
#         'PO4','PO8',
#         'TP8']
# clu4 = ['P1','P2','P4','Pz',
#         'PO3','PO4','POz',
#         'O1','O2','Oz']
# clu5 = ['AF3',
#         'F2','F3','Fz',
#         'FC3','FC5','FCz',
#         'C3']
# clu6 = ['C5','CP5']
# picks = clu1+clu3
# picks.remove('TP8')
# --- --- ---
# similar pattern
pick_tag = 'simi'
picks = ['P5','P6','P7','P8',
         'PO3','PO4','PO7','PO8','PO9','PO10',
         'O1','O2']
# --- --- ---
# # top-3
# pick_tag = 'top3'
# top3_data = pd.read_csv(
#     os.path.join(resPath,'peak_3elec_info.csv'),
#     sep=',')
# --- --- ---

glmFigPath = set_filepath(glmFigPath,'%s'%pick_tag)

erp_stats_all = pd.DataFrame()
df_glm = pd.DataFrame()
glm_subj,glm_cond,glm_coeff = [],[],[]
# for subjN in subjList_final:
for subjN in subjList:
    # times = top3_data.loc[
    #     (top3_data['subj']==subjN)&
    #     (top3_data['comp']=='p2'),
    #     'time'].tolist()

    # --- --- ---
    # # top-3
    # picks = top3_data.loc[
    #     (top3_data['subj']==subjN)&
    #     (top3_data['comp']=='p2'),
    #     'chan'].tolist()
    # --- --- ---

    # cp_tag = 'p1'
    # t0,t1 = 0.1,0.16
    cp_tag = 'n1'
    t0,t1 = 0.16,0.2
    # t0,t1 = 0.196,0.264
    # t0,t1 = 0.196,0.232

    # cp_tag = 'p2'
    # t0,t1 = 0.2,0.3
    # t0,t1 = 0.2,0.25
    # t0,t1 = 0.25,0.3

    # t0,t1 = 0.172,0.456
    # t0,t1 = 0.1,0.28 # top-3 inter: 0.049
    # t0,t1 = 0.19,0.26 # top-3 inter: 0.043, simi: 0.053
    # t0,t1 = 0.2,0.25  # po7/8
    # t0,t1 = 0.24,0.29
    # t0,t1 = 0.192,0.264

    print('SUBJECT %d STARTS'%subjN)

    fname = 'subj%d_epo.fif' % subjN
    subj_epo_all = mne.read_epochs(
        os.path.join(epoDataPath,fname))
    subj_epo = subj_epo_all[recog_labels]
    df_epo = subj_epo.copy().crop(
        tmin=t0,tmax=t1).to_data_frame()
    df_epo[['cond','img_cate','setsize']] = \
        df_epo['condition'].str.split('/',n=2,expand=True)
    df_epo['setsize'] = df_epo['setsize'].astype('int')
    # fit_tag = 'lm'
    fit_tag = 'log'
    # df_epo['setsize'] = df_epo['setsize'].apply(np.log2)
    df_subj = df_epo[['time','epoch','cond','setsize']+picks]
    df_subj = df_subj.copy()
    df_subj['cond_trans'] = np.where(df_subj['cond']=='w',1,-1)
    df_subj[pick_tag] = df_subj[picks].mean(axis=1).values

    erp_stats = df_subj.groupby(
        ['epoch','cond_trans','setsize'])[
        pick_tag].agg(np.mean).reset_index()
    erp_stats.dropna(inplace=True)
    erp_stats[pick_tag+'_Z'] = preprocessing.scale(
        erp_stats[pick_tag].values)

    erp_stats['cond_Z'] = preprocessing.scale(
        erp_stats['cond_trans'].values)

    erp_stats['setsize_Z'] = preprocessing.scale(
        erp_stats['setsize'].values)

    erp_stats['inter'] = erp_stats['cond_Z']*erp_stats['setsize']
    erp_stats['inter_Z'] = preprocessing.scale(erp_stats['inter'].values)
    erp_stats['subj'] = [subjN]*len(erp_stats)

    erp_stats_all = pd.concat(
        [erp_stats_all,erp_stats],axis=0,ignore_index=True)

    # model = glm(
    #     formula=pick_tag+"_Z"+"~C(cond_Z)+setsize_Z+C(cond_Z)*setsize_Z",
    #     data=erp_stats).fit()
    # print(model.summary())

    y = erp_stats[pick_tag+'_Z']
    X = erp_stats[['cond_Z','setsize_Z','inter_Z']]
    X = sm.add_constant(X)
    model = sm.GLM(y,X,family=sm.families.Gaussian()).fit()
    # print(model.summary())

    glm_subj += [subjN]*4
    glm_cond.append('intc')
    glm_coeff.append(model.params[0])
    glm_cond.append('cond')
    glm_coeff.append(model.params[1])
    glm_cond.append('setsize')
    glm_coeff.append(model.params[2])
    glm_cond.append('inter')
    glm_coeff.append(model.params[3])

    model_cond = glm(
        formula=pick_tag+"_Z"+"~C(cond_Z):setsize_Z",
        data=erp_stats).fit()

df_glm['subj'] = glm_subj
df_glm['cond'] = glm_cond
df_glm['coeff'] = glm_coeff
df_glm['cp'] = [cp_tag]*len(df_glm)
df_glm['fit'] = [fit_tag]*len(df_glm)

if cp_tag=='p1':
    tailed = 'greater'
else:
    tailed = 'less'
# tailed = 'two-sided'
pd.set_option('display.max_columns',None)
print('category effect')
res = pg.ttest(
    df_glm.loc[df_glm['cond']=='cond','coeff'].values,0,
    alternative=tailed,correction='auto')
pg.print_table(res)
print('setsize effect')
res = pg.ttest(
    df_glm.loc[df_glm['cond']=='setsize','coeff'].values,0,
    alternative=tailed,correction='auto')
pg.print_table(res)
print('interaction')
res = pg.ttest(
    df_glm.loc[df_glm['cond']=='inter','coeff'].values,0,
    alternative='less',correction='auto')
pg.print_table(res)

# glm_file = os.path.join(resPath,'glm_coeff.csv')
# if os.path.isfile(glm_file):
#     df_glm.to_csv(glm_file,sep=',',mode='a',header=False,index=False)
# else:
#     df_glm.to_csv(glm_file,sep=',',mode='w',header=True,index=False)

# plot
fig,ax = plt.subplots(figsize=(9,6))
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
a_cate = df_glm.loc[df_glm['cond']=='cond','coeff'].mean(axis=0)
a_size = df_glm.loc[df_glm['cond']=='setsize','coeff'].mean(axis=0)
a_inter = df_glm.loc[df_glm['cond']=='inter','coeff'].mean(axis=0)

b = df_glm.loc[df_glm['cond']=='intc','coeff'].mean(axis=0)

x_cate_w = np.array([1]*4)
x_cate_b = np.array([-1]*4)
if fit_tag=='log':
    x_size = preprocessing.scale(np.log2(sizeList))
else:
    x_size = np.array(sizeList)
x_inter_w = x_cate_w*x_size
x_inter_b = x_cate_b*x_size
y_w = a_cate*x_cate_w+a_size*x_size+a_inter*x_inter_w+b
y_b = a_cate*x_cate_b+a_size*x_size+a_inter*x_inter_b+b
plt.plot(x_size,y_w,marker='^',color='orange',label='w')
plt.plot(x_size,y_b,marker='o',color='green',label='b')
if fit_tag=='log':
    plt.xticks(x_size,np.log2(sizeList))
    plt.xlabel(xlabel='Memory set size (Logarithmical scale)')
else:
    plt.xticks(sizeList)
    plt.xlabel(xlabel='Memory set size')
plt.ylabel(ylabel='Amplitude (Z-score)')
plt.legend(loc='best',ncol=1)
plt.grid(linestyle =':')
plt.legend(loc='best')
fig.tight_layout()
figName = os.path.join(
    glmFigPath,
    'stats_glm_cond_%s_%s_%s'%(pick_tag,fit_tag,cp_tag))
save_fig(fig,figName)


'''
# each top-3
erp_stats_all = pd.DataFrame()
df_glm = pd.DataFrame()
glm_subj,glm_cate,glm_coeff = [],[],[]
for subjN in subjList_final:
    picks = top3_data.loc[
        (top3_data['subj']==subjN)&
        (top3_data['comp']=='p2'),
        'chan'].tolist()
    times = top3_data.loc[
        (top3_data['subj']==subjN)&
        (top3_data['comp']=='p2'),
        'time'].tolist()
    pick_tag = 'top3'

    print('SUBJECT %d STARTS'%subjN)

    t_space = 0.05

    fname = 'subj%d_epo.fif' % subjN
    subj_epo_all = mne.read_epochs(
        os.path.join(epoDataPath,fname))
    subj_epo = subj_epo_all[recog_labels]
    df_epo = subj_epo.copy().to_data_frame()
    df_epo[['cond','img_cate','setsize']] = \
        df_epo['condition'].str.split('/',n=2,expand=True)
    df_epo['setsize'] = df_epo['setsize'].astype('int')
    df_subj_epo = df_epo[['time','epoch','cond','setsize']+picks]
    df_subj_epo = df_subj_epo.copy()
    df_subj_epo['cond_Z'] = np.where(df_subj_epo['cond']=='w',1,-1)
    df_subj = pd.DataFrame()
    for n,chan in enumerate(picks):
        t0 = times[0]-t_space
        t1 = times[0]+t_space
        chan_time = df_subj_epo[(df_subj_epo['time']<=t1)&(
                df_subj_epo['time']>=t0)].groupby(
            ['epoch','cond_Z','setsize'])[
            chan].agg(np.mean).reset_index()
        if n==0:
            df_subj = chan_time
        else:
            df_subj = pd.merge(df_subj,chan_time,on=['epoch','cond_Z','setsize'])
    df_subj[pick_tag] = df_subj[picks].mean(axis=1).values

    erp_stats = df_subj.groupby(
        ['epoch','cond_Z','setsize'])[
        pick_tag].agg(np.mean).reset_index()
    erp_stats.dropna(inplace=True)
    x = preprocessing.scale(erp_stats[pick_tag].values)
    erp_stats[pick_tag+'_Z'] = x
    erp_stats['subj'] = [subjN]*len(erp_stats)

    erp_stats_all = pd.concat(
        [erp_stats_all,erp_stats],axis=0,ignore_index=True)

    model = glm(
        formula=pick_tag+"_Z"+"~C(cond_Z)+setsize+C(cond_Z)*setsize",
        data=erp_stats).fit()
    # print(grp_model.summary())

    model_cond = glm(
        formula=pick_tag+"_Z"+"~C(cond_Z):setsize",
        data=erp_stats).fit()
    # print(grp_model.summary())

    glm_subj += [subjN]*3
    glm_cate.append('cond')
    glm_coeff.append(model.params[1])
    glm_cate.append('setsize')
    glm_coeff.append(model.params[2])
    glm_cate.append('inter')
    glm_coeff.append(model.params[3])
df_glm['subj'] = glm_subj
df_glm['cond'] = glm_cate
df_glm['setsize'] = glm_cate
df_glm['coeff'] = glm_coeff

pd.set_option('display.max_columns',None)
print('category effect')
res = pg.ttest(df_glm.loc[df_glm['cond']=='cond','coeff'].values,0)
print(res)
print('setsize effect')
res = pg.ttest(df_glm.loc[df_glm['cond']=='setsize','coeff'].values,0)
print(res)
print('interaction')
res = pg.ttest(df_glm.loc[df_glm['cond']=='inter','coeff'].values,0)
print(res)

palette = sns.xkcd_palette(['faded green','amber'])
sns.set_style('whitegrid')
fig = sns.lmplot(y=pick_tag+'_Z',x='setsize',
                 hue='cond_Z',data=erp_stats_all,
                 ci=95,palette=palette,
                 x_estimator= np.mean,
                 markers=['o','^'],
                 legend=True,
                 legend_out=False,)
fig.fig.set_size_inches(9,6)
plt.xticks(sizeList)
plt.xlabel('Memory Set Size')
plt.ylabel('Amplitude (Z-scores)')
# plt.legend(labels=['between','within'])
fig.tight_layout()
figName = os.path.join(glmFigPath,'stats_glm_fit_%s_z')%pick_tag
save_fig(fig,figName)

palette = sns.xkcd_palette(['faded green','amber'])
sns.set_style('whitegrid')
fig = sns.lmplot(y=pick_tag,x='setsize',
                 hue='cond_Z',data=erp_stats_all,
                 ci=95,palette=palette,
                 x_estimator= np.mean,
                 markers=['o','^'],
                 legend=True,
                 legend_out=False,)
fig.fig.set_size_inches(9,6)
plt.xticks(sizeList)
plt.xlabel('Memory Set Size')
plt.ylabel('Amplitude')
# plt.legend(labels=['between','within'])
fig.tight_layout()
figName = os.path.join(glmFigPath,'stats_glm_fit_%s_amp')%pick_tag
save_fig(fig,figName)
erp_stats_all.to_csv(
    os.path.join(resPath,'erp_stats_top3.csv'),
    sep=',')
'''