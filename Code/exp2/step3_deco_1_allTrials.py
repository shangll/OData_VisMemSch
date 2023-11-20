#!/usr/bin/env python
#-*-coding:utf-8 -*-

# EXP. 4-3-1 (EEG): Decoding
# Temporal dynamics of non-target processing vary with
# memory set size in different category conditions
# 2022.Jan.18
# linlin.shang@donders.ru.nl


from exp4_config import decoDataPath_all,decoFigPath_all,\
    resPath,subjList,subjAllN,wb_names,targ_names,\
    targ_plt_ord,dict_to_arr,set_filepath,\
    fdN,jobN,scoring,p_crit,chance_crit,n_permutations,\
    tag_savefile,tag_savefig,show_flg
from mne.stats import permutation_cluster_1samp_test, \
    permutation_cluster_test,f_threshold_mway_rm,f_mway_rm
from mne.decoding import cross_val_multiscore,\
     LinearModel,GeneralizingEstimator

import numpy as np
import pandas as pd
from scipy.stats import sem
import scipy.stats
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression

import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns

import os


# --- --- --- --- --- --- --- --- --- Set Global Parameters --- --- --- --- --- --- --- --- --- #

# epoch
fileName = 'deco_data_mean.csv'
fileAllName = 'deco_data_all.csv'

t_list = np.load(file=os.path.join(
    decoDataPath_all,'t_list.npy'),allow_pickle=True)
t_points = len(t_list)

t0_indx,t1_indx = np.where(t_list>=0.1)[0][0],\
                  np.where(t_list<=0.4)[0][-1]
t_indx = t_list[t0_indx:t1_indx]

l_wid = 2
l_wid_acc = 1.5



# --- --- --- --- --- --- --- --- --- Sub-Functions --- --- --- --- --- --- --- --- --- #

def gen_decoding_in(X,y):
    clf = make_pipeline(
        StandardScaler(),
        LinearModel(
            LogisticRegression(
                solver='liblinear')))
    time_gen = GeneralizingEstimator(
        clf,n_jobs=jobN,
        scoring=scoring,
        verbose=True)

    # scores = time_gen.fit(X=X,y=y)
    scores = cross_val_multiscore(
        time_gen,X,y,cv=fdN,n_jobs=jobN)
    mean_score = np.mean(scores,axis=0)
    mean_score_diag = np.diag(mean_score)

    return scores,mean_score,mean_score_diag

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

def clu_permu_1samp_t(acc_data):
    acc_data = acc_data[:,t0_indx:t1_indx]
    tail = 0
    t_obs,clu,clu_p,H0 = permutation_cluster_1samp_test(
        acc_data-chance_crit,n_permutations=n_permutations,
        tail=tail,out_type='indices',verbose=True)
    acc_sig,grp_sig = find_sig(clu,clu_p)
    return acc_sig,grp_sig



# --- --- --- --- --- --- --- --- --- Main Function --- --- --- --- --- --- --- --- --- #

pick_tag = 'eeg'
if pick_tag == 'eeg':
    decoFigPath_ch = set_filepath(decoFigPath_all,'allChans')
    dataFileName = '_dat_all.npy'
    labelFileName = '_lab_all.npy'
elif pick_tag == 'n2pc':
    decoFigPath_ch = set_filepath(decoFigPath_all,'n2pcChans')
    dataFileName = '_dat_n2pc.npy'
    labelFileName = '_lab_n2pc.npy'

acc_df = pd.DataFrame(
    columns=['chans', 'pred', 'type', 'subj', 'time',
             'acc', 'sig_label', 'grp_label'])
acc_df_all = pd.DataFrame(
    columns=['chans','pred','type','subj','time','acc'])

# LOAD FILES --- --- --- --- --- --- --- --- ---
print('*** LOADING FILES ***')

data_dict,labels_dict = dict(),dict()
for cond in targ_names+wb_names:
    data_dict[cond] = np.load(file=os.path.join(
        decoDataPath_all,'%s'%cond.replace('/','')+dataFileName),
        allow_pickle=True)
    labels_dict[cond] = np.load(file=os.path.join(
        decoDataPath_all,'%s'%cond.replace('/','')+labelFileName),
        allow_pickle=True)
    print('CONDITION %s LOADED ***' % cond)
print('*')

print('EACH CONDIION LOADED ***')
print('*')

print('*** GENERALIZATION DECODING ***')

print('2. EACH CONDITION DECODING ***')
acc_mean = dict()
acc_subjAll_dict,mean_scores,sem_scores = \
    dict(),dict(),dict()

for label in targ_names+wb_names:
    acc_subjAll = np.zeros(
        [subjAllN,t_points,t_points])
    acc_subjAll_diag = np.zeros(
        [subjAllN,t_points])

    for n in range(subjAllN):
        scores,acc_subjAll[n],acc_subjAll_diag[n] = \
            gen_decoding_in(data_dict[label][n],
                            labels_dict[label][n])

        acc_df_tmp = pd.DataFrame(
            columns=['chans','pred','type','subj',
                     'time','acc'])
        acc_df_tmp['pred'] = ['each']*t_points
        acc_df_tmp['type'] = [label]*t_points
        acc_df_tmp['subj'] = [n]*t_points
        acc_df_tmp['time'] = t_list
        acc_df_tmp['acc'] = acc_subjAll_diag[n]
        acc_df_tmp['chans'] = [pick_tag]*t_points
        acc_df_all = pd.concat([acc_df_all,acc_df_tmp],
                               axis=0,ignore_index=True)
        print('Subject %d Finished'%n)

    print('Condition %s Finished' % label)
    print('*')
    print('*')

    acc_mean[label] = np.mean(acc_subjAll,axis=0)
    mean_scores[label] = np.diag(acc_mean[label])
    sem_scores[label] = sem(acc_subjAll_diag)
    acc_subjAll_dict[label] = acc_subjAll_diag

    # t-test
    acc_sig,grp_sig = clu_permu_1samp_t(acc_subjAll_diag)
    # save
    acc_df_tmp = pd.DataFrame(
        columns=['chans','pred','type','subj',
                 'time','acc','sig_label','grp_label'])
    acc_df_tmp['pred'] = ['each']*t_points
    acc_df_tmp['type'] = [label]*t_points
    acc_df_tmp['subj'] = ['mean']*t_points
    acc_df_tmp['time'] = t_list
    acc_df_tmp['acc'] = mean_scores[label]
    acc_df_tmp['sig_label'] = acc_sig
    acc_df_tmp['grp_label'] = grp_sig
    acc_df_tmp['chans'] = [pick_tag]*t_points
    acc_df = pd.concat([acc_df,acc_df_tmp],
                       axis=0,ignore_index=True)
acc_subjAll_cond = dict_to_arr(acc_subjAll_dict)
np.save(file=os.path.join(resPath,'deco_acc_all.npy'),
        arr=acc_subjAll_cond)

# acc_subjAll_cond = np.load(file=os.path.join(
#         resPath,'deco_acc_all.npy'),
#         allow_pickle=True)
# data_dict,labels_dict = dict(),dict()
# for cond in targ_names+wb_names:
#     data_dict[cond] = np.load(file=os.path.join(
#         decoDataPath_all,'%s'%cond.replace('/','')+dataFileName),
#         allow_pickle=True)
#     labels_dict[cond] = np.load(file=os.path.join(
#         decoDataPath_all,'%s'%cond.replace('/','')+labelFileName),
#         allow_pickle=True)
#     print('CONDITION %s LOADED ***' % cond)
# print('*')



# plot the full (generalization) matrix
fig,ax = plt.subplots(
    3,4,sharex=True,sharey=True,figsize=(12,9))
ax = ax.ravel()
for indx,label in enumerate(targ_names+wb_names):
    im = ax[indx].imshow(
        acc_mean[label],interpolation='lanczos',
        origin='lower',cmap='RdBu_r',
        extent=t_list[[0,-1,0,-1]],
        vmin=0.,vmax=1.)
    ax[indx].axvline(0,color='k')
    ax[indx].axhline(0,color='k')
    ax[indx].set_title('%s'%(label))
cb_ax = fig.add_axes([1.0,0.1,0.02,0.8])
cbar = fig.colorbar(im,cax=cb_ax)
cbar.set_label('AUC')
fig.supxlabel('Testing Time (s)')
fig.supylabel('Training Time (s)')
plt.suptitle('Temporal Generalization')
plt.tight_layout()

title = 'gen_AUC_subjAvg.png'
if tag_savefig==1:
    fig.savefig(os.path.join(decoFigPath_ch,title),
                bbox_inches='tight',dpi=300)
plt.show(block=show_flg)
plt.close('all')

# plot time-by-time decoding
acc_df_deco = acc_df[
    (acc_df['pred']=='each') &
    (acc_df['subj']=='mean')]
acc_subjAll_cond = dict_to_arr(acc_subjAll_dict)
acc_subjAll_cond = acc_subjAll_cond.astype(np.float64)
# sig_grp_all = []
# for k in [0,2,4,6]:
#     acc_sig, sig_grp = clu_permu_cond(acc_subjAll_cond[k:k+2])
#     sig_grp_all += sig_grp
# acc_df_deco.loc[:,'diff_sig'] = sig_grp_all

def stat_fun(*args):
    factor_levels = [2,4]
    effects = 'A:B'
    return f_mway_rm(
        np.array(args).transpose(1,0,2),
        factor_levels=factor_levels,
        effects=effects,return_pvals=False)[0]
def clu_permu_cond(acc_data):
    acc_data = acc_data[:,:,t0_indx:t1_indx]
    tail = 0
    factor_levels = [2,4]
    effects = 'A:B'
    f_thresh = f_threshold_mway_rm(
        subjAllN,factor_levels,effects,p_crit)
    f_obs,clu,clu_p,H0 = permutation_cluster_test(
        acc_data,stat_fun=stat_fun,threshold=f_thresh,
        tail=tail,n_jobs=None,n_permutations=n_permutations,
        buffer_size=None,out_type='indices')
    acc_sig,grp_sig = find_sig(clu,clu_p)
    print(clu)
    print(clu_p)
    return acc_sig,grp_sig

acc_sig,sig_grp = clu_permu_cond(acc_subjAll_cond[0:8])
acc_df_deco.loc[:,'diff_sig'] = sig_grp*12

sig_df = acc_df_deco[
    (acc_df_deco['sig_label']==1) &
    (acc_df_deco['acc']>chance_crit)]

mpl.rcParams.update({'font.size':26})
clr = ['crimson','deepskyblue']
fig, ax = plt.subplots(
    2,2,sharex=True,sharey=True,figsize=(20,12))
ax = ax.ravel()
for indx,label_list in enumerate(targ_plt_ord):
    ax[indx].axhline(0.5,color='k',linestyle='--')
    ax[indx].axvline(0.0,color='k',linestyle=':')
    for n,label in enumerate(label_list):
        if label[0:2]!='wb':
            ls = '-'
            clr_cond = clr[n]
        else:
            ls = '--'
            clr_cond = 'grey'
        ax[indx].plot(t_list,mean_scores[label],
                      color=clr_cond,linestyle=ls,
                      linewidth=l_wid,label=label[0:2])

        # ymin, ymax = ax[indx].get_ylim()
        ymax = 0.75
        if (label[0]=='w')&(label[1]!='b'):
            ymin_show = 0.47
        elif label[0]=='b':
            ymin_show = 0.475
        elif label[0:2]=='wb':
            ymin_show = 0.48
        for k in set(sig_df['grp_label']):
            sig_times = sig_df.loc[
                (sig_df['grp_label']==k) &
                (sig_df['type']==label),'time']
            ax[indx].plot(sig_times,[ymin_show]*len(sig_times),
                          color=clr_cond,linewidth=l_wid_acc)

        ax[indx].fill_between(t_list,
                              mean_scores[label]-sem_scores[label],
                              mean_scores[label]+sem_scores[label],
                              color=clr_cond,alpha=0.1,
                              edgecolor='none')
        ax[indx].set_xticks(np.arange(-0.2,0.5,0.1))
        ax[indx].set_yticks(np.arange(0.45,0.8,0.15))
        ax[indx].spines['right'].set_visible(False)
        ax[indx].spines['top'].set_visible(False)

    diff_times = acc_df_deco.loc[
        (acc_df_deco['diff_sig']>=1)&
        (acc_df_deco['type']==label),['time','diff_sig']]
    diff_times.reset_index(drop=True,inplace=True)
    if len(diff_times)!=0:
        for k in set(diff_times['diff_sig']):
            ax[indx].fill_betweenx(
                (chance_crit,ymax),diff_times.loc[0,'time'],
                diff_times.loc[len(diff_times)-1,'time'],
                color='grey',alpha=0.3)
    ax[indx].set_title('MMS %s'%(label[-1]))
# h,_ = ax.get_legend_handles_labels()
# ax[indx].legend(h,['within-target','between-target'],
#                 loc='best',ncol=1).set_title('Category')
ax[indx].legend(loc='best',ncol=1)
fig.text(0.5,0,'Time (sec)',ha='center')
fig.text(0,0.5,'AUC',va='center',rotation='vertical')
plt.tight_layout()
title = 'within_recog_AUC_subjAvg.png'
if tag_savefig == 1:
    fig.savefig(os.path.join(decoFigPath_ch,title),
                bbox_inches='tight',dpi=300)
plt.show(block=show_flg)
plt.close('all')

print('EACH CONDITION DECODING FINISHED ***')
print('***')
print('**')
print('*')

# category
def stat_fun_maineff(*args):
    factor_levels = [2,4]
    effects = 'A'
    return f_mway_rm(
        np.array(args).transpose(1,0,2),
        factor_levels=factor_levels,
        effects=effects,return_pvals=False)[0]
tail = 0
factor_levels = [2,4]
effects = 'A'
f_thresh = f_threshold_mway_rm(
    subjAllN,factor_levels,effects,p_crit)
f_obs,clu,clu_p,H0 = permutation_cluster_test(
    acc_subjAll_cond[0:8],stat_fun=stat_fun_maineff,
    threshold=f_thresh,
    tail=tail,n_jobs=None,n_permutations=n_permutations,
    buffer_size=None,out_type='indices')
acc_sig,grp_sig = find_sig(clu,clu_p)
print(clu)
print(clu_p)

grp_start_cate,grp_end_cate = [],[]
for c,p in zip(clu,clu_p):
    if p<p_crit:
        grp_start_cate.append(t_list[c[0]][0])
        grp_end_cate.append(t_list[c[0]][-2])
print(grp_start_cate,grp_end_cate)
# set size
def stat_fun_maineff(*args):
    factor_levels = [2,4]
    effects = 'B'
    return f_mway_rm(
        np.array(args).transpose(1,0,2),
        factor_levels=factor_levels,
        effects=effects,return_pvals=False)[0]
tail = 0
factor_levels = [2,4]
effects = 'B'
f_thresh = f_threshold_mway_rm(
    subjAllN,factor_levels,effects,p_crit)
f_obs,clu,clu_p,H0 = permutation_cluster_test(
    acc_subjAll_cond[0:8,:,t0_indx:t1_indx],
    stat_fun=stat_fun_maineff,
    threshold=f_thresh,
    tail=tail,n_jobs=None,n_permutations=n_permutations,
    buffer_size=None,out_type='indices')
acc_sig,grp_sig = find_sig(clu,clu_p)
print(clu)
print(clu_p)

grp_start_size,grp_end_size = [],[]
for c,p in zip(clu,clu_p):
    if p<p_crit:
        grp_start_size.append(t_indx[c[0]][0])
        grp_end_size.append(t_indx[c[0]][-2])
print(grp_start_size,grp_end_size)

print('MAIN EFFECT DECODING FINISHED ***')



print('3. EACH CATEGORY DECODING ***')
print('*** LOADING FILES ***')

data_dict,labels_dict = dict(),dict()
for cond in ['wt','bt']:
    data_dict[cond] = np.load(file=os.path.join(
        decoDataPath_all,'cate_%s'%cond.replace('/','')+dataFileName),
        allow_pickle=True)
    labels_dict[cond] = np.load(file=os.path.join(
        decoDataPath_all,'cate_%s'%cond.replace('/','')+labelFileName),
        allow_pickle=True)
    print('CONDITION %s LOADED ***'%cond)
print('*')

print('LOADED ***')
print('*')

acc_mean = dict()
acc_subjAll_dict,mean_scores,sem_scores =\
    dict(),dict(),dict()

for label in ['wt','bt']:
    acc_subjAll = np.zeros(
        [subjAllN,t_points,t_points])
    acc_subjAll_diag = np.zeros(
        [subjAllN,t_points])

    for n in range(subjAllN):
        scores,acc_subjAll[n],acc_subjAll_diag[n] =\
            gen_decoding_in(data_dict[label][n],
                            labels_dict[label][n])
        acc_df_tmp = pd.DataFrame(
            columns=['chans','pred','type','subj',
                     'time','acc'])
        acc_df_tmp['pred'] = ['cate']*t_points
        acc_df_tmp['type'] = [label]*t_points
        acc_df_tmp['subj'] = [n]*t_points
        acc_df_tmp['time'] = t_list
        acc_df_tmp['acc'] = acc_subjAll_diag[n]
        acc_df_tmp['chans'] = [pick_tag]*t_points
        acc_df_all = pd.concat([acc_df_all,acc_df_tmp],
                               axis=0,ignore_index=True)
        print('Subject %d Finished'%n)


    print('Condition %s Finished'%label)
    print('*')
    print('*')

    acc_mean[label] = np.mean(acc_subjAll,axis=0)
    mean_scores[label] = np.diag(acc_mean[label])
    sem_scores[label] = sem(acc_subjAll_diag)
    acc_subjAll_dict[label] = acc_subjAll_diag

    # t-test
    acc_sig,grp_sig = clu_permu_1samp_t(acc_subjAll_diag)
    # save
    acc_df_tmp = pd.DataFrame(
        columns=['chans','pred','type','subj',
                 'time','acc','sig_label','grp_label'])
    acc_df_tmp['pred'] = ['cate']*t_points
    acc_df_tmp['type'] = [label]*t_points
    acc_df_tmp['subj'] = ['mean']*t_points
    acc_df_tmp['time'] = t_list
    acc_df_tmp['acc'] = mean_scores[label]
    acc_df_tmp['sig_label'] = acc_sig
    acc_df_tmp['grp_label'] = grp_sig
    acc_df_tmp['chans'] = [pick_tag]*t_points
    acc_df = pd.concat([acc_df,acc_df_tmp],
                       axis=0,ignore_index=True)

# plot the full (generalization) matrix
fig,ax = plt.subplots(
    1,1,sharex=True,sharey=True,figsize=(12,9))
for indx,label in enumerate(['wt','bt']):
    im = ax.imshow(
        acc_mean[label],interpolation='lanczos',
        origin='lower',cmap='RdBu_r',
        extent=t_list[[0,-1,0,-1]],
        vmin=0.,vmax=1.)
    ax.axvline(0,color='k')
    ax.axhline(0,color='k')
    ax.set_title('wt vs bt')
cb_ax = fig.add_axes([1.0,0.1,0.02,0.8])
cbar = fig.colorbar(im,cax=cb_ax)
cbar.set_label('AUC')
fig.supxlabel('Testing Time (s)')
fig.supylabel('Training Time (s)')
plt.suptitle('Temporal Generalization')
plt.tight_layout()

title = 'gen_AUC_subjAvg_cate.png'
if tag_savefig==1:
    fig.savefig(os.path.join(decoFigPath_ch,title),
                bbox_inches='tight',dpi=300)
plt.show(block=show_flg)
plt.close('all')

# plot time-by-time decoding
acc_df_deco = acc_df[
    (acc_df['pred']=='cate')&
    (acc_df['subj']=='mean')]
acc_subjAll_cond = dict_to_arr(acc_subjAll_dict)
acc_subjAll_cond = acc_subjAll_cond.astype(np.float64)

tail = 0
t_obs,clu,clu_p,H0 = permutation_cluster_1samp_test(
    acc_subjAll_cond[0,:,t0_indx:t1_indx]-acc_subjAll_cond[1,:,t0_indx:t1_indx],
    n_permutations=n_permutations,
    tail=tail,out_type='indices',verbose=True)
print(clu)
print(clu_p)
grp_start_cate,grp_end_cate = [],[]
for c,p in zip(clu,clu_p):
    if p<p_crit:
        grp_start_cate.append(t_indx[c[0]][0])
        grp_end_cate.append(t_indx[c[0]][-1])
print(grp_start_cate,grp_end_cate)

acc_sig,grp_sig = find_sig(clu,clu_p)
acc_df_deco.loc[:,'diff_sig'] = acc_sig*2
sig_df = acc_df_deco[
    (acc_df_deco['sig_label']==1) &
    (acc_df_deco['acc']>chance_crit)]

mpl.rcParams.update({'font.size':26})
clr = ['crimson','deepskyblue']
fig, ax = plt.subplots(
    1,1,sharex=True,sharey=True,figsize=(20,12))
for n,label in enumerate(['wt','bt']):
    ax.axhline(0.5,color='k',linestyle='--')
    ax.axvline(0.0,color='k',linestyle=':')
    if label[0:2]!='wb':
        ls = '-'
        clr_cond = clr[n]
    else:
        ls = '--'
        clr_cond = 'grey'
    ax.plot(t_list,mean_scores[label],
            color=clr_cond,linestyle=ls,
            linewidth=l_wid,label=label)

    # ymin, ymax = ax[indx].get_ylim()
    ymax = 0.75
    if (label[0]=='w')&(label[1]!='b'):
        ymin_show = 0.47
    elif label[0]=='b':
        ymin_show = 0.475
    elif label[0:2]=='wb':
        ymin_show = 0.48
    for k in set(sig_df['grp_label']):
        sig_times = sig_df.loc[
            (sig_df['grp_label']==k) &
            (sig_df['type']==label),'time']
        ax.plot(sig_times,[ymin_show]*len(sig_times),
                color=clr_cond,linewidth=l_wid_acc)

    ax.fill_between(t_list,
                    mean_scores[label]-sem_scores[label],
                    mean_scores[label]+sem_scores[label],
                    color=clr_cond,alpha=0.1,
                    edgecolor='none')
    ax.set_xticks(np.arange(-0.2,0.5,0.1))
    ax.set_yticks(np.arange(0.45,0.8,0.15))
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)

diff_times = acc_df_deco.loc[
    (acc_df_deco['diff_sig']>=1)&
    (acc_df_deco['type']==label),['time','diff_sig']]
diff_times.reset_index(drop=True,inplace=True)
if len(diff_times)!=0:
    for k in set(diff_times['diff_sig']):
        ax.fill_betweenx(
            (chance_crit,ymax),diff_times.loc[0,'time'],
            diff_times.loc[len(diff_times)-1,'time'],
            color='grey',alpha=0.3)
ax.set_title('Category Effect')
# h,_ = ax.get_legend_handles_labels()
# ax[indx].legend(h,['within-target','between-target'],
#                 loc='best',ncol=1).set_title('Category')
ax.legend(loc='best',ncol=1)
fig.text(0.5,0,'Time (sec)',ha='center')
fig.text(0,0.5,'AUC',va='center',rotation='vertical')
plt.tight_layout()
title = 'within_recog_AUC_subjAvg_cate.png'
if tag_savefig == 1:
    fig.savefig(os.path.join(decoFigPath_ch,title),
                bbox_inches='tight',dpi=300)
plt.show(block=show_flg)
plt.close('all')

print('CATEGORY CONDITION DECODING FINISHED ***')
print('***')
print('**')
print('*')

print('4. SET SIZE CONDITION DECODING ***')
print('*** LOADING FILES ***')

data_dict,labels_dict = dict(),dict()
for cond in ['1','2','4','8']:
    data_dict[cond] = np.load(file=os.path.join(
        decoDataPath_all,'size_%s'%cond.replace('/','')+dataFileName),
        allow_pickle=True)
    labels_dict[cond] = np.load(file=os.path.join(
        decoDataPath_all,'size_%s'%cond.replace('/','')+labelFileName),
        allow_pickle=True)
    print('CONDITION %s LOADED ***'%cond)
print('*')

print('LOADED ***')
print('*')

acc_mean = dict()
acc_subjAll_dict,mean_scores,sem_scores =\
    dict(),dict(),dict()

for label in ['1','2','4','8']:
    acc_subjAll = np.zeros(
        [subjAllN,t_points,t_points])
    acc_subjAll_diag = np.zeros(
        [subjAllN,t_points])

    for n in range(subjAllN):
        scores,acc_subjAll[n],acc_subjAll_diag[n] =\
            gen_decoding_in(data_dict[label][n],
                            labels_dict[label][n])
        acc_df_tmp = pd.DataFrame(
            columns=['chans','pred','type','subj',
                     'time','acc'])
        acc_df_tmp['pred'] = ['size']*t_points
        acc_df_tmp['type'] = [label]*t_points
        acc_df_tmp['subj'] = [n]*t_points
        acc_df_tmp['time'] = t_list
        acc_df_tmp['acc'] = acc_subjAll_diag[n]
        acc_df_tmp['chans'] = [pick_tag]*t_points
        acc_df_all = pd.concat([acc_df_all,acc_df_tmp],
                               axis=0,ignore_index=True)
        print('Subject %d Finished'%n)

    np.save(file=os.path.join(
        decoDataPath_all,
        'in_auc_recog_%s'%label.replace('/','')+dataFileName),
        arr=acc_subjAll)

    print('Condition %s Finished'%label)
    print('*')
    print('*')

    acc_mean[label] = np.mean(acc_subjAll,axis=0)
    mean_scores[label] = np.diag(acc_mean[label])
    sem_scores[label] = sem(acc_subjAll_diag)
    acc_subjAll_dict[label] = acc_subjAll_diag

    # t-test
    acc_sig,grp_sig = clu_permu_1samp_t(acc_subjAll_diag)
    # save
    acc_df_tmp = pd.DataFrame(
        columns=['chans','pred','type','subj',
                 'time','acc','sig_label','grp_label'])
    acc_df_tmp['pred'] = ['size']*t_points
    acc_df_tmp['type'] = [label]*t_points
    acc_df_tmp['subj'] = ['mean']*t_points
    acc_df_tmp['time'] = t_list
    acc_df_tmp['acc'] = mean_scores[label]
    acc_df_tmp['sig_label'] = acc_sig
    acc_df_tmp['grp_label'] = grp_sig
    acc_df_tmp['chans'] = [pick_tag]*t_points
    acc_df = pd.concat([acc_df,acc_df_tmp],
                       axis=0,ignore_index=True)

# plot the full (generalization) matrix
fig,ax = plt.subplots(
    1,1,sharex=True,sharey=True,figsize=(12,9))
for indx,label in enumerate(['1','2','4','8']):
    im = ax.imshow(
        acc_mean[label],interpolation='lanczos',
        origin='lower',cmap='RdBu_r',
        extent=t_list[[0,-1,0,-1]],
        vmin=0.,vmax=1.)
    ax.axvline(0,color='k')
    ax.axhline(0,color='k')
    ax.set_title('MSS')
cb_ax = fig.add_axes([1.0,0.1,0.02,0.8])
cbar = fig.colorbar(im,cax=cb_ax)
cbar.set_label('AUC')
fig.supxlabel('Testing Time (s)')
fig.supylabel('Training Time (s)')
plt.suptitle('Temporal Generalization')
plt.tight_layout()

title = 'gen_AUC_subjAvg_size.png'
if tag_savefig==1:
    fig.savefig(os.path.join(decoFigPath_ch,title),
                bbox_inches='tight',dpi=300)
plt.show(block=show_flg)
plt.close('all')

# plot time-by-time decoding
acc_df_deco = acc_df[
    (acc_df['pred']=='size')&
    (acc_df['subj']=='mean')]
acc_subjAll_cond = dict_to_arr(acc_subjAll_dict)
acc_subjAll_cond = acc_subjAll_cond.astype(np.float64)


def stat_fun(*args):
    factor_levels = [4,1]
    effects = 'A'
    return f_mway_rm(
        np.array(args).transpose(1,0,2),
        factor_levels=factor_levels,
        effects=effects,return_pvals=False)[0]


tail = 0
factor_levels = [4,1]
effects = 'A'
f_thresh = f_threshold_mway_rm(
    subjAllN,factor_levels,effects,p_crit)
f_obs,clu,clu_p,H0 = permutation_cluster_test(
    acc_subjAll_cond[:,:,t0_indx:t1_indx],stat_fun=stat_fun,threshold=f_thresh,
    tail=tail,n_jobs=None,n_permutations=n_permutations,
    buffer_size=None,out_type='indices')
acc_sig,grp_sig = find_sig(clu,clu_p)
print(clu)
print(clu_p)

grp_start_size,grp_end_size = [],[]
for c,p in zip(clu,clu_p):
    if p<p_crit:
        grp_start_size.append(t_indx[c[0]][0])
        grp_end_size.append(t_indx[c[0]][-2])
print(grp_start_size,grp_end_size)

acc_df_deco.loc[:,'diff_sig'] = acc_sig*4

sig_df = acc_df_deco[
    (acc_df_deco['sig_label']==1)&
    (acc_df_deco['acc']>chance_crit)]

mpl.rcParams.update({'font.size':26})
clrs_all_b = sns.color_palette('Blues',n_colors=35)
clrs_all = sns.color_palette('GnBu_d')
clr = [clrs_all_b[28],clrs_all_b[20],clrs_all[2],clrs_all[0]]
fig,ax = plt.subplots(
    1,1,sharex=True,sharey=True,figsize=(20,12))
for n,label in enumerate(['1','2','4','8']):
    ax.axhline(0.5,color='k',linestyle='--')
    ax.axvline(0.0,color='k',linestyle=':')
    ls = '-'
    clr_cond = clr[n]
    ax.plot(t_list,mean_scores[label],
            color=clr_cond,linestyle=ls,
            linewidth=l_wid,label=label)
    ymax = 0.75
    if label=='1':
        ymin_show = 0.48
    elif label=='2':
        ymin_show = 0.475
    elif label=='4':
        ymin_show = 0.47
    elif label=='8':
        ymin_show = 0.465
    for k in set(sig_df['grp_label']):
        sig_times = sig_df.loc[
            (sig_df['grp_label']==k)&
            (sig_df['type']==label),'time']
        ax.plot(sig_times,[ymin_show]*len(sig_times),
                color=clr_cond,linewidth=l_wid_acc)
    ax.fill_between(t_list,
                    mean_scores[label]-sem_scores[label],
                    mean_scores[label]+sem_scores[label],
                    color=clr_cond,alpha=0.1,
                    edgecolor='none')
ax.set_xticks(np.arange(-0.2,0.5,0.1))
ax.set_yticks(np.arange(0.45,0.8,0.15))
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)

diff_times = acc_df_deco.loc[
    (acc_df_deco['diff_sig']>=1)&
    (acc_df_deco['type']==label),['time','diff_sig']]
diff_times.reset_index(drop=True,inplace=True)
if len(diff_times)!=0:
    for k in set(diff_times['diff_sig']):
        ax.fill_betweenx(
            (chance_crit,ymax),diff_times.loc[0,'time'],
            diff_times.loc[len(diff_times)-1,'time'],
            color='grey',alpha=0.3)
ax.set_title('Memory Set Size Effect')
# h,_ = ax.get_legend_handles_labels()
# ax[indx].legend(h,['within-target','between-target'],
#                 loc='best',ncol=1).set_title('Category')
ax.legend(loc='best',ncol=1)
fig.text(0.5,0,'Time (sec)',ha='center')
fig.text(0,0.5,'AUC',va='center',rotation='vertical')
plt.tight_layout()
title = 'within_recog_AUC_subjAvg_size.png'
if tag_savefig==1:
    fig.savefig(os.path.join(decoFigPath_ch,title),
                bbox_inches='tight',dpi=300)
plt.show(block=show_flg)
plt.close('all')

print('SET SIZE CONDITION DECODING FINISHED ***')
print('***')
print('**')
print('*')



# save ACC files
if tag_savefile==1:
    acc_df.to_csv(os.path.join(resPath,fileName),
                  mode='w',header=True,index=False)
if tag_savefile==1:
    acc_df_all.to_csv(os.path.join(resPath,fileAllName),
                      mode='w',header=True,index=False)
print('%s FINISHED ***'%pick_tag)


print('*** GENERALIZATION DECODING FINISHED ***')
print('***')
print('***')
print('***')
print('')
