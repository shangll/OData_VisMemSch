#!/usr/bin/env python
# -*-coding:utf-8 -*-

# EXP. 3-2 (EEG): Decoding
# Temporal dynamics of non-target processing vary with
# memory set size in different category conditions
# 2022.Jan.18
# linlin.shang@donders.ru.nl


#%% --- --- --- --- --- --- libraries --- --- --- --- --- ---
from eeg_config import decoDataPath,allResFigPath,\
    decoFigPath,subjAllN,cond_label_list,sizeList,\
    save_fig,tag_savefig,resPath,recog_label_list,\
    chance_crit,p_crit,p_show,n_permutations,\
    set_filepath,dict_to_arr

from mne.stats import permutation_cluster_1samp_test,\
    permutation_cluster_test,f_mway_rm
from mne.decoding import GeneralizingEstimator

from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

import numpy as np
import pandas as pd
from scipy.stats import sem

import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns

import os

# --- --- --- --- --- --- --- --- --- Set Global Parameters --- --- --- --- --- --- --- --- --- #

subjAllN_final = subjAllN

# epoch
fileName = 'deco_data.csv'
figPath = set_filepath(decoFigPath,'epo')
tfr_tag = 'erp'

t_list = np.load(file=os.path.join(decoDataPath,'t_list.npy'),
                 allow_pickle=True)
t_points = len(t_list)
t0_indx,t1_indx = np.where(t_list>=0.0)[0][0],\
    np.where(t_list<=0.6)[0][-1]

l_wid = 2.5
l_wid_acc = 2.5


# --- --- --- --- --- --- --- --- --- Sub-Functions --- --- --- --- --- --- --- --- --- #

def gen_decoding_cx(task1,task2,labels1,labels2):
    clf = make_pipeline(StandardScaler(),
                        LogisticRegression(
                            solver='liblinear'))
    time_gen = GeneralizingEstimator(
        clf,scoring='roc_auc',
        n_jobs=None,verbose=True)

    time_gen.fit(X=task1,y=labels1)
    scores = time_gen.score(X=task2,y=labels2)
    scores_diag = np.diag(scores)

    return scores,scores_diag


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


def clu_permu_cond(acc_data_all,stat_fun):
    acc_data = acc_data_all[:,t0_indx:t1_indx]

    threshold = None
    f_obs,clu,clu_p,H0 = permutation_cluster_test(
        acc_data,
        n_permutations=n_permutations,
        threshold=threshold,tail=0,
        stat_fun=stat_fun,n_jobs=None,
        out_type='indices')
    print(clu)
    print(clu_p)

    acc_sig,grp_sig = find_sig(clu,clu_p)
    return acc_sig,grp_sig


def clu_permu_1samp_t(acc_data_all):
    acc_data = acc_data_all[:,t0_indx:t1_indx]
    threshold = None
    tail = 0
    # degrees_of_freedom = len(acc_data) - 1
    # threshold = scipy.stats.t.ppf(
    #     1-p_crit/2,df=degrees_of_freedom)

    t_obs,clu,clu_p,H0 = permutation_cluster_1samp_test(
        acc_data-chance_crit,n_permutations=n_permutations,
        threshold=threshold,tail=tail,
        out_type='indices',verbose=True)
    print(clu)
    print(clu_p)

    acc_sig,grp_sig = find_sig(clu,clu_p)
    return acc_sig,grp_sig


# --- --- --- --- --- --- --- --- --- Main Function --- --- --- --- --- --- --- --- --- #

pick_chs = 'simi'
decoFigPath_ch = set_filepath(figPath,'simiChans')
dataFileName = '_simi.npy'
labelFileName = '_lab_simi.npy'

'''
acc_df = pd.DataFrame(
    columns=['tfr','chans','pred',
             'type','subj','time',
             'acc','sig_label','grp_label'])
acc_df_subj = pd.DataFrame(
    columns=['tfr','chans','pred',
             'type','subj','time',
             'acc'])

odd_data = np.load(
    file=os.path.join(decoDataPath,'odd'+dataFileName),
    allow_pickle=True)
odd_labels = np.load(
    file=os.path.join(decoDataPath,'odd'+labelFileName),
    allow_pickle=True)
print('ODDBALL TASK LOADED ***')
print('*')

recogCate_data_dict,recogCate_labels_dict,\
    recogSize_data_dict,recogSize_labels_dict =\
    dict(),dict(),dict(),dict()

for cond in cond_label_list:
    recogCate_data_dict[cond] = np.load(
        file=os.path.join(
            decoDataPath,'recog_%s'%cond+dataFileName),
        allow_pickle=True)
    recogCate_labels_dict[cond] = np.load(
        file=os.path.join(
            decoDataPath,'recog_%s'%cond+labelFileName),
        allow_pickle=True)
    print('CONDITION %s LOADED ***'%cond)
print('*')

for cond in sizeList:
    recogSize_data_dict[cond] = np.load(
        file=os.path.join(
            decoDataPath,'recog_%d'%cond+dataFileName),
        allow_pickle=True)
    recogSize_labels_dict[cond] = np.load(
        file=os.path.join(
            decoDataPath,'recog_%d'%cond+labelFileName),
        allow_pickle=True)
    print('CONDITION %s LOADED ***'%cond)
print('*')

print('RECOGNITION TASK LOADED ***')
print('*')

print('*** ALL THE FILES LOADED ***')
print('***')
print('***')
print('***')
print('')

print('CATEGORY DECODING START **')
acc_subjAll_dict,mean_scores,acc_mean,\
    acc_sem,acc_sig_dict,sig_grp_dict =\
    dict(),dict(),dict(),dict(),dict(),dict()

for label in cond_label_list:
    acc_subjAll = np.zeros(
        [subjAllN_final,t_points,t_points])
    acc_subjAll_diag = np.zeros(
        [subjAllN_final,t_points])

    for n in range(subjAllN_final):
        acc_df_tmp = pd.DataFrame(
            columns=['tfr','chans','pred',
                     'type','subj',
                     'time','acc'])
        acc_subjAll[n],acc_subjAll_diag[n] =\
            gen_decoding_cx(
                odd_data[n],recogCate_data_dict[label][n],
                odd_labels[n],recogCate_labels_dict[label][n])

        print('Subject %d Finished'%n)
        acc_df_tmp['tfr'] = [tfr_tag]*t_points
        acc_df_tmp['pred'] = ['o2r_cate']*t_points
        acc_df_tmp['type'] = [label]*t_points
        acc_df_tmp['subj'] = [n]*t_points
        acc_df_tmp['time'] = t_list
        acc_df_tmp['acc'] = acc_subjAll_diag[n]
        acc_df_tmp['chans'] = [pick_chs]*t_points
        acc_df_subj = pd.concat([acc_df_subj,acc_df_tmp],
                                axis=0,ignore_index=True)

        print('Subject %d Finished'%(n))

    print('Condition %s Finished'%label)
    print('*')
    print('*')

    mean_scores[label] = np.mean(acc_subjAll,axis=0)
    acc_mean[label] = np.diag(mean_scores[label])
    acc_sem[label] = sem(acc_subjAll_diag)
    acc_subjAll_dict[label] = acc_subjAll_diag

    acc_df_tmp = pd.DataFrame(
        columns=['tfr','chans','pred','type',
                 'subj','time','acc',
                 'sig_label','grp_label'])
    # t-test
    acc_sig_dict[label],sig_grp_dict[label] =\
        clu_permu_1samp_t(acc_subjAll_diag)
    # save
    acc_df_tmp['tfr'] = [tfr_tag]*t_points
    acc_df_tmp['pred'] = ['o2r_cate']*t_points
    acc_df_tmp['type'] = [label]*t_points
    acc_df_tmp['subj'] = ['mean']*t_points
    acc_df_tmp['time'] = t_list
    acc_df_tmp['acc'] = acc_mean[label]
    acc_df_tmp['sig_label'] = acc_sig_dict[label]
    acc_df_tmp['grp_label'] = sig_grp_dict[label]
    acc_df_tmp['chans'] = [pick_chs]*t_points

    acc_df = pd.concat([acc_df,acc_df_tmp],
                       axis=0,ignore_index=True)

acc_df_deco = acc_df[acc_df['pred']=='o2r_cate']
acc_subjAll_cond = dict_to_arr(acc_subjAll_dict)
np.save(file=os.path.join(resPath,'deco_cate.npy'),
        arr=acc_subjAll_cond)
print('CATEGORY DECODING FINISHED ***')
print('***')
print('**')
print('*')

print('SETSIZE DECODING START **')
acc_subjAll_dict,mean_scores,acc_mean,\
    acc_sem,acc_sig_dict,sig_grp_dict =\
    dict(),dict(),dict(),dict(),dict(),dict()

for label in sizeList:
    acc_subjAll = np.zeros(
        [subjAllN_final,t_points,t_points])
    acc_subjAll_diag = np.zeros(
        [subjAllN_final,t_points])

    for n in range(subjAllN_final):
        acc_df_tmp = pd.DataFrame(
            columns=['tfr','chans','pred',
                     'type','subj',
                     'time','acc'])
        acc_subjAll[n],acc_subjAll_diag[n] =\
            gen_decoding_cx(
                odd_data[n],recogSize_data_dict[label][n],
                odd_labels[n],recogSize_labels_dict[label][n])

        print('Subject %d Finished'%n)
        acc_df_tmp['tfr'] = [tfr_tag]*t_points
        acc_df_tmp['pred'] = ['o2r_size']*t_points
        acc_df_tmp['type'] = [label]*t_points
        acc_df_tmp['subj'] = [n]*t_points
        acc_df_tmp['time'] = t_list
        acc_df_tmp['acc'] = acc_subjAll_diag[n]
        acc_df_tmp['chans'] = [pick_chs]*t_points
        acc_df_subj = pd.concat([acc_df_subj,acc_df_tmp],
                                axis=0,ignore_index=True)
        print('Subject %d Finished'%(n))

    # np.save(file=os.path.join(
    #     decoDataPath,
    #     'cx_auc_o2r_%d'%label+dataFileName),
    #     arr=acc_subjAll)

    print('Condition: %d Finished'%label)
    print('*')
    print('*')

    mean_scores[label] = np.mean(acc_subjAll,axis=0)
    acc_mean[label] = np.diag(mean_scores[label])
    acc_sem[label] = sem(acc_subjAll_diag)
    acc_subjAll_dict[label] = acc_subjAll_diag

    acc_df_tmp = pd.DataFrame(
        columns=['tfr','chans','pred','type',
                 'subj','time','acc',
                 'sig_label','grp_label'])
    # t-test
    acc_sig_dict[label],sig_grp_dict[label] =\
        clu_permu_1samp_t(acc_subjAll_diag)
    acc_df_tmp['tfr'] = [tfr_tag]*t_points
    acc_df_tmp['pred'] = ['o2r_size']*t_points
    acc_df_tmp['type'] = [label]*t_points
    acc_df_tmp['subj'] = ['mean']*t_points
    acc_df_tmp['time'] = t_list
    acc_df_tmp['acc'] = acc_mean[label]
    acc_df_tmp['sig_label'] = acc_sig_dict[label]
    acc_df_tmp['grp_label'] = sig_grp_dict[label]
    acc_df_tmp['chans'] = [pick_chs]*t_points
    acc_df = pd.concat([acc_df,acc_df_tmp],
                       axis=0,ignore_index=True)

acc_df_deco = acc_df[acc_df['pred']=='o2r_size']
acc_subjAll_cond = dict_to_arr(acc_subjAll_dict)

np.save(file=os.path.join(resPath,'deco_size.npy'),
        arr=acc_subjAll_cond)

print('SETSIZE DECODING FINISHED ***')
print('***')
print('**')
print('*')
'''


acc_df = pd.read_csv(os.path.join(resPath,fileName),sep=',')
acc_df_deco = acc_df[acc_df['pred']=='o2r_cate']
acc_subjAll_cond = np.load(
    file=os.path.join(resPath,'deco_cate.npy'),
    allow_pickle=True)
acc_mean,acc_sem = dict(),dict()
for n,label in enumerate(cond_label_list):
    acc_mean[label] = np.mean(acc_subjAll_cond[n],axis=0)
    acc_sem[label] = sem(acc_subjAll_cond[n])


def stat_fun_1way(*args):
    factor_levels = [2,1]
    effects = 'A'
    return f_mway_rm(
        np.array(args).transpose(1,0,2),
        factor_levels=factor_levels,
        effects=effects,return_pvals=False)[0]


acc_sig,sig_grp = clu_permu_cond(
    acc_subjAll_cond,
    stat_fun_1way)
acc_df_deco.loc[:,'diff_sig'] = sig_grp*2

sig_df = acc_df_deco[
    (acc_df_deco['sig_label']==1)&
    (acc_df_deco['acc']>chance_crit)]



#
#
# plot
mpl.rcParams.update({'font.size':21})
fig = plt.figure(figsize=(18,12))
plt.subplots_adjust(hspace=0.35)

plt.subplot(2,2,1)
# plot time-by-time decoding
clrs_all = sns.color_palette('Blues',n_colors=35)
clrList = [clrs_all[20],clrs_all[26]]
lineStyList = ['-','--']

ymin_show = 0.48
y = 0.635
yy = 0.653
y_title = 1.02
count = 0
text_x = [0.6,0.58]
text_y = [0.479,0.47]
for lineSty,clr,label in zip(lineStyList,clrList,cond_label_list):
    plt.plot(t_list,acc_mean[label],color=clr,
             linestyle=lineSty,
             linewidth=l_wid,label=label)
plt.legend(['within','between'],
           loc='upper right',ncol=1,fontsize=18,
           frameon=False).set_title(None)
for lineSty,clr,label in zip(lineStyList,clrList,cond_label_list):
    ymin,ymax = 0.46,0.62
    for k in set(sig_df['grp_label']):
        sig_times = sig_df.loc[
            (sig_df['grp_label']==k)&
            (sig_df['type']==label),'time']
        plt.plot(sig_times,[ymin_show]*len(sig_times),
                 color=clr,linestyle=lineSty,
                 linewidth=l_wid_acc)
    ymin_show -= p_show
    diff_times = sig_df.loc[
        (sig_df['diff_sig']>=1)&
        (sig_df['type']==label),'time']
    diff_times.reset_index(drop=True,inplace=True)
    if len(diff_times)!=0:
        for k in set(diff_times['grp_label']):
            plt.fill_betweenx(
                (chance_crit,ymax),diff_times[0],
                diff_times[len(diff_times)-1],
                color='orange',alpha=0.3)
    plt.fill_between(t_list,
                     acc_mean[label]-acc_sem[label],
                     acc_mean[label]+acc_sem[label],
                     color=clr,alpha=0.2,edgecolor='none')
    plt.text(text_x[count],text_y[count],
             '%.2f-%.2f'%(sig_times.tolist()[0],
                          sig_times.tolist()[-1]),
             color=clr,fontsize=15)
    count += 1
plt.axhline(0.5,color='k',linestyle=':',
            label='Chance level')
plt.axvline(0.0,color='k',linestyle=':')
plt.ylim(0.46,y)
plt.yticks(np.arange(0.46,0.63,0.04))
plt.ylabel('AUC')
plt.xlabel('Time (sec)')
plt.title('Cross-Task Decoding for Each Category',fontsize=17,y=y_title)
plt.axvline(0.1,ls='--',color='grey')
plt.axvline(0.16,ls='--',color='grey')
plt.axvline(0.2,ls='--',color='grey')
plt.axvline(0.3,ls='--',color='grey')
p1_text = 'P1'
n1_text = 'N1'
p2_text = 'P2'
plt.text(0.112,0.62,p1_text,color='grey',fontsize=13)
plt.text(0.161,0.62,n1_text,color='grey',fontsize=13)
plt.text(0.232,0.62,p2_text,color='grey',fontsize=13)
plt.text(-0.3,yy,'(A)',ha='center',
         va='top',color='k')


acc_df_deco = acc_df[acc_df['pred']=='o2r_size']
acc_subjAll_cond = np.load(
    file=os.path.join(resPath,'deco_size.npy'),
    allow_pickle=True)
acc_mean,acc_sem = dict(),dict()
for n,label in enumerate(sizeList):
    acc_mean[label] = np.mean(acc_subjAll_cond[n],axis=0)
    acc_sem[label] = sem(acc_subjAll_cond[n])

def stat_fun_1way(*args):
    factor_levels = [4,1]
    effects = 'A'
    return f_mway_rm(
        np.array(args).transpose(1,0,2),factor_levels=factor_levels,
        effects=effects,return_pvals=False)[0]


acc_sig,sig_grp = clu_permu_cond(
    acc_subjAll_cond,stat_fun_1way)
acc_df_deco.loc[:,'diff_sig'] = sig_grp*4

sig_df = acc_df_deco[
    (acc_df_deco['sig_label']==1)&
    (acc_df_deco['acc']>chance_crit)]

clrs_all_b = sns.color_palette('Blues',n_colors=35)
clrs_all = sns.color_palette('GnBu_d')
clrList = [clrs_all_b[28],clrs_all_b[20],clrs_all[2],clrs_all[0]]

ymin_show = 0.48
count = 0
text_x = [0.58]*4
text_y = [0.483,0.475,0.468,0.46]
plt.subplot(2,2,2)

y_loc = [0.75,0.7,0.65,0.6]
for clr,label in zip(clrList,sizeList):
    plt.plot(t_list,acc_mean[label],color=clr,
             linewidth=l_wid,label='MSS %d'%label)
plt.legend(['MSS 1','MSS 2','MSS 4','MSS 8'],
           loc='upper right',ncol=1,fontsize=15,
           frameon=False).set_title(None)
for clr,label in zip(clrList,sizeList):
    for k in set(sig_df['grp_label']):
        sig_times = sig_df.loc[
            (sig_df['grp_label']==k)&
            (sig_df['type']==str(label)),'time']
        plt.plot(sig_times,[ymin_show]*len(sig_times),
                 color=clr,linewidth=l_wid_acc)
    ymin_show -= p_show

    plt.fill_between(t_list,
                     acc_mean[label]-acc_sem[label],
                     acc_mean[label]+acc_sem[label],
                     color=clr,alpha=0.1,edgecolor='none')
    plt.text(text_x[count],text_y[count],
             '%.2f-%.2f'%(sig_times.tolist()[0],
                          sig_times.tolist()[-1]),
             color=clr,fontsize=15)
    count += 1
plt.axhline(0.5,color='k',linestyle=':')
plt.axvline(0.0,color='k',linestyle=':')
plt.ylim(0.46,y)
plt.yticks(np.arange(0.46,0.63,0.04),labels=[])
plt.ylabel(None)
plt.xlabel('Time (sec)')
plt.title('Cross-Task Decoding for Each Memory Set Size',fontsize=17,y=y_title)
plt.axvline(0.1,ls='--',color='grey')
plt.axvline(0.16,ls='--',color='grey')
plt.axvline(0.2,ls='--',color='grey')
plt.axvline(0.3,ls='--',color='grey')
p1_text = 'P1'
n1_text = 'N1'
p2_text = 'P2'
plt.text(0.112,0.62,p1_text,color='grey',fontsize=13)
plt.text(0.161,0.62,n1_text,color='grey',fontsize=13)
plt.text(0.232,0.62,p2_text,color='grey',fontsize=13)
plt.text(-0.3,yy,'(B)',ha='center',
         va='top',color='k')
# fig.text(0.5,0.49,'Time (sec)',ha='center')

# barplot
pick_tag = 'simi'
pred = 'o2r'
deco_subj_all = pd.read_csv(
    os.path.join(resPath,'deco_data_subj.csv'),sep=',')
deco_subj = deco_subj_all[
    (deco_subj_all['chans']==pick_tag)&
    (deco_subj_all['pred']==pred)&
    (deco_subj_all['type'].isin(recog_label_list))].reset_index(drop=True)
deco_subj['cond'] = deco_subj['type'].str.split('',expand=True)[1]
deco_subj['setsize'] = deco_subj['type'].str.split('',expand=True)[2]

lab_size = 18
cate_clr = [sns.color_palette('Blues')[1],sns.color_palette('Blues')[3]]*4
for n,cp_tag in enumerate(['P1','N1','P2']):
    indx = n+1
    if cp_tag=='P1':
        fig_lab = '(C)'
        t0,t1 = 0.1,0.16
    elif cp_tag=='N1':
        fig_lab = '(D)'
        t0,t1 = 0.16,0.2
    else:
        fig_lab = '(E)'
        t0,t1 = 0.2,0.3
    deco_cp = deco_subj[(deco_subj['time']<t1)&(deco_subj['time']>=t0)]

    plt.subplot(2,3,n+4)
    ax = sns.barplot(data=deco_cp,x='setsize',y='acc',
                     hue='cond',hue_order=['w','b'],
                     palette='Blues',saturation=0.75,
                     errorbar='se',capsize=0.15,errcolor='grey',
                     errwidth=1.5)
    patches = [mpl.patches.Patch(color=cate_clr[h],label=t) for h,t in
               enumerate(t.get_text() for t in ax.get_xticklabels())]
    plt.xlabel('Memory Set Size')
    if n==1:
        plt.ylabel(None)
        plt.yticks(np.arange(0.5,0.63,0.04),labels=[])
        plt.legend(title=[],labels=[],frameon=False)
    elif n==0:
        # plt.xlabel(None)
        plt.ylabel('AUC')
        plt.yticks(np.arange(0.5,0.63,0.04))
        plt.legend(handles=patches,labels=['within','between'],frameon=False)
    else:
        # plt.xlabel(None)
        plt.ylabel(None)
        plt.yticks(np.arange(0.5,0.63,0.04),labels=[])
        plt.legend(title=[],labels=[],frameon=False)
    plt.ylim(0.5,0.63)
    plt.title(cp_tag,fontsize=18)
    plt.text(-0.7,0.64,fig_lab,ha='center',
             va='top',color='k',fontsize=lab_size)
# fig.text(0.05,0.5,'AUC',va='center',rotation='vertical')
sns.despine(offset=10,trim=True)
plt.subplots_adjust(hspace=0.5)

if tag_savefig==1:
    figName = os.path.join(allResFigPath,'simi','erp_simi_deco.png')
    save_fig(fig,figName)

