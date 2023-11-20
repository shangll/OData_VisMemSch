#!/usr/bin/env python
# -*-coding:utf-8 -*-

# EXP. 3-2 (EEG): Decoding
# Temporal dynamics of non-target processing vary with
# memory set size in different category conditions
# 2022.Jan.18
# linlin.shang@donders.ru.nl


#%% --- --- --- --- --- --- libraries --- --- --- --- --- ---
from eeg_config import decoDataPath,resPath,\
    decoFigPath,subjAllN_final,\
    recog_label_list,\
    cond_label_list,sizeList,\
    tag_savefile,tag_savefig,show_flg,\
    chance_crit,p_crit,p_show,n_permutations,\
    set_filepath,dict_to_arr,\
    scoring,fdN,jobN

from mne.stats import permutation_cluster_1samp_test, \
    permutation_cluster_test,f_threshold_mway_rm,f_mway_rm
from mne.decoding import GeneralizingEstimator,\
    cross_val_multiscore,LinearModel,GeneralizingEstimator
from sklearn.metrics import roc_auc_score
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

import numpy as np
import pandas as pd
from scipy.stats import sem
import scipy.stats

import matplotlib.pyplot as plt

import os



# --- --- --- --- --- --- --- --- --- Set Global Parameters --- --- --- --- --- --- --- --- --- #

# epoch
fileName = 'deco_data.csv'
figPath = set_filepath(decoFigPath,'epo')
tfr_tag = 'erp'

t_list = np.load(file=os.path.join(decoDataPath, 't_list.npy'),
                 allow_pickle=True)
t_points = len(t_list)

l_wid = 2
l_wid_acc = 1.5


# --- --- --- --- --- --- --- --- --- Sub-Functions --- --- --- --- --- --- --- --- --- #

def cx_decoding(task1,task2,labels1,labels2):
    accs = np.zeros([t_points])
    data_train = np.mean(task1,axis=2)

    # decoding beased on epoch
    for t in range(t_points):
        data_test = task2[:,:,t]

        scaler = StandardScaler()
        x_train = scaler.fit_transform(data_train)
        x_test = scaler.transform(data_test)
        clf = SVC(
            kernel='rbf',class_weight='balanced',max_iter=-1)
        clf.fit(x_train,labels1)
        pred = clf.predict(x_test)
        accs[t] = roc_auc_score(labels2,pred)

    return accs

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

    scores = cross_val_multiscore(
        time_gen,X,y,cv=fdN,n_jobs=jobN)
    mean_score = np.mean(scores,axis=0)
    mean_score_diag = np.diag(mean_score)

    return scores, mean_score, mean_score_diag

def gen_decoding_cx(task1, task2, labels1, labels2):
    clf = make_pipeline(StandardScaler(),
                        LogisticRegression(
                            solver='liblinear'))
    time_gen = GeneralizingEstimator(
        clf,scoring='roc_auc',
        n_jobs=None,verbose=True)

    time_gen.fit(X=task1, y=labels1)
    scores = time_gen.score(X=task2, y=labels2)
    scores_diag = np.diag(scores)

    return scores, scores_diag

def find_sig(clu,clu_p,k):
    acc_sig, grp_sig = k*[0],k*[0]
    grp_label = 0

    for c,p in zip(clu, clu_p):
        if p < p_crit:
            grp_label += 1
            acc_sig[c[0][0]:(c[0][-1]+1)] = \
                [1]*len(c[0])
            grp_sig[c[0][0]:(c[0][-1]+1)] = \
                [grp_label]*len(c[0])
    return acc_sig, grp_sig

def clu_permu_cond(acc_data):
    threshold = None
    f_obs, clu, clu_p, H0 = permutation_cluster_test(
        acc_data,n_permutations=n_permutations,
        threshold=threshold,tail=1,n_jobs=None,
        out_type='indices')
    print(clu)
    print(clu_p)

    acc_sig, grp_sig = find_sig(
        clu,clu_p,acc_data.shape[-1])
    return acc_sig, grp_sig

# def stat_fun(*args):
#     factor_levels = [4,2]
#     effects = 'A*B'
#     return f_mway_rm(
#         np.swapaxes(np.array(args),1,0),
#         factor_levels=factor_levels,
#         effects=effects,return_pvals=False)[0]
#
# def stat_fun_1way(*args):
#     factor_levels = [4,1]
#     effects = 'A'
#     return f_mway_rm(
#         np.array(args),factor_levels=factor_levels,
#         effects=effects,return_pvals=False)[0]
#
# def clu_permu_cond(erp_data):
#     tail = 0
#     # pthresh = 0.001
#     factor_levels = [4,2]
#     effects = 'A:B'
#     f_thresh = f_threshold_mway_rm(
#         subjAllN_final,factor_levels,effects,p_crit)
#     f_obs,clu,clu_p,h0 = permutation_cluster_test(
#         erp_data,stat_fun=stat_fun,threshold=f_thresh,
#         tail=tail,n_jobs=None,
#         n_permutations=n_permutations,
#         buffer_size=None,out_type='indices')
#     print(clu)
#     print(clu_p)
#
#     acc_sig, grp_sig = find_sig(clu,clu_p,erp_data.shape[-1])
#     return acc_sig, grp_sig

def clu_permu_1samp_t(acc_data):
    threshold = None
    tail = 0
    degrees_of_freedom = len(acc_data) - 1
    t_thresh = scipy.stats.t.ppf(
        1-p_crit/2,df=degrees_of_freedom)

    t_obs, clu, clu_p, H0 = permutation_cluster_1samp_test(
        acc_data - chance_crit, n_permutations=n_permutations,
        threshold=t_thresh, tail=tail,
        out_type='indices', verbose=True)
    print(clu)
    print(clu_p)

    acc_sig,grp_sig = find_sig(clu,clu_p,acc_data.shape[-1])
    return acc_sig,grp_sig



# --- --- --- --- --- --- --- --- --- Main Function --- --- --- --- --- --- --- --- --- #

# for pick_chs in ['eeg', 'post', 'front']:
for pick_chs in ['simi']:
    if pick_chs == 'eeg':
        decoFigPath_ch = set_filepath(figPath, 'allChans')
        dataFileName = '_all.npy'
        labelFileName = '_lab_all.npy'
    elif pick_chs == 'front':
        decoFigPath_ch = set_filepath(figPath, 'frontChans')
        dataFileName = '_front.npy'
        labelFileName = '_lab_front.npy'
    elif pick_chs == 'fcp':
        decoFigPath_ch = set_filepath(figPath, 'fcpChans')
        dataFileName = '_fcp.npy'
        labelFileName = '_lab_fcp.npy'
    elif pick_chs == 'post':
        decoFigPath_ch = set_filepath(figPath, 'postChans')
        dataFileName = '_post.npy'
        labelFileName = '_lab_post.npy'
    elif pick_chs == 'simi':
        decoFigPath_ch = set_filepath(figPath,'simiChans')
        dataFileName = '_simi.npy'
        labelFileName = '_lab_simi.npy'

    acc_df = pd.DataFrame(
        columns=['tfr', 'chans', 'pred',
                 'type', 'subj', 'time',
                 'acc', 'sig_label', 'grp_label'])
    acc_df_subj = pd.DataFrame(
        columns=['tfr','chans','pred',
                 'type','subj','time',
                 'acc'])

    # LOAD FILES --- --- --- --- --- --- --- --- ---
    print('*** LOADING FILES ***')

    odd_data = np.load(
        file=os.path.join(decoDataPath, 'odd' + dataFileName),
        allow_pickle=True)
    odd_labels = np.load(
        file=os.path.join(decoDataPath, 'odd' + labelFileName),
        allow_pickle=True)
    print('ODDBALL TASK LOADED ***')
    print('*')

    recogAll_data = np.load(
        file=os.path.join(decoDataPath, 'recogAll' + dataFileName),
        allow_pickle=True)
    recogAll_labels = np.load(
        file=os.path.join(decoDataPath, 'recogAll' + labelFileName),
        allow_pickle=True)
    print('RECOGNITION (NON-CONDITION) TASK LOADED ***')
    print('*')

    recog_data_dict, recog_labels_dict, \
    recogCate_data_dict, recogCate_labels_dict, \
    recogSize_data_dict, recogSize_labels_dict = \
        dict(), dict(), dict(), dict(), dict(), dict()
    for cond in recog_label_list:
        recog_data_dict[cond] = np.load(
            file=os.path.join(
                decoDataPath, 'recog_%s' % cond + dataFileName),
            allow_pickle=True)
        recog_labels_dict[cond] = np.load(
            file=os.path.join(
                decoDataPath, 'recog_%s' % cond + labelFileName),
            allow_pickle=True)
        print('CONDITION %s LOADED ***' % cond)
    print('*')

    for cond in cond_label_list:
        recogCate_data_dict[cond] = np.load(
            file=os.path.join(
                decoDataPath, 'recog_%s' % cond + dataFileName),
            allow_pickle=True)
        recogCate_labels_dict[cond] = np.load(
            file=os.path.join(
                decoDataPath, 'recog_%s' % cond + labelFileName),
            allow_pickle=True)
        print('CONDITION %s LOADED ***' % cond)
    print('*')

    for cond in sizeList:
        recogSize_data_dict[cond] = np.load(
            file=os.path.join(
                decoDataPath, 'recog_%d' % cond + dataFileName),
            allow_pickle=True)
        recogSize_labels_dict[cond] = np.load(
            file=os.path.join(
                decoDataPath, 'recog_%d' % cond + labelFileName),
            allow_pickle=True)
        print('CONDITION %s LOADED ***' % cond)
    print('*')

    print('RECOGNITION TASK LOADED ***')
    print('*')

    print('*** ALL THE FILES LOADED ***')
    print('***')
    print('***')
    print('***')
    print('')

    '''
    # WITHIN TASKS --- --- --- --- --- --- --- --- ---

    acc_subjAll = np.zeros(
        [subjAllN_final,t_points,t_points])
    acc_subjAll_diag = np.zeros(
        [subjAllN_final,t_points])
    for n in range(subjAllN_final):
        scores,acc_subjAll[n],acc_subjAll_diag[n] = \
            gen_decoding_in(odd_data[n],odd_labels[n])
    acc_mean = np.mean(acc_subjAll,axis=0)
    odd_mean_scores = np.diag(acc_mean)
    odd_sem_scores = sem(acc_subjAll_diag)
    # t-test
    acc_sig,grp_sig = clu_permu_1samp_t(acc_subjAll_diag)
    sig_indx = np.where(np.array(acc_sig)==1)[0]
    crop_tag = 'sigOdd'
    t0_indx,t1_indx = sig_indx[0],sig_indx[-1]
    t0,t1 = t_list[sig_indx][0],t_list[sig_indx][-1]

    acc_df_odd = pd.DataFrame(
        columns=['tfr','chans','pred','type','subj',
                 'time','acc','sig_label','grp_label'])
    # save
    acc_df_odd['tfr'] = [tfr_tag]*t_points
    acc_df_odd['pred'] = ['odd']*t_points
    acc_df_odd['subj'] = ['mean']*t_points
    acc_df_odd['type'] = ['mean']*t_points
    acc_df_odd['time'] = t_list
    acc_df_odd['acc'] = odd_mean_scores
    acc_df_odd['sig_label'] = acc_sig
    acc_df_odd['grp_label'] = grp_sig
    acc_df_odd['chans'] = [pick_chs]*t_points

    # plot time-by-time decoding
    sig_df = acc_df_odd[
        (acc_df_odd['pred']=='odd')&
        (acc_df_odd['subj']=='mean')&
        (acc_df_odd['sig_label']==1)&
        (acc_df_odd['acc']>chance_crit)]
    clr = 'crimson'
    fig,ax = plt.subplots(
        1,1,sharex=True,sharey=True,figsize=(9,6))
    for indx,label in enumerate(['Mean']):
        ax.axhline(0.5,color='k',linestyle='--',
                   label='Chance level')
        ax.axvline(0.0,color='k',linestyle=':')

        ax.plot(t_list,odd_mean_scores,clr,
                linewidth=l_wid,label=label)

        ymin,ymax = ax.get_ylim()
        ymin_show = 0.48
        for k in set(sig_df['grp_label']):
            sig_times = sig_df.loc[sig_df['grp_label']==k,'time']
            ax.plot(sig_times,[ymin]*len(sig_times),
                    clr,linewidth=l_wid_acc,
                    label=label)
        ax.fill_between(t_list,odd_mean_scores-odd_sem_scores,
                        odd_mean_scores+odd_sem_scores,
                        color=clr,alpha=0.1,
                        edgecolor='none')
    title = 'localizer_AUC_subjAvg.png'
    # if tag_savefig==1:
    #     fig.savefig(os.path.join(decoFigPath_ch,title),
    #                 bbox_inches='tight',dpi=300)
    plt.show(block=show_flg)
    plt.close('all')

    print('ODDBALL DECODING FINISHED ***')
    '''
    # # p2
    # t0,t1 = 0.2,0.3
    # crop_tag = 'p2Odd'
    # # all 3 cps
    # crop_tag = 'cp'
    # t0,t1 = 0.1,0.3
    # # overlap
    # crop_tag = 'overlap'
    # t0,t1 = 0.192,0.256
    # # 0.0-0.6 cps
    crop_tag = 'all'
    t0,t1 = 0.0,0.6

    t0_indx,t1_indx = np.where(t_list>=t0)[0][0],np.where(t_list<=t1)[0][-1]

    # CROSS TASKS --- --- --- --- --- --- --- --- ---
    print('*** CROSS TASKS DECODING ***')


    print('1. NON-CONDITION DECODING START **')

    acc_subjAll_diag = np.zeros(
        [subjAllN_final,t_points])

    for n in range(subjAllN_final):
        train_data = odd_data[n][:,:,t0_indx:(t1_indx+1)]

        acc_subjAll_diag[n] = \
            cx_decoding(
            train_data, recogAll_data[n],
            odd_labels[n], recogAll_labels[n])
        print('Subject %d Finished' % n)

    # np.save(file=os.path.join(
    #     decoDataPath,'cx_auc_o2rAll'+dataFileName),
    #     arr=acc_subjAll)

    acc_mean = np.mean(acc_subjAll_diag,axis=0)
    acc_sem = sem(acc_subjAll_diag)

    acc_df_tmp = pd.DataFrame(
        columns=['tfr', 'chans', 'pred', 'type', 'subj',
                 'time', 'acc', 'sig_label', 'grp_label'])
    # t-test
    acc_sig, grp_sig = clu_permu_1samp_t(acc_subjAll_diag)
    # save
    acc_df_tmp['tfr'] = [tfr_tag] * t_points
    acc_df_tmp['pred'] = ['o2r_all'] * t_points
    acc_df_tmp['type'] = ['mean'] * t_points
    acc_df_tmp['subj'] = ['mean'] * t_points
    acc_df_tmp['time'] = t_list
    acc_df_tmp['acc'] = acc_mean
    acc_df_tmp['sig_label'] = acc_sig
    acc_df_tmp['grp_label'] = grp_sig
    acc_df_tmp['chans'] = [pick_chs] * t_points
    acc_df = pd.concat([acc_df, acc_df_tmp],
                       axis=0, ignore_index=True)

    # plot time-by-time decoding
    acc_df_deco = acc_df[acc_df['pred'] == 'o2r_all']
    sig_df = acc_df_deco[
        (acc_df_deco['sig_label'] == 1) &
        (acc_df_deco['acc'] > chance_crit)]

    clr = 'crimson'
    fig, ax = plt.subplots(
        1, 1, sharex=True, sharey=True, figsize=(9, 6))
    for indx, label in enumerate(['Mean']):
        ax.axhline(0.5, color='k', linestyle='--',
                   label='Chance level')
        ax.axvline(0.0, color='k', linestyle=':')

        ax.plot(t_list, acc_mean, clr,
                linewidth=l_wid, label=label)

        ymin, ymax = ax.get_ylim()
        ymin_show = 0.48
        for k in set(sig_df['grp_label']):
            sig_times = sig_df.loc[
                sig_df['grp_label'] == k, 'time']
            ax.plot(sig_times, [ymin] * len(sig_times),
                    clr, linewidth=l_wid_acc)
        ax.fill_between(t_list, acc_mean - acc_sem,
                        acc_mean + acc_sem,
                        color=clr, alpha=0.1,
                        edgecolor='none')
    plt.suptitle('Cross-Task Decoding Based on Average Data')
    plt.xlabel('Time (sec)')
    plt.ylabel('AUC')
    plt.legend()
    title = '%s_o2rAll_AUC_subjAvg.png'%crop_tag
    if tag_savefig == 1:
        fig.savefig(os.path.join(decoFigPath_ch, title),
                    bbox_inches='tight', dpi=300)
    plt.show(block=show_flg)
    plt.close('all')

    print('NON-CONDITION DECODING FINISHED ***')
    print('***')
    print('**')
    print('*')


    print('2. EACH CONDITION DECODING START **')
    acc_subjAll_dict, mean_scores, acc_mean, \
    acc_sem, acc_sig_dict, sig_grp_dict = \
        dict(), dict(), dict(), dict(), dict(), dict()

    for label in recog_label_list:
        acc_subjAll_diag = np.zeros(
            [subjAllN_final,t_points])

        for n in range(subjAllN_final):
            train_data = odd_data[n][:,:,t0_indx:(t1_indx+1)]
            acc_df_tmp = pd.DataFrame(
                columns=['tfr','chans','pred','type','subj',
                         'time','acc'])
            acc_subjAll_diag[n] = \
                cx_decoding(
                    train_data,recog_data_dict[label][n],
                    odd_labels[n],recog_labels_dict[label][n])
            print('Subject %d Finished' % n)
            acc_df_tmp['tfr'] = [tfr_tag]*t_points
            acc_df_tmp['pred'] = ['o2r']*t_points
            acc_df_tmp['type'] = [label]*t_points
            acc_df_tmp['subj'] = [n]*t_points
            acc_df_tmp['time'] = t_list
            acc_df_tmp['acc'] = acc_subjAll_diag[n]
            acc_df_tmp['chans'] = [pick_chs]*t_points
            acc_df_subj = pd.concat([acc_df_subj,acc_df_tmp],
                                    axis=0,ignore_index=True)

        # np.save(file=os.path.join(
        #     decoDataPath,
        #     'cx_auc_o2r_%s'%label.replace('/','')+dataFileName),
        #     arr=acc_subjAll)

        print('Condition %s Finished' % label)
        print('*')
        print('*')

        acc_mean[label] = np.mean(acc_subjAll_diag,axis=0)
        acc_sem[label] = sem(acc_subjAll_diag)
        acc_subjAll_dict[label] = acc_subjAll_diag

        acc_df_tmp = pd.DataFrame(
            columns=['tfr', 'chans', 'pred', 'type', 'subj',
                     'time', 'acc', 'sig_label', 'grp_label'])
        # t-test
        acc_sig_dict[label], sig_grp_dict[label] = \
            clu_permu_1samp_t(acc_subjAll_diag)
        # save
        acc_df_tmp['tfr'] = [tfr_tag]*t_points
        acc_df_tmp['pred'] = ['o2r']*t_points
        acc_df_tmp['type'] = [label]*t_points
        acc_df_tmp['subj'] = ['mean']*t_points
        acc_df_tmp['time'] = t_list
        acc_df_tmp['acc'] = acc_mean[label]
        acc_df_tmp['sig_label'] = acc_sig_dict[label]
        acc_df_tmp['grp_label'] = sig_grp_dict[label]
        acc_df_tmp['chans'] = [pick_chs] * t_points
        acc_df = pd.concat([acc_df, acc_df_tmp],
                           axis=0, ignore_index=True)

    acc_df_deco = acc_df[acc_df['pred'] == 'o2r']
    acc_subjAll_cond = dict_to_arr(acc_subjAll_dict)
    acc_sig, sig_grp = clu_permu_cond(acc_subjAll_cond)
    acc_df_deco['diff_sig'] = sig_grp * 8

    sig_df = acc_df_deco[
        (acc_df_deco['sig_label']==1)&
        (acc_df_deco['acc']>chance_crit)]

    # plot time-by-time decoding
    clr = 'crimson'
    fig, ax = plt.subplots(
        2, 4, sharex=True, sharey=True, figsize=(9, 6))
    ax = ax.ravel()
    for indx, label in enumerate(recog_label_list):

        ax[indx].axhline(0.5, color='k', linestyle='--',
                         label='Chance level')
        ax[indx].axvline(0.0, color='k', linestyle=':')

        ax[indx].plot(t_list, acc_mean[label], clr,
                      linewidth=l_wid, label=label)

        ymin, ymax = ax[indx].get_ylim()
        ymin_show = 0.48
        for k in set(sig_df['grp_label']):
            sig_times = sig_df.loc[
                (sig_df['grp_label'] == k) &
                (sig_df['type'] == label), 'time']
            ax[indx].plot(sig_times, [ymin_show] * len(sig_times),
                          clr, linewidth=l_wid_acc,
                          label=label)

        diff_times = sig_df.loc[
            (sig_df['diff_sig'] >= 1) &
            (sig_df['type'] == label), 'time']
        diff_times.reset_index(drop=True, inplace=True)

        if len(diff_times) != 0:
            for k in set(diff_times['grp_label']):
                ax[indx].fill_betweenx(
                    (chance_crit, ymax), diff_times[0],
                    diff_times[len(diff_times) - 1],
                    color='orange', alpha=0.3)

        ax[indx].fill_between(t_list,
                              acc_mean[label] - acc_sem[label],
                              acc_mean[label] + acc_sem[label],
                              color=clr, alpha=0.1,
                              edgecolor='none')
        ax[indx].set_title('%s' % (label))
    plt.suptitle('Cross-Task Decoding for Each Condition')
    fig.text(0.5,0,'Time (sec)',ha='center')
    fig.text(0,0.5,'AUC',va='center',rotation='vertical')
    title = '%s_o2r_AUC_subjAvg.png'%crop_tag
    if tag_savefig == 1:
        fig.savefig(os.path.join(decoFigPath_ch, title),
                    bbox_inches='tight', dpi=300)
    plt.show(block=show_flg)
    plt.close('all')

    print('O2R DECODING FINISHED ***')
    print('***')
    print('**')
    print('*')

    
    print('3. CATEGORY DECODING START **')
    acc_subjAll_dict, mean_scores, acc_mean, \
    acc_sem, acc_sig_dict, sig_grp_dict = \
        dict(), dict(), dict(), dict(), dict(), dict()

    for label in cond_label_list:
        acc_subjAll_diag = np.zeros(
            [subjAllN_final, t_points])

        for n in range(subjAllN_final):
            train_data = odd_data[n][:,:,t0_indx:(t1_indx+1)]
            acc_subjAll_diag[n] = \
                cx_decoding(
                    train_data, recogCate_data_dict[label][n],
                    odd_labels[n], recogCate_labels_dict[label][n])
            print('Subject %d Finished' % (n))

        # np.save(file=os.path.join(
        #     decoDataPath,
        #     'cx_auc_o2r_%s'%label+dataFileName),
        #     arr=acc_subjAll)
        print('Condition %s Finished' % label)
        print('*')
        print('*')

        acc_mean[label] = np.mean(acc_subjAll_diag,axis=0)
        acc_sem[label] = sem(acc_subjAll_diag)
        acc_subjAll_dict[label] = acc_subjAll_diag

        acc_df_tmp = pd.DataFrame(
            columns=['tfr', 'chans', 'pred', 'type',
                     'subj', 'time', 'acc',
                     'sig_label', 'grp_label'])
        # t-test
        acc_sig_dict[label], sig_grp_dict[label] = \
            clu_permu_1samp_t(acc_subjAll_diag)
        # save
        acc_df_tmp['tfr'] = [tfr_tag]*t_points
        acc_df_tmp['pred'] = ['o2r_cate'] * t_points
        acc_df_tmp['type'] = [label] * t_points
        acc_df_tmp['subj'] = ['mean'] * t_points
        acc_df_tmp['time'] = t_list
        acc_df_tmp['acc'] = acc_mean[label]
        acc_df_tmp['sig_label'] = acc_sig_dict[label]
        acc_df_tmp['grp_label'] = sig_grp_dict[label]
        acc_df_tmp['chans'] = [pick_chs] * t_points
        acc_df = pd.concat([acc_df, acc_df_tmp],
                           axis=0, ignore_index=True)

    # plot time-by-time decoding
    clrList = ['orange', 'g']
    fig, ax = plt.subplots(
        1, 1, sharex=True, sharey=True, figsize=(9, 6))
    ax.axhline(0.5, color='k', linestyle='--',
               label='Chance level')
    ax.axvline(0.0, color='k', linestyle=':')

    ymin_show = 0.48
    for clr, label in zip(clrList, cond_label_list):
        ax.plot(t_list, acc_mean[label], clr,
                linewidth=l_wid, label=label)

        ymin, ymax = ax.get_ylim()
        for k in set(sig_df['grp_label']):
            sig_times = sig_df.loc[
                (sig_df['grp_label'] == k) &
                (sig_df['type'] == label), 'time']
            ax.plot(sig_times, [ymin_show] * len(sig_times),
                    clr, linewidth=l_wid_acc)
        ymin_show -= p_show

        diff_times = sig_df.loc[
            (sig_df['diff_sig'] >= 1) &
            (sig_df['type'] == label), 'time']
        diff_times.reset_index(drop=True, inplace=True)

        if len(diff_times) != 0:
            for k in set(diff_times['grp_label']):
                ax.fill_betweenx(
                    (chance_crit, ymax), diff_times[0],
                    diff_times[len(diff_times) - 1],
                    color='orange', alpha=0.3)

        ax.fill_between(t_list,
                        acc_mean[label] - acc_sem[label],
                        acc_mean[label] + acc_sem[label],
                        color=clr, alpha=0.1,
                        edgecolor='none')

    plt.suptitle('Cross-Task Decoding Based on Category Condition')
    fig.text(0.5,0,'Time (sec)',ha='center')
    fig.text(0,0.5,'AUC',va='center',rotation='vertical')
    plt.legend()
    title = '%s_o2rCate_AUC_subjAvg.png'%crop_tag
    if tag_savefig == 1:
        fig.savefig(os.path.join(decoFigPath_ch, title),
                    bbox_inches='tight', dpi=300)
    plt.show(block=show_flg)
    plt.close('all')

    print('CATEGORY DECODING FINISHED ***')
    print('***')
    print('**')
    print('*')

    print('4. SETSIZE DECODING START **')
    acc_subjAll_dict, mean_scores, acc_mean, \
    acc_sem, acc_sig_dict, sig_grp_dict = \
        dict(), dict(), dict(), dict(), dict(), dict()

    for label in sizeList:
        acc_subjAll_diag = np.zeros(
            [subjAllN_final,t_points])

        for n in range(subjAllN_final):
            train_data = odd_data[n][:,:,t0_indx:(t1_indx+1)]

            acc_subjAll_diag[n] = \
                cx_decoding(
                    train_data, recogSize_data_dict[label][n],
                    odd_labels[n], recogSize_labels_dict[label][n])
            print('Subject %d Finished' % (n))

        # np.save(file=os.path.join(
        #     decoDataPath,
        #     'cx_auc_o2r_%d'%label+dataFileName),
        #     arr=acc_subjAll)

        print('Condition: %d Finished' % label)
        print('*')
        print('*')

        acc_mean[label] = np.mean(acc_subjAll_diag, axis=0)
        acc_sem[label] = sem(acc_subjAll_diag)
        acc_subjAll_dict[label] = acc_subjAll_diag

        acc_df_tmp = pd.DataFrame(
            columns=['tfr', 'chans', 'pred', 'type',
                     'subj', 'time', 'acc',
                     'sig_label', 'grp_label'])
        # t-test
        acc_sig_dict[label], sig_grp_dict[label] = \
            clu_permu_1samp_t(acc_subjAll_diag)
        acc_df_tmp['tfr'] = [tfr_tag]*t_points
        acc_df_tmp['pred'] = ['o2r_size'] * t_points
        acc_df_tmp['type'] = [label] * t_points
        acc_df_tmp['subj'] = ['mean'] * t_points
        acc_df_tmp['time'] = t_list
        acc_df_tmp['acc'] = acc_mean[label]
        acc_df_tmp['sig_label'] = acc_sig_dict[label]
        acc_df_tmp['grp_label'] = sig_grp_dict[label]
        acc_df_tmp['chans'] = [pick_chs] * t_points
        acc_df = pd.concat([acc_df,acc_df_tmp],
                           axis=0, ignore_index=True)

    # plt.suptitle('Temporal Generalization')
    acc_df_deco = acc_df[acc_df['pred'] == 'o2r_size']
    acc_subjAll_cond = dict_to_arr(acc_subjAll_dict)
    acc_sig, sig_grp = clu_permu_cond(acc_subjAll_cond)
    acc_df_deco.loc[:, 'diff_sig'] = sig_grp * 4

    sig_df = acc_df_deco[
        (acc_df_deco['sig_label'] == 1) &
        (acc_df_deco['acc'] > chance_crit)]

    clrList = ['crimson', 'gold', 'darkturquoise', 'dodgerblue']
    fig, ax = plt.subplots(
        1, 1, sharex=True, sharey=True, figsize=(9, 6))
    ax.axhline(0.5, color='k', linestyle='--',
               label='Chance level')
    ax.axvline(0.0, color='k', linestyle=':')

    ymin_show = 0.48
    for clr, label in zip(clrList, sizeList):
        ax.plot(t_list, acc_mean[label], clr,
                linewidth=l_wid, label=label)

        ymin, ymax = ax.get_ylim()
        for k in set(sig_df['grp_label']):
            sig_times = sig_df.loc[
                (sig_df['grp_label'] == k) &
                (sig_df['type'] == label), 'time']
            ax.plot(sig_times, [ymin_show] * len(sig_times),
                    clr, linewidth=l_wid_acc)
        ymin_show -= p_show

        diff_times = sig_df.loc[
            (sig_df['diff_sig'] >= 1) &
            (sig_df['type'] == label), 'time']
        diff_times.reset_index(drop=True, inplace=True)

        if len(diff_times) != 0:
            for k in set(diff_times['grp_label']):
                ax.fill_betweenx(
                    (chance_crit, ymax), diff_times[0],
                    diff_times[len(diff_times) - 1],
                    color='orange', alpha=0.3)

        ax.fill_between(t_list,
                        acc_mean[label] - acc_sem[label],
                        acc_mean[label] + acc_sem[label],
                        color=clr, alpha=0.1,
                        edgecolor='none')
    plt.suptitle('Cross-Task Decoding Based on Memory Set Size')
    fig.text(0.5,0,'Time (sec)',ha='center')
    fig.text(0,0.5,'AUC',va='center',rotation='vertical')
    plt.legend()
    title = '%s_o2rSize_AUC_subjAvg.png'%crop_tag
    if tag_savefig == 1:
        fig.savefig(os.path.join(decoFigPath_ch, title),
                    bbox_inches='tight', dpi=300)
    plt.show(block=show_flg)
    plt.close('all')

    print('SETSIZE DECODING FINISHED ***')
    print('***')
    print('**')
    print('*')
    

    # # save ACC files
    # tag_savefile = 0
    # if tag_savefile==1:
    #     if os.path.isfile(os.path.join(resPath,fileName)):
    #         acc_df.to_csv(os.path.join(resPath,fileName),
    #                       mode='a',header=False,index=False)
    #     else:
    #         acc_df.to_csv(os.path.join(resPath,fileName),
    #                       mode='w',header=True,index=False)
    #
    # if tag_savefile==1:
    #     if os.path.isfile(
    #             os.path.join(resPath,'deco_data_subj.csv')):
    #         acc_df_subj.to_csv(
    #             os.path.join(resPath,'deco_data_subj.csv'),
    #             mode='a',header=False,index=False)
    #     else:
    #         acc_df_subj.to_csv(
    #             os.path.join(resPath,'deco_data_subj.csv'),
    #             mode='w',header=True,index=False)
    print('%s FINISHED ***'%pick_chs)

print('*** GENERALIZATION DECODING FINISHED ***')
print('***')
print('***')
print('***')
print('')