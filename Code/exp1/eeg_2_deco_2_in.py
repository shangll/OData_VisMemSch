#!/usr/bin/env python
#-*-coding:utf-8 -*-

# EXP. 3-2 (EEG): Decoding
# Temporal dynamics of non-target processing vary with
# memory set size in different category conditions
# 2022.Jan.18
# linlin.shang@donders.ru.nl


from eeg_config import decoDataPath,resPath,decoFigPath,subjAllN,\
    recog_label_list,cond_label_list,sizeList,\
    tag_savefile,tag_savefig,show_flg,\
    chance_crit,p_crit,p_show,n_permutations,scoring,jobN,fdN,\
    set_filepath,dict_to_arr

from mne.stats import permutation_cluster_1samp_test, \
    permutation_cluster_test
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

import os


# --- --- --- --- --- --- --- --- --- Set Global Parameters --- --- --- --- --- --- --- --- --- #

subjAllN_final = subjAllN

# epoch
fileName = 'deco_data.csv'
figPath = set_filepath(decoFigPath,'epo')
tfr_tag = 'erp'

t_list = np.load(file=os.path.join(
    decoDataPath, 't_list.npy'),allow_pickle=True)
t_points = len(t_list)

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

    scores = cross_val_multiscore(
        time_gen,X,y,cv=fdN,n_jobs=jobN)
    mean_score = np.mean(scores,axis=0)
    mean_score_diag = np.diag(mean_score)

    return scores, mean_score, mean_score_diag

def find_sig(clu, clu_p):
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

def clu_permu_cond(acc_data):
    threshold = None
    f_obs, clu, clu_p, H0 = permutation_cluster_test(
        acc_data - chance_crit,
        n_permutations=n_permutations,
        threshold=threshold, tail=1, n_jobs=None,
        out_type='indices')
    acc_sig, grp_sig = find_sig(clu, clu_p)
    return acc_sig, grp_sig

def clu_permu_1samp_t(acc_data):
    threshold = None
    tail = 0
    degrees_of_freedom = len(acc_data) - 1
    t_thresh = scipy.stats.t.ppf(
        1 - p_crit / 2, df=degrees_of_freedom)

    t_obs, clu, clu_p, H0 = permutation_cluster_1samp_test(
        acc_data - chance_crit, n_permutations=n_permutations,
        threshold=t_thresh, tail=tail,
        out_type='indices', verbose=True)
    acc_sig,grp_sig = find_sig(clu, clu_p)
    return acc_sig,grp_sig



# --- --- --- --- --- --- --- --- --- Main Function --- --- --- --- --- --- --- --- --- #

for pick_tag in ['eeg', 'postChans']:
    if pick_tag == 'eeg':
        decoFigPath_ch = set_filepath(figPath, 'allChans')
        dataFileName = '_all.npy'
        labelFileName = '_lab_all.npy'
    elif pick_tag == 'frontChans':
        decoFigPath_ch = set_filepath(figPath, 'frontChans')
        dataFileName = '_front.npy'
        labelFileName = '_lab_front.npy'
    elif pick_tag == 'fcpChans':
        decoFigPath_ch = set_filepath(figPath, 'fcpChans')
        dataFileName = '_fcp.npy'
        labelFileName = '_lab_fcp.npy'
    elif pick_tag == 'postChans':
        decoFigPath_ch = set_filepath(figPath, 'postChans')
        dataFileName = '_post.npy'
        labelFileName = '_lab_post.npy'

    acc_df = pd.DataFrame(
        columns=['tfr','chans', 'pred', 'type', 'subj', 'time',
                 'acc', 'sig_label', 'grp_label'])

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

    print('*** GENERALIZATION DECODING ***')
    print('ODDBALL TASK ***')
    acc_subjAll = np.zeros(
        [subjAllN_final, t_points, t_points])
    acc_subjAll_diag = np.zeros(
        [subjAllN_final, t_points])

    for n in range(subjAllN_final):
        scores, acc_subjAll[n], acc_subjAll_diag[n] = \
            gen_decoding_in(odd_data[n], odd_labels[n])

    acc_mean = np.mean(acc_subjAll, axis=0)
    odd_mean_scores = np.diag(acc_mean)
    odd_sem_scores = sem(acc_subjAll_diag)

    acc_df_tmp = pd.DataFrame(
        columns=['tfr','chans', 'pred', 'type', 'subj',
                 'time', 'acc', 'sig_label', 'grp_label'])
    # t-test
    acc_sig, grp_sig = clu_permu_1samp_t(acc_subjAll_diag)
    # save
    acc_df_tmp['tfr'] = [tfr_tag]*t_points
    acc_df_tmp['pred'] = ['odd'] * t_points
    acc_df_tmp['subj'] = ['mean'] * t_points
    acc_df_tmp['type'] = ['mean'] * t_points
    acc_df_tmp['time'] = t_list
    acc_df_tmp['acc'] = odd_mean_scores
    acc_df_tmp['sig_label'] = acc_sig
    acc_df_tmp['grp_label'] = grp_sig
    acc_df_tmp['chans'] = [pick_tag] * t_points
    acc_df = pd.concat([acc_df, acc_df_tmp],
                       axis=0, ignore_index=True)

    # plot the full (generalization) matrix
    fig, ax = plt.subplots(1, 1)
    im = ax.imshow(acc_mean, interpolation='lanczos',
                   origin='lower', cmap='RdBu_r',
                   extent=t_list[[0, -1, 0, -1]],
                   vmin=0., vmax=1.)
    ax.set_xlabel('Testing Time (s)')
    ax.set_ylabel('Training Time (s)')
    # ax.set_title('Temporal Generalization (Average)')
    ax.axvline(0, color='k')
    ax.axhline(0, color='k')
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('AUC')
    title = 'gen_odd_AUC_subjAvg.png'
    if tag_savefig == 1:
        fig.savefig(os.path.join(decoFigPath_ch, title),
                    bbox_inches='tight', dpi=300)
    plt.show(block=show_flg)
    plt.close('all')

    # plot time-by-time decoding
    sig_df = acc_df[
        (acc_df['pred'] == 'odd') &
        (acc_df['subj'] == 'mean') &
        (acc_df['sig_label'] == 1) &
        (acc_df['acc'] > chance_crit)]

    clr = 'crimson'
    fig, ax = plt.subplots(
        1, 1, sharex=True, sharey=True, figsize=(9, 6))
    for indx, label in enumerate(['Mean']):
        ax.axhline(0.5, color='k', linestyle='--',
                   label='Chance level')
        ax.axvline(0.0, color='k', linestyle=':')

        ax.plot(t_list, odd_mean_scores, clr,
                linewidth=l_wid, label=label)

        ymin, ymax = ax.get_ylim()
        ymin_show = 0.48
        for k in set(sig_df['grp_label']):
            sig_times = sig_df.loc[sig_df['grp_label'] == k, 'time']
            ax.plot(sig_times, [ymin] * len(sig_times),
                    clr, linewidth=l_wid_acc,
                    label=label)
        ax.fill_between(t_list, odd_mean_scores - odd_sem_scores,
                        odd_mean_scores + odd_sem_scores,
                        color=clr, alpha=0.1,
                        edgecolor='none')
    title = 'within_odd_AUC_subjAvg.png'
    if tag_savefig == 1:
        fig.savefig(os.path.join(decoFigPath_ch, title),
                    bbox_inches='tight', dpi=300)
    plt.show(block=show_flg)
    plt.close('all')

    print('ODDBALL DECODING FINISHED ***')
    print('***')
    print('**')
    print('*')

    print('1. RECOGNITION NON-CONDITION ***')
    acc_subjAll = np.zeros(
        [subjAllN_final,t_points,t_points])
    acc_subjAll_diag = np.zeros(
        [subjAllN_final,t_points])

    for n in range(subjAllN_final):
        scores, acc_subjAll[n], acc_subjAll_diag[n] = \
            gen_decoding_in(recogAll_data[n],recogAll_labels[n])

    np.save(file=os.path.join(
        decoDataPath,'in_auc_recog'+dataFileName),
        arr=acc_subjAll)

    acc_mean = np.mean(acc_subjAll, axis=0)
    recogAll_mean_scores = np.diag(acc_mean)
    recogAll_sem_scores = sem(acc_subjAll_diag)

    acc_df_tmp = pd.DataFrame(
        columns=['tfr','chans', 'pred', 'type', 'subj',
                 'time', 'acc', 'sig_label', 'grp_label'])
    # t-test
    acc_sig, grp_sig = clu_permu_1samp_t(acc_subjAll_diag)
    # save
    acc_df_tmp['tfr'] = [tfr_tag]*t_points
    acc_df_tmp['pred'] = ['recogAll'] * t_points
    acc_df_tmp['type'] = ['mean'] * t_points
    acc_df_tmp['subj'] = ['mean'] * t_points
    acc_df_tmp['time'] = t_list
    acc_df_tmp['acc'] = recogAll_mean_scores
    acc_df_tmp['sig_label'] = acc_sig
    acc_df_tmp['grp_label'] = grp_sig
    acc_df_tmp['chans'] = [pick_tag] * t_points
    acc_df = pd.concat([acc_df, acc_df_tmp],
                       axis=0, ignore_index=True)

    # plot the full (generalization) matrix
    fig, ax = plt.subplots(1, 1)
    im = ax.imshow(acc_mean,interpolation='lanczos',
                   origin='lower',cmap='RdBu_r',
                   extent=t_list[[0,-1,0,-1]],
                   vmin=0.,vmax=1.)
    ax.set_xlabel('Testing Time (s)')
    ax.set_ylabel('Training Time (s)')
    # ax.set_title('Temporal Generalization (Average)')
    ax.axvline(0,color='k')
    ax.axhline(0,color='k')
    cbar = plt.colorbar(im,ax=ax)
    cbar.set_label('AUC')
    title = 'gen_All_AUC_subjAvg.png'
    if tag_savefig == 1:
        fig.savefig(os.path.join(decoFigPath_ch, title),
                    bbox_inches='tight', dpi=300)
    plt.show(block=show_flg)
    plt.close('all')

    # plot time-by-time decoding
    sig_df = acc_df[
        (acc_df['pred'] == 'recogAll') &
        (acc_df['subj'] == 'mean') &
        (acc_df['sig_label'] == 1) &
        (acc_df['acc'] > chance_crit)]

    clr = 'crimson'
    fig, ax = plt.subplots(
        1, 1, sharex=True, sharey=True, figsize=(9, 6))
    for indx, label in enumerate(['Mean']):
        ax.axhline(0.5, color='k', linestyle='--',
                   label='Chance level')
        ax.axvline(0.0, color='k', linestyle=':')

        ax.plot(t_list, recogAll_mean_scores, clr,
                linewidth=l_wid, label=label)

        ymin, ymax = ax.get_ylim()
        ymin_show = 0.48
        for k in set(sig_df['grp_label']):
            sig_times = sig_df.loc[sig_df['grp_label'] == k, 'time']
            ax.plot(sig_times, [ymin] * len(sig_times),
                    clr, linewidth=l_wid_acc,
                    label=label)
        ax.fill_between(t_list,
                        recogAll_mean_scores - recogAll_sem_scores,
                        recogAll_mean_scores + recogAll_sem_scores,
                        color=clr, alpha=0.1,
                        edgecolor='none')
        # ax.set_title('Decoding EEG sensors over time (average)')
    title = 'within_recogAll_AUC_subjAvg.png'
    if tag_savefig == 1:
        fig.savefig(os.path.join(decoFigPath_ch, title),
                    bbox_inches='tight', dpi=300)
    plt.show(block=show_flg)
    plt.close('all')

    print('NON-CONDITION DECODING FINISHED ***')
    print('***')
    print('**')
    print('*')


    print('2. EACH CONDITION DECODING FINISHED ***')
    acc_mean = dict()
    acc_subjAll_dict, recog_mean_scores, recog_sem_scores = \
        dict(), dict(), dict()

    for label in recog_label_list:
        acc_subjAll = np.zeros(
            [subjAllN_final, t_points, t_points])
        acc_subjAll_diag = np.zeros(
            [subjAllN_final, t_points])

        for n in range(subjAllN_final):
            scores, acc_subjAll[n], acc_subjAll_diag[n] = \
                gen_decoding_in(recog_data_dict[label][n],
                                recog_labels_dict[label][n])
            print('Subject %d Finished' % n)

        np.save(file=os.path.join(
            decoDataPath,
            'in_auc_recog_%s'%label.replace('/','')+dataFileName),
            arr=acc_subjAll)

        print('Condition %s Finished' % label)
        print('*')
        print('*')

        acc_mean[label] = np.mean(acc_subjAll, axis=0)
        recog_mean_scores[label] = np.diag(acc_mean[label])
        recog_sem_scores[label] = sem(acc_subjAll_diag)
        acc_subjAll_dict[label] = acc_subjAll_diag

        # t-test
        acc_sig, grp_sig = clu_permu_1samp_t(acc_subjAll_diag)
        # save
        acc_df_tmp = pd.DataFrame(
            columns=['tfr','chans', 'pred', 'type', 'subj',
                     'time', 'acc', 'sig_label', 'grp_label'])
        acc_df_tmp['tfr'] = [tfr_tag]*t_points
        acc_df_tmp['pred'] = ['recog'] * t_points
        acc_df_tmp['type'] = [label] * t_points
        acc_df_tmp['subj'] = ['mean'] * t_points
        acc_df_tmp['time'] = t_list
        acc_df_tmp['acc'] = recog_mean_scores[label]
        acc_df_tmp['sig_label'] = acc_sig
        acc_df_tmp['grp_label'] = grp_sig
        acc_df_tmp['chans'] = [pick_tag] * t_points
        acc_df = pd.concat([acc_df, acc_df_tmp],
                           axis=0, ignore_index=True)

    # plot the full (generalization) matrix
    fig, ax = plt.subplots(
        2, 4, sharex=True, sharey=True, figsize=(9, 6))
    ax = ax.ravel()
    for indx, label in enumerate(recog_label_list):
        im = ax[indx].imshow(
            acc_mean[label], interpolation='lanczos',
            origin='lower', cmap='RdBu_r',
            extent=t_list[[0, -1, 0, -1]],
            vmin=0., vmax=1.)
        ax[indx].axvline(0, color='k')
        ax[indx].axhline(0, color='k')
        ax[indx].set_title('%s' % (label))
    cb_ax = fig.add_axes([1.0, 0.1, 0.02, 0.8])
    cbar = fig.colorbar(im, cax=cb_ax)
    cbar.set_label('AUC')
    fig.supxlabel('Testing Time (s)')
    fig.supylabel('Training Time (s)')
    plt.suptitle('Temporal Generalization')
    plt.tight_layout()

    title = 'gen_AUC_subjAvg.png'
    if tag_savefig == 1:
        fig.savefig(os.path.join(decoFigPath_ch, title),
                    bbox_inches='tight', dpi=300)
    plt.show(block=show_flg)
    plt.close('all')

    # plot time-by-time decoding
    acc_df_deco = acc_df[
        (acc_df['pred'] == 'recog') &
        (acc_df['subj'] == 'mean')]
    acc_subjAll_cond = dict_to_arr(acc_subjAll_dict)
    acc_sig, sig_grp = clu_permu_cond(acc_subjAll_cond)
    acc_df_deco.loc[:, 'diff_sig'] = sig_grp * 8

    sig_df = acc_df_deco[
        (acc_df_deco['sig_label'] == 1) &
        (acc_df_deco['acc'] > chance_crit)]

    clr = 'crimson'
    fig, ax = plt.subplots(
        2, 4, sharex=True, sharey=True, figsize=(9, 6))
    ax = ax.ravel()
    for indx, label in enumerate(recog_label_list):
        ax[indx].axhline(0.5, color='k', linestyle='--',
                         label='Chance level')
        ax[indx].axvline(0.0, color='k', linestyle=':')

        ax[indx].plot(t_list, recog_mean_scores[label], clr,
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
                              recog_mean_scores[label] - recog_sem_scores[label],
                              recog_mean_scores[label] + recog_sem_scores[label],
                              color=clr, alpha=0.1,
                              edgecolor='none')
    title = 'within_recog_AUC_subjAvg.png'
    if tag_savefig == 1:
        fig.savefig(os.path.join(decoFigPath_ch, title),
                    bbox_inches='tight', dpi=300)
    plt.show(block=show_flg)
    plt.close('all')

    print('EACH CONDITION DECODING FINISHED ***')
    print('***')
    print('**')
    print('*')

    print('3. CATEGORY DECODING START **')
    acc_mean = dict()
    acc_subjAll_dict, recogCate_mean_scores, recogCate_sem_scores = \
        dict(), dict(), dict()
    for label in cond_label_list:
        acc_subjAll = np.zeros(
            [subjAllN_final, t_points, t_points])
        acc_subjAll_diag = np.zeros(
            [subjAllN_final, t_points])

        for n in range(subjAllN_final):
            scores, acc_subjAll[n], acc_subjAll_diag[n] = \
                gen_decoding_in(recogCate_data_dict[label][n],
                                recogCate_labels_dict[label][n])
        print('Subject %d Finished' % n)

        np.save(file=os.path.join(
            decoDataPath,
            'in_auc_recog_%s'%label+dataFileName),
            arr=acc_subjAll)

        print('Condition %s Finished' % label)
        print('*')
        print('*')

        acc_mean[label] = np.mean(acc_subjAll, axis=0)
        recogCate_mean_scores[label] = np.diag(acc_mean[label])
        recogCate_sem_scores[label] = sem(acc_subjAll_diag)
        acc_subjAll_dict[label] = acc_subjAll_diag

        # t-test
        acc_sig, grp_sig = clu_permu_1samp_t(acc_subjAll_diag)
        # check
        acc_df_tmp = pd.DataFrame(
            columns=['tfr','chans', 'pred', 'type', 'subj',
                     'time', 'acc', 'sig_label', 'grp_label'])
        acc_df_tmp['tfr'] = [tfr_tag]*t_points
        acc_df_tmp['pred'] = ['recogCate'] * t_points
        acc_df_tmp['type'] = [label] * t_points
        acc_df_tmp['subj'] = ['mean'] * t_points
        acc_df_tmp['time'] = t_list
        acc_df_tmp['acc'] = recogCate_mean_scores[label]
        acc_df_tmp['sig_label'] = acc_sig
        acc_df_tmp['grp_label'] = grp_sig
        acc_df_tmp['chans'] = [pick_tag] * t_points
        acc_df = pd.concat([acc_df, acc_df_tmp],
                           axis=0, ignore_index=True)

    # plot the full (generalization) matrix
    fig, ax = plt.subplots(
        1, 2, sharex=True, sharey=True, figsize=(9, 6))
    ax = ax.ravel()
    for indx, label in enumerate(cond_label_list):
        im = ax[indx].imshow(
            acc_mean[label], interpolation='lanczos',
            origin='lower', cmap='RdBu_r',
            extent=t_list[[0, -1, 0, -1]],
            vmin=0., vmax=1.)
        ax[indx].axvline(0, color='k')
        ax[indx].axhline(0, color='k')
        ax[indx].set_title('%s' % (label))
    cb_ax = fig.add_axes([1.0, 0.1, 0.02, 0.8])
    cbar = fig.colorbar(im, cax=cb_ax)
    cbar.set_label('AUC')
    fig.supxlabel('Testing Time (s)')
    fig.supylabel('Training Time (s)')
    plt.suptitle('Temporal Generalization')
    plt.tight_layout()

    title = 'gen_Cate_AUC_subjAvg.png'
    if tag_savefig == 1:
        fig.savefig(os.path.join(decoFigPath_ch, title),
                    bbox_inches='tight', dpi=300)
    plt.show(block=show_flg)
    plt.close('all')

    # plot time-by-time decoding
    acc_df_deco = acc_df[
        (acc_df['pred'] == 'recogCate') &
        (acc_df['subj'] == 'mean')]
    acc_subjAll_cond = dict_to_arr(acc_subjAll_dict)
    acc_sig, sig_grp = clu_permu_cond(acc_subjAll_cond)
    acc_df_deco.loc[:, 'diff_sig'] = sig_grp * 2

    sig_df = acc_df_deco[
        (acc_df_deco['sig_label'] == 1) &
        (acc_df_deco['acc'] > chance_crit)]

    clrList = ['orange', 'g']
    fig, ax = plt.subplots(
        1, 1, sharex=True, sharey=True, figsize=(9, 6))
    ax.axhline(0.5, color='k', linestyle='--',
               label='Chance level')
    ax.axvline(0.0, color='k', linestyle=':')

    ymin_show = 0.48
    for clr, label in zip(clrList, cond_label_list):
        ax.plot(t_list, recogCate_mean_scores[label], clr,
                linewidth=l_wid, label=label)

        ymin, ymax = ax.get_ylim()
        for k in set(sig_df['grp_label']):
            sig_times = sig_df.loc[
                (sig_df['grp_label'] == k) &
                (sig_df['type'] == label), 'time']
            ax.plot(sig_times, [ymin_show] * len(sig_times),
                    clr, linewidth=l_wid_acc,
                    label=label)
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
                        recogCate_mean_scores[label] - recogCate_sem_scores[label],
                        recogCate_mean_scores[label] + recogCate_sem_scores[label],
                        color=clr, alpha=0.1,
                        edgecolor='none')
    title = 'within_recogCate_AUC_subjAvg.png'
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
    acc_mean = dict()
    acc_subjAll_dict, recogSize_mean_scores, recogSize_sem_scores = \
        dict(), dict(), dict()
    for label in sizeList:
        acc_subjAll = np.zeros(
            [subjAllN_final, t_points, t_points])
        acc_subjAll_diag = np.zeros(
            [subjAllN_final, t_points])

        for n in range(subjAllN_final):
            scores, acc_subjAll[n], acc_subjAll_diag[n] = \
                gen_decoding_in(recogSize_data_dict[label][n],
                                recogSize_labels_dict[label][n])
        print('Subject %d Finished' % n)

        np.save(file=os.path.join(
            decoDataPath,
            'in_auc_recog_%d'%label+dataFileName),
            arr=acc_subjAll)

        print('Condition %s Finished' % label)
        print('*')
        print('*')

        acc_mean[label] = np.mean(acc_subjAll, axis=0)
        recogSize_mean_scores[label] = np.diag(acc_mean[label])
        recogSize_sem_scores[label] = sem(acc_subjAll_diag)
        acc_subjAll_dict[label] = acc_subjAll_diag

        # t-test
        acc_sig, grp_sig = clu_permu_1samp_t(acc_subjAll_diag)
        # save
        acc_df_tmp = pd.DataFrame(
            columns=['tfr','chans', 'pred', 'type', 'subj',
                     'time', 'acc', 'sig_label', 'grp_label'])
        acc_df_tmp['tfr'] = [tfr_tag]*t_points
        acc_df_tmp['pred'] = ['recogSize'] * t_points
        acc_df_tmp['type'] = [label] * t_points
        acc_df_tmp['subj'] = ['mean'] * t_points
        acc_df_tmp['time'] = t_list
        acc_df_tmp['acc'] = recogSize_mean_scores[label]
        acc_df_tmp['sig_label'] = acc_sig
        acc_df_tmp['grp_label'] = grp_sig
        acc_df_tmp['chans'] = [pick_tag] * t_points
        acc_df = pd.concat([acc_df, acc_df_tmp],
                           axis=0, ignore_index=True)

    # plot the full (generalization) matrix
    fig, ax = plt.subplots(
        2, 2, sharex=True, sharey=True, figsize=(9, 9))
    ax = ax.ravel()
    for indx, label in enumerate(sizeList):
        im = ax[indx].imshow(
            acc_mean[label], interpolation='lanczos',
            origin='lower', cmap='RdBu_r',
            extent=t_list[[0, -1, 0, -1]],
            vmin=0., vmax=1.)
        ax[indx].axvline(0, color='k')
        ax[indx].axhline(0, color='k')
        ax[indx].set_title('%s' % (label))
    cb_ax = fig.add_axes([1.0, 0.1, 0.02, 0.8])
    cbar = fig.colorbar(im, cax=cb_ax)
    cbar.set_label('AUC')
    fig.supxlabel('Testing Time (s)')
    fig.supylabel('Training Time (s)')
    plt.suptitle('Temporal Generalization')
    plt.tight_layout()

    title = 'gen_Size_AUC_subjAvg.png'
    if tag_savefig == 1:
        fig.savefig(os.path.join(decoFigPath_ch, title),
                    bbox_inches='tight', dpi=300)
    plt.show(block=show_flg)
    plt.close('all')

    # plot time-by-time decoding
    acc_df_deco = acc_df[
        (acc_df['pred'] == 'recogSize') &
        (acc_df['subj'] == 'mean')]
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
        ax.plot(t_list, recogSize_mean_scores[label],
                clr, linewidth=l_wid, label=label)

        ymin, ymax = ax.get_ylim()
        for k in set(sig_df['grp_label']):
            sig_times = sig_df.loc[
                (sig_df['grp_label'] == k) &
                (sig_df['type'] == label), 'time']
            ax.plot(sig_times, [ymin_show] * len(sig_times),
                    clr, linewidth=l_wid_acc,
                    label=label)
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
                        recogSize_mean_scores[label] - recogSize_sem_scores[label],
                        recogSize_mean_scores[label] + recogSize_sem_scores[label],
                        color=clr, alpha=0.1,
                        edgecolor='none')
    title = 'within_recogSize_AUC_subjAvg_all.png'
    if tag_savefig == 1:
        fig.savefig(os.path.join(decoFigPath_ch, title),
                    bbox_inches='tight', dpi=300)
    plt.show(block=show_flg)
    plt.close('all')

    print('SETSIZE DECODING FINISHED ***')
    print('***')
    print('**')
    print('*')

    # save ACC files
    if tag_savefile==1:
        if os.path.isfile(os.path.join(resPath,fileName)):
            acc_df.to_csv(os.path.join(resPath,fileName),
                          mode='a',header=False,index=False)
        else:
            acc_df.to_csv(os.path.join(resPath,fileName),
                          mode='w',header=True,index=False)
    print('%s FINISHED ***'%pick_tag)


print('*** GENERALIZATION DECODING FINISHED ***')
print('***')
print('***')
print('***')
print('')
