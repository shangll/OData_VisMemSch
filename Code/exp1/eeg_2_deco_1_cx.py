#!/usr/bin/env python
# -*-coding:utf-8 -*-

# EXP. 3-2 (EEG): Decoding
# Temporal dynamics of non-target processing vary with
# memory set size in different category conditions
# 2022.Jan.18
# linlin.shang@donders.ru.nl


#%% --- --- --- --- --- --- libraries --- --- --- --- --- ---
from eeg_config import decoDataPath,resPath,\
    decoFigPath,subjAllN,\
    recog_label_list,\
    cond_label_list,sizeList,\
    tag_savefile,tag_savefig,show_flg,\
    chance_crit,p_crit,p_show,n_permutations,\
    set_filepath,dict_to_arr

from mne.stats import permutation_cluster_1samp_test, \
    permutation_cluster_test,f_threshold_mway_rm,f_mway_rm
from mne.decoding import GeneralizingEstimator

from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

import numpy as np
import pandas as pd
from scipy.stats import sem
import scipy.stats

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

t_list = np.load(file=os.path.join(decoDataPath, 't_list.npy'),
                 allow_pickle=True)
t_points = len(t_list)
t0_indx,t1_indx = np.where(t_list>=0.0)[0][0],\
                  np.where(t_list<=0.6)[0][-1]

l_wid = 4.5
l_wid_acc = 4.5


# --- --- --- --- --- --- --- --- --- Sub-Functions --- --- --- --- --- --- --- --- --- #

def gen_decoding_cx(task1,task2,labels1,labels2):
    clf = make_pipeline(StandardScaler(),
                        LogisticRegression(
                            solver='liblinear'))
    time_gen = GeneralizingEstimator(
        clf,scoring='roc_auc',
        n_jobs=None,verbose=True)

    time_gen.fit(X=task1, y=labels1)
    scores = time_gen.score(X=task2,y=labels2)
    scores_diag = np.diag(scores)

    return scores, scores_diag

def find_sig(clu,clu_p):
    acc_sig, grp_sig = t_points*[0],t_points*[0]
    grp_label = 0

    for c,p in zip(clu, clu_p):
        if p < p_crit:
            grp_label += 1
            acc_sig[(t0_indx+c[0][0]):(t0_indx+c[0][-1]+1)] = \
                [1]*len(c[0])
            grp_sig[(t0_indx+c[0][0]):(t0_indx+c[0][-1]+1)] = \
                [grp_label]*len(c[0])
    return acc_sig, grp_sig

def clu_permu_cond(acc_data_all,stat_fun):
    acc_data = acc_data_all[:,:,t0_indx:t1_indx]

    threshold = None
    f_obs,clu,clu_p,H0 = permutation_cluster_test(
        acc_data,
        n_permutations=n_permutations,
        threshold=threshold,tail=0,
        stat_fun=stat_fun,n_jobs=None,
        out_type='indices')
    print(clu)
    print(clu_p)
    grp_start,grp_end = [],[]
    for c,p in zip(clu,clu_p):
        if p<p_crit:
            grp_start.append(t0_indx[c[0]][0])
            grp_end.append(t0_indx[c[0]][-1])
    print(grp_start,grp_end)

    acc_sig,grp_sig = find_sig(clu,clu_p)
    return acc_sig,grp_sig

def clu_permu_1samp_t(acc_data_all):
    acc_data_all.shape
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


    # CROSS TASKS --- --- --- --- --- --- --- --- ---
    print('*** CROSS TASKS DECODING ***')



    print('1. NON-CONDITION DECODING START **')

    acc_subjAll = np.zeros(
        [subjAllN_final, t_points, t_points])
    acc_subjAll_diag = np.zeros(
        [subjAllN_final, t_points])

    for n in range(subjAllN_final):
        acc_subjAll[n], acc_subjAll_diag[n] = \
            gen_decoding_cx(
            odd_data[n], recogAll_data[n],
            odd_labels[n], recogAll_labels[n])
        print('Subject %d Finished' % n)

    # np.save(file=os.path.join(
    #     decoDataPath,'cx_auc_o2rAll'+dataFileName),
    #     arr=acc_subjAll)

    mean_scores = np.mean(acc_subjAll, axis=0)
    acc_mean = np.diag(mean_scores)
    acc_sem = sem(acc_subjAll_diag)

    acc_df_tmp = pd.DataFrame(
        columns=['tfr', 'chans', 'pred', 'type', 'subj',
                 'time', 'acc', 'sig_label', 'grp_label'])
    # t-test
    acc_sig,grp_sig = clu_permu_1samp_t(acc_subjAll_diag)
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

    # plot the full (generalization) matrix
    fig, ax = plt.subplots(1, 1)
    im = ax.matshow(mean_scores, vmin=0, vmax=1.,
                    cmap='RdBu_r', origin='lower',
                    extent=t_list[[0, -1, 0, -1]])
    ax.axhline(0., color='k')
    ax.axvline(0., color='k')
    ax.xaxis.set_ticks_position('bottom')
    ax.set_xlabel('Testing Time (s)')
    ax.set_ylabel('Training Time (s)')
    # ax.set_title('Generalization across time and condition')
    plt.colorbar(im, ax=ax)
    title = 'gen_o2rAll_AUC_subjAvg.png'
    if tag_savefig == 1:
        fig.savefig(os.path.join(decoFigPath_ch, title),
                    bbox_inches='tight', dpi=300)
    plt.show(block=show_flg)
    plt.close('all')

    # plot time-by-time decoding
    acc_df_deco = acc_df[acc_df['pred'] == 'o2r_all']
    sig_df = acc_df_deco[
        (acc_df_deco['sig_label'] == 1) &
        (acc_df_deco['acc'] > chance_crit)]

    clr = 'crimson'
    mpl.rcParams.update({'font.size':26})
    fig, ax = plt.subplots(
        1, 1, sharex=True, sharey=True, figsize=(20,12))
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
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    plt.suptitle('Cross-Task Decoding Based on Average Data')
    fig.text(0.5,0,'Time (sec)',ha='center')
    fig.text(0,0.5,'AUC',va='center',rotation='vertical')
    plt.axvline(0.1,ls='--',color='grey')
    plt.axvline(0.16,ls='--',color='grey')
    plt.axvline(0.2,ls='--',color='grey')
    plt.axvline(0.3,ls='--',color='grey')
    p1_text = 'P1'
    n1_text = 'N1'
    p2_text = 'P2'
    plt.text(0.115,0.615,p1_text,color='grey',fontsize=18,fontweight='bold')
    plt.text(0.165,0.615,n1_text,color='grey',fontsize=18,fontweight='bold')
    plt.text(0.232,0.615,p2_text,color='grey',fontsize=18,fontweight='bold')
    plt.tight_layout()
    title = 'o2rAll_AUC_subjAvg.png'
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
        acc_subjAll = np.zeros(
            [subjAllN_final, t_points, t_points])
        acc_subjAll_diag = np.zeros(
            [subjAllN_final, t_points])

        for n in range(subjAllN_final):
            acc_df_tmp = pd.DataFrame(
                columns=['tfr','chans','pred','type','subj',
                         'time','acc'])
            acc_subjAll[n], acc_subjAll_diag[n] = \
                gen_decoding_cx(
                    odd_data[n], recog_data_dict[label][n],
                    odd_labels[n], recog_labels_dict[label][n])
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


        mean_scores[label] = np.mean(acc_subjAll,axis=0)
        acc_mean[label] = np.diag(mean_scores[label])
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
        acc_df_tmp['pred'] = ['o2r'] * t_points
        acc_df_tmp['type'] = [label] * t_points
        acc_df_tmp['subj'] = ['mean'] * t_points
        acc_df_tmp['time'] = t_list
        acc_df_tmp['acc'] = acc_mean[label]
        acc_df_tmp['sig_label'] = acc_sig_dict[label]
        acc_df_tmp['grp_label'] = sig_grp_dict[label]
        acc_df_tmp['chans'] = [pick_chs] * t_points
        acc_df = pd.concat([acc_df, acc_df_tmp],
                           axis=0, ignore_index=True)

    
    # plot the full (generalization) matrix
    fig, ax = plt.subplots(
        2, 4, sharex=True, sharey=True, figsize=(20,12))
    ax = ax.ravel()
    for indx, label in enumerate(recog_label_list):
        im = ax[indx].matshow(
            mean_scores[label], vmin=0, vmax=1.,
            cmap='RdBu_r', origin='lower',
            extent=t_list[[0, -1, 0, -1]])
        ax[indx].axhline(0., color='k')
        ax[indx].axvline(0., color='k')
        ax[indx].xaxis.set_ticks_position('bottom')
        # ax[indx].set_xlabel('Testing Time (s)')
        # ax[indx].set_ylabel('Training Time (s)')
        ax[indx].set_title('%s' % (label))
    cb_ax = fig.add_axes([1.0, 0.1, 0.02, 0.8])
    cbar = fig.colorbar(im, cax=cb_ax)
    cbar.set_label('AUC')
    fig.supxlabel('Testing Time (s)')
    fig.supylabel('Training Time (s)')
    # plt.suptitle('Temporal Generalization')
    plt.tight_layout()
    title = 'gen_o2r_AUC_subjAvg.png'
    if tag_savefig == 1:
        fig.savefig(os.path.join(decoFigPath_ch, title),
                    bbox_inches='tight', dpi=300)
    plt.show(block=show_flg)
    plt.close('all')

    acc_df_deco = acc_df[acc_df['pred'] == 'o2r']
    acc_subjAll_cond = dict_to_arr(acc_subjAll_dict)
    def stat_fun_1way(*args):
        factor_levels = [2,4]
        effects = 'A:B'
        return f_mway_rm(
            np.array(args).transpose(1,0,2),factor_levels=factor_levels,
            effects=effects,return_pvals=False)[0]
    acc_sig, sig_grp = clu_permu_cond(acc_subjAll_cond,stat_fun_1way)
    acc_df_deco['diff_sig'] = sig_grp * 8

    sig_df = acc_df_deco[
        (acc_df_deco['sig_label']==1)&
        (acc_df_deco['acc']>chance_crit)]

    # plot time-by-time decoding
    clr = 'crimson'
    mpl.rcParams.update({'font.size':26})
    fig, ax = plt.subplots(
        2, 4, sharex=True, sharey=True, figsize=(20,12))
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
                          clr, linewidth=l_wid_acc)

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
        if label[0]=='w':
            label_pre = 'within/'
        else:
            label_pre = 'between/'
        ax[indx].set_title(label_pre+'%s' % (label[-1]))
        ax[indx].spines['right'].set_visible(False)
        ax[indx].spines['top'].set_visible(False)
    plt.suptitle('Cross-Task Decoding for Each Condition')
    fig.text(0.5,0,'Time (sec)',ha='center')
    fig.text(0,0.5,'AUC',va='center',rotation='vertical')
    plt.tight_layout()
    title = 'o2r_AUC_subjAvg.png'
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
        acc_subjAll = np.zeros(
            [subjAllN_final, t_points, t_points])
        acc_subjAll_diag = np.zeros(
            [subjAllN_final, t_points])

        for n in range(subjAllN_final):
            acc_df_tmp = pd.DataFrame(
                columns=['tfr','chans','pred',
                         'type','subj',
                         'time','acc'])
            acc_subjAll[n], acc_subjAll_diag[n] = \
                gen_decoding_cx(
                    odd_data[n], recogCate_data_dict[label][n],
                    odd_labels[n], recogCate_labels_dict[label][n])

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

            print('Subject %d Finished' % (n))

        # np.save(file=os.path.join(
        #     decoDataPath,
        #     'cx_auc_o2r_%s'%label+dataFileName),
        #     arr=acc_subjAll)
        print('Condition %s Finished' % label)
        print('*')
        print('*')

        mean_scores[label] = np.mean(acc_subjAll, axis=0)
        acc_mean[label] = np.diag(mean_scores[label])
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

    # plot the full (generalization) matrix
    fig, ax = plt.subplots(
        1, 2, sharex=True, sharey=True, figsize=(20, 12))
    ax = ax.ravel()
    for indx, label in enumerate(cond_label_list):
        im = ax[indx].matshow(
            mean_scores[label], vmin=0, vmax=1.,
            cmap='RdBu_r', origin='lower',
            extent=t_list[[0, -1, 0, -1]])
        ax[indx].axhline(0., color='k')
        ax[indx].axvline(0., color='k')
        ax[indx].xaxis.set_ticks_position('bottom')
        # ax[indx].set_xlabel('Testing Time (s)')
        # ax[indx].set_ylabel('Training Time (s)')
        ax[indx].set_title('%s' % (label))
    cb_ax = fig.add_axes([1.0, 0.1, 0.02, 0.8])
    cbar = fig.colorbar(im, cax=cb_ax)
    cbar.set_label('AUC')
    fig.supxlabel('Testing Time (s)')
    fig.supylabel('Training Time (s)')
    # plt.suptitle('Temporal Generalization')
    plt.tight_layout()
    title = 'gen_o2rCate_AUC_subjAvg.png'
    if tag_savefig == 1:
        fig.savefig(os.path.join(decoFigPath_ch, title),
                    bbox_inches='tight', dpi=300)
    plt.show(block=show_flg)
    plt.close('all')

    acc_df_deco = acc_df[acc_df['pred'] == 'o2r_cate']
    acc_subjAll_cond = dict_to_arr(acc_subjAll_dict)

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
    acc_df_deco.loc[:,'diff_sig'] = sig_grp * 2

    sig_df = acc_df_deco[
        (acc_df_deco['sig_label'] == 1) &
        (acc_df_deco['acc'] > chance_crit)]

    # plot time-by-time decoding
    clrs_all = sns.color_palette('Blues',n_colors=35)
    clrList = [clrs_all[20],clrs_all[26]]
    lineStyList = ['-','--']
    mpl.rcParams.update({'font.size':26})
    fig, ax = plt.subplots(
        1, 1, sharex=True, sharey=True, figsize=(20,12))
    ymin_show = 0.48
    count = 0
    text_x = [0.62,0.6]
    text_y = [0.481,0.472]
    for lineSty,clr,label in zip(lineStyList,clrList,cond_label_list):
        ax.plot(t_list, acc_mean[label],color=clr,linestyle=lineSty,
                linewidth=l_wid,label=label)
        ymin, ymax = ax.get_ylim()
        for k in set(sig_df['grp_label']):
            sig_times = sig_df.loc[
                (sig_df['grp_label'] == k) &
                (sig_df['type'] == label), 'time']
            ax.plot(sig_times, [ymin_show] * len(sig_times),
                    color=clr,linestyle=lineSty,linewidth=l_wid_acc)
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
                        color=clr, alpha=0.2,
                        edgecolor='none')
        plt.text(text_x[count],text_y[count],
                 '%.2f-%.2f'%(sig_times.tolist()[0],
                              sig_times.tolist()[-1]),
                 color=clr)
        count += 1
    # ymin,ymax = ax.get_ylim()
    # ax.fill_between(
    #     t_list,ymin,ymax,
    #         where=(t_list>=0.16)&(t_list<0.2),
    #         color='grey',alpha=0.1)
    # ax.grid(True)
    ax.legend(loc='best')
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    h,_ = ax.get_legend_handles_labels()
    ax.legend(h,['within','between'],
              loc='best',ncol=1).set_title('Category')
    ax.axhline(0.5,color='k',linestyle=':',
               label='Chance level')
    ax.axvline(0.0,color='k',linestyle=':')
    # plt.ylim(0.46,0.66)
    ax.set_yticks(np.arange(0.46,0.66,0.04))
    plt.suptitle('Cross-Task Decoding for Each Category')
    fig.text(0.5,0,'Time (sec)',ha='center')
    fig.text(0,0.5,'AUC',va='center',rotation='vertical')
    plt.axvline(0.1,ls='--',color='grey')
    plt.axvline(0.16,ls='--',color='grey')
    plt.axvline(0.2,ls='--',color='grey')
    plt.axvline(0.3,ls='--',color='grey')
    p1_text = 'P1'
    n1_text = 'N1'
    p2_text = 'P2'
    plt.text(0.115,0.615,p1_text,color='grey',fontsize=18,fontweight='bold')
    plt.text(0.165,0.615,n1_text,color='grey',fontsize=18,fontweight='bold')
    plt.text(0.232,0.615,p2_text,color='grey',fontsize=18,fontweight='bold')
    plt.tight_layout()
    title = 'o2rCate_AUC_subjAvg.png'
    if tag_savefig == 1:
        fig.savefig(os.path.join(decoFigPath_ch, title),
                    bbox_inches='tight',dpi=300)
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
        acc_subjAll = np.zeros(
            [subjAllN_final,t_points,t_points])
        acc_subjAll_diag = np.zeros(
            [subjAllN_final,t_points])

        for n in range(subjAllN_final):
            acc_df_tmp = pd.DataFrame(
                columns=['tfr','chans','pred',
                         'type','subj',
                         'time','acc'])
            acc_subjAll[n], acc_subjAll_diag[n] = \
                gen_decoding_cx(
                    odd_data[n], recogSize_data_dict[label][n],
                    odd_labels[n], recogSize_labels_dict[label][n])

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
            print('Subject %d Finished' % (n))

        # np.save(file=os.path.join(
        #     decoDataPath,
        #     'cx_auc_o2r_%d'%label+dataFileName),
        #     arr=acc_subjAll)

        print('Condition: %d Finished' % label)
        print('*')
        print('*')

        mean_scores[label] = np.mean(acc_subjAll, axis=0)
        acc_mean[label] = np.diag(mean_scores[label])
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
        acc_df = pd.concat([acc_df, acc_df_tmp],
                           axis=0, ignore_index=True)

    # plot the full (generalization) matrix
    fig, ax = plt.subplots(
        2, 2, sharex=True, sharey=True, figsize=(20,12))
    ax = ax.ravel()
    for indx, label in enumerate(sizeList):
        im = ax[indx].matshow(
            mean_scores[label],vmin=0, vmax=1.,
            cmap='RdBu_r', origin='lower',
            extent=t_list[[0, -1, 0, -1]])
        ax[indx].axhline(0., color='k')
        ax[indx].axvline(0., color='k')
        ax[indx].xaxis.set_ticks_position('bottom')
        # ax[indx].set_xlabel('Testing Time (s)')
        # ax[indx].set_ylabel('Training Time (s)')
        ax[indx].set_title('%s' % (label))
    cb_ax = fig.add_axes([1.0, 0.1, 0.02, 0.8])
    cbar = fig.colorbar(im, cax=cb_ax)
    cbar.set_label('AUC')
    fig.supxlabel('Testing Time (s)')
    fig.supylabel('Training Time (s)')
    # plt.figure(figsize=(20,15))
    # plt.suptitle('Temporal Generalization')
    plt.tight_layout()
    title = 'gen_o2rSize_AUC_subjAvg.png'
    if tag_savefig == 1:
        fig.savefig(os.path.join(decoFigPath_ch, title),
                    bbox_inches='tight', dpi=300)
    plt.show(block=show_flg)
    plt.close('all')

    acc_df_deco = acc_df[acc_df['pred'] == 'o2r_size']
    acc_subjAll_cond = dict_to_arr(acc_subjAll_dict)


    def stat_fun_1way(*args):
        factor_levels = [4,1]
        effects = 'A'
        return f_mway_rm(
            np.array(args).transpose(1,0,2),factor_levels=factor_levels,
            effects=effects,return_pvals=False)[0]


    acc_sig,sig_grp = clu_permu_cond(
        acc_subjAll_cond,stat_fun_1way)
    acc_df_deco.loc[:, 'diff_sig'] = sig_grp * 4

    sig_df = acc_df_deco[
        (acc_df_deco['sig_label'] == 1) &
        (acc_df_deco['acc'] > chance_crit)]

    # clrList = ['crimson', 'gold', 'darkturquoise', 'dodgerblue']
    mpl.rcParams.update({'font.size':26})
    clrs_all_b = sns.color_palette('Blues',n_colors=35)
    clrs_all = sns.color_palette('GnBu_d')
    clrList = [clrs_all_b[28],clrs_all_b[20],clrs_all[2],clrs_all[0]]
    fig, ax = plt.subplots(
        1,1,sharex=True,sharey=True,figsize=(20,12))
    ymin_show = 0.48
    count = 0
    text_x = [0.6,0.6,0.6,0.6]
    text_y = [0.482,0.476,0.47,0.464]
    for clr,label in zip(clrList,sizeList):
        ax.plot(t_list,acc_mean[label],color=clr,
                linewidth=l_wid,label=label)
        ymin,ymax = ax.get_ylim()
        for k in set(sig_df['grp_label']):
            sig_times = sig_df.loc[
                (sig_df['grp_label'] == k) &
                (sig_df['type'] == label), 'time']
            ax.plot(sig_times,[ymin_show]*len(sig_times),
                    color=clr,linewidth=l_wid_acc)
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
        plt.text(text_x[count],text_y[count],
                 '%.2f-%.2f'%(sig_times.tolist()[0],
                              sig_times.tolist()[-1]),
                 color=clr)
        count += 1
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    h,_ = ax.get_legend_handles_labels()
    ax.legend(h,['size 1','size 2','size 4','size 8'],
              loc='best',ncol=2).set_title('Memory Set Size')
    ax.axhline(0.5,color='k',linestyle=':')
    ax.axvline(0.0,color='k',linestyle=':')
    ax.set_yticks(np.arange(0.46,0.66,0.04))
    plt.suptitle('Cross-Task Decoding for Each Memory Set Size')
    fig.text(0.5,0,'Time (sec)',ha='center')
    fig.text(0,0.5,'AUC',va='center',rotation='vertical')
    fig.text(0.5,0,'Time (sec)',ha='center')
    fig.text(0,0.5,'AUC',va='center',rotation='vertical')
    plt.axvline(0.1,ls='--',color='grey')
    plt.axvline(0.16,ls='--',color='grey')
    plt.axvline(0.2,ls='--',color='grey')
    plt.axvline(0.3,ls='--',color='grey')
    p1_text = 'P1'
    n1_text = 'N1'
    p2_text = 'P2'
    plt.text(0.115,0.62,p1_text,color='grey',fontsize=18,fontweight='bold')
    plt.text(0.165,0.62,n1_text,color='grey',fontsize=18,fontweight='bold')
    plt.text(0.232,0.62,p2_text,color='grey',fontsize=18,fontweight='bold')
    plt.tight_layout()
    title = 'o2rSize_AUC_subjAvg.png'
    if tag_savefig == 1:
        fig.savefig(os.path.join(decoFigPath_ch, title),
                    bbox_inches='tight',dpi=300)
    plt.show(block=show_flg)
    plt.close('all')

    print('SETSIZE DECODING FINISHED ***')
    print('***')
    print('**')
    print('*')
    

    # save ACC files
    tag_savefile = 0
    if tag_savefile==1:
        if os.path.isfile(os.path.join(resPath,fileName)):
            acc_df.to_csv(os.path.join(resPath,fileName),
                          mode='a',header=False,index=False)
        else:
            acc_df.to_csv(os.path.join(resPath,fileName),
                          mode='w',header=True,index=False)
    if tag_savefile==1:
        if os.path.isfile(
                os.path.join(resPath,'deco_data_subj.csv')):
            acc_df_subj.to_csv(
                os.path.join(resPath,'deco_data_subj.csv'),
                mode='a',header=False,index=False)
        else:
            acc_df_subj.to_csv(
                os.path.join(resPath,'deco_data_subj.csv'),
                mode='w',header=True,index=False)
    print('%s FINISHED ***'%pick_chs)

print('*** GENERALIZATION DECODING FINISHED ***')
print('***')
print('***')
print('***')
print('')
