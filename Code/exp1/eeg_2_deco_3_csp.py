#!/usr/bin/env python
# -*-coding:utf-8 -*-

# EXP. 3-2 (EEG): Decoding
# Temporal dynamics of non-target processing vary with
# memory set size in different category conditions
# 2022.Oct.10
# linlin.shang@donders.ru.nl



#%% --- --- --- --- --- --- libraries --- --- --- --- --- ---
from eeg_config import epoDataPath,decoDataPath,\
    decoFigPath,subjList_final,subjAllN_final,\
    recog_label_list,recog_labels,aList,oList,\
    tmin,tmax,show_flg,set_filepath,save_fig

import mne
from mne import create_info
from mne.decoding import CSP
from mne.time_frequency import AverageTFR

import numpy as np
import matplotlib.pyplot as plt
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import LabelEncoder

import os



# --- --- --- --- --- --- --- --- --- Sub-Functions --- --- --- --- --- --- --- --- --- #

def change_label(labels):
    arr = np.zeros(labels.shape)
    indx = []

    for label_old in aList:
        indx += np.where(labels==label_old)[0].tolist()
    arr[indx] = 1
    return arr



# --- --- --- --- --- --- Main Function --- --- --- --- --- --- #

fig_pref = os.path.join(
    set_filepath(decoFigPath,'csp','indv'),'csp_')

# Classification & time-frequency parameters
n_freqs = 6  # how many frequency bins to use
min_freq = 1
max_freq = 40
n_cycles = 10.  # how many complete cycles: used to define window size
# freqs = np.linspace(min_freq,max_freq,n_freqs)
freqs = np.array([min_freq,4,8,12,20,max_freq]) # assemble frequencies

for subjN in subjList_final:
    fname = 'subj%d_ICA_ref_epo.fif'%subjN
    fpath = os.path.join(epoDataPath,fname)
    epo = mne.read_epochs(fpath)
    epo_recog = epo[recog_labels]
    print('Subject %d loaded' % subjN)
    # Extract information from the raw file
    sfreq = epo_recog.info['sfreq']

    clf = make_pipeline(CSP(n_components=4,reg=None,log=True,norm_trace=False),
                        LinearDiscriminantAnalysis())
    n_splits = 3  # for cross-validation, 5 is better, here we use 3 for speed
    cv = StratifiedKFold(n_splits=n_splits,shuffle=True,random_state=42)

    # Assemble list of frequency range tuples
    freq_ranges = list(zip(freqs[:-1],freqs[1:]))  # make freqs list of tuples

    # Infer window spacing from the max freq and number of cycles to avoid gaps
    window_spacing = (n_cycles/np.max(freqs)/2.)
    centered_w_times = np.arange(tmin,tmax,window_spacing)[1:]
    n_windows = len(centered_w_times)

    # Instantiate label encoder
    le = LabelEncoder()

    freq_scores = np.zeros((n_freqs-1,))

    # Loop through each frequency range of interest
    for freq,(fmin,fmax) in enumerate(freq_ranges):
        # Infer window size based on the frequency being used
        w_size = n_cycles/((fmax+fmin)/2.)  # in seconds

        # Apply band-pass filter to isolate the specified frequencies
        epochs = epo_recog.copy().filter(
            fmin,fmax,fir_design='firwin',
            skip_by_annotation='edge')

        # Extract epochs from filtered data, padded by window size
        epochs = epochs.crop(tmin=tmin-w_size,tmax=tmax+w_size)
        epo_labels = change_label(epochs.events[:,2])
        y = le.fit_transform(epo_labels)
        X = epochs.get_data()

        # Save mean scores over folds for each frequency and time window
        freq_scores[freq] = np.mean(cross_val_score(
            estimator=clf,X=X,y=y,scoring='roc_auc',cv=cv),axis=0)

    # Plot
    fig,ax = plt.subplots(
        1,1,sharex=True,sharey=True,figsize=(9,6))
    ax.bar(freqs[:-1],freq_scores,width=np.diff(freqs)[0],
            align='edge',edgecolor='black')
    ax.set_xticks(freqs)
    ax.set_ylim([0,1])
    ax.axhline(len(epochs['a'])/len(epochs),color='k',
               linestyle='--',label='chance level')
    ax.legend()
    ax.set_xlabel('Frequency (Hz)')
    ax.set_ylabel('Decoding Scores')
    ax.set_title('Subject %d: Frequency Decoding Scores'%subjN)
    figName = fig_pref+'avg_chk_subj%d.png'%(subjN)
    save_fig(fig,figName)

    # init scores
    tf_scores = np.zeros((n_freqs-1,n_windows))
    fig,ax = plt.subplots(
        1,1,sharex=True,sharey=True,figsize=(9,6))
    for freq,(fmin,fmax) in enumerate(freq_ranges):
        # Infer window size based on the frequency being used
        w_size = n_cycles/((fmax+fmin)/2.)  # in seconds

        # Apply band-pass filter to isolate the specified frequencies
        epochs = epo_recog.copy().filter(
            fmin,fmax,fir_design='firwin',
            skip_by_annotation='edge')

        # Extract epochs from filtered data, padded by window size
        epochs = epochs.crop(tmin=tmin-w_size,tmax=tmax+w_size)
        epo_labels = change_label(epochs.events[:,2])
        y = le.fit_transform(epo_labels)

        # Roll covariance, csp and lda over time
        for t,w_time in enumerate(centered_w_times):
            # Center the min and max of the window
            w_tmin = w_time-w_size/2.
            w_tmax = w_time+w_size/2.

            # Crop data into time-window of interest
            X = epochs.copy().crop(w_tmin,w_tmax).get_data()

            # Save mean scores over folds for each frequency and time window
            tf_scores[freq,t] = np.mean(cross_val_score(
                estimator=clf,X=X,y=y,scoring='roc_auc',cv=cv),axis=0)

    av_tfr = AverageTFR(create_info(['freq'],sfreq),tf_scores[np.newaxis,:],
                        centered_w_times,freqs[1:],1)
    chance = np.mean(y)  # set chance level to white in the plot
    av_tfr.plot([0],vmin=chance,show=show_flg,
                title='Subject %d: Time-Frequency Decoding Scores'%subjN,
                axes=ax,cmap=plt.cm.Reds)
    figName = fig_pref+'avg_deco_subj%d.png'%(subjN)
    save_fig(fig,figName)


    # Each Condition
    fig,ax = plt.subplots(2,4,figsize=(12,9))
    ax = ax.ravel()
    for indx,label in enumerate(recog_labels):
        freq_scores = np.zeros((n_freqs-1,))

        for freq,(fmin,fmax) in enumerate(freq_ranges):
            # Infer window size based on the frequency being used
            w_size = n_cycles/((fmax+fmin)/2.)  # in seconds

            # Apply band-pass filter to isolate the specified frequencies
            epochs = epo_recog[label].copy().filter(
                fmin,fmax,fir_design='firwin',
                skip_by_annotation='edge')

            # Extract epochs from filtered data, padded by window size
            epochs = epochs.crop(tmin=tmin-w_size,tmax=tmax+w_size)
            epo_labels = change_label(epochs.events[:,2])
            y = le.fit_transform(epo_labels)
            X = epochs.get_data()

            # Save mean scores over folds for each frequency and time window
            freq_scores[freq] = np.mean(cross_val_score(
                estimator=clf,X=X,y=y,scoring='roc_auc',cv=cv),axis=0)

        ax[indx].bar(freqs[:-1],freq_scores,width=np.diff(freqs)[0],
                     align='edge',edgecolor='black')
        ax[indx].set_xticks(freqs)
        ax[indx].set_ylim([0,1])
        ax[indx].axhline(len(epochs['a'])/len(epochs),color='k',
                         linestyle='--',label='chance level')
        # ax[indx].legend()
        # ax[indx].set_xlabel('Frequency (Hz)')
        # ax[indx].set_ylabel('Decoding Scores')
        ax[indx].set_title('%s'%label)
    fig.supxlabel('Frequency (Hz)')
    fig.supylabel('Decoding Scores')
    plt.tight_layout()
    figName = fig_pref+'cond_chk_subj%d.png'%(subjN)
    save_fig(fig,figName)

    fig,ax = plt.subplots(2,4,figsize=(12,9))
    ax = ax.ravel()
    for indx,label in enumerate(recog_labels):
        freq_scores = np.zeros((n_freqs-1,))

        for freq,(fmin,fmax) in enumerate(freq_ranges):
            # Infer window size based on the frequency being used
            w_size = n_cycles/((fmax+fmin)/2.)  # in seconds

            # Apply band-pass filter to isolate the specified frequencies
            epochs = epo_recog[label].copy().filter(
                fmin,fmax,fir_design='firwin',
                skip_by_annotation='edge')

            # Extract epochs from filtered data, padded by window size
            epochs = epochs.crop(tmin=tmin-w_size,tmax=tmax+w_size)
            epo_labels = change_label(epochs.events[:,2])
            y = le.fit_transform(epo_labels)

            # Roll covariance, csp and lda over time
            for t,w_time in enumerate(centered_w_times):
                # Center the min and max of the window
                w_tmin = w_time-w_size/2.
                w_tmax = w_time+w_size/2.

                # Crop data into time-window of interest
                X = epochs.copy().crop(w_tmin,w_tmax).get_data()

                # Save mean scores over folds for each frequency and time window
                tf_scores[freq,t] = np.mean(cross_val_score(
                    estimator=clf,X=X,y=y,scoring='roc_auc',cv=cv),axis=0)

        av_tfr = AverageTFR(create_info(['freq'],sfreq),tf_scores[np.newaxis,:],
                            centered_w_times,freqs[1:],1)
        chance = np.mean(y)  # set chance level to white in the plot

        av_tfr.plot([0],vmin=chance,show=show_flg,
                    title='%s'%label,axes=ax[indx],cmap=plt.cm.Reds)
    fig.supxlabel('Frequency (Hz)')
    fig.supylabel('Times')
    plt.tight_layout()
    plt.suptitle('Subject %d: Time-Frequency Decoding Scores'%subjN)
    figName = fig_pref+'cond_deco_subj%d.png'%(subjN)
    save_fig(fig,figName)
