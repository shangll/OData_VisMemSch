#!/usr/bin/env python
# -*-coding:utf-8 -*-

# EXP. 3-3.1 (EEG): evoke
# Temporal dynamics of non-target processing vary with
# memory set size in different category conditions
# 2021.Nov.24
# linlin.shang@donders.ru.nl



#%% --- --- --- --- --- --- libraries --- --- --- --- --- ---
import pandas as pd

from eeg_config import epoDataPath,resPath,\
    tfrFigPath,subjList_final,\
    recog_labels,\
    tag_savefile,show_flg,\
    baseline,bands,save_fig

import mne
from mne.time_frequency import tfr_morlet

import numpy as np

import matplotlib.pyplot as plt

import os



# --- --- --- --- --- --- Set Global Parameters --- --- --- --- --- --- #

# mode = 'mean'
mode = 'logratio'
fmin,fmax = 1,40
freqs = np.arange(fmin,fmax)
n_cycles = freqs/2.

tag_dB = False

pick_chs = 'eeg'
chanN = 60
tfr = pd.DataFrame()
itc_df = pd.DataFrame()



# --- --- --- --- --- --- Main Function --- --- --- --- --- --- #

epochs_list = []
pwr_allSubj,itc_allSubj = dict(),dict()

for subjN in subjList_final:
    fname = 'subj%d_ICA_ref_epo.fif'%subjN
    fpath = os.path.join(epoDataPath,fname)
    epo = mne.read_epochs(fpath)
    epochs_list.append(epo[recog_labels])
    print('Subject %d loaded' % subjN)
epochs = mne.concatenate_epochs(epochs_list)
t_list = epo.times
t_points = len(t_list)



# --- --- --- --- --- --- Frequency

fig_pref = os.path.join(tfrFigPath,'freq_')

# frequency
for tag,tag_figname in zip(
        [True,False],['avg','each']):
    epo = epochs.copy().apply_baseline(baseline)
    fig = epo.copy().pick(
        'eeg').plot_psd(fmin=fmin,fmax=fmax,
                        average=tag,
                        spatial_colors=True)
    # plt.title('EEG')
    figName = fig_pref +'eeg_avg_%s.png'%tag_figname
    save_fig(fig,figName)

# topo
fig = epochs.plot_psd_topomap(
    bands=bands,ch_type='eeg',normalize=False)
figName = fig_pref +'eeg_band_avg.png'
save_fig(fig,figName)



# --- --- --- --- --- --- Time-Frequency

fig_pref = os.path.join(tfrFigPath,'tfr_')

# Plot
# average condition
power_avg,itc_avg = mne.time_frequency.tfr_morlet(
    epochs,n_cycles=n_cycles,use_fft=True,
    return_itc=True,freqs=freqs,
    picks=pick_chs,average=True)

fig = power_avg.plot_topo(
    baseline=baseline,mode=mode,dB=tag_dB,
    title='Average Power')
figName = fig_pref +'pwr_topo_avg.png'
save_fig(fig,figName)

fig = itc_avg.plot_topo(
    title='Inter-Trial coherence',cmap='RdBu_r')
figName = fig_pref +'itc_topo_avg.png'
save_fig(fig,figName)

# 8 conditions
chan = 'PO7'
chan = 'PO8'
fig, ax = plt.subplots(
    2,4,sharex=True,sharey=True,figsize=(12,6))
ax = ax.ravel()
power_dict,itc_dict = dict(),dict()
for indx,label in enumerate(recog_labels):
    power_dict[label],itc_dict[label] = \
        mne.time_frequency.tfr_morlet(
            epochs[label],n_cycles=n_cycles,
            use_fft=True,return_itc=True,
            freqs=freqs,picks=pick_chs,
            average=True)

    power_dict[label].plot(
        picks=chan,baseline=baseline,
        mode=mode,cmap='RdBu_r',
        axes=ax[indx],colorbar=True,
        show=show_flg,
        dB=tag_dB)
    ax[indx].xaxis.set_ticks_position('bottom')
    ax[indx].yaxis.set_ticks_position('left')
    ax[indx].set_xlabel('')
    ax[indx].set_ylabel('')
    ax[indx].set_title('%s'%label)
fig.supxlabel('Time (s)')
fig.supylabel('Frequency (Hz)')
plt.suptitle('TF')
plt.tight_layout()
figName = fig_pref +'pwr_%s.png'%chan
save_fig(fig,figName)


# get power/itc data
for l_band,h_band,title in bands:
    freqs = np.arange(l_band,h_band)
    n_cycles = freqs/2.

    if title=='Delta (1-3.5 Hz)':
        band_tag = 'd'
    elif title=='Theta (4-7 Hz)':
        band_tag = 't'
    elif title=='Alpha (8-12 Hz)':
        band_tag = 'a'
    elif title=='Beta (13-28 Hz)':
        band_tag = 'b'
    else:
        band_tag = 'g'

    for n,label in enumerate(recog_labels):
        for k,subjN in enumerate(subjList_final):
            tfr_subj = pd.DataFrame(columns=['band','type','subj','time'])
            itc_subj = pd.DataFrame(columns=['band','type','subj','time'])
            fname_epo_ICA = 'subj%d_ICA_ref_epo.fif'%subjN
            epochs = mne.read_epochs(
                os.path.join(epoDataPath,fname_epo_ICA))
            epo = epochs[recog_labels]
            del epochs

            power,itc = tfr_morlet(
                epo[label],freqs=freqs,
                n_cycles=n_cycles,
                use_fft=True,return_itc=True,
                picks=pick_chs,average=True)
            power.apply_baseline(baseline=baseline,mode=mode)
            for ch_indx,ch in enumerate(power.ch_names):
                pwr = np.mean(power.data,axis=1)
                itc_phase = np.mean(itc.data,axis=1)
                tfr_subj['band'] = [band_tag]*t_points
                tfr_subj['type'] = [label]*t_points
                tfr_subj['subj'] = [subjN]*t_points
                tfr_subj['time'] = t_list
                tfr_subj[ch] = pwr[ch_indx]

                itc_subj['band'] = [band_tag]*t_points
                itc_subj['type'] = [label]*t_points
                itc_subj['subj'] = [subjN]*t_points
                itc_subj['time'] = t_list
                itc_subj[ch] = itc_phase[ch_indx]

            tfr = pd.concat(
                [tfr,tfr_subj],axis=0,ignore_index=True)
            itc_df = pd.concat(
                [itc_df,itc_subj],axis=0,ignore_index=True)

if tag_savefile == 1:
    tfr.to_csv(os.path.join(resPath,'pwr_eeg.csv'),
               mode='w',header=True,index=False)
    itc_df.to_csv(os.path.join(resPath,'itc_eeg.csv'),
               mode='w',header=True,index=False)
