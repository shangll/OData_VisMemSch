#!/usr/bin/env python
# -*-coding:utf-8 -*-

# EXP.4 (EEG): configure
# 2023.Mar.9
# linlin.shang@donders.ru.nl


from exp4_config import subjAllN,subjList,sizeList,cateList,condList,\
    filePath,rawEEGPath,epoDataPath,resPath,preFigPath,fileList_EEG,\
    montageFile,set_filepath,badChanDict,event_dict_all,\
    baseline,tmin,tmax,fmin,fmax,power_line_noise_freq,\
    sampleNum,chanN,ica_num,random_num,reject_criteria,\
    p_crit,crit_rt,crit_sd,crit_acc,save_fig,show_flg

import mne
import random
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl

import os

from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = "all"


#%% --- --- --- --- --- --- SUB FUNCTIONS --- --- --- --- --- ---
def raw_check(rawData,title,figName):
    subjN = title.strip('Subject ')
    subjN_indx = subjN.find(':')
    subjN = int(subjN[0:subjN_indx])

    events,evt_id = mne.events_from_annotations(rawData)
    events = mne.pick_events(
        events,include=list(event_dict_all.values()))
    epochs_check = mne.Epochs(
        rawData,events,event_id=event_dict_all,
        tmin=tmin,tmax=tmax,baseline=None,preload=True)
    # PSD
    fig,ax = plt.subplots(1,figsize=(11,5))
    epochs_check.plot_psd(ax=ax,
                          fmax=fmax,
                          average=False,
                          spatial_colors=True,
                          show=show_flg)
    ax.set_title(title)
    plt.tight_layout()
    save_fig(fig,figName+'psd_subj%d.png'%subjN)
    # evoked
    fig,ax = plt.subplots(1,figsize=(11,5))
    epochs_check.copy().pick_channels(
        []).average().plot(axes=ax,
                           titles=title,
                           show=show_flg,
                           spatial_colors=True)
    save_fig(fig,figName+'evk_subj%d.png'%subjN)

    del epochs_check

def epo_check(epoData,title,figName):
    subjN = title.strip('Subject ')
    subjN_indx = subjN.find(':')
    subjN = int(subjN[0:subjN_indx])
    # PSD
    fig,ax = plt.subplots(1,figsize=(11,5))
    epoData.copy().plot_psd(ax=ax,
                            fmax=fmax,
                            average=False,
                            spatial_colors=True,
                            show=show_flg)
    ax.set_title(title)
    plt.tight_layout()
    save_fig(fig,figName+'psd_subj%d.png'%subjN)
    # evoked
    fig,ax = plt.subplots(1,figsize=(11,5))
    epoData.copy().pick_channels(
        []).average().plot(axes=ax,
                           titles=title,
                           show=show_flg,
                           spatial_colors=True)
    save_fig(fig,figName+'evk_subj%d.png'%subjN)

def find_bad_chans(subjN):
    badChans = {subjN: []}

    flag_find = True
    while flag_find:
        badChan = str(
            input('enter bad channel one by one:'))
        if badChan == '':
            flag_find = False
        else:
            if subjN in badChans.keys():
                badChans[subjN].append(badChan)
            else:
                badChans[subjN] = [badChan]
    return badChans

def get_ICA(epoData,subjN):
    fig_pref = os.path.join(preFigPath,'7_ICA_')
    fig_suff = '_subj%d.png'%subjN

    # --- --- --- 8.1 EOG
    ica = mne.preprocessing.ICA(
        n_components=ica_num,
        random_state=random_num)
    ica.fit(epoData)

    # choose components： auto-
    ica.exclude = []
    eog_indices,eog_scores = \
        ica.find_bads_eog(epoData)
    print('*** *** ***')
    print('excluded EOG:')
    print('*** *** ***')
    print(eog_indices)

    # plot sources
    fig = ica.plot_sources(
        epoData,show=show_flg,
        show_scrollbars=True,
        title='Subject %d: EOG Sources'%subjN)
    figName = fig_pref+'eog_source'+fig_suff
    save_fig(fig,figName)

    # plot all components
    fig = ica.plot_components(show=show_flg)
    figName = fig_pref+'eog_cps'+fig_suff
    save_fig(fig,figName)

    # plot each property
    fig = ica.plot_properties(
        epoData,picks=list(
            range(0,len(eog_scores[1]))),
        show=show_flg)
    figName = fig_pref+'eog_props'+fig_suff
    save_fig(fig,figName)

    # plot scores
    fig = ica.plot_scores(
        eog_scores,
        title='Subject %d: eog scores'%subjN,
        show=show_flg)
    figName = fig_pref+'eog_scores'+fig_suff
    save_fig(fig,figName)

    # --- --- --- 8.2 remove noises
    # choose components： manually
    del_indices = []
    flag_del = True
    while flag_del:
        del_cpn = str(input('enter components one by one:'))
        if del_cpn == '':
            flag_del = False
        else:
            del_indices.append(int(del_cpn))
    ica.exclude = del_indices

    # plot overlay
    fig = ica.plot_overlay(
        epoData.copy().average(),exclude=del_indices,
        picks='eeg',title='Subject %d: overlay'%subjN,
        show=show_flg)
    figName = fig_pref+'overlay'+fig_suff
    save_fig(fig,figName)

    print('*** *** ***')
    print('excluded ICAs (all):')
    print('*** *** ***')
    print(ica.exclude)

    ica.apply(epoData)
    return epoData

def findDelIndx(subjN, df):
    if subjN in df['subj'].tolist():
        dropList = df.loc[
            df['subj']==subjN,'del_indx'].tolist()
    else:
        dropList = []
    return dropList


#%% --- --- --- --- --- --- MAIN FUNCTION --- --- --- --- --- ---

del_indx_df = pd.read_csv(os.path.join(resPath,'del_indx.csv'),sep=',')

print('***')
print('*** 1. PRE-PROCESSING START ***')
print('***')

subjList = [7]
fileList_EEG = ['subj%d.vhdr'%subjList[0]]

for subjN,fileName in zip(subjList,fileList_EEG):

    # --- --- --- --- --- --- 1.1 Loading Files

    figName = os.path.join(preFigPath,'1_raw_')

    print('***')
    print('*** 1.1 SUBJECT %d, LOADING RAW FILE %s ***'
          %(subjN,fileName))

    fullDataPath = os.path.join(rawEEGPath,fileName)
    raw = mne.io.read_raw_brainvision(
        fullDataPath,eog=(),preload=True,verbose=False)

    title = 'Subject %d: Raw Data'%subjN
    raw_check(raw,title,figName)

    fig = raw.copy().plot(
        n_channels=64,title=title,show=show_flg)
    save_fig(fig,figName+'epo_subj%d.png'%subjN)

    print('1.1 FILE IMPORT FINISHED ***')

    # --- --- --- --- --- --- 1.2 Channel Location Assignment

    figName = os.path.join(preFigPath,'2_reset_')

    print('***')
    print('*** 1.2 Channel Location Assignment ***')
    print(raw.info)

    # set bipolar re-reference EOGs
    # (from electrodes taken from cap):
    # Fp1, Fp2 (vertical);
    # FT10, FT9 (horizontal))
    raw = mne.set_bipolar_reference(raw,['HEOGR','VEOGUP'],
                                    ['HEOGL','VEOGDO'],
                                    ch_name=['hEOG','vEOG'])
    raw.set_channel_types({'vEOG':'eog','hEOG':'eog'})
    custom64 = mne.channels.read_custom_montage(montageFile)

    # set montage
    raw = raw.set_montage(montage=custom64,match_case=True)

    # plot real sensor locations
    fig = raw.plot_sensors(
        title='Subject %d: Real Montage'%subjN,
        show_names=True,show=show_flg)
    save_fig(fig,figName+'montage_subj%d.png'%subjN)

    # plot raw data: PSD & evoked check
    title = 'Subject %d: Re-Set Sensor Locations'%subjN
    raw_check(raw,title,figName)
    # epochs
    evt_check,evt_id_check = mne.events_from_annotations(raw)
    evt_check = mne.pick_events(
        evt_check,include=list(event_dict_all.values()))
    epochs_check = mne.Epochs(
        raw,evt_check,event_id=event_dict_all,
        tmin=tmin,tmax=tmax,baseline=None,preload=True)
    epochs_check.copy().plot(
        n_channels=64,title=title,show=show_flg)
    plt.show(block=show_flg)
    plt.close('all')
    del evt_check,evt_id_check,epochs_check

    print('1.2 SET MONTAGE FINISHED ***')

    # --- --- --- --- --- --- 1.3 Interpolation

    figName = os.path.join(preFigPath,'3_intpl_')

    print('***')
    print('*** 1.3 Interpolation ***')

    # ±200 μV
    events_check,evt_id_check = mne.events_from_annotations(raw)
    events_check = mne.pick_events(
        events_check,include=list(event_dict_all.values()))
    epochs_check = mne.Epochs(
        raw,events_check,event_id=event_dict_all,
        tmin=tmin,tmax=0.5,baseline=None,preload=True)
    epochs_check.drop_bad(reject=reject_criteria)
    epochs_check.plot_drop_log(subject='Subject %d'%subjN,
                               show=show_flg)
    plt.show(block=True)
    del epochs_check

    # remove bad channels
    print('*** *** ***')
    # check bad channels
    # flag_bad = int(input(
    #     '1 to add bad channels one by one; 0 to not:'))
    flag_bad = 1
    # flag_bad = 0
    if flag_bad==1:
        # add bad channels
        print(badChanDict[subjN])
        badChans_add = find_bad_chans(subjN)
        badChanDict[subjN] = badChanDict[subjN]+badChans_add[subjN]

    for badChan in badChanDict[subjN]:
        raw.info['bads'].append(badChan)
    print(badChanDict[subjN])

    # interpolation
    print('*** *** ***')
    if len(badChanDict[subjN])>0:
        raw = raw.interpolate_bads(reset_bads=True)
        # plot raw data: PSD & evoked check
        title = 'Subject %d: Interpolation'%subjN
        raw_check(raw,title,figName)

    print('1.3 INTERPOLATION FINISHED ***')

    # --- --- --- --- --- --- 1.4 Filter: Bandstop: 50Hz plus + H&L

    figName = os.path.join(preFigPath,'4_filt_')

    print('***')
    print('*** 1.4 Filter: Bandstop: 50Hz plus + H&L ***')
    # 1.4.1 Notch
    raw_nch = raw.copy().notch_filter(
        freqs=power_line_noise_freq)
    del raw
    # plot raw data: PSD & evoked check
    title = 'Subject %d: Notch 50Hz+'%subjN
    raw_check(raw_nch,title,figName+'nch_')

    # 1.4.2 high- & low-pass
    raw_nch_filt = raw_nch.copy().filter(
        l_freq=fmin,h_freq=fmax)
    del raw_nch
    # plot raw data: PSD & evoked check
    title = 'Subject %d: High- & Low-Pass'%subjN
    raw_check(raw_nch_filt,title,figName+'pass_')

    print('1.4 ALL FILTER FINISHED ***')

    # --- --- --- --- --- --- 1.5 Down sampling to 250

    figName = os.path.join(preFigPath,'5_ds_')

    print('***')
    print('*** 1.5 Down sampling to 250 ***')

    raw_nch_filt_ds = raw_nch_filt.copy().resample(sfreq=sampleNum)
    del raw_nch_filt

    # plot raw data: PSD & evoked check
    title = 'Subject %d: Down Sample'%subjN
    raw_check(raw_nch_filt_ds,title,figName)

    print('1.5 RESAMPLE TO 250 FINISHED ***')

    # --- --- --- --- --- --- 1.6 Epoch Segmentation

    figName = os.path.join(preFigPath,'6_epo_')

    print('***')
    print('*** 1.6 Epoch Segmentation ***')

    # get epoch parameters
    events,evt_id = mne.events_from_annotations(raw_nch_filt_ds)
    events_all = mne.pick_events(
        events,include=list(event_dict_all.values()))
    picks = mne.pick_types(raw_nch_filt_ds.info,
                           eeg=True,eog=True,
                           exclude='bads')

    # plot events
    clrList = random.sample(list(mpl.colors.cnames.keys()),
                            len(event_dict_all))
    clrDict = {list(event_dict_all.values())[x]:
                   clrList[x] for x in range(len(clrList))}
    fig = mne.viz.plot_events(
        events_all,sfreq=raw_nch_filt_ds.info['sfreq'],
        first_samp=raw_nch_filt_ds.first_samp,
        show=show_flg,event_id=event_dict_all,
        color=clrDict)
    save_fig(fig,figName+'evt_subj%d.png'%subjN)

    # 1.6.1 get epochs
    epochs = mne.Epochs(raw_nch_filt_ds,events,
                        event_id=event_dict_all,
                        tmin=tmin,tmax=tmax,
                        picks=picks,baseline=None,
                        reject_by_annotation=True,
                        preload=True)
    del raw_nch_filt_ds

    # plot PSD & evoked check
    title = 'Subject %d: Epoch'%subjN
    epo_check(epochs,title,figName)

    # 1.6.2 reject incorrect epochs
    dropList = findDelIndx(subjN,del_indx_df)
    epochs.drop(dropList,reason='ACC=0')
    fig = epochs.plot_drop_log(
        subject='Subject %d'%subjN,show=show_flg)
    save_fig(fig,figName+'rej_log_acc_subj%d.png'%subjN)

    # plot PSD & evoked check
    title = 'Subject %d: Reject Incorrect Epochs'%subjN
    epo_check(epochs,title,figName+'rej_acc_')

    # 1.6.3 reject epochs for cleaning
    '''
    # drop ±300
    epochs.drop_bad(reject=reject_criteria)
    fig = epochs.plot_drop_log(subject='%d'%subjN,
                               ignore=('IGNORED','ACC=0'),
                               show=show_flg)
    save_fig(fig,figName+'_rej_log_500_subj%d.png'%subjN)
    '''
    # trial by trial manually
    title = 'Subject %d'%subjN
    epochs.plot(n_channels=chanN,
                events=epochs.events,
                title=title,show=show_flg)
    plt.show(block=show_flg)
    plt.close('all')
    fig = epochs.plot(n_channels=chanN,
                      events=epochs.events,
                      title=title,show=False)
    plt.show(block=False)
    fig.savefig(figName+'rej_bad_manu_subj%d.png'%subjN,
                bbox_inches='tight',
                dpi=300)
    plt.close('all')

    fig = epochs.plot_drop_log(subject='%d'%subjN,
                               ignore=('IGNORED','ACC=0'),
                               show=show_flg)
    save_fig(fig,figName+'rej_log_bad_subj%d.png'%subjN)

    # plot PSD & evoked check
    title = 'Subject %d: Reject Bad Epochs'%subjN
    epo_check(epochs,title,figName+'rej_bad_')

    print('1.6 SEGMENTATION FINISHED ***')

    # --- --- --- --- --- ---
    # SAVE FILES (NO ICA)
    fname_epo = 's%d_nch_filt_ds_epo.fif'%subjN
    epochs.save(os.path.join(
        epoDataPath,fname_epo),overwrite=True)
    print('FILE (NO ICA) SAVED ***')
    print('***')

    # --- --- --- --- --- --- 1.7 Re-Reference to Average

    figName = os.path.join(preFigPath,'7_ref_')

    print('***')
    print('*** 1.7 Re-Reference to Average ***')

    # Average
    epo = epochs.set_eeg_reference(ref_channels='average')

    '''
    # REST
    epochs.del_proj()
    sphere = mne.make_sphere_model(
        'auto','auto',epochs.info)
    src = mne.setup_volume_source_space(
        sphere=sphere,exclude=30.,pos=15.)
    forward = mne.make_forward_solution(
        epochs.info,trans=None,
        src=src,bem=sphere)

    epo = epo_ICA.set_eeg_reference('REST',forward=forward)

    # plot raw data: PSD & evoked check
    title = 'Subject %d: Re-Reference (REST)'%subjN
    epo_check(epo,title,figName)
    '''

    # plot raw data: PSD & evoked check
    title = 'Subject %d: Re-Reference (Average)'%subjN
    epo_check(epo,title,figName)

    print('1.7 RE-REFERENCE FINISHED ***')

    # --- --- --- --- --- ---
    # SAVE FILES (no ICA)
    fname_epo = 'subj%d_ref_epo.fif'%subjN
    epo.save(os.path.join(
        epoDataPath,fname_epo),overwrite=True)
    print('FILE (no ICA) SAVED ***')
    print('***')

    # --- --- --- --- --- --- 1.8 Baseline Correction

    figName = os.path.join(preFigPath,'8_detr_')

    epo.pick_types(eeg=True).apply_baseline(
        baseline=baseline)

    # plot PSD & evoked check
    title = 'Subject %d: Baseline Correction'%subjN
    epo_check(epo,title,figName)

    # --- --- --- --- --- ---
    # SAVE FILES (AFTER ICA)
    fname = 'subj%d_epo.fif'%subjN
    fpath = os.path.join(epoDataPath,fname)
    epo.save(fpath,overwrite=True)
    print('PREPROCESSED FILE SAVED ***')
    print('***')

    print('Subject %d FINISHED, ALL FILES SAVED ***'%subjN)
    print('***\n')
print('1. PRE-PROCESSING FINISHED ***')
print('***')
print('**')
print('*')