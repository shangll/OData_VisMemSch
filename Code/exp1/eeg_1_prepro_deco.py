#!/usr/bin/env python
# -*-coding:utf-8 -*-

# EXP. 3 (EEG): 1-pre-processing
# Temporal dynamics of non-target processing vary with
# memory set size in different category conditions
# 2022.04.29
# linlin.shang@donders.ru.nl

#%% --- --- --- --- --- --- libraries --- --- --- --- --- ---
from eeg_config import rawEEGPath,epoDataPath,\
    resPath,preFigPath,\
    fileList_EEG,subjList,resetList,montageFile,reset_elect,\
    badChanDict,event_dict_all,\
    baseline,tmin,tmax,tmax_ica,fmin,fmax,power_line_noise_freq,\
    sampleNum,chanN,ica_num,random_num,reject_criteria,save_fig

import mne
import random
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl

import os

from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = "all"

#%% --- --- --- --- --- --- SET PARAMETERS --- --- --- --- --- ---
del_indx_df = pd.read_csv(
    os.path.join(resPath,'del_indx.csv'),sep=',')

tag_savefig = 1
show_flg = True


#%% --- --- --- --- --- --- SUB FUNCTIONS --- --- --- --- --- ---
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



#%% --- --- --- --- --- --- MAIN FUNCTION --- --- --- --- --- ---
print('***')
print('*** 1. PRE-PROCESSING START ***')
print('***')

for subjN,fileName in zip(subjList,fileList_EEG):
    if subjN in resetList:
        reset_tag = 1
    else:
        reset_tag = 0

    # --- --- --- --- --- --- 1.1 Loading Files

    print('*** SUBJECT %d, LOADING RAW FILE %s ***'
          %(subjN,fileName))

    fullDataPath = os.path.join(rawEEGPath,fileName)
    raw = mne.io.read_raw_brainvision(
        fullDataPath,eog=(),preload=True,verbose=False)

    print('1.1 FILE IMPORT FINISHED ***')

    # --- --- --- --- --- --- 1.2 Channel Location Assignment

    if reset_tag:
        raw = raw.rename_channels(mapping=reset_elect)
    print(raw.info)

    raw = mne.set_bipolar_reference(raw,['HEOGR','VEOGUP'],
                                    ['HEOGL','VEOGDO'],
                                    ch_name=['hEOG','vEOG'])
    raw.set_channel_types({'vEOG':'eog','hEOG':'eog'})
    custom64 = mne.channels.read_custom_montage(montageFile)

    # set montage
    raw = raw.set_montage(montage=custom64,match_case=True)

    print('1.2 SET MONTAGE FINISHED ***')

    # --- --- --- --- --- --- 1.3 Interpolation

    # remove bad channels
    print('*** *** ***')
    # check bad channels
    # flag_bad = int(input(
    #     '1 to add bad channels one by one; 0 to not:'))
    # flag_bad = 1
    flag_bad = 0
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

    print('1.3 INTERPOLATION FINISHED ***')

    # --- --- --- --- --- --- 1.4 Filter: Bandstop: 50Hz plus + H&L

    # 1.4.1 Notch
    raw_nch = raw.copy().notch_filter(
        freqs=power_line_noise_freq)
    del raw

    # 1.4.2 high- & low-pass
    raw_nch_filt = raw_nch.copy().filter(
        l_freq=fmin,h_freq=fmax)
    del raw_nch

    print('1.4 ALL FILTER FINISHED ***')

    # --- --- --- --- --- --- 1.5 Down sampling to 250

    raw_nch_filt_ds = raw_nch_filt.copy().resample(sfreq=sampleNum)
    del raw_nch_filt

    print('1.5 RESAMPLE TO 250 FINISHED ***')

    # --- --- --- --- --- --- 1.6 Epoch Segmentation

    # get epoch parameters
    events,evt_id = mne.events_from_annotations(raw_nch_filt_ds)
    events_all = mne.pick_events(
        events,include=list(event_dict_all.values()))
    picks = mne.pick_types(raw_nch_filt_ds.info,
                           eeg=True,eog=True,
                           exclude='bads')
    # 1.6.1 get epochs
    epochs = mne.Epochs(raw_nch_filt_ds,events,
                        event_id=event_dict_all,
                        tmin=tmin,tmax=tmax_ica,
                        picks=picks,baseline=None,
                        reject_by_annotation=True,
                        preload=True)
    del raw_nch_filt_ds

    print('1.6 SEGMENTATION FINISHED ***')
    print('***')

    # --- --- --- --- --- --- 1.9 Baseline Correction

    epochs.pick_types(eeg=True).apply_baseline(
        baseline=baseline)

    # --- --- --- --- --- ---
    # SAVE FILES (AFTER ICA)
    fname = 'subj%d_epo_full.fif'%subjN
    fpath = os.path.join(epoDataPath,fname)
    epochs.save(fpath,overwrite=True)
    print('PREPROCESSED FILE SAVED ***')
    print('***')

    print('Subject %d FINISHED, ALL FILES SAVED ***'%subjN)
    print('***\n')
print('1. PRE-PROCESSING FINISHED ***')
print('***')
print('**')
print('*')