#!/usr/bin/env python
# -*-coding:utf-8 -*-

# EXP. 3 (EEG): 1-pre-processing
# Temporal dynamics of non-target processing vary with
# memory set size in different category conditions
# 2022.04.29
# linlin.shang@donders.ru.nl

# %% --- --- --- --- --- --- libraries --- --- --- --- --- ---
from eeg_config import rawEEGPath,epoDataPath,\
    resPath,chkFigPath,\
    fileList_EEG,subjList,montageFile,reset_elect,\
    badChanDict,event_dict_all,\
    tmin,tmax,fmin,fmax,power_line_noise_freq,\
    ica_num,sampleNum,random_num,save_fig

import mne
import random
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl

import os

from IPython.core.interactiveshell import InteractiveShell

InteractiveShell.ast_node_interactivity = "all"

# %% --- --- --- --- --- --- SET PARAMETERS --- --- --- --- --- ---
del_indx_df = pd.read_csv(
    os.path.join(resPath,'del_indx.csv'),sep=',')

tag_savefig = 1
show_flg = True


# %% --- --- --- --- --- --- SUB FUNCTIONS --- --- --- --- --- ---
def raw_check(rawData,title,figName):
    events,evt_id = mne.events_from_annotations(rawData)
    events = mne.pick_events(
        events,include=list(event_dict_all.values()))
    epochs_check = mne.Epochs(
        rawData,events,event_id=event_dict_all,
        tmin=tmin,tmax=tmax,baseline=None,preload=True)
    # evoked
    fig,ax = plt.subplots(1,figsize=(11,5))
    epochs_check.copy().pick_channels(
        []).average().plot(axes=ax,
                           titles=title,
                           show=show_flg,
                           spatial_colors=True)
    save_fig(fig,figName+'.png')
    del epochs_check


def epo_check(epoData,title,figName):
    # evoked
    '''
    fig,ax = plt.subplots(1,figsize=(11,5))
    epoData.copy().pick_channels(
        []).average().plot(axes=ax,
                           titles=title,
                           show=show_flg,
                           spatial_colors=True)
    '''
    fig = epoData.average().plot_joint(
        times=[0.15,0.18,0.2,0.25,0.3,0.4],
        title=title,show=show_flg)
    save_fig(fig,figName+'.png')


def find_bad_chans(subjN):
    badChans = {subjN:[]}

    flag_find = True
    while flag_find:
        badChan = str(
            input('enter bad channel one by one:'))
        if badChan=='':
            flag_find = False
        else:
            if subjN in badChans.keys():
                badChans[subjN].append(badChan)
            else:
                badChans[subjN] = [badChan]
    return badChans


def get_ICA(epoData,subjN,end_tag):
    if end_tag=='Resetting':
        round_tag= 'new'
    else:
        round_tag = 'old'
    figName = os.path.join(
        chkFigPath,'3_ICA_subj%d_%s.png'%(subjN,round_tag))

    # --- --- --- 8.1 EOG
    ica = mne.preprocessing.ICA(
        n_components=ica_num,
        random_state=random_num)
    ica.fit(epoData)

    # plot all components
    fig = ica.plot_components(show=show_flg)
    save_fig(fig,figName)


def findDelIndx(subjN,df):
    if subjN in df['subj'].tolist():
        dropList = df.loc[
            df['subj']==subjN,'del_indx'].tolist()
    else:
        dropList = []
    return dropList


# %% --- --- --- --- --- --- MAIN FUNCTION --- --- --- --- --- ---

print('***')
print('*** 0.1 CHECKING START ***')
print('***')

for subjN,fileName in zip(subjList,fileList_EEG):
    fullDataPath = os.path.join(rawEEGPath,fileName)

    for reset_tag in [0,1]:
        badChanDict[subjN] = []

        # --- --- --- --- --- --- 0.1 Loading Files

        print('*** SUBJECT %d, LOADING RAW FILE %s ***'
              %(subjN,fileName))

        raw = mne.io.read_raw_brainvision(
            fullDataPath,eog=(),preload=True,verbose=False)

        print('0.1 FINISHED ***')

        # --- --- --- --- --- --- 1.2 Channel Location Assignment

        if reset_tag:
            raw = raw.rename_channels(mapping=reset_elect)
            figName1 = os.path.join(chkFigPath,'1_eyeMovChk_subj%d_new'%subjN)
            figName2 = os.path.join(chkFigPath,'2_evk_subj%d_new'%subjN)
            title = 'Subject %d: New'%subjN
            end_tag = 'Resetting'
        else:
            figName1 = os.path.join(chkFigPath,'1_eyeMovChk_subj%d_old'%subjN)
            figName2 = os.path.join(chkFigPath,'2_evk_subj%d_old'%subjN)
            title = 'Subject %d: Old'%subjN
            end_tag = 'Original'

        print('%s ***'%end_tag)
        print('***')
        print('**')
        print('*')

        raw_check(raw,title,figName1)

        raw = mne.set_bipolar_reference(raw,['HEOGR','VEOGUP'],
                                        ['HEOGL','VEOGDO'],
                                        ch_name=['hEOG','vEOG'])
        raw.set_channel_types({'vEOG':'eog','hEOG':'eog'})
        custom64 = mne.channels.read_custom_montage(montageFile)
        raw = raw.set_montage(montage=custom64,match_case=True)

        # --- --- --- --- --- --- 0.3 Interpolation

        flag_bad = 1
        if flag_bad==1:
            # add bad channels
            print(badChanDict[subjN])
            badChans_add = find_bad_chans(subjN)
            badChanDict[subjN] = badChanDict[subjN]+badChans_add[subjN]
        for badChan in badChanDict[subjN]:
            raw.info['bads'].append(badChan)
        print(badChanDict[subjN])
        # interpolation
        if len(badChanDict[subjN])>0:
            raw = raw.interpolate_bads(reset_bads=True)
        print('0.3 INTERPOLATION FINISHED ***')

        # --- --- --- --- --- --- 0.4 Filter: Bandstop: 50Hz plus + H&L

        # 1.4.1 Notch
        raw_nch = raw.copy().notch_filter(
            freqs=power_line_noise_freq)
        del raw
        # 1.4.2 high- & low-pass
        raw_nch_filt = raw_nch.copy().filter(
            l_freq=fmin,h_freq=fmax)
        del raw_nch

        print('0.4 ALL FILTER FINISHED ***')

        # --- --- --- --- --- --- 0.5 Down sampling to 250

        raw_nch_filt_ds = raw_nch_filt.copy().resample(sfreq=sampleNum)
        del raw_nch_filt

        print('0.5 RESAMPLE TO 250 FINISHED ***')

        # --- --- --- --- --- --- 0.6 Epoch Segmentation

        # get epoch parameters
        events,evt_id = mne.events_from_annotations(raw_nch_filt_ds)
        events_all = mne.pick_events(
            events,include=list(event_dict_all.values()))
        picks = mne.pick_types(raw_nch_filt_ds.info,
                               eeg=True,eog=True,
                               exclude='bads')
        # 0.6.1 get epochs
        epochs = mne.Epochs(raw_nch_filt_ds,events,
                            event_id=event_dict_all,
                            tmin=tmin,tmax=tmax,
                            picks=picks,baseline=None,
                            reject_by_annotation=True,
                            preload=True)
        del raw_nch_filt_ds
        # 0.6.2 reject incorrect epochs
        dropList = findDelIndx(subjN,del_indx_df)
        epochs.drop(dropList,reason='ACC=0')
        # plot PSD & evoked check
        epo_check(epochs,title,figName2)

        # --- --- --- --- --- --- 0.7 ICA
        # ICA
        get_ICA(epochs,subjN,end_tag)

        print('%s FINISHED ***'%end_tag)
        print('***\n')
    print('Subject %d FINISHED ***'%subjN)
    print('***\n')
print('0. SENSOR LOCATION CHECK FINISHED ***')
print('***')
print('**')
print('*')