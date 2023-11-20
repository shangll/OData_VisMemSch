#!/usr/bin/env python
# -*-coding:utf-8 -*-

# EXP. 3 (EEG): 1-pre-processing
# Temporal dynamics of non-target processing vary with
# memory set size in different category conditions
# 2022.04.29
# linlin.shang@donders.ru.nl



#%% --- --- --- --- --- --- libraries --- --- --- --- --- ---

from eeg_config import epoDataPath,decoDataPath,\
    subjList_final,odd_label_list,recog_labels,\
    cond_label_list,size_label_list,size_labels,\
    frontChans,fcpChans,postChans,bands,baseline,\
    dict_to_arr

import mne
from mne.time_frequency import tfr_morlet
import numpy as np

import os

from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = "all"



# --- --- --- --- --- --- --- --- --- Set Global Parameters --- --- --- --- --- --- --- --- --- #

return_tag = False
# mode = 'mean'
mode = 'logratio'


# --- --- --- --- --- --- --- --- --- Main Function --- --- --- --- --- --- --- --- --- #

num = 10

for pick_tag in ['eeg','post','front']:
    if pick_tag == 'eeg':
        pick_chs = 'eeg'
        ch_n = 60
    elif pick_tag == 'front':
        pick_chs = frontChans
        ch_n = len(frontChans)
    elif pick_tag == 'fcp':
        pick_chs = fcpChans
        ch_n = len(fcpChans)
    elif pick_tag == 'post':
        pick_chs = postChans
        ch_n = len(postChans)


    # recognition all conditions
    for l_band,h_band,title in bands:
        # freqs = np.logspace(
        #     *np.log10([l_band,h_band]),num=num)
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

        subj_dict = dict()
        for subjN in subjList_final:
            fname_epo_ICA = 'subj%d_ICA_ref_epo.fif'%subjN
            epochs = mne.read_epochs(
                os.path.join(epoDataPath,fname_epo_ICA))
            epo = epochs[recog_labels]
            epo.resample(sfreq=100)
            del epochs

            if return_tag==True:
                power,itc = tfr_morlet(
                    epo,freqs=freqs,
                    n_cycles=n_cycles,
                    use_fft=True,return_itc=True,
                    picks=pick_chs,average=True)
            else:
                power = tfr_morlet(
                    epo,freqs=freqs,
                    n_cycles=n_cycles,
                    use_fft=True,return_itc=return_tag,
                    picks=pick_chs,average=return_tag)
            power.apply_baseline(baseline=baseline,mode=mode)
            subj_dict[subjN] = np.mean(power.data,axis=2)
        power_arr = dict_to_arr(subj_dict)
        np.save(file=os.path.join(
            decoDataPath,
            'tfr_%s_recogAll_%s.npy'%(
                pick_tag,band_tag)),
            arr=power_arr)

    # recognition 8 conditions
    for label in recog_labels:
        for l_band,h_band,title in bands:
            # freqs = np.logspace(
            #     *np.log10([l_band,h_band]),num=num)
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

            subj_dict = dict()
            for subjN in subjList_final:
                fname_epo_ICA = 'subj%d_ICA_ref_epo.fif'%subjN
                epochs = mne.read_epochs(
                    os.path.join(epoDataPath,fname_epo_ICA))
                epo = epochs[recog_labels]
                epo.resample(sfreq=100)
                del epochs

                if return_tag==True:
                    power,itc = tfr_morlet(
                        epo[label],freqs=freqs,
                        n_cycles=n_cycles,
                        use_fft=True,return_itc=True,
                        picks=pick_chs,average=True)
                else:
                    power = tfr_morlet(
                        epo[label],freqs=freqs,
                        n_cycles=n_cycles,
                        use_fft=True,return_itc=return_tag,
                        picks=pick_chs,average=return_tag)
                power.apply_baseline(baseline=baseline,mode=mode)
                subj_dict[subjN] = np.mean(power.data,axis=2)
            power_arr = dict_to_arr(subj_dict)
            np.save(file=os.path.join(
                decoDataPath,
                'tfr_%s_recog_%s_%s.npy'%(
                    pick_tag,band_tag,
                    label.replace('/',''))),
                arr=power_arr)

    # category conditions
    for label in cond_label_list:
        for l_band,h_band,title in bands:
            # freqs = np.logspace(
            #     *np.log10([l_band,h_band]),num=num)
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

            subj_dict = dict()
            for subjN in subjList_final:
                fname_epo_ICA = 'subj%d_ICA_ref_epo.fif'%subjN
                epochs = mne.read_epochs(
                    os.path.join(epoDataPath,fname_epo_ICA))
                epo = epochs[recog_labels]
                epo.resample(sfreq=100)
                del epochs

                if return_tag==True:
                    power,itc = tfr_morlet(
                        epo[label],freqs=freqs,
                        n_cycles=n_cycles,
                        use_fft=True,return_itc=True,
                        picks=pick_chs,average=True)
                else:
                    power = tfr_morlet(
                        epo[label],freqs=freqs,
                        n_cycles=n_cycles,
                        use_fft=True,return_itc=return_tag,
                        picks=pick_chs,average=return_tag)
                power.apply_baseline(baseline=baseline,mode=mode)
                subj_dict[subjN] = np.mean(power.data,axis=2)
            power_arr = dict_to_arr(subj_dict)
            np.save(file=os.path.join(
                decoDataPath,
                'tfr_%s_recog_%s_%s.npy'%(
                    pick_tag,band_tag,label)),
                arr=power_arr)

    # setsize conditions
    indx = 0
    for label in size_label_list:
        for l_band,h_band,title in bands:
            # freqs = np.logspace(
            #     *np.log10([l_band,h_band]),num=num)
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

            subj_dict = dict()
            for subjN in subjList_final:
                fname_epo_ICA = 'subj%d_ICA_ref_epo.fif'%subjN
                epochs = mne.read_epochs(
                    os.path.join(epoDataPath,fname_epo_ICA))
                epo = epochs[label]
                epo.resample(sfreq=100)
                del epochs

                if return_tag==True:
                    power,itc = tfr_morlet(
                        epo,freqs=freqs,
                        n_cycles=n_cycles,
                        use_fft=True,return_itc=True,
                        picks=pick_chs,average=True)
                else:
                    power = tfr_morlet(
                        epo,freqs=freqs,
                        n_cycles=n_cycles,
                        use_fft=True,return_itc=return_tag,
                        picks=pick_chs,average=return_tag)
                power.apply_baseline(baseline=baseline,mode=mode)
                subj_dict[subjN] = np.mean(power.data,axis=2)
            power_arr = dict_to_arr(subj_dict)
            np.save(file=os.path.join(
                decoDataPath,
                'tfr_%s_recog_%s_%s.npy'%(
                    pick_tag,band_tag,
                    size_labels[indx])),
                arr=power_arr)
        indx += 1

    # ODDBALL TASK
    for l_band,h_band,title in bands:
        # freqs = np.logspace(
        #     *np.log10([l_band,h_band]),num=num)
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

        subj_dict = dict()
        for subjN in subjList_final:
            fname_epo_ICA = 'subj%d_ICA_ref_epo.fif'%subjN
            epochs = mne.read_epochs(
                os.path.join(epoDataPath,fname_epo_ICA))
            epo = epochs[odd_label_list]
            epo.resample(sfreq=100)
            del epochs

            if return_tag==True:
                power,itc = tfr_morlet(
                    epo,freqs=freqs,
                    n_cycles=n_cycles,
                    use_fft=True,return_itc=True,
                    picks=pick_chs,average=True)
            else:
                power = tfr_morlet(
                    epo,freqs=freqs,
                    n_cycles=n_cycles,
                    use_fft=True,return_itc=return_tag,
                    picks=pick_chs,average=return_tag)
            power.apply_baseline(baseline=baseline,mode=mode)
            subj_dict[subjN] = np.mean(power.data,axis=2)
        power_arr = dict_to_arr(subj_dict)
        np.save(file=os.path.join(
            decoDataPath,
            'tfr_%s_odd_%s.npy'%(
                pick_tag,band_tag)),
            arr=power_arr)


print('*** ALL THE FILES SAVED***')
print('***')
print('**')
print('*')
print('')
