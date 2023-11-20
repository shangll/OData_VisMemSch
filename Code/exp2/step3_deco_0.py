#!/usr/bin/env python
# -*-coding:utf-8 -*-

# EXP.4 (EEG): configure
# 2023.Mar.13
# linlin.shang@donders.ru.nl


#%% --- --- --- --- --- --- libraries --- --- --- --- --- ---
from exp4_config import epoDataPath,decoDataPath,subjList,\
    subjList,subjAllN,sizeList,condList,cond_list,cond_size_list,\
    targ_names,wb_list,wb_names,l_list,r_list,targ_list,dict_to_arr
import mne
import numpy as np

import os
import warnings
#warnings.filterwarnings('ignore')


# --- --- --- --- --- --- --- --- --- Set Global Parameters --- --- --- --- --- --- --- --- --- #

annot_kwargs = dict(fontsize=12,fontweight='bold',\
                    xycoords="axes fraction",\
                    ha='right',va='center')

# --- --- --- --- --- --- --- --- --- Sub-Functions --- --- --- --- --- --- --- --- --- #

def mne_to_dict(fname,pick_chs,labels):
    epoData_dict,epoData_avgt_dict,label_dict,info_dict = \
        dict(),dict(),dict(),dict()
    
    for subjN in subjList:
        subj_epo = mne.read_epochs(os.path.join(
            epoDataPath,'subj%d'%subjN+fname))
        subj_epo.resample(sfreq=100)
        
        subj_epo = subj_epo[labels]
        print(subj_epo)
        if pick_chs=='eeg':
            subj_epo = subj_epo.pick_channels(subj_epo.ch_names)
        else:
            subj_epo = subj_epo.pick_channels(pick_chs) 
        #subj_epo.equalize_event_counts(subj_epo.event_id)
        
        info_dict[subjN] = subj_epo.info
        epoData_dict[subjN] = subj_epo.get_data(picks=pick_chs)
        # trials, channels, & timepoints
        print(epoData_dict[subjN].shape)
        
        epoData_avgt_dict[subjN] = epoData_dict[subjN]
        print(epoData_avgt_dict[subjN].shape)
        
        label_dict[subjN] = subj_epo.events[:,2]
        print(label_dict[subjN].shape)
    
    times = subj_epo.times
    return info_dict,epoData_avgt_dict,label_dict,times

def change_label(arr,labels,label_new):
    for label_old in labels:
        for n in range(arr.shape[0]):
            arr_tmp = arr[n]
            arr[n] = np.where(arr_tmp==label_old,label_new,arr_tmp)
    return arr



# --- --- --- --- --- --- --- --- --- Main Function --- --- --- --- --- --- --- --- --- #

chans = ['PO7','PO8']

for pick_tag in ['n2pc','eeg']:
    if pick_tag == 'eeg':
        dataFileName = '_dat_all.npy'
        labelFileName = '_lab_all.npy'
        pick_chs = 'eeg'
    elif pick_tag == 'n2pc':
        dataFileName = '_dat_n2pc.npy'
        labelFileName = '_lab_n2pc.npy'
        pick_chs = chans

    print('*** 3.1 TARGET CONDITION ***')
    fname = '_epo.fif'
    for k,cond in enumerate(targ_list):
        data_info_all,data_dict_cond,label_dict_cond,data_times = \
            mne_to_dict(fname,pick_chs,cond)
        data_cond = dict_to_arr(data_dict_cond)
        label_cond = dict_to_arr(label_dict_cond)

        label_cond = change_label(label_cond,l_list,1)
        label_cond = change_label(label_cond,r_list,0)

        np.save(file=os.path.join(
            decoDataPath,'%s'%targ_names[k]+dataFileName),
                arr=data_cond)
        np.save(file=os.path.join(
            decoDataPath,'%s'%targ_names[k]+labelFileName),
                arr=label_cond)
    print('ALL THE DATA SAVED FOR EACH CONDITION***')
    print('**')

np.save(file=os.path.join(decoDataPath,'t_list.npy'),
        arr=data_times)

for pick_tag in ['n2pc','eeg']:
    if pick_tag == 'eeg':
        dataFileName = '_dat_all.npy'
        labelFileName = '_lab_all.npy'
        pick_chs = 'eeg'
    elif pick_tag == 'n2pc':
        dataFileName = '_dat_n2pc.npy'
        labelFileName = '_lab_n2pc.npy'
        pick_chs = chans

    print('*** 3.1 WB CONDITION ***')
    fname = '_epo.fif'
    for k,cond in enumerate(wb_list):
        data_info_all,data_dict_cond,label_dict_cond,data_times = \
            mne_to_dict(fname,pick_chs,cond)
        data_cond = dict_to_arr(data_dict_cond)
        label_cond = dict_to_arr(label_dict_cond)

        label_cond = change_label(label_cond,l_list,1)
        label_cond = change_label(label_cond,r_list,0)

        np.save(file=os.path.join(
            decoDataPath,'%s'%wb_names[k]+dataFileName),
                arr=data_cond)
        np.save(file=os.path.join(
            decoDataPath,'%s'%wb_names[k]+labelFileName),
                arr=label_cond)
    print('ALL THE DATA SAVED FOR EACH CONDITION***')
    print('**')

print('*** ALL THE FILES LOADED***')
print('***')
print('***')
print('***')
print('')
