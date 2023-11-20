#!/usr/bin/env python
#-*-coding:utf-8 -*-

# EXP. 3-2 (EEG): Decoding
# Temporal dynamics of non-target processing vary with
# memory set size in different category conditions
# 2022.Jan.18
# linlin.shang@donders.ru.nl

#%% --- --- --- --- --- --- libraries --- --- --- --- --- ---
from eeg_config import epoDataPath,decoDataPath,\
    subjList,\
    frontChans,fcpChans,postChans,\
    odd_label_list,recog_labels,\
    cond_label_list,size_label_list,\
    aList,oList,tag_savefile,\
    dict_to_arr
import mne
import numpy as np

import os
import warnings
#warnings.filterwarnings('ignore')



# --- --- --- --- --- --- --- --- --- Set Global Parameters --- --- --- --- --- --- --- --- --- #

annot_kwargs = dict(fontsize=12,fontweight='bold',\
                    xycoords="axes fraction",\
                    ha='right',va='center')

subjList_final = subjList

# --- --- --- --- --- --- --- --- --- Sub-Functions --- --- --- --- --- --- --- --- --- #

def mne_to_dict(fname,pick_chs,labels):
    epoData_dict,epoData_avgt_dict,label_dict,info_dict = \
        dict(),dict(),dict(),dict()
    
    for subjN in subjList_final:
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

simiChans = ['P5','P6','P7','P8',
             'PO3','PO4','PO7','PO8','PO9','PO10',
             'O1','O2']
# for pick_tag in ['eeg','post','front','simi']:
for pick_tag in ['simi']:
    if pick_tag == 'eeg':
        dataFileName = '_all.npy'
        labelFileName = '_lab_all.npy'
        pick_chs = 'eeg'
    elif pick_tag == 'front':
        dataFileName = '_front.npy'
        labelFileName = '_lab_front.npy'
        pick_chs = frontChans
    elif pick_tag == 'fcp':
        dataFileName = '_fcp.npy'
        labelFileName = '_lab_fcp.npy'
        pick_chs = fcpChans
    elif pick_tag == 'post':
        dataFileName = '_post.npy'
        labelFileName = '_lab_post.npy'
        pick_chs = postChans
    elif pick_tag == 'simi':
        dataFileName = '_simi.npy'
        labelFileName = '_lab_simi.npy'
        pick_chs = simiChans

    print('*** 2.1 LOADING FILES ***')
    fname = '_epo.fif'

    print('2.1.1 ODDBALL ***')
    # oddball task
    odd_info_dict, odd_data_dict, odd_label_dict, odd_times = \
        mne_to_dict(fname, pick_chs, odd_label_list)
    odd_data = dict_to_arr(odd_data_dict)
    odd_labels = dict_to_arr(odd_label_dict)
    odd_labels = change_label(odd_labels, [2], 0)

    np.save(file=os.path.join(
        decoDataPath,'odd'+dataFileName),
        arr=odd_data)
    np.save(file=os.path.join(
        decoDataPath,'odd'+labelFileName),
        arr=odd_labels)

    print(' ODDBALL TASK FINISHED***')
    print('**')

    # --- --- --- --- --- --- Animate vs Inanimate --- --- --- --- --- ---

    print('2.1.2 RECOGNITION ***')
    # recognition task
    for cond in recog_labels:
        recog_info_dict_cond, recog_data_dict_cond, \
        recog_label_dict_cond, recog_times = \
            mne_to_dict(fname, pick_chs, cond)
        recog_data_cond = dict_to_arr(recog_data_dict_cond)
        recog_label_cond = dict_to_arr(recog_label_dict_cond)

        recog_label_cond = change_label(recog_label_cond, aList, 1)
        recog_label_cond = change_label(recog_label_cond, oList, 0)

        np.save(file=os.path.join(
            decoDataPath, 'recog_%s' % cond.replace('/','') + dataFileName),
                arr=recog_data_cond)
        np.save(file=os.path.join(
            decoDataPath, 'recog_%s' % cond.replace('/','') + labelFileName),
                arr=recog_label_cond)
    print('RECOGNITION TASK FINISHED***')
    print('**')

    print('2.1.3 NON-CONDITION RECOGNITION ***')
    # non-condition
    recog_info_all, recog_data_all, recog_label_all, recog_times_all = \
        mne_to_dict(fname, pick_chs, recog_labels)
    recog_data_allCond = dict_to_arr(recog_data_all)
    recog_labels_allCond = dict_to_arr(recog_label_all)
    recog_labels_allCond = change_label(recog_labels_allCond, aList, 1)
    recog_labels_allCond = change_label(recog_labels_allCond, oList, 0)

    np.save(file=os.path.join(decoDataPath, 'recogAll' + dataFileName),
            arr=recog_data_allCond)
    np.save(file=os.path.join(decoDataPath, 'recogAll' + labelFileName),
            arr=recog_labels_allCond)

    print('NON-CONDITION RECOGNITION TASK FINISHED***')
    print('**')


    print('2.1.4 CATEGORY ***')
    # category
    for cond in cond_label_list:
        recog_info_dict_cond, recog_data_dict_cond,\
        recog_label_dict_cond, recog_times = \
            mne_to_dict(fname, pick_chs, cond)
        recog_data_cond = dict_to_arr(recog_data_dict_cond)
        recog_label_cond = dict_to_arr(recog_label_dict_cond)
        recog_label_cond = change_label(recog_label_cond, aList, 1)
        recog_label_cond = change_label(recog_label_cond, oList, 0)

        np.save(file=os.path.join(
            decoDataPath, 'recog_%s' % cond + dataFileName),
                arr=recog_data_cond)
        np.save(file=os.path.join(
            decoDataPath, 'recog_%s' % cond + labelFileName),
                arr=recog_label_cond)

    print('CATEGORY FINISHED***')
    print('**')

    print('2.1.5 SETSIZE***')
    # setsize
    for cond_size in size_label_list:
        if 'w/1' in cond_size:
            cond = '1'
        elif 'w/2' in cond_size:
            cond = '2'
        elif 'w/4' in cond_size:
            cond = '4'
        elif 'w/8' in cond_size:
            cond = '8'
        recog_info_dict_cond, recog_data_dict_cond, \
        recog_label_dict_cond, recog_times = \
            mne_to_dict(fname, pick_chs, cond_size)
        recog_data_size = dict_to_arr(recog_data_dict_cond)
        recog_label_size = dict_to_arr(recog_label_dict_cond)
        recog_label_size = change_label(recog_label_size, aList, 1)
        recog_label_size = change_label(recog_label_size, oList, 0)

        np.save(file=os.path.join(
            decoDataPath, 'recog_%s' % cond + dataFileName),
                arr=recog_data_size)
        np.save(file=os.path.join(
            decoDataPath, 'recog_%s' % cond + labelFileName),
                arr=recog_label_size)

    print('SETSIZE FINISHED***')
    print('**')

    if tag_savefile == 1:
        np.save(file=os.path.join(decoDataPath, 't_list.npy'),
                arr=recog_times)


print('*** ALL THE FILES LOADED***')
print('***')
print('***')
print('***')
print('')
