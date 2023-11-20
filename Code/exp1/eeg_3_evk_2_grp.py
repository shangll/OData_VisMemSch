#!/usr/bin/env python
# -*-coding:utf-8 -*-

# EXP. 3-3.1 (EEG): evoke
# Temporal dynamics of non-target processing vary with
# memory set size in different category conditions
# 2021.Nov.24
# linlin.shang@donders.ru.nl



#%% --- --- --- --- --- --- libraries --- --- --- --- --- ---
from eeg_config import epoDataPath,evkDataPath,resPath,grpFigPath,\
    subjList,frontChans,fcpChans,postChans,recog_labels,\
    show_flg,tag_savefile,set_filepath,save_fig
import mne
import numpy as np
import os

from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = "all"



# --- --- --- --- --- --- Set Global Parameters --- --- --- --- --- --- #
topo_t = np.linspace(0,0.6,9)
lineSty = {}
clr = {}
for cond, line_s, clr_s in zip(recog_labels,
                               ['solid']*4+['dashed']*4,
                               ['crimson','gold','darkturquoise',
                                'dodgerblue'] * 2):
    lineSty[cond] = line_s
    clr[cond] = clr_s



# --- --- --- --- --- --- Main Function --- --- --- --- --- --- #
all_evokeds = [list() for _ in range(len(recog_labels))]
fname_grand = 'grand_avg_recog_ave.fif'

print('***')
print('***')
print('*** 3-2. GROUP LEVEL ***')
print('***')

# -- --- --- --- --- --- 8 conditions
# for subjN in subjList_final:
for subjN in subjList:
    fname = 'subj%d_recog_ave.fif'%subjN
    fpath = os.path.join(evkDataPath,fname)
    evokeds = mne.read_evokeds(fpath)

    assert len(evokeds) == len(all_evokeds)
    for idx,evoked in enumerate(evokeds):
        all_evokeds[idx].append(evoked)

for idx,evokeds in enumerate(all_evokeds):
    all_evokeds[idx] = mne.combine_evoked(evokeds,'equal')

if tag_savefile == 1:
    mne.write_evokeds(os.path.join(evkDataPath,fname_grand),
                      all_evokeds,overwrite=True)

evokeds = mne.read_evokeds(os.path.join(evkDataPath,fname_grand))
mapping = dict()
for n,cond in enumerate(recog_labels):
    mapping[cond] = evokeds[n]

# PLOT
fig_pref = os.path.join(grpFigPath,'grp_')

for labelName,ch_num,pick_chans in zip(['eeg','post','front','fcp'],
                                       ['60 Channels','Posterior Channels',
                                        'Frontal Channels','fcp'],
                                       ['eeg',postChans,frontChans,fcpChans]):
    for combine in ('mean','median','gfp'):
        if combine=='gfp':
            ymin,ymax = 0,7
        else:
            if labelName!='post':
                ymin,ymax = -5,3
            else:
                ymin,ymax = -2,7
        fig = mne.viz.plot_compare_evokeds(mapping,picks=pick_chans,colors=clr,
                                           truncate_xaxis=False,
                                           ylim=dict(eeg=[ymin,ymax]),
                                           linestyles=lineSty,show=show_flg,
                                           show_sensors='upper right',
                                           combine=combine,
                                           title='%s'%(ch_num))
        figName = fig_pref+'erp_%s_%s.png'%(labelName,combine)
        save_fig(fig,figName)

# plot each channel topo
fig = mne.viz.plot_compare_evokeds(mapping,picks='eeg',colors=clr,
                                   linestyles=lineSty,axes='topo',
                                   show=show_flg,
                                   legend=False)
figName = fig_pref+'erp_each.png'
save_fig(fig,figName)

# plot each channel
chans = evokeds[0].ch_names
for chan in chans:
    fig = mne.viz.plot_compare_evokeds(mapping,picks=chan,colors=clr,
                                       linestyles=lineSty,
                                       truncate_xaxis=False,
                                       ylim=dict(eeg=[-7,12]),
                                       show=show_flg,
                                       show_sensors='upper right',
                                       title='%s'%(chan))
    figName = os.path.join(
        set_filepath(grpFigPath,'chan'),'erp_ch_%s.png'%chan)
    save_fig(fig,figName)

'''
print('*** CONTRAST ***')
print('***')
diff_size_labels = ['w8-w4','w4-w2','w2-w1',
                    'b8-b4','b4-b2','b2-b1']
diff_evokeds = [list() for _ in range(len(diff_size_labels))]
grand_size = 'grand_avg_recog_diff_size_ave.fif'

# for subjN in subjList_final:
for subjN in subjList:
    fname = 'subj%d_recog_diff_size_ave.fif'%subjN
    fpath = os.path.join(evkDataPath,fname)
    evokeds = mne.read_evokeds(fpath)

    assert len(evokeds) == len(diff_evokeds)
    for idx,evoked in enumerate(evokeds):
        diff_evokeds[idx].append(evoked)

for idx,evokeds in enumerate(diff_evokeds):
    diff_evokeds[idx] = mne.combine_evoked(evokeds,'equal')

if tag_savefile == 1:
    mne.write_evokeds(os.path.join(evkDataPath,grand_size),
    diff_evokeds,overwrite=True)


print('*** *** ***')
diff_cate_labels = ['b8-w8','b4-w4','b2-w2','b1-w1']
diff_evokeds = [list() for _ in range(len(diff_cate_labels))]
grand_cate = 'grand_avg_recog_diff_cate_ave.fif'

# for subjN in subjList_final:
for subjN in subjList:
    fname = 'subj%d_recog_diff_cate_ave.fif'%subjN
    fpath = os.path.join(evkDataPath,fname)
    evokeds = mne.read_evokeds(fpath)

    assert len(evokeds) == len(diff_evokeds)
    for idx,evoked in enumerate(evokeds):
        diff_evokeds[idx].append(evoked)

for idx,evokeds in enumerate(diff_evokeds):
    diff_evokeds[idx] = mne.combine_evoked(evokeds,'equal')

if tag_savefile == 1:
    mne.write_evokeds(os.path.join(evkDataPath,grand_cate),
    diff_evokeds,overwrite=True)

# plot
evokeds = mne.read_evokeds(os.path.join(evkDataPath,grand_size))
mapping = dict()
for n,cond in enumerate(diff_size_labels):
    mapping[cond] = evokeds[n]
fig = mne.viz.plot_compare_evokeds(mapping,picks='eeg',axes='topo',
                                   show=show_flg)
figName = '%s_' + 'recog_diff_size.png'
saveFigs(fig, figName, figPars, tag_savefig)

evokeds = mne.read_evokeds(os.path.join(evkDataPath,grand_cate))
mapping = dict()
for n,cond in enumerate(diff_cate_labels):
    mapping[cond] = evokeds[n]
fig = mne.viz.plot_compare_evokeds(mapping,picks='eeg',axes='topo',
                                   show=show_flg)
figName = '%s_' + 'recog_diff_cate.png'
saveFigs(fig, figName, figPars, tag_savefig)
'''

# -- --- --- --- --- --- averaging all conditions
fig_pref = os.path.join(grpFigPath,'avg_grp_')
evk_allCond = []
# for subjN in subjList_final:
for subjN in subjList:
    fname_epo = 'subj%d_epo.fif'%subjN
    fpath_epo = os.path.join(epoDataPath,fname_epo)
    epo =  mne.read_epochs(fpath_epo)
    evk = epo[recog_labels].average()
    evk_allCond.append(evk)

evks_allCond = mne.combine_evoked(evk_allCond,'equal')
fname_grand_allCond = 'grand_avg_recogAllCond_ave.fif'
mne.evoked.write_evokeds(
    os.path.join(evkDataPath,
                 fname_grand_allCond),
    evks_allCond,overwrite=True)

evks = mne.read_evokeds(os.path.join(evkDataPath,
                                     fname_grand_allCond))
fig = mne.viz.plot_compare_evokeds({'All Condiditons':evks},
                                   picks='eeg',axes='topo',
                                   show=show_flg,
                                   legend=False)
figName = fig_pref+'erp.png'
save_fig(fig,figName)

# butterfly
fig = evks_allCond.plot_joint(title='All Conditions',
                              picks='eeg',show=show_flg)
figName = fig_pref+'btf.png'
save_fig(fig,figName)

# topo
fig = evks_allCond.plot_topomap(times=topo_t,
                                ch_type='eeg',
                                show=show_flg,
                                title='All Condiditons')
figName = fig_pref+'topo.png'
save_fig(fig,figName)

fig = mne.viz.plot_compare_evokeds(
    {'All Condiditons':evks},
    picks=postChans,
    truncate_xaxis=False,
    ylim=dict(eeg=[-7,12]),combine='mean',
    show=show_flg,
    show_sensors='upper right',title='post')
figName = fig_pref+'erp_each_post.png'
save_fig(fig,figName)

# each electrode
# chans = evks[0].ch_names
for chan in chans:
    fig = mne.viz.plot_compare_evokeds(
        {'All Condiditons':evks},picks=chan,
        truncate_xaxis=False,
        ylim=dict(eeg=[-7,12]),show=show_flg,
        show_sensors='upper right',title='%s'%(chan))
    figName = os.path.join(
        set_filepath(grpFigPath,'chan'),'avg_erp_ch_%s.png'%chan)
    save_fig(fig,figName)

print('ALL FINISHED')