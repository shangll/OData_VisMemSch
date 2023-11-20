#!/usr/bin/env python
# -*-coding:utf-8 -*-

# EXP. 3-3.1 (EEG): evoke
# Temporal dynamics of non-target processing vary with
# memory set size in different category conditions
# 2021.Nov.24
# linlin.shang@donders.ru.nl



#%% --- --- --- --- --- --- libraries --- --- --- --- --- ---
from eeg_config import epoDataPath,evkDataPath,resPath,indvFigPath,\
    subjList_final,subjList,frontChans,fcpChans,postChans,recog_labels,\
    show_flg,tag_savefile,save_fig
import mne
import numpy as np
import pandas as pd

import os

from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = "all"


tag_savefile=1
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

pick_chs = 'eeg'



# --- --- --- --- --- --- Main Function --- --- --- --- --- --- #
df = pd.DataFrame()

print('***')
print('*** 3. INDIVIDUAL LEVEL ***')
print('***')
# for subjN in subjList_final:
for subjN in subjList:

    print('SUBJECT %d STARTS'%subjN)
    df_tmp = pd.DataFrame()
    fig_suff = '_subj%d.png'%subjN
    fname = 'subj%d_epo.fif' % subjN

    print('3.1 ODD BALL TASK ***')
    # oddball
    fig_pref = os.path.join(indvFigPath,'odd_')
    fname_odd_evk = 'odd_subj%d_ave.fif' % subjN
    subj_odd_evk_path = os.path.join(evkDataPath,fname_odd_evk)
    subj_epo = mne.read_epochs(os.path.join(epoDataPath,fname))

    subj_odd_anim = subj_epo['anim'].average()
    subj_odd_obj = subj_epo['obj'].average()
    subj_odd_anim.comment = 'anim'
    subj_odd_obj.comment = 'obj'
    # subj_odd_evk = mne.combine_evoked([subj_odd_anim,subj_odd_obj],'nave')
    subj_odd_evk = [subj_odd_anim,subj_odd_obj]
    mapping_odd = dict(anim=subj_odd_anim,obj=subj_odd_obj)
    for cond,evk in zip(mapping_odd.keys(),mapping_odd.values()):
        chans = evk.ch_names
        times = evk.times

        df_tmp['subj'] = [subjN] * len(times)
        df_tmp['time'] = times
        df_tmp['type'] = [cond] * len(times)
        for chan in chans:
            df_tmp[chan] = evk.get_data(picks=chan)[0]
        df = pd.concat([df,df_tmp],axis=0,ignore_index=True)

    if tag_savefile == 1:
        mne.write_evokeds(subj_odd_evk_path,subj_odd_evk,overwrite=True)

    for combine in ('mean','median','gfp'):
        fig = mne.viz.plot_compare_evokeds(mapping_odd,
                                           picks=postChans,show=show_flg,
                                           show_sensors='upper right',
                                           combine=combine,
                                           title='Subj %d Oddball %s'% (subjN,' 60 chs'))
        figName = fig_pref+'erp_post_%s'%(combine)+fig_suff
        save_fig(fig,figName)

    fig = mne.viz.plot_compare_evokeds(mapping_odd,picks=pick_chs,
                                       axes='topo',
                                       show=show_flg,legend=False)
    figName = fig_pref+'erp_each'+fig_suff
    save_fig(fig,figName)

    fig = subj_odd_obj.plot_joint(title='Subj %d'%subjN,
                                  picks='eeg',show=show_flg)
    figName = fig_pref+'btf'+fig_suff
    save_fig(fig,figName)

    fig = subj_odd_obj.plot_topomap(times=topo_t,
                                    ch_type='eeg',show=show_flg,
                                    title='Subj %d Odd'%subjN)
    figName = fig_pref+'topo'+fig_suff
    save_fig(fig,figName)

    odd_diff = mne.combine_evoked([subj_odd_anim,subj_odd_obj],
                                  weights=[1,-1])
    # if tag_savefile == 1:
    #     mne.write_evokeds(os.path.join(evkDataPath,
    #                                    'subj%d_odd_diff_ave.fif'%subjN),
    #                       odd_diff,overwrite=True)

    fig = odd_diff.plot_topo(title='Subj %d Odd_Diff'%subjN,show=show_flg)
    figName = fig_pref+'diff_btf'+fig_suff
    save_fig(fig,figName)

    print('ODD BALL FINISHED ***')


    print('3.2 RECOGNITION TASK ***')
    fig_pref = os.path.join(indvFigPath,'recog_')
    # recognition
    subj_w1 = subj_epo['w/1'].average()
    subj_w2 = subj_epo['w/2'].average()
    subj_w4 = subj_epo['w/4'].average()
    subj_w8 = subj_epo['w/8'].average()
    subj_b1 = subj_epo['b/1'].average()
    subj_b2 = subj_epo['b/2'].average()
    subj_b4 = subj_epo['b/4'].average()
    subj_b8 = subj_epo['b/8'].average()
    subj_targ1 = subj_epo['targ/1'].average()
    subj_targ2 = subj_epo['targ/2'].average()
    subj_targ4 = subj_epo['targ/4'].average()
    subj_targ8 = subj_epo['targ/8'].average()
    subj_w1.comment = 'w/1'
    subj_w2.comment = 'w/2'
    subj_w4.comment = 'w/4'
    subj_w8.comment = 'w/8'
    subj_b1.comment = 'b/1'
    subj_b2.comment = 'b/2'
    subj_b4.comment = 'b/4'
    subj_b8.comment = 'b/8'
    subj_targ1.comment = 'targ/1'
    subj_targ2.comment = 'targ/2'
    subj_targ4.comment = 'targ/4'
    subj_targ8.comment = 'targ/8'
    subj_recog_evk_cmb = mne.combine_evoked([subj_w1,subj_w2,subj_w4,subj_w8,
                                             subj_b1,subj_b2,subj_b4,subj_b8],
                                            'nave')
    subj_targ_evk_cmb = mne.combine_evoked([subj_targ1,subj_targ2,
                                            subj_targ4,subj_targ8],
                                           'nave')
    subj_recog_evk = [subj_w1,subj_w2,subj_w4,subj_w8,
                      subj_b1,subj_b2,subj_b4,subj_b8]
    subj_targ_evk = [subj_targ1,subj_targ2,
                     subj_targ4,subj_targ8]
    fname_recog_evk = 'subj%d_recog_ave.fif' % subjN
    fname_targ_evk = 'subj%d_targ_ave.fif' % subjN
    subj_recog_evk_path = os.path.join(evkDataPath,fname_recog_evk)
    subj_targ_evk_path = os.path.join(evkDataPath,fname_targ_evk)
    if tag_savefile == 1:
        mne.write_evokeds(subj_recog_evk_path,subj_recog_evk,overwrite=True)
        mne.write_evokeds(subj_targ_evk_path,subj_targ_evk,overwrite=True)

    mapping = {'w/1':subj_w1,'w/2':subj_w2,'w/4':subj_w4,'w/8':subj_w8,
               'b/1':subj_b1,'b/2':subj_b2,'b/4':subj_b4,'b/8':subj_b8}
    for cond,evk in zip(mapping.keys(),mapping.values()):
        chans = evk.ch_names
        times = evk.times

        df_tmp['subj'] = [subjN] * len(times)
        df_tmp['time'] = times
        df_tmp['type'] = [cond] * len(times)
        for chan in chans:
            df_tmp[chan] = evk.get_data(picks=chan)[0]
        df = pd.concat([df,df_tmp],axis=0,ignore_index=True)

    mapping_targ = {'targ/1':subj_targ1,'targ/2':subj_targ2,
                    'targ/4':subj_targ4,'targ/8':subj_targ8}
    for cond, evk in zip(mapping_targ.keys(),mapping_targ.values()):
        chans = evk.ch_names
        times = evk.times

        df_tmp['subj'] = [subjN] * len(times)
        df_tmp['time'] = times
        df_tmp['type'] = [cond] * len(times)
        for chan in chans:
            df_tmp[chan] = evk.get_data(picks=chan)[0]
        df = pd.concat([df,df_tmp],axis=0,ignore_index=True)

    # plot
    # ERP
    fig = mne.viz.plot_compare_evokeds(
        {'average':subj_epo[recog_labels].copy().average()},
        picks=postChans,show=show_flg,show_sensors='upper right',
        combine='mean',title='Subject %d Recog %s'%(subjN,' post'))
    figName = fig_pref+'distr_avg_erp_post'+fig_suff
    save_fig(fig,figName)
    for combine in ('mean','median','gfp'):
        fig = mne.viz.plot_compare_evokeds(mapping,picks=postChans,colors=clr,
                                           linestyles=lineSty,show=show_flg,
                                           show_sensors='upper right',
                                           combine=combine,
                                           title='Subject %d Recog %s'\
                                                 %(subjN,' post'))
        figName = fig_pref+'distr_erp_post_%s'%combine+fig_suff
        save_fig(fig,figName)
    # each channel
    fig = mne.viz.plot_compare_evokeds(mapping,picks=pick_chs,colors=clr,
                                       linestyles=lineSty,axes='topo',
                                       show=show_flg,legend=False)
    figName = fig_pref+'distr_erp_each'+fig_suff
    save_fig(fig,figName)

    # butterfly
    fig = subj_recog_evk_cmb.plot_joint(title='Subj %d Recog'%subjN,
                                        picks=pick_chs,show=show_flg)
    figName = fig_pref+'distr_btf'+fig_suff
    save_fig(fig,figName)

    # topo
    fig = subj_recog_evk_cmb.plot_topomap(times=topo_t,
                                          ch_type=pick_chs,show=show_flg,
                                          title='Subj %d Recog'%subjN)
    figName = fig_pref+'distr_topo'+fig_suff
    save_fig(fig,figName)
    '''
    # contrast
    recog_diff_size = [mne.combine_evoked([subj_w8,subj_w4],weights=[1,-1]),
                       mne.combine_evoked([subj_w4,subj_w2],weights=[1,-1]),
                       mne.combine_evoked([subj_w2,subj_w1],weights=[1,-1]),
                       mne.combine_evoked([subj_b8,subj_b4],weights=[1,-1]),
                       mne.combine_evoked([subj_b4,subj_b2],weights=[1,-1]),
                       mne.combine_evoked([subj_b2,subj_b1],weights=[1,-1])]
    recog_diff_cate = [mne.combine_evoked([subj_b8,subj_w8],weights=[1,-1]),
                       mne.combine_evoked([subj_b4,subj_w4],weights=[1,-1]),
                       mne.combine_evoked([subj_b2,subj_w2],weights=[1,-1]),
                       mne.combine_evoked([subj_b1,subj_w1],weights=[1,-1])]
    mapping_size = {'w8-w4':mne.combine_evoked([subj_w8,subj_w4],weights=[1,-1]),
                    'w4-w4':mne.combine_evoked([subj_w4,subj_w2],weights=[1,-1]),
                    'w2-w4':mne.combine_evoked([subj_w2,subj_w1],weights=[1,-1]),
                    'b8-b4':mne.combine_evoked([subj_b8,subj_b4],weights=[1,-1]),
                    'b4-b2':mne.combine_evoked([subj_b4,subj_b2],weights=[1,-1]),
                    'b2-b1':mne.combine_evoked([subj_b2,subj_b1],weights=[1,-1])}
    mapping_cate = {'b8-w8':mne.combine_evoked([subj_b8,subj_w8],weights=[1,-1]),
                    'b4-w4':mne.combine_evoked([subj_b4,subj_w4],weights=[1,-1]),
                    'b2-w2':mne.combine_evoked([subj_b2,subj_w2],weights=[1,-1]),
                    'b1-w1':mne.combine_evoked([subj_b1,subj_w1],weights=[1,-1])}

    if tag_savefile == 1:
        mne.write_evokeds(os.path.join(evkDataPath,
                                       'subj%d_recog_diff_size_ave.fif'%subjN),
                          recog_diff_size,overwrite=True)
        mne.write_evokeds(os.path.join(evkDataPath,
                                       'subj%d_recog_diff_cate_ave.fif' % subjN),
                          recog_diff_cate,overwrite=True)

    fig = mne.viz.plot_compare_evokeds(mapping_size,picks='eeg',axes='topo',
                                       show=show_flg)
    figName = '%s_' + 'recog_diff_size.png'
    saveFigs(fig, figName, figPars, tag_savefig)

    fig = mne.viz.plot_compare_evokeds(mapping_cate,picks='eeg',axes='topo',
                                       show=show_flg)
    figName = '%s_' + 'recog_diff_cate.png'
    saveFigs(fig, figName, figPars, tag_savefig)'''

    # target
    for combine in ('mean','median','gfp'):
        fig = mne.viz.plot_compare_evokeds(mapping_targ,
                                           picks=postChans,show=show_flg,
                                           show_sensors='upper right',
                                           combine=combine,
                                           title='Subject %d Targ %s'\
                                                 %(subjN,' 60 chs'))
        figName = fig_pref+'targ_erp_%s'%combine+fig_suff
        save_fig(fig,figName)

    fig = subj_targ_evk_cmb.plot_joint(title='Subj %d Targ'%subjN,
                                       picks='eeg',show=show_flg)
    figName = fig_pref+'targ_btf'+fig_suff
    save_fig(fig,figName)

    fig = subj_targ_evk_cmb.plot_topomap(times=topo_t,
                                         ch_type='eeg',show=show_flg,
                                         title='Subj %d Targ_'%subjN)
    figName = fig_pref+'targ_topo'+fig_suff
    save_fig(fig,figName)

    '''
    targ_diff = [mne.combine_evoked([subj_targ8,subj_targ4],weights=[1,-1]),
                 mne.combine_evoked([subj_targ4,subj_targ2],weights=[1,-1]),
                 mne.combine_evoked([subj_targ2,subj_targ1],weights=[1,-1])]
    mapping_targ_diff = {'8-4':mne.combine_evoked([subj_targ8,subj_targ4],weights=[1,-1]),
                         '4-2':mne.combine_evoked([subj_targ4,subj_targ2],weights=[1,-1]),
                         '2-1':mne.combine_evoked([subj_targ2,subj_targ1],weights=[1,-1])}
    if tag_savefile == 1:
        mne.write_evokeds(os.path.join(evkDataPath,
                                       'subj%d_targ_diff_ave.fif' % subjN),
                          targ_diff,overwrite=True)

    fig = mne.viz.plot_compare_evokeds(mapping_targ_diff,
                                       picks='eeg',axes='topo',
                                       show=show_flg)
    figName = '%s_' + 'targ_diff.png'
    saveFigs(fig, figName, figPars, tag_savefig)
    '''

    print('SUBJECT %d: EVOKED DATA SAVED ***' % subjN)


    # Special Channels
    for labelName,pick_chans in zip(['front','fcp'],
                                    [frontChans,fcpChans]):
        for combine in ('mean','median','gfp'):
            fig = mne.viz.plot_compare_evokeds(mapping,picks=pick_chans,colors=clr,
                                               linestyles=lineSty,show=show_flg,
                                               show_sensors='upper right',
                                               combine=combine,
                                               title='Subject %d Recog %s' \
                                                     % (subjN,labelName))
            figName = fig_pref+'distr_erp_%s_%s'%(combine,labelName)+fig_suff
            save_fig(fig,figName)

if tag_savefile == 1:
    fileName = 'erp_eeg.csv'
    df.to_csv(os.path.join(resPath,fileName),
              mode='w',header=True,index=False)
print('ALL FINISHED')
print('***')
print('**')
print('*')