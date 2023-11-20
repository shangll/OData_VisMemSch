#!/usr/bin/env python
# -*-coding:utf-8 -*-

# EXP. 3-3.1 (EEG): evoke
# Temporal dynamics of non-target processing vary with
# memory set size in different category conditions
# 2021.Nov.24
# linlin.shang@donders.ru.nl

#%% --- --- --- --- --- --- libraries --- --- --- --- --- ---

from eeg_config import epoDataPath,evkDataPath,resPath,grpFigPath,\
    subjList,subjList_final,event_dict_distr,cond_label_list,\
    recog_labels,n_permutations,save_fig,set_filepath
import mne
from mne.stats import spatio_temporal_cluster_test,f_mway_rm
from mne.channels import find_ch_adjacency
import numpy as np
import pandas as pd
import pingouin as pg
import scipy.stats as stats
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

import os

from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = "all"
model = LinearRegression()



# --- --- --- --- --- --- Sub-Functions --- --- --- --- --- --- #

def getCoeff(df,x_var,y_var):
    dfCoeffNew = pd.DataFrame(columns=['subj','cond','coeff','r2'])
    count = 0
    # for n in subjList_final:
    for n in subjList:
        for var in cond_label_list:
            dfCoeff = pd.DataFrame()
            dfCoeff.loc[count,'subj'] = n
            dfCoeff.loc[count,'cond'] = var
            x = df.loc[(df['subj']==n)&(df['cond']==var),x_var].values
            x = x.astype('int')
            x = np.log2(x)
            y = df.loc[(df['subj']==n)&(df['cond']==var),y_var].values
            model.fit(x.reshape(-1, 1), y)
            dfCoeff.loc[dfCoeff['subj']==n,'coeff'] = model.coef_
            dfCoeff.loc[dfCoeff['subj']==n,'r2'] = model.score(x.reshape(-1,1),y)
            dfCoeffNew = pd.concat([dfCoeffNew,dfCoeff],axis=0,ignore_index=True)
            count += 1
    dfCoeffNew.index = range(len(dfCoeffNew))
    return dfCoeffNew



# --- --- --- --- --- --- Main Function --- --- --- --- --- --- #
fig_pref = os.path.join(
    set_filepath(grpFigPath,'Cluster'),'stats_clu_eeg_')
fname = '_epo.fif'

epochs_list = []
# for subjN in subjList_final:
for subjN in subjList:
    fname_subj = 'subj%d'%subjN+fname
    fpath = os.path.join(epoDataPath,fname_subj)
    epo = mne.read_epochs(fpath)
    epochs_list.append(epo[recog_labels])
    print('Subject %d loaded' % subjN)

epochs = mne.concatenate_epochs(epochs_list)
# epochs.equalize_event_counts(event_dict_distr)

# n_epochs × n_times × n_channels
X = [epochs[event_name].get_data() for event_name in recog_labels]
X = [np.transpose(x,(0,2,1)) for x in X]
adjacency, ch_names = find_ch_adjacency(epochs.info,ch_type='eeg')

fig = mne.viz.plot_ch_adjacency(epochs.info,adjacency,ch_names)
figName = fig_pref+'adjacency.png'
save_fig(fig,figName)

# --- --- --- --- --- --- permutation statistic
'''
tail = 1
# run the cluster based permutation analysis
def stat_fun(*args):
    factor_levels = [8,1]
    effects = 'A'
    return f_mway_rm(
        np.array(args),factor_levels=factor_levels,
        effects=effects,return_pvals=False)[0]
f_thresh = None
cluster_stats = spatio_temporal_cluster_test(
    X,n_permutations=n_permutations,threshold=f_thresh,
    tail=tail,stat_fun=stat_fun,n_jobs=None,
    buffer_size=None,adjacency=adjacency)
F_obs,clusters,p_values,_ = cluster_stats

'''
tail = 1
alpha_cluster_forming = 0.001
n_conditions = len(recog_labels)
n_observations = len(X[0])
dfn = n_conditions-1
dfd = n_observations-n_conditions

f_thresh = stats.f.ppf(1-alpha_cluster_forming,dfn=dfn,dfd=dfd)

# run the cluster based permutation analysis
cluster_stats = spatio_temporal_cluster_test(X,n_permutations=n_permutations,
                                             threshold=f_thresh,tail=tail,
                                             n_jobs=None,buffer_size=None,
                                             adjacency=adjacency)
F_obs,clusters,p_values,_ = cluster_stats


# plot cluster
p_accept = 0.005
good_cluster_inds = np.where(p_values<p_accept)[0]

# configure variables for visualization
colors,linestyles = dict(),dict()
for cond,line_s,clr_s in zip(recog_labels,
                             ['solid']*4+['dashed']*4,
                             ['crimson','gold','darkturquoise',
                              'dodgerblue'] * 2):
    linestyles[cond] = line_s
    colors[cond] = clr_s

# organize data for plotting
evokeds = {cond:epochs[cond].average() for cond in recog_labels}

# loop over clusters
clu_dict = {'time':[],'chans':[]}
for i_clu, clu_idx in enumerate(good_cluster_inds):
    # unpack cluster information, get unique indices
    time_inds, space_inds = np.squeeze(clusters[clu_idx])
    ch_inds = np.unique(space_inds)
    time_inds = np.unique(time_inds)

    # get topography for F stat
    f_map = F_obs[time_inds, ...].mean(axis=0)

    # get signals at the sensors contributing to the cluster
    sig_times = epochs.times[time_inds]
    clu_dict['time'].append(sig_times)
    chan_list = []
    for indx in ch_inds:
        chan_list.append(epochs.ch_names[indx])
    clu_dict['chans'].append(chan_list)

    # create spatial mask
    mask = np.zeros((f_map.shape[0],1),dtype=bool)
    mask[ch_inds,:] = True

    # initialize figure
    fig,ax_topo = plt.subplots(1,1,figsize=(12,4))

    # plot average test statistic and mark significant sensors
    f_evoked = mne.EvokedArray(f_map[:,np.newaxis],
                               epochs.info,tmin=0)
    f_evoked.plot_topomap(times=0,mask=mask,axes=ax_topo,
                          cmap='Reds',vmin=np.min,
                          vmax=np.max,show=False,
                          colorbar=False,
                          mask_params=dict(markersize=10))
    image = ax_topo.images[0]

    # remove the title that would otherwise say "0.000 s"
    ax_topo.set_title('')

    # create additional axes (for ERF and colorbar)
    divider = make_axes_locatable(ax_topo)

    # add axes for colorbar
    ax_colorbar = divider.append_axes('right',size='5%',pad=0.05)
    plt.colorbar(image,cax=ax_colorbar)
    ax_topo.set_xlabel(
        'Averaged F-map ({:0.3f} - {:0.3f} s)'.format(*sig_times[[0,-1]]))

    # add new axis for time courses and plot time courses
    ax_signals = divider.append_axes('right',size='300%',pad=1.2)
    title = 'Cluster #{0}, {1} sensor'.format(i_clu+1,len(ch_inds))
    if len(ch_inds) > 1:
        title += 's (mean)'
    mne.viz.plot_compare_evokeds(
        evokeds,title=title,picks=ch_inds,combine='mean',
        axes=ax_signals,colors=colors,
        linestyles=linestyles,show=False,
        split_legend=False,truncate_yaxis='auto')
    figName = fig_pref+'%d.png'%(i_clu+1)

    # plot temporal cluster extent
    ymin, ymax = ax_signals.get_ylim()
    ax_signals.fill_betweenx((ymin,ymax),sig_times[0],sig_times[-1],
                             color='orange',alpha=0.3)

    # clean up viz
    mne.viz.tight_layout(fig=fig)
    fig.subplots_adjust(bottom=.05)
    # plt.show(block=show_flg)
    save_fig(fig,figName)

    fig = epo.copy().pick(
        chan_list).plot_image(
        combine='mean',
        title='Cluster %d'%(i_clu+1))
    figName = os.path.join(
        grpFigPath,'Cluster','trials_amp_clu%d.png'%(i_clu+1))
    save_fig(fig,figName)

    print('Cluster %d Finished'%(i_clu+1))

print('')
print('*** *** ***')
print(clu_dict)
print('*** *** ***')
print('')

# ANOVA: ERP
erp_data = pd.read_csv(
    os.path.join(resPath,'erp_eeg.csv'), sep=',')
times = erp_data.loc[
    (erp_data['subj']==1)&
    (erp_data['type']=='w/1'),
    'time'].values

erp_recog = erp_data[erp_data['type'].isin(recog_labels)]
erp_recog.reset_index(drop=True,inplace=True)
# erp_recog = erp_recog.copy()

clu_n = len(clu_dict['chans'])
clu_tag = ['clu%d'%(k+1) for k in range(clu_n)]
for pick_tag,pick_chs in zip(clu_tag,clu_dict['chans']):
    erp_recog.loc[:,pick_tag] = \
        erp_recog[pick_chs].mean(axis=1).values
erp_recog.loc[:,['cond','setsize']] = \
    erp_recog['type'].str.split('/',expand=True).values

all_chans = [k for k in erp_recog.columns.tolist() \
             if k not in ['subj','time','type','cond','setsize']]
erp_recog_mean = erp_recog.groupby(['time','type']
                                   )[all_chans].agg(np.mean).reset_index()
erp_recog_mean.loc[:,['cond','setsize']] = \
    erp_recog_mean['type'].str.split('/',expand=True).values

erp_coeff = pd.DataFrame()
for pick_tag,pick_times in zip(clu_tag,clu_dict['time']):
    t0 = pick_times[0]
    t1 = pick_times[-1]
    erp_stats = erp_recog[(erp_recog['time']>=t0)&
                          (erp_recog['time'] <= t1)]
    erp_stats.reset_index(drop=True,inplace=True)

    # coeff_t = getCoeff(
    #     erp_stats,'setsize',pick_tag)
    coeff_t = getCoeff(
        erp_stats,'setsize',pick_tag)
    coeff_t['chans'] = [pick_tag] * len(coeff_t)
    erp_coeff = pd.concat([erp_coeff,coeff_t],
                          axis=0,ignore_index=True)

    aov = pg.rm_anova(dv=pick_tag,within=['cond','setsize'],
                      subject='subj',data=erp_stats,
                      detailed=True,effsize='np2')

    pwc1 = pg.pairwise_tests(dv=pick_tag,within=['cond','setsize'],
                             subject='subj',data=erp_stats,
                             padjust='bonf',effsize='hedges')
    pwc2 = pg.pairwise_tests(dv=pick_tag,within=['setsize','cond'],
                             subject='subj',data=erp_stats,
                             padjust='bonf',effsize='hedges')

    pd.set_option('display.max_columns',None)
    print(pick_tag)
    print('--- --- ---')
    print(aov)
    # print('--- --- ---')
    # print(pwc1)
    # print('--- --- ---')
    # print(pwc2)
    # print('--- --- ---')

'''
# ANOVA: Coefficients
fileName = 'sch_mean.csv'
df_sch_mean = pd.read_csv(
    os.path.join(resPath,fileName),sep=',')
# behav_coeff = getCoeff(
#     df_sch_mean,'setsize','rt')
behav_coeff = getCoeff(
    df_sch_mean,'setsize','rt')

df_corr = pd.DataFrame()
for pick_tag in clu_tag:
    erp_stats = erp_coeff[erp_coeff['chans']==pick_tag]

    t_val = pg.ttest(erp_stats.loc[erp_stats['cond']=='w','coeff'],
                     erp_stats.loc[erp_stats['cond']=='b','coeff'],
                     paired=True,correction=True)
    print(pick_tag)
    print('--- --- ---')
    print(t_val)
    print('--- --- ---')

    for cond in cond_label_list:
        behav_val = behav_coeff.loc[
            behav_coeff['cond']==cond,'coeff'].values
        erp_val = erp_stats.loc[
            erp_stats['cond'] == cond,'coeff'].values
        coeff_corr = pg.corr(erp_val,behav_val,
                                method='spearman')
        coeff_corr['cond'] = [cond]*len(coeff_corr)
        coeff_corr['chans'] = [pick_tag]*len(coeff_corr)
        df_corr = pd.concat([df_corr,coeff_corr],
                            axis=0,ignore_index=True)
pd.set_option('display.max_rows',None)
print(df_corr)
'''