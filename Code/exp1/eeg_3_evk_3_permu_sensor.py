#!/usr/bin/env python
# -*-coding:utf-8 -*-

# EXP. 3-3.1 (EEG): evoke
# Temporal dynamics of non-target processing vary with
# memory set size in different category conditions
# 2021.Nov.24
# linlin.shang@donders.ru.nl

# %% --- --- --- --- --- --- libraries --- --- --- --- --- ---

from eeg_config import epoDataPath,evkDataPath,resPath,grpFigPath,\
    subjList_final,event_dict_distr,cond_label_list,\
    recog_labels,n_permutations,save_fig,set_filepath

import numpy as np
import mne
from mne.stats import permutation_t_test

import os
import matplotlib.pyplot as plt

fig_pref = os.path.join(
    set_filepath(grpFigPath,'SensorClu'),'clu_')
fname = '_epo.fif'

epochs_list = []
for subjN in subjList_final:
    fname_subj = 'subj%d'%subjN+fname
    fpath = os.path.join(epoDataPath,fname_subj)
    epo = mne.read_epochs(fpath)
    epochs_list.append(epo[recog_labels])
    print('Subject %d loaded' % subjN)
epochs = mne.concatenate_epochs(epochs_list)

picks = mne.pick_types(epochs.info,eeg=True)
data = epochs.get_data()
times = epochs.times

temporal_mask = np.logical_and(0.2<=times,times<=0.3)
data = np.mean(data[:,:,temporal_mask],axis=2)

n_permutations = 50000
T0, p_values, H0 = permutation_t_test(
    data,n_permutations,n_jobs=None)

significant_sensors = picks[p_values<=0.05]
significant_sensors_names = [
    epochs.ch_names[k] for k in significant_sensors]

print("Number of significant sensors : %d"%len(significant_sensors))
print("Sensors names : %s" %significant_sensors_names)

evoked = mne.EvokedArray(
    -np.log10(p_values)[:,np.newaxis],
    epochs.info,tmin=0.)

# Extract mask and indices of active sensors in the layout
stats_picks = mne.pick_channels(
    evoked.ch_names,significant_sensors_names)
mask = p_values[:,np.newaxis]<0.05

evoked.plot_topomap(
    times=[0],scalings=1,time_format=None,
    cmap='Reds',
    units='-log10(p)',cbar_fmt='-%0.1f',mask=mask,
    size=3,show_names=lambda x: x[4:] + ' ' * 20,
    time_unit='s')
plt.show(block=True)
plt.close('all')



from mne.stats import linear_regression, fdr_correction
from mne.viz import plot_compare_evokeds

import pandas as pd

name = "Concreteness"
df = epochs.metadata
df[name] = pd.cut(df[name], 11, labels=False) / 10
colors = {str(val): val for val in df[name].unique()}
epochs.metadata = df.assign(Intercept=1)  # Add an intercept for later
evokeds = {val: epochs[name + " == " + val].average() for val in colors}
plot_compare_evokeds(evokeds, colors=colors, split_legend=True,
                     cmap=(name + " Percentile", "viridis"))
plt.show(block=True)
plt.close('all')

names = ["Intercept", name]
res = linear_regression(epochs, epochs.metadata[names], names=names)
for cond in names:
    res[cond].beta.plot_joint(title=cond, ts_args=dict(time_unit='s'),
                              topomap_args=dict(time_unit='s'))
plt.show(block=True)
plt.close('all')

reject_H0, fdr_pvals = fdr_correction(res["Concreteness"].p_val.data)
evoked = res["Concreteness"].beta
evoked.plot_image(mask=reject_H0, time_unit='s')
plt.show(block=True)
plt.close('all')
