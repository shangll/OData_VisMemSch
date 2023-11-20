#!/usr/bin/env python
# -*-coding:utf-8 -*-

# EXP. 3-3.1 (EEG): evoke
# Temporal dynamics of non-target processing vary with
# memory set size in different category conditions
# 2021.Nov.24
# linlin.shang@donders.ru.nl



#%% --- --- --- --- --- --- libraries --- --- --- --- --- ---

from eeg_config import resPath,allResFigPath,\
    recog_label_list,show_flg,evkDataPath,\
    recog_labels,save_fig
import mne
import os
import numpy as np
import pandas as pd

from sklearn.linear_model import LinearRegression
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.pyplot import MultipleLocator
import seaborn as sns

from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = "all"
model = LinearRegression()

scale = (1e+6)


picks = ['P5','P6','P7','P8',
         'PO3','PO4','PO7','PO8','PO9','PO10',
         'O1','O2']
pick_tag = 'simi'



erp_data = pd.read_csv(os.path.join(resPath,'erp_eeg.csv'), sep=',')
times = erp_data.loc[(erp_data['subj']==1)&
                     (erp_data['type']=='w/1'),'time'].values
erp_recog = erp_data[erp_data['type'].isin(recog_labels)]
erp_recog.reset_index(drop=True,inplace=True)
erp_recog.loc[:,pick_tag] = erp_recog[picks].mean(axis=1).values
erp_recog.loc[:,['cond','setsize']] = \
    erp_recog['type'].str.split('/',expand=True).values
erp_recog['setsize'] = erp_recog['setsize'].astype('int')
erp_recog['simi'] = erp_recog['simi']*scale

fname_grand = 'grand_avg_recogAllCond_ave.fif'
evks = mne.read_evokeds(os.path.join(evkDataPath,fname_grand))



# bar plot for erp components
lab_size = 25
mpl.rcParams.update({'font.size':23})
fig,ax = plt.subplots(2,2,figsize=(16,12))
ax = ax.ravel()
#
mne.viz.plot_compare_evokeds(
    {'Averaged ERP':evks},picks=picks,truncate_xaxis=False,
    ylim=dict(eeg=[-3,8]),show=show_flg,combine='mean',
    show_sensors='upper right',title='12 Sensors',
    legend=False,axes=ax[0])
ax[0].set_ylabel('μV')
ax[0].axvline(0.1,ls='--',color='grey')
ax[0].axvline(0.16,ls='--',color='grey')
ax[0].axvline(0.2,ls='--',color='grey')
ax[0].axvline(0.3,ls='--',color='grey')
p1_text = 'P1'
n1_text = 'N1'
p2_text = 'P2'
ax[0].text(0.115,-2,p1_text,color='grey',fontsize=12,fontweight='bold')
ax[0].text(0.16,-2,n1_text,color='grey',fontsize=12,fontweight='bold')
ax[0].text(0.232,-2,p2_text,color='grey',fontsize=12,fontweight='bold')
ax[0].text(-0.28,8.6,'(A)',fontsize=lab_size)
#
for n,cp_tag in enumerate(['P1','N1','P2']):
    indx = n+1
    if cp_tag=='P1':
        fig_lab = '(B)'
        t0,t1 = 0.1,0.16
        y = 4.2
        y_loc = 0.9
    elif cp_tag=='N1':
        fig_lab = '(C)'
        t0,t1 = 0.16,0.2
        y = 1.84
        y_loc = 0.5
    else:
        fig_lab = '(D)'
        t0,t1 = 0.2,0.3
        y = 8.2
        y_loc = 2.3
    erp_cp = erp_recog[(erp_recog['time']<t1)&(erp_recog['time']>=t0)]
    sns.barplot(data=erp_cp,x='setsize',y=pick_tag,
                hue='cond',hue_order=['w','b'],
                palette='Blues',saturation=0.75,
                errorbar='se',capsize=0.15,errcolor='grey',
                errwidth=1.5,ax=ax[indx])
    ax[indx].set_title(cp_tag,fontsize=lab_size)
    ax[indx].set_xlabel('Memory Set Size')
    ax[indx].set_ylabel(None)
    y_major_locator = MultipleLocator(y_loc)
    ax[indx].yaxis.set_major_locator(y_major_locator)
    ax[indx].text(-0.7,y,fig_lab,ha='center',
                  va='top',color='k',fontsize=lab_size)
    if indx!=1:
        ax[indx].get_legend().remove()
h,_ = ax[1].get_legend_handles_labels()
ax[1].legend(h,['within','between'],loc='lower left',ncol=1,
             fontsize=18,
             frameon=True).set_title(None)
ax[1].set_xticklabels([])
ax[1].set_xlabel(None)
ax[2].set_ylabel('μV')
# fig.text(0,0.5,'μV',va='center',rotation='vertical')
sns.despine(offset=15,trim=True)
plt.tight_layout()
figName = os.path.join(allResFigPath,'simi','erp_simi_3cps.png')
save_fig(fig,figName)

# erp permutation
yy = 8.3
title_size = 23
mpl.rcParams.update({'font.size':26})
fig,ax = plt.subplots(1,2,figsize=(20,8),sharey=True)
ax = ax.ravel()
sig_start,sig_end = 0.152,0.260
erp_recog_mean = erp_recog.groupby(
    ['time','cond'])[pick_tag].agg(np.mean).reset_index()
sns.lineplot(data=erp_recog_mean,x='time',y=pick_tag,
             hue='cond',hue_order=['w','b'],
             style='cond',style_order=['w','b'],
             lw=3.5,markers=False,palette='Blues',ax=ax[0])
ymin,ymax_ = ax[0].get_ylim()
ymax = 7.8
ax[0].fill_between(times,ymin,ymax,
                   where=(times>=sig_start)&(times<sig_end),
                   color='grey',alpha=0.1)
ax[0].text(0.27,-1.5,'%.3f-%.3f sec'%(sig_start,sig_end),
           color='grey',fontsize=20)
ax[0].text(-0.28,yy,'(A)',ha='center',
           va='top',color='k')
ax[0].set_title('Main Effect of Category',fontsize=title_size)
ax[0].set_xlabel('Time (sec)')
ax[0].set_ylabel('μV')
ax[0].set_ylim(-3,ymax)
ax[0].set_yticklabels(np.arange(-2,9,2))
h,_ = ax[0].get_legend_handles_labels()
ax[0].legend(h,['within','between'],loc='upper right',
             ncol=1,fontsize=22,
             frameon=False).set_title(None)
#
sig_start,sig_end = 0.188,0.292
erp_recog_mean = erp_recog.groupby(
    ['time','setsize'])[pick_tag].agg(np.mean).reset_index()
clrs_all_b = sns.color_palette('Blues',n_colors=35)
clrs_all = sns.color_palette('GnBu_d')
clrs = [clrs_all_b[28],clrs_all_b[20],clrs_all[2],clrs_all[0]]
sns.lineplot(data=erp_recog_mean,x='time',y=pick_tag,
             hue='setsize',hue_order=[1,2,4,8],
             lw=3,markers=False,palette=clrs,ax=ax[1])
ymin,ymax_ = ax[1].get_ylim()
ymax = 7.8
ax[1].fill_between(times,ymin,ymax,
                   where=(times>=sig_start)&(times<sig_end),
                   color='grey',alpha=0.1)
ax[1].text(0.3,-1.5,'%.3f-%.3f sec'%(sig_start,sig_end),
           color='grey',fontsize=20)
ax[1].text(-0.28,yy,'(B)',ha='center',
           va='top',color='k')
ax[1].set_title('Main Effect of Memory Set Size',fontsize=title_size)
ax[1].set_xlabel('Time (sec)')
ax[1].set_ylabel(None)
h,_ = ax[1].get_legend_handles_labels()
ax[1].legend(h,['MSS 1','MSS 2','MSS 4','MSS 8'],
             loc='upper right',ncol=1,fontsize=22,
             frameon=False).set_title(None)
# fig.text(0.5,0,'Time (sec)',ha='center')
sns.despine(offset=10,trim=True)
plt.tight_layout()
figName = os.path.join(allResFigPath,'simi','erp_simi_permu.png')
save_fig(fig,figName)

'''
# decoding
pred = 'o2r'
deco_subj_all = pd.read_csv(
    os.path.join(resPath,'deco_data_subj.csv'),sep=',')
deco_subj = deco_subj_all[
    (deco_subj_all['chans']==pick_tag)&
    (deco_subj_all['pred']==pred)&
    (deco_subj_all['type'].isin(recog_label_list))].reset_index(drop=True)
deco_subj['cond'] = deco_subj['type'].str.split('',expand=True)[1]
deco_subj['setsize'] = deco_subj['type'].str.split('',expand=True)[2]

mpl.rcParams.update({'font.size':18})
fig,ax = plt.subplots(1,2,figsize=(18,9))
ax = ax.ravel()
deco_recog_mean = deco_subj.groupby(
    ['time','cond'])['acc'].agg(np.mean).reset_index()
sns.lineplot(data=deco_recog_mean,x='time',y='acc',
             hue='cond',hue_order=['w','b'],
             style='cond',style_order=['w','b'],
             lw=3,markers=False,palette='Blues',ax=ax[0])
ax[0].axvline(0.1,ls='--',color='grey')
ax[0].axvline(0.16,ls='--',color='grey')
ax[0].axvline(0.2,ls='--',color='grey')
ax[0].axvline(0.3,ls='--',color='grey')
ax[0].text(0.115,0.615,p1_text,color='grey',fontsize=18,fontweight='bold')
ax[0].text(0.165,0.615,n1_text,color='grey',fontsize=18,fontweight='bold')
ax[0].text(0.232,0.615,p2_text,color='grey',fontsize=18,fontweight='bold')
ax[0].set_title('Cross-Task Decoding for Each Category')
ax[0].set_xlabel(None)
ax[0].set_ylabel('AUC')
ax[0].set_ylim(0.48,0.62)
h,_ = ax[0].get_legend_handles_labels()
ax[0].legend(h,['within','between'],loc='upper right',
             ncol=1,fontsize=18,
             frameon=False).set_title(None)
#
deco_recog_mean = deco_subj.groupby(
    ['time','setsize'])['acc'].agg(np.mean).reset_index()
clrs_all_b = sns.color_palette('Blues',n_colors=35)
clrs_all = sns.color_palette('GnBu_d')
clrs = [clrs_all_b[28],clrs_all_b[20],clrs_all[2],clrs_all[0]]
sns.lineplot(data=deco_recog_mean,x='time',y='acc',
             hue='setsize',hue_order=['1','2','4','8'],
             lw=2.5,markers=False,palette=clrs,ax=ax[1])
ax[1].axvline(0.1,ls='--',color='grey')
ax[1].axvline(0.16,ls='--',color='grey')
ax[1].axvline(0.2,ls='--',color='grey')
ax[1].axvline(0.3,ls='--',color='grey')
ax[1].text(0.115,0.615,p1_text,color='grey',fontsize=18,fontweight='bold')
ax[1].text(0.165,0.615,n1_text,color='grey',fontsize=18,fontweight='bold')
ax[1].text(0.232,0.615,p2_text,color='grey',fontsize=18,fontweight='bold')
ax[1].set_title('Cross-Task Decoding for Each Memory Set Size')
ax[1].set_xlabel(None)
ax[1].set_yticklabels([])
ax[1].set_ylim(0.48,0.62)
ax[1].set_ylabel(None)
h,_ = ax[1].get_legend_handles_labels()
ax[1].legend(h,['MSS 1','MSS 2','MSS 4','MSS 8'],
             loc='upper right',ncol=1,fontsize=18,
             frameon=False).set_title(None)
fig.text(0.5,0,'Time (sec)',ha='center')

sns.despine(offset=15,trim=True)
plt.tight_layout()
figName = os.path.join(allResFigPath,'simi','erp_simi_deco.png')
save_fig(fig,figName)
'''









