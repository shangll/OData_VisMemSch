#!/usr/bin/env python
#-*-coding:utf-8 -*-

# EXP. 3 (Behavioural):
# Temporal dynamics of non-target processing vary with
# memory set size in different category conditions
# 2021.Nov.24
# linlin.shang@donders.ru.nl

from eeg_config import resPath,outliers,sizeList,\
    cateList,condList,p_crit,set_filepath,save_fig

import os
from math import log
import random
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.pyplot import MultipleLocator
import seaborn as sns
import statsmodels.api as sm
from sklearn.linear_model import LinearRegression
import pingouin as pg
import statannot

from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = "all"

model = LinearRegression()



# --- --- --- Set Global Parameters --- --- --- #

exp_tag_list = ['exp1a','exp3']
expResPath = set_filepath('U:\Documents\DCC\exp3','AllExpRes')
figPath = set_filepath(expResPath,'Figs','behav')

exp_mean = os.path.join(expResPath,'data_allExp_mean.csv')
data_mean_all = pd.read_csv(exp_mean,sep=',')
exp_data_mean = data_mean_all[(data_mean_all['exp']=='exp1a')|
                         (data_mean_all['exp']=='exp3')]
data_plt = exp_data_mean.groupby(
    ['exp','setsize','cond'])[['rt','lm','log']].agg(np.mean).reset_index()
lm_df = data_plt[['exp','setsize','cond','lm']].reset_index(drop=True)
lm_df.rename(columns={'lm':'rt'},inplace=True)
lm_df['Fit'] = ['linear']*len(lm_df)
log_df = data_plt[['exp','setsize','cond','log']].reset_index(drop=True)
log_df.rename(columns={'log':'rt'},inplace=True)
log_df['Fit'] = ['log2']*len(log_df)
pred_plt = pd.concat([lm_df,log_df],axis=0,ignore_index=True)
pred_plt.rename(columns={'cond':'Category'},inplace=True)
data_plt.rename(columns={'cond':'Category'},inplace=True)

text_labels = {}
text_labels['exp1a'] = {'rtbetween':['k',0.485],'rtwithin':['grey',0.602],
                        'linearbetween':['dodgerblue',0.559],
                        'linearwithin':['lightskyblue',0.7],
                        'log2between':['dodgerblue',0.524],
                        'log2within':['lightskyblue',0.623]}
text_labels['exp3'] = {'rtbetween':['k',0.46],'rtwithin':['grey',0.542],
                       'linearbetween':['dodgerblue',0.556],
                       'linearwithin':['lightskyblue',0.576],
                       'log2between':['dodgerblue',0.475],
                       'log2within':['lightskyblue',0.51]}
for exp_tag in exp_tag_list:
    for cate_tag in condList:
        text_label = data_plt.loc[(data_plt['exp']==exp_tag)&
            (data_plt['Category']==cate_tag),'rt'].tolist()[-1]
        text_labels[exp_tag]['rt'+cate_tag].append(text_label)
        for pred_tag in ['linear','log2']:
            text_label = pred_plt.loc[(pred_plt['exp']==exp_tag)&
                (pred_plt['Category']==cate_tag)&
                (pred_plt['Fit']==pred_tag),'rt'].tolist()[-1]
            text_labels[exp_tag][pred_tag+cate_tag].append(text_label)

exp_all = os.path.join(expResPath,'data_allExp.csv')
data_all = pd.read_csv(exp_all,sep=',')
exp_data_1a = data_all[(data_all['exp']=='exp1a')].groupby(
        ['subj','cond','setsize'])['rt'].agg(np.mean).reset_index()
exp_data_1a['exp'] = 'exp1a'
exp_data_3 = data_all[(data_all['exp']=='exp3')].groupby(
        ['subj','cond','setsize'])['rt'].agg(np.mean).reset_index()
exp_data_3['exp'] = 'exp3'
exp_data = pd.concat([exp_data_1a,exp_data_3],axis=0,ignore_index=True)

#
# barplot for mean RT
mpl.rcParams.update({'font.size':18})
fig,ax = plt.subplots(2,2,figsize=(12,9))
ax = ax.ravel()
#
for indx, exp_tag in enumerate(exp_tag_list):
    if exp_tag=='exp1a':
        sub_title = 'Experiment 1a'
        fig_lab = '(A)'
    else:
        sub_title = 'Experiment 1b'
        fig_lab = '(B)'
    sns.barplot(data=exp_data[exp_data['exp']==exp_tag],
                x='setsize',y='rt',
                hue='cond',hue_order=['within','between'],
                palette='Blues',saturation=0.75,
                errorbar='se',capsize=0.15,errcolor='grey',
                errwidth=1.5,ax=ax[indx])
    # ax[indx].set_xticklabels([])
    ax[indx].set_xlabel('')
    ax[indx].set_ylabel('')
    ax[indx].set_ylim(0.0,0.62)
    ax[indx].set_title(sub_title,fontsize=15)
    ax[indx].text(-0.85,0.68,fig_lab,ha='center',
                  va='top',color='k',fontsize=15)
ax[0].get_legend().remove()
h,_ = ax[1].get_legend_handles_labels()
ax[1].legend(h,['within','between'],loc='upper left',ncol=1,fontsize=13,
             frameon=False).set_title(None)
ax[1].set_yticklabels([])
ax[0].set_ylabel('RT (sec)')
#
for indx,exp_tag in enumerate(exp_tag_list):
    if exp_tag=='exp1a':
        sub_title = 'Experiment 1a'
        fig_lab = '(C)'
    else:
        sub_title = 'Experiment 1b'
        fig_lab = '(D)'
    sns.lineplot(data=pred_plt[pred_plt['exp']==exp_tag],x='setsize',y='rt',
                 hue='Category',hue_order=['within','between'],
                 style='Fit',style_order=['log2','linear'],
                 err_style=None,lw=3,markersize=10,
                 markers=False,
                 palette='Blues',ax=ax[indx+2])
    sns.scatterplot(data=data_plt[data_plt['exp']==exp_tag],
                    x='setsize',y='rt',hue='Category',markers=['o','X'],
                    style='Category',style_order=['within','between'],
                    palette={'within':'grey','between':'black'},
                    ax=ax[indx+2])
    ax[indx+2].collections[0].set_sizes([100])
    if exp_tag=='exp1a':
        text_x = 16
        tick_size = [1,2,4,8,16]
        plt_size = [str(x) for x in tick_size]
        x = -1
    if exp_tag=='exp3':
        text_x = 8.35
        tick_size = [1,2,4,8]
        plt_size = [str(x) for x in tick_size]
        x = -0.27
    for key,value in text_labels[exp_tag].items():
        num = round(value[-1],3)
        ax[indx+2].text(text_x,value[1],str(num),ha='center',
                      va='bottom',color=value[0],fontsize=13)
    ax[indx+2].set_xticks(tick_size)
    ax[indx+2].set_xticklabels(plt_size)
    ax[indx+2].set_xlabel('')
    ax[indx+2].set_ylabel('')
    ax[indx+2].set_yticks(np.arange(0.45,0.76,0.1))
    ax[indx+2].set_title(sub_title,fontsize=15)
    ax[indx+2].text(x,0.8,fig_lab,ha='center',
                    va='top',color='k',fontsize=15)
ax[2].get_legend().remove()
handles,labels = ax[3].get_legend_handles_labels()
ax[3].legend(handles=handles[1:3]+handles[4:],
             labels=labels[1:3]+labels[4:],title=None,
             frameon=False,loc='upper left',
             ncol=1,fontsize=13)
ax[3].set_yticklabels([])
# fig.text(0.5,0,'Memory Set Size',ha='center')
# fig.text(0,0.5,'RT (sec)',va='center',rotation='vertical')
ax[2].set_xlabel('Memory Set Size')
ax[3].set_xlabel('Memory Set Size')
ax[2].set_ylabel('RT (sec)')

sns.despine(offset=15,trim=True)
plt.tight_layout()

figName = os.path.join(figPath,'behav_descr_fit.tif')
save_fig(fig,figName)
plt.show(block=True)
plt.close('all')
