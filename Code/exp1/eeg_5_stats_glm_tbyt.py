#!/usr/bin/env python
# -*-coding:utf-8 -*-

# EXP. 3-3.1 (EEG): evoke
# Temporal dynamics of non-target processing vary with
# memory set size in different category conditions
# 2021.Nov.24
# linlin.shang@donders.ru.nl



#%% --- --- --- --- --- --- libraries --- --- --- --- --- ---

from eeg_config import epoDataPath,resPath,glmFigPath,\
    subjList_final,subjAllN_final,comp_list,condList,\
    cond_label_list,recog_label_list,sizeList,recog_labels,\
    postChans,frontChans,n_permutations,chance_crit,p_crit,\
    save_fig,set_filepath
import mne
from mne.stats import permutation_cluster_1samp_test
import os
import numpy as np
import pandas as pd
import pingouin as pg
import scipy.stats
import statsmodels.api as sm
from statsmodels.formula.api import glm
from sklearn import preprocessing
import seaborn as sns
import statannot
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.pyplot import MultipleLocator



# --- --- --- --- --- --- --- --- --- Set Global Parameters --- --- --- --- --- --- --- --- --- #

erp_data = pd.read_csv(
    os.path.join(resPath,'erp_eeg.csv'),sep=',')
times = erp_data.loc[(erp_data['subj']==1)&
                     (erp_data['type']=='w/1'),
                     'time'].values.tolist()
erp_recog = erp_data[erp_data['type'].isin(recog_labels)]
erp_recog.reset_index(drop=True,inplace=True)
erp_recog = erp_recog.copy()


# similar pattern
simiChans = ['P5','P6','P7','P8',
             'PO3','PO4','PO7','PO8','PO9','PO10',
             'O1','O2']
# # cluster
# clu1 = ['P5','P7','P8',
#         'PO3','PO7','PO9',
#         'O1']
# clu2 = ['AF3','AF4','AF7','AF8',
#         'F1','F3','F5','F7',
#         'FC5']
# clu3 = ['P6','P8',
#         'PO4','PO8',
#         'TP8']
# clu4 = ['P1','P2','P4','Pz',
#         'PO3','PO4','POz',
#         'O1','O2','Oz']
# clu5 = ['AF3',
#         'F2','F3','Fz',
#         'FC3','FC5','FCz',
#         'C3']
# clu6 = ['C5','CP5']
# cluChans = clu1+clu3
# cluChans.remove('TP8')

# for pick_tag,picks in zip(['po78'],[['PO7','PO8']]):
for pick_tag, picks in zip(['simi'],[simiChans]):
# for pick_tag, picks in zip(['clu_post'],[cluChans]):
    erp_recog.loc[:,[pick_tag]] = erp_recog[picks].mean(axis=1).values
    erp_recog.loc[:,['cond','setsize']] = \
        erp_recog['type'].str.split('/',expand=True).values
    erp_recog_mean = erp_recog.groupby(
        ['time','type'])[pick_tag].agg(np.mean).reset_index()
    erp_recog_mean.loc[:,['cond','setsize']] = \
        erp_recog_mean['type'].str.split('/',expand=True).values
'''
# top-3
pick_tag = 'top3'
top3_data = pd.read_csv(
    os.path.join(resPath,'peak_3elec_info.csv'),
    sep=',')
top3_amps = []
for subjN in subjList_final:
    picks = top3_data.loc[
        (top3_data['subj']==subjN)&
        (top3_data['comp']=='p2'),
        'chan'].tolist()
    top3_amps += erp_recog.loc[
        erp_recog['subj']==subjN,
        picks].mean(axis=1).values.tolist()
erp_recog[pick_tag] = top3_amps
erp_recog.loc[:,['cond','setsize']] = \
    erp_recog['type'].str.split('/',expand=True).values
erp_recog_mean = erp_recog.groupby(
    ['time','type'])[pick_tag].agg(np.mean).reset_index()
erp_recog_mean.loc[:,['cond','setsize']] = \
    erp_recog_mean['type'].str.split('/',expand=True).values
'''


# --- --- --- --- --- --- --- --- --- Sub-Functions --- --- --- --- --- --- --- --- --- #

def find_sig(clu,clu_p,t_points):
    acc_sig, grp_sig = t_points*[0],t_points*[0]
    grp_label = 0

    for c,p in zip(clu, clu_p):
        if p <= p_crit:
            grp_label += 1
            acc_sig[c[0][0]:(c[0][-1]+1)] = \
                [1]*len(c[0])
            grp_sig[c[0][0]:(c[0][-1]+1)] = \
                [grp_label]*len(c[0])
    return acc_sig,grp_sig

def clu_permu_1samp_t(acc_data):
    t_thresh = None
    tail = -1

    t_obs, clu, clu_p, H0 = permutation_cluster_1samp_test(
        acc_data,n_permutations=n_permutations,
        threshold=t_thresh, tail=tail,
        out_type='indices', verbose=True)
    print(clu)
    print(clu_p)

    acc_sig,grp_sig = find_sig(clu,clu_p,acc_data.shape[1])
    return acc_sig,grp_sig

def plotERP(evkData,x,acc_df,tag,title_name):
    mpl.style.use('default')

    # scale = 1e6
    scale = 1
    clrList = ['crimson','gold','darkturquoise','dodgerblue'] * 2
    lineStyList = ['-']*4+['--']*4


    fig,ax = plt.subplots(1,figsize=(20,16))
    for cond,clr,lineSty in zip(recog_labels,clrList,
                                lineStyList):
        print('*')
        print('* %s' % (cond))

        y = evkData.loc[(evkData['type']==cond),tag]
        ax.plot(x,y*scale,color=clr,linestyle=lineSty,
                linewidth=3,label=cond,alpha=0.8)
        # ci_low,ci_up = bootstrap_confidence_interval(evkData[cond].data,\
        # random_state=0,\
        # stat_fun=sum2_func)
        # ci_low = rescale(ci_low,x,baseline=(-0.2,0))
        # ci_up = rescale(ci_up,x,baseline=(-0.2,0))
        # ax.fill_between(x,y+ci_up,y-ci_low,color=clr,alpha=0.05)
    '''
    if tag=='post':
        tag_n = -0.5
    else:
        tag_n = 0.5
    for n in range(len(sig_start)):
        x_sig = x[(x>=sig_start[n])&(x<=sig_end[n])]
        ax.plot(x_sig,[tag_n*1e-6]*len(x_sig),color='gray')
    '''
    ymin,ymax = ax.get_ylim()
    grp_sig = list(set(acc_df['grp'].values))
    for k in grp_sig:
        if k!=0:
            t0 = acc_df.loc[acc_df['grp']==k,'time'].values[0]
            t1 = acc_df.loc[acc_df['grp']==k,'time'].values[-1]
            ax.fill_betweenx(
                (ymin,ymax),t0,t1,
                color='orange',alpha=0.3)
            print('%f,%f'%(t0,t1))
    ax.grid(True)

    # ax.set_xlim(xmin=tmin,xmax=tmax)
    ax.set_title(title_name)
    ax.set_xlabel(xlabel='Time (sec)')
    ax.set_ylabel(ylabel='μV')
    x_major_locator = MultipleLocator(0.1)
    ax.xaxis.set_major_locator(x_major_locator)
    ax.axvline(0, ls='--',color='k')
    ax.axhline(0, ls='--',color='k')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.legend(loc='best',ncol=2)
    plt.grid(linestyle=':')
    fig.tight_layout()

    return fig



#%%  --- --- --- --- --- --- Main Function --- --- --- --- --- --- #

df_subj_all = pd.DataFrame()
df_glm = pd.DataFrame()
glm_subj,glm_t,glm_cond,glm_coeff = [],[],[],[]

for subjN in subjList_final:
    print('SUBJECT %d STARTS'%subjN)

    fname = 'subj%d_epo.fif'%subjN
    subj_epo_all = mne.read_epochs(
        os.path.join(epoDataPath,fname))
    subj_epo = subj_epo_all[recog_labels]
    df_subj_epo = subj_epo.to_data_frame()
    df_subj_epo[['cond','img_cate','setsize']] = \
        df_subj_epo['condition'].str.split('/',n=2,expand=True)
    df_subj_epo['setsize'] = df_subj_epo['setsize'].astype('int')
    # fit_tag = 'lm'
    fit_tag = 'log'
    df_subj_epo['setsize'] = df_subj_epo['setsize'].apply(np.log2)
    df_subj_epo['subj'] = [subjN]*len(df_subj_epo)
    df_subj_all = pd.concat(
        [df_subj_all,df_subj_epo],
        axis=0,ignore_index=True)
    print('SUBJECT %d FINISHED'%subjN)
    print('***')
    print('**')
    print('*')
'''
df_subj_all.to_csv(
    os.path.join(resPath,'epo_data.csv'),
    sep=',')
'''

erp_stats_all = pd.DataFrame()
for t in times:
    for subjN in subjList_final:
        # --- --- ---
        # # top-3
        # picks = top3_data.loc[
        #     (top3_data['subj']==subjN)&
        #     (top3_data['comp']=='p2'),
        #     'chan'].tolist()
        # pick_tag = 'top3'
        # --- --- ---

        df_subj = df_subj_all.loc[
            (df_subj_all['subj']==subjN)&
            (df_subj_all['time']==t),
            ['epoch','cond','setsize']+picks]
        df_subj = df_subj.copy()
        df_subj['cond_trans'] = np.where(df_subj['cond']=='w',1,-1)
        df_subj[pick_tag] = df_subj[picks].mean(axis=1).values
        df_subj['subj'] = [subjN]*len(df_subj)

        erp_stats = df_subj[['subj','cond_trans','setsize',pick_tag]]
        erp_stats = erp_stats.copy()
        erp_stats[pick_tag+'_Z'] = preprocessing.scale(
            erp_stats[pick_tag].values)
        erp_stats['cond_Z'] = preprocessing.scale(
            erp_stats['cond_trans'].values)
        erp_stats['setsize_Z'] = preprocessing.scale(
            erp_stats['setsize'].values)
        erp_stats['inter'] = erp_stats['cond_Z']*erp_stats['setsize']
        erp_stats['inter_Z'] = preprocessing.scale(erp_stats['inter'].values)
        erp_stats['time'] = [t]*len(erp_stats)
        erp_stats_all = pd.concat(
            [erp_stats_all,erp_stats],axis=0,ignore_index=True)

        # GLM fit
        y = erp_stats[pick_tag+'_Z']
        X = erp_stats[['cond_Z','setsize_Z','inter_Z']]
        X = sm.add_constant(X)
        model = sm.GLM(y,X,family=sm.families.Gaussian()).fit()
        glm_subj += [subjN]*4
        glm_t += [t]*4
        glm_cond.append('intc')
        glm_coeff.append(model.params[0])
        glm_cond.append('cond')
        glm_coeff.append(model.params[1])
        glm_cond.append('setsize')
        glm_coeff.append(model.params[2])
        glm_cond.append('inter')
        glm_coeff.append(model.params[3])

df_glm['subj'] = glm_subj
df_glm['time'] = glm_t
df_glm['cond'] = glm_cond
df_glm['coeff'] = glm_coeff

df_glm.to_csv(os.path.join(resPath,
                           'df_glm_tbyt_log.csv'),sep=',')

t0_stats,t1_stats = 0.1,0.3
acc_df = pd.DataFrame()
cond_acc,accs,grp_acc,t_acc = [],[],[],[]
df_glm = df_glm[(df_glm['time']>=t0_stats)&
                (df_glm['time']<=t1_stats)].reset_index(drop=True)
t = df_glm.loc[(df_glm['subj']==1)&(df_glm['cond']=='cond'),'time'].tolist()
for cond in ['cond','setsize','inter']:
    # coeff_arr = np.zeros(
    #     [subjAllN_final,len(times)])
    coeff_arr = np.zeros(
        [subjAllN_final,len(t)])

    print(cond)
    for n in range(subjAllN_final):
        subjN = subjList_final[n]
        coeff_arr[n,:] = df_glm.loc[
            (df_glm['subj']==subjN)&
            (df_glm['cond']==cond),
            'coeff'].values

    acc_sig,grp_sig = clu_permu_1samp_t(coeff_arr)
    accs += acc_sig
    grp_acc += grp_sig
    # cond_acc += [cond]*len(times)
    # t_acc += times
    cond_acc += [cond]*len(t)
    t_acc += t

acc_df['cond'] = cond_acc
acc_df['time'] = t_acc
acc_df['p_sig'] = accs
acc_df['grp'] = grp_acc

# # Plot
# fig_pref = os.path.join(glmFigPath,'glm_clu_')
# for cond in ['cond','setsize','inter']:
#     fig = plotERP(erp_recog_mean,times,
#                   acc_df[acc_df['cond']==cond],pick_tag,
#                   '%s (%s)'%(cond,pick_tag))
#     figName = fig_pref+'%s_%s.png'%(pick_tag,cond)
#     save_fig(fig,figName)
# Plot
fig_pref = os.path.join(glmFigPath,'glm_clu_')
for cond in ['cond','setsize','inter']:
    print(cond)
    fig = plotERP(erp_recog_mean,times,
                  acc_df[acc_df['cond']==cond],pick_tag,
                  '%s (%s)'%(cond,pick_tag))
    figName = fig_pref+'%s_%s_%s.png'%(pick_tag,fit_tag,cond)
    save_fig(fig,figName)
