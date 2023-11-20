#!/usr/bin/env python
# -*-coding:utf-8 -*-

# EXP. 3-3.1 (EEG): evoke
# Temporal dynamics of non-target processing vary with
# memory set size in different category conditions
# 2021.Nov.24
# linlin.shang@donders.ru.nl



#%% --- --- --- --- --- --- libraries --- --- --- --- --- ---

from eeg_config import epoDataPath,resPath,aovFigPath,\
    subjList_final,subjAllN_final,sizeList,cond_label_list,\
    recog_labels,postChans,tmin,tmax,t_space,show_flg,\
    n_permutations,p_crit,save_fig
import mne
from mne.stats import permutation_cluster_1samp_test,\
    permutation_cluster_test
from mne.stats import f_threshold_mway_rm,f_mway_rm,fdr_correction
import os
import copy
import numpy as np
import pandas as pd
import pingouin as pg
from scipy import stats
import statsmodels.api as sm
from statsmodels.formula.api import glm
from statsmodels.gam.api import GLMGam, BSplines
from statsmodels.graphics.factorplots import interaction_plot
from scipy.signal import argrelextrema
from sklearn.linear_model import LinearRegression
from sklearn import preprocessing
import torch
import seaborn as sns
import statannot
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.pyplot import MultipleLocator

from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = "all"
model = LinearRegression()



# --- --- --- --- --- --- Sub-Functions --- --- --- --- --- --- #

def find_sig(clu,clu_p,t_points):
    acc_sig, grp_sig = t_points*[0],t_points*[0]
    grp_label = 0

    for c,p in zip(clu, clu_p):
        if p < p_crit:
            grp_label += 1
            acc_sig[c[0][0]:(c[0][-1]+1)] = \
                [1]*len(c[0])
            grp_sig[c[0][0]:(c[0][-1]+1)] = \
                [grp_label]*len(c[0])
    return acc_sig,grp_sig

def stat_fun(*args):
    factor_levels = [4,2]
    effects = 'A:B'
    return f_mway_rm(
        np.array(args),factor_levels=factor_levels,
        effects=effects,return_pvals=False)[0]

def stat_fun_1way(*args):
    factor_levels = [4,1]
    effects = 'A'
    return f_mway_rm(
        np.array(args),factor_levels=factor_levels,
        effects=effects,return_pvals=False)[0]

def clu_permu_cond(erp_data):
    tail = 0
    # pthresh = 0.001
    factor_levels = [4,2]
    effects = 'A:B'
    f_thresh = f_threshold_mway_rm(
        subjAllN_final,factor_levels,effects,p_crit)
    f_thresh = None
    f_obs,clu,clu_p,h0 = permutation_cluster_test(
        erp_data,stat_fun=stat_fun,threshold=f_thresh,
        tail=tail,n_jobs=None,
        n_permutations=n_permutations,
        buffer_size=None,out_type='indices')
    print(clu)
    print(clu_p)

    acc_sig, grp_sig = find_sig(clu,clu_p,erp_data.shape[-1])
    return acc_sig, grp_sig

def check_sig(time_list):
    sig_start,sig_end = [],[]
    count = 1
    for n in range(len(time_list)):
        if ((n==0) and (time_list[n]>0)) or\
                ((n>0) and (time_list[n]>0) and
                 (round(time_list[n]-time_list[n-1],3)>
                  0.004)):
            sig_start.append(time_list[n])
            if count>1:
                sig_end.append(time_list[n-1])
            count += 1

    if len(time_list)!=0:
        sig_end.append(time_list[len(time_list)-1])

    return sig_start,sig_end

def clu_permu_1samp_t(acc_data):
    t_thresh = None
    tail = -1
    degrees_of_freedom = len(acc_data) - 1
    # t_thresh = scipy.stats.t.ppf(
    #     1 - p_crit / 2, df=degrees_of_freedom)

    t_obs, clu, clu_p, H0 = permutation_cluster_1samp_test(
        acc_data,n_permutations=n_permutations,
        threshold=t_thresh, tail=tail,
        out_type='indices', verbose=True)
    print(clu)
    print(clu_p)

    acc_sig,grp_sig = find_sig(clu,clu_p,acc_data.shape[1])
    return acc_sig,grp_sig

def plotERP(evkData,x,sig_start,sig_end,tag,title_name):
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
        y = evkData.loc[evkData['type']==cond,tag]
        ax.plot(x,y*scale,color=clr,linestyle=lineSty,
                linewidth=5,label=cond,alpha=0.8)
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
    for n in range(len(sig_start)):
        ax.fill_between(
            x,ymin,ymax,
            where=(x>=sig_start[n])&(x<=sig_end[n]),
            color='orange',alpha=0.3)

    ax.grid(True)

    ax.set_xlim(xmin=tmin,xmax=tmax)
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

def plot_fit(df,y_tag,y_name,title_name):
    mpl.style.use('default')

    fig, ax = plt.subplots(1,figsize=(20, 16))
    for cond,clr,dot in zip(cond_label_list,['orange','g'],['^','o']):
        x = df.loc[df['cond']==cond,'setsize'].astype(int).values
        # x = list(map(lambda x:log(x,4),sizeList))
        y = df.loc[df['cond']==cond,y_tag].values
        ax.plot(x,y,color=clr,linewidth=3,
                marker=dot,
                label=cond,alpha=0.8)
    ax.set_title(title_name)
    ax.set_xlabel(xlabel='Memory Set Size')
    ax.set_ylabel(ylabel=y_name)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.legend(loc='best',ncol=2)
    plt.grid(linestyle=':')
    fig.tight_layout()

    return fig

# def predFunc(df_train,df,subjList_final):
#     for k in subjList_final:
#         for h in condList:
#             dfSub_train = df_train[(df_train['subj']==k)&(df_train['type']==h)]
#             x = dfSub_train['setsize'].values
#             y = list(dfSub_train['rt'])
#             model.fit(x.reshape(-1,1),y)
#             pred_value = df.loc[(df['subj']==k)&(df['type']==h),'setsize'].values
#             df.loc[(df['subj']==k)&(df['type']==h),'lmPred'] = model.predict(pred_value.reshape(-1,1))
#             x = np.array(logSizeList[0:-1])
#             model.fit(x.reshape(-1,1),y)
#             pred_value = np.array(logSizeList)
#             df.loc[(df['subj']==k)&(df['type']==h),'logPred'] = model.predict(pred_value.reshape(-1,1))
#     return df

def getCoeff(df,x_var,y_var,subjList_final):
    dfCoeffNew = pd.DataFrame(columns=['subj','cond','coeff','r2'])
    count = 0
    for n in subjList_final:
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

erp_data = pd.read_csv(os.path.join(resPath,'erp_eeg.csv'), sep=',')
times = erp_data.loc[(erp_data['subj']==1)&
                     (erp_data['type']=='w/1'),'time'].values

erp_recog = erp_data[erp_data['type'].isin(recog_labels)]
erp_recog.reset_index(drop=True,inplace=True)
erp_recog = erp_recog.copy()

# for pick_tag,pick_chs in zip(['post','front','fcp'],
#                              [postChans,frontChans,fcpChans]):
for pick_tag, pick_chs in zip(['post'],[postChans]):
    erp_recog.loc[:,pick_tag] = erp_recog[pick_chs].mean(axis=1).values
    erp_recog.loc[:,['cond','setsize']] = \
        erp_recog['type'].str.split('/', expand=True).values

    erp_recog_mean = erp_recog.groupby(['time','type']
                                       )[pick_tag].agg(np.mean).reset_index()
    erp_recog_mean.loc[:,['cond','setsize']] = \
        erp_recog_mean['type'].str.split('/',expand=True).values

    erp_recog_avg = erp_recog.groupby(['time']
                                      )[pick_tag].agg(np.mean).reset_index()

    # top-3 electrodes
    evk_peak = pd.DataFrame()
    evk_peak_avg = pd.DataFrame()
    # plt.plot(erp_recog_avg['time'], erp_recog_avg[pick_tag])
    # plt.show(block=show_flg)
    # plt.close('all')

    erp_chans = []
    erp_times = []
    erp_amps = []
    erp_subjs = []
    erp_subj_avg,erp_cond_avg,erp_amp_avg,erp_cp_avg = [],[],[],[]
    erp_cp = []
    cp_tag_list = ['p1','n1','p2']
    for subjN in subjList_final:
        for cp,t0,t1,mode in [('p1',0.08,0.18,'pos'),
                              ('n1',0.15,0.25,'abs'),
                              ('p2',0.2,0.3,'pos')]:
            picks = copy.deepcopy(pick_chs)
            erp_recog_subj = erp_recog[
                (erp_recog['subj']==subjN)&
                (erp_recog['time']>=t0)&
                (erp_recog['time']<=t1)]
            erp_recog_subj.reset_index(drop=True,inplace=True)
            erp_recog_subj_avg = erp_recog_subj.groupby(
                ['subj','time'])[pick_chs].agg(np.mean).reset_index()

            for k in range(3):
                amp_df = pd.DataFrame()
                chan_list,amp_list,time_list = [],[],[]
                for chan in picks:
                    amp_indx_df = pd.DataFrame()
                    if cp=='n1':
                        amp_indx_list = argrelextrema(
                            erp_recog_subj_avg[chan].values,
                            np.less)[0].tolist()
                    else:
                        amp_indx_list = argrelextrema(
                            erp_recog_subj_avg[chan].values,
                            np.greater)[0].tolist()
                    if amp_indx_list!=[]:
                        if cp=='n1':
                            asc_tag = True
                        else:
                            asc_tag = False
                        amp_indx_df['indx'] = amp_indx_list
                        amp_indx_df['amp'] = erp_recog_subj_avg.loc[
                            amp_indx_list,chan].values
                        amp_indx_df = amp_indx_df.sort_values(
                            'amp',ascending=asc_tag,ignore_index=True)
                        amp_indx = amp_indx_df.loc[0,'indx']
                        amp_list.append(erp_recog_subj_avg.loc[amp_indx,chan])
                        chan_list.append(chan)
                        time_list.append(erp_recog_subj_avg.loc[amp_indx,'time'])

                amp_df['chan'] = chan_list
                amp_df['amp'] = amp_list
                amp_df['time'] = time_list
                if cp=='n1':
                    asc_tag = True
                else:
                    asc_tag = False
                amp_df = amp_df.sort_values(
                    'amp',ascending=asc_tag,ignore_index=True)
                t = amp_df.loc[0,'time']
                erp_times.append(t)
                best_chan = amp_df.loc[0,'chan']
                erp_chans.append(best_chan)
                print()
                erp_amps.append(amp_df.loc[0,'amp'])
                erp_cp.append(cp)
                erp_subjs.append(subjN)

                for cond in recog_labels:
                    amp_avg = erp_recog_subj.loc[
                        (erp_recog_subj['type']==cond)&
                        (erp_recog_subj['time']<=(t+t_space))&
                        (erp_recog_subj['time']>=(t-t_space)),
                        best_chan].mean()
                    erp_amp_avg.append(amp_avg)
                    erp_cond_avg.append(cond)
                    erp_subj_avg.append(subjN)
                    erp_cp_avg.append(cp)

                picks.remove(best_chan)

    evk_peak['subj'] = erp_subjs
    evk_peak['comp'] = erp_cp
    evk_peak['chan'] = erp_chans
    evk_peak['time'] = erp_times
    evk_peak['peak_amp'] = erp_amps
    evk_peak.to_csv(os.path.join(resPath,'peak_3elec_info.csv'),
                    mode='w',header=True,index=False)

    evk_peak_avg['subj'] = erp_subj_avg
    evk_peak_avg['comp'] = erp_cp_avg
    evk_peak_avg['type'] = erp_cond_avg
    evk_peak_avg['avg_amp'] = erp_amp_avg
    evk_peak_avg.loc[:,['cond','setsize']] = \
        evk_peak_avg['type'].str.split('/',expand=True).values
    evk_peak_avg.to_csv(os.path.join(resPath,'peak_3elec_amp.csv'),
                        mode='w',header=True,index=False)

    pd.set_option('display.max_columns',None)

    # Statistics
    pick_tag = 'top3'
    for cp in cp_tag_list:
        evk_cp = evk_peak_avg[(evk_peak_avg['comp']==cp)].copy().reset_index()

        # ERP
        aov = pg.rm_anova(dv='avg_amp',within=['cond','setsize'],
                          subject='subj',
                          data=evk_cp,
                          detailed=True,effsize='np2')
        # plot fit
        fig_pref = os.path.join(aovFigPath,'aov_top3_')
        evk_cp_avg = evk_cp.groupby(
            ['cond','setsize'])['avg_amp'].agg(np.mean).reset_index()
        fig = plot_fit(evk_cp_avg,'avg_amp','μV',cp.title())
        figName = fig_pref+'fit_%s.png'%(cp)
        save_fig(fig,figName)

        print('%s'%cp)
        print(aov)
        print('---***---***---')

# --- --- ---
amp_avg_list = []
for subjN in subjList_final:
    best_chan_list = evk_peak.loc[
        (evk_peak['subj']==subjN)&
        (evk_peak['comp']=='p2'),'chan'].values.tolist()
    amp_avg_list += \
        erp_recog[erp_recog['subj']==subjN][
            best_chan_list].mean(axis=1).values.tolist()
erp_recog['top3'] = amp_avg_list

# time-by-time
top3_mean = erp_recog.groupby(
    ['time','type'])[pick_tag].agg(np.mean).reset_index()
erp_recog_mean[pick_tag] = top3_mean[pick_tag]
df_aov_t = pd.DataFrame()
for t in times:
    # t = round(tt,3)
    aov = pg.rm_anova(
        dv=pick_tag,within=['cond','setsize'],subject='subj',
        data=erp_recog[erp_recog['time']==t],
        detailed=True,effsize='np2')
    aov['time'] = [t]*len(aov)
    df_aov_t = pd.concat([df_aov_t,aov],axis=0,ignore_index=True)

# get significant time points
# anova
print('--- *** --- *** --- *** ---')
sigList = df_aov_t.loc[
    (df_aov_t['Source']=='cond * setsize')&
    (df_aov_t['p-GG-corr']<p_crit),'time']
pd.set_option('display.max_columns',None)
sig_aov = np.array(sigList)
grp_aov_start,grp_aov_end = check_sig(sig_aov)
print(sig_aov)
print(grp_aov_start)
print(grp_aov_end)
# category
print('--- *** --- *** --- *** ---')
sigList = df_aov_t.loc[
    (df_aov_t['Source']=='cond')&
    (df_aov_t['p-GG-corr']<p_crit),'time']
pd.set_option('display.max_columns',None)
print('Category')
sig_cate = np.array(sigList)
grp_cate_start,grp_cate_end = check_sig(sig_cate)
print(sig_cate)
print(grp_cate_start)
print(grp_cate_end)
# set size
print('--- *** --- *** --- *** ---')
sigList = df_aov_t.loc[
    (df_aov_t['Source']=='setsize')&
    (df_aov_t['p-GG-corr']<p_crit),'time']
pd.set_option('display.max_columns',None)
print('Set Size')
sig_size = np.array(sigList)
grp_size_start,grp_size_end = check_sig(sig_size)
print(sig_size)
print(grp_size_start)
print(grp_size_end)
print('--- *** --- *** --- *** ---')
# Plot
fig_pref = os.path.join(aovFigPath,'aov_tbyt_')
fig = plotERP(erp_recog_mean,times,grp_cate_start,grp_cate_end,
              pick_tag,'Category Effect (%s)'%pick_tag)
figName = fig_pref+'%s_cate.png'%pick_tag
save_fig(fig,figName)
fig = plotERP(erp_recog_mean,times,grp_size_start,grp_size_end,
              pick_tag,'Set Size Effect (%s)'%pick_tag)
figName = fig_pref+'%s_size.png'%pick_tag
save_fig(fig,figName)


# 10-ms bins
# t_step = 0.01
t_step = 0.015
df_aov = pd.DataFrame()
for t0 in times:
    t1 = t0+t_step
    if t1<=times[-1]:
        erp_stats = erp_recog[
            (erp_recog['time']<=t1)&
            (erp_recog['time']>=t0)].groupby(
            ['subj','cond','setsize'])[pick_tag].agg(np.mean).reset_index()
        aov = pg.rm_anova(
            dv=pick_tag,within=['cond','setsize'],subject='subj',
            data=erp_stats,detailed=True,effsize='np2')
        pwc = pg.pairwise_tests(
            dv=pick_tag,within=['cond','setsize'],subject='subj',
            data=erp_stats,padjust='bonf',effsize='hedges')

        aov['time'] = [t0]*len(aov)
        df_aov = pd.concat([df_aov,aov],axis=0,ignore_index=True)

sigList = df_aov.loc[(df_aov['Source']=='cond * setsize')&
                     (df_aov['p-unc']<p_crit),'time']
print(sigList)
sigList = df_aov.loc[(df_aov['Source']=='cond * setsize')&
                     (df_aov['p-GG-corr']<p_crit),'time']
print(sigList)

# 200-300 ms
# t0,t1 = 0.2,0.3
# t0,t1 = 0.25,0.3
# t0,t1 = 0.172,0.264 # category
# t0,t1 = 0.196,0.324 # setsize
# t0,t1 = 0.19,0.26 # glm inter: 0.043
# t0,t1 = 0.19,0.24
# t0,t1 = 0.196,0.264
# t0,t1 = 0.24,0.29

t0,t1 = 0.2,0.212
erp_stats = erp_recog[
    (erp_recog['time']<=t1)&
    (erp_recog['time']>=t0)].groupby(
    ['subj','cond','setsize'])[pick_tag].agg(np.mean).reset_index()
aov = pg.rm_anova(
    dv=pick_tag,within=['cond','setsize'],subject='subj',
    data=erp_stats,detailed=True,effsize='np2')
pwc1 = pg.pairwise_tests(
    dv=pick_tag,within=['cond','setsize'],subject='subj',
    alternative='two-sided',data=erp_stats,
    padjust='bonf',effsize='hedges')
pwc2 = pg.pairwise_tests(
    dv=pick_tag,within=['setsize','cond'],subject='subj',
    alternative='two-sided',data=erp_stats,
    padjust='bonf',effsize='hedges')
print('*** *** ***')
print(aov)
print('*** *** ***')
print(pwc1)
print('*** *** ***')
print(pwc2)
# 0.196-0.216: 0.049152(p-unc)/0.056103(p-GG-cor)

erp_coeff = pd.DataFrame()
erp_coeff = getCoeff(
    erp_stats,'setsize',pick_tag,subjList_final)
dfCoeffNew = pd.DataFrame(columns=['subj','cond','coeff','r2'])
count = 0
for n in subjList_final:
    dfCoeff = pd.DataFrame()
    dfCoeff.loc[count,'subj'] = n
    dfCoeff.loc[count,'cond'] = 'both'
    x = erp_stats.loc[(erp_stats['subj']==n),'setsize'].values
    x = x.astype('int')
    x = np.log2(x)
    y = erp_stats.loc[(erp_stats['subj']==n),pick_tag].values
    model.fit(x.reshape(-1, 1), y)
    dfCoeff.loc[dfCoeff['subj']==n,'coeff'] = model.coef_
    dfCoeff.loc[dfCoeff['subj']==n,'r2'] = model.score(x.reshape(-1,1),y)
    dfCoeffNew = pd.concat([dfCoeffNew,dfCoeff],axis=0,ignore_index=True)
    count += 1
dfCoeffNew.index = range(len(dfCoeffNew))
erp_coeff = pd.concat([erp_coeff,dfCoeffNew],axis=0,ignore_index=True)

# cluster
top3_amps = []
for subjN in subjList_final:
    picks = evk_peak.loc[
        (evk_peak['subj']==subjN)&
        (evk_peak['comp']=='p2'),
        'chan'].tolist()
    pick_tag = 'top3'
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


t0_stats,t1_stats = 0.196,0.264
df_glm = erp_recog[(erp_recog['time']>=t0_stats)&
                   (erp_recog['time']<=t1_stats)].reset_index(drop=True)
t = df_glm.loc[(df_glm['subj']==1)&(df_glm['type']=='w/1'),'time'].tolist()

# interaction
erp_arr = np.zeros(
        [subjAllN_final,len(recog_labels),len(t)])
for n in range(subjAllN_final):
    subjN = subjList_final[n]
    for k,cond in enumerate(recog_labels):
        erp_arr[n,k,:] = df_glm.loc[
            (df_glm['subj']==subjN)&
            (df_glm['type']==cond),
            pick_tag].values
tail = 0
# pthresh = 0.001
factor_levels = [4,2]
effects = 'A:B'

f_thresh = f_threshold_mway_rm(
    subjAllN_final,factor_levels,effects,p_crit)

f_obs,clu,clu_p,h0 = permutation_cluster_test(
    erp_arr,stat_fun=stat_fun,threshold=f_thresh,
    tail=tail,n_jobs=None,n_permutations=n_permutations,
    buffer_size=None,out_type='mask')
print(clu)
print(clu_p)
grp_start,grp_end = [],[]
for c,p in zip(clu,clu_p):
    if p < p_crit:
        grp_start.append(t[c[0]][0])
        grp_end.append(t[c[0]][-2])
print(grp_start,grp_end)

# set size
erp_arr = np.zeros(
        [subjAllN_final,len(sizeList),len(t)])
for n in range(subjAllN_final):
    subjN = subjList_final[n]
    for k,cond in enumerate(sizeList):
        erp_arr[n,k,:] = df_glm[
            (df_glm['subj']==subjN)&
            (df_glm['setsize']==str(cond))
        ].groupby(['time'])[pick_tag].agg(np.mean).values
tail = 0
# pthresh = 0.001
factor_levels = [4,1]
effects = 'A'
f_thresh = f_threshold_mway_rm(
    subjAllN_final,factor_levels,effects,p_crit)
f_obs,clu,clu_p,h0 = permutation_cluster_test(
    erp_arr,stat_fun=stat_fun_1way,threshold=f_thresh,
    tail=tail,n_jobs=None,n_permutations=n_permutations,
    buffer_size=None,out_type='mask')
print(clu)
print(clu_p)
grp_start_size,grp_end_size = [],[]
for c,p in zip(clu,clu_p):
    if p < p_crit:
        grp_start_size.append(t[c[0]][0])
        grp_end_size.append(t[c[0]][-2])
print(grp_start_size,grp_end_size)

fig = plotERP(erp_recog_mean,times,grp_start_size,grp_end_size,
              pick_tag,'Set Size Effect (%s)'%pick_tag)
fig_pref = os.path.join(aovFigPath,'aov_clu_')
figName = fig_pref+'%s_size.png'%pick_tag
save_fig(fig,figName)

# category
erp_arr = np.zeros(
        [subjAllN_final,len(cond_label_list),len(t)])
for n in range(subjAllN_final):
    subjN = subjList_final[n]
    for k,cond in enumerate(cond_label_list):
        erp_arr[n,k,:] = df_glm[
            (df_glm['subj']==subjN)&
            (df_glm['cond']==cond)
        ].groupby(['time'])[pick_tag].agg(np.mean).values

t_thresh = None
tail = -1
degrees_of_freedom = len(erp_arr) - 1

t_obs,clu,clu_p,H0 = permutation_cluster_1samp_test(
    erp_arr[:,0,:]-erp_arr[:,1,:],
    n_permutations=n_permutations,threshold=t_thresh,
    tail=tail,out_type='mask',verbose=True)
print(clu)
print(clu_p)
grp_start_cate,grp_end_cate = [],[]
for c,p in zip(clu,clu_p):
    if p < p_crit:
        grp_start_cate.append(t[c[0]][0])
        grp_end_cate.append(t[c[0]][-2])
print(grp_start_cate,grp_end_cate)

fig = plotERP(erp_recog_mean,times,grp_start_cate,grp_end_cate,
              pick_tag,'Category Effect (%s)'%pick_tag)
figName = fig_pref+'%s_cate.png'%pick_tag
save_fig(fig,figName)

# plot comparison
mpl.style.use('default')
# scale = 1e6
scale = 1
clrList = ['crimson','gold','darkturquoise','dodgerblue'] * 2
lineStyList = ['-']*4+['--']*4
x = times
fig,ax = plt.subplots(1,figsize=(20,16))
for cond,clr,lineSty in zip(recog_labels,clrList,
                            lineStyList):
    print('*')
    print('* %s' % (cond))
    y = erp_recog_mean.loc[erp_recog_mean['type']==cond,pick_tag]
    ax.plot(x,y*scale,color=clr,linestyle=lineSty,
            linewidth=3,label=cond,alpha=0.8)
ymin,ymax = ax.get_ylim()

cate_sig,size_sig = [],[]
for sig_tag,sig_list in zip(
        ['category','set size'],[cate_sig,size_sig]):
    if sig_tag=='category':
        sig_start,sig_end = grp_start_cate,grp_end_cate
        y_loc = -0.0000002
        sig_clr = 'grey'
    else:
        sig_start,sig_end = grp_start_size,grp_end_size
        y_loc = -0.0000004
        sig_clr = 'orange'
    for n in range(len(sig_start)):
        x_sig = np.arange(sig_start[n],sig_end[n],0.004)
        ax.plot(x_sig,[y_loc*scale]*len(x_sig),color=sig_clr,
                linewidth=6,label=sig_tag,alpha=0.8)
        # plt.axhline(y=y_loc,xmin=sig_start,
        #             xmax=sig_end,color=sig_clr,lw=2)

ax.grid(True)
ax.set_xlim(xmin=tmin,xmax=tmax)
ax.set_title('Main Effects')
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
fig_pref = os.path.join(aovFigPath,'aov_clu_')
figName = fig_pref+'%s_main.png'%pick_tag
save_fig(fig,figName)


print('correlate ERP with behavioural data--- --- --- --- --- --- ***')
# get behavioural coeffcients
fileName = 'sch_mean.csv'
df_sch_mean = pd.read_csv(os.path.join(resPath,fileName),sep=',')
behav_coeff = getCoeff(df_sch_mean,'setsize','rt',subjList_final)

dfCoeffNew = pd.DataFrame(columns=['subj','cond','coeff','r2'])
count = 0
for n in subjList_final:
    dfCoeff = pd.DataFrame()
    dfCoeff.loc[count,'subj'] = n
    dfCoeff.loc[count,'cond'] = 'both'
    x = df_sch_mean.loc[(df_sch_mean['subj']==n),'setsize'].values
    x = x.astype('int')
    x = np.log2(x)
    y = df_sch_mean.loc[(df_sch_mean['subj']==n),'rt'].values
    model.fit(x.reshape(-1, 1), y)
    dfCoeff.loc[dfCoeff['subj']==n,'coeff'] = model.coef_
    dfCoeff.loc[dfCoeff['subj']==n,'r2'] = model.score(x.reshape(-1,1),y)
    dfCoeffNew = pd.concat([dfCoeffNew,dfCoeff],axis=0,ignore_index=True)
    count += 1
dfCoeffNew.index = range(len(dfCoeffNew))
behav_coeff = pd.concat([behav_coeff,dfCoeffNew],axis=0,ignore_index=True)

# coefficients correlation
df_corr = pd.DataFrame()
erp_both = erp_coeff[
    erp_coeff['cond']=='both'].reset_index(drop=True)
behav_both = behav_coeff[
    behav_coeff['cond']=='both'].reset_index(drop=True)
coeff_corr = pg.corr(erp_both['coeff'],
                     behav_both['coeff'],method='spearman')
print(coeff_corr)
# coeff_Z_corr = pg.corr(
#     preprocessing.scale(erp_both['coeff']),
#     preprocessing.scale(behav_both['coeff']),method='spearman')
# print(coeff_Z_corr)

corr_data = pd.DataFrame()
corr_data['erp'] = erp_both['coeff'].values
corr_data['behav'] = behav_both['coeff'].values

fig = sns.lmplot(x='behav',y='erp',data=corr_data)
fig.fig.set_size_inches(9,6)
plt.title('Slope Coefficients of Behavioural data vs ERP data')
fig.tight_layout()
figName = os.path.join(aovFigPath,'corr_%s'%pick_tag)
save_fig(fig,figName)

behav_both['task'] = 'behav'
erp_both['task'] = 'erp'
both_data = pd.concat([erp_both,behav_both],axis=0,ignore_index=True)
fig = sns.pairplot(both_data,hue ='task',markers=['o','s'])
fig.tight_layout()
figName = os.path.join(aovFigPath,'corr_paird_%s'%pick_tag)
save_fig(fig,figName)

# print('slope coefficients --- --- --- --- --- --- ***')
# # get ERP coefficients
# erp_coeff = pd.DataFrame()
# for cp in cp_tag_list:
#     coeff_t = getCoeff(
#         evk_peak_avg[evk_peak_avg['comp']==cp],
#         'setsize','avg_amp',subjList_final)
#     coeff_t['comp'] = [cp]*len(coeff_t)
#     erp_coeff = pd.concat([erp_coeff,coeff_t],axis=0,ignore_index=True)
#
# # coefficients anova
# df_aov = pd.DataFrame()
# for cp in cp_tag_list:
#     t_val = pg.ttest(
#         erp_coeff.loc[(erp_coeff['cond']=='w')&
#                       (erp_coeff['comp']==cp),'coeff'],
#         erp_coeff.loc[(erp_coeff['cond']=='b')&
#                       (erp_coeff['comp']==cp),'coeff'],
#         paired=True, correction=True)
#     print(cp)
#     print('--- --- ---')
#     print(t_val)
#     print('--- --- ---')
#
# print('correlate ERP with behavioural data--- --- --- --- --- --- ***')
# # get behavioural coeffcients
# fileName = 'sch_mean.csv'
# df_sch_mean = pd.read_csv(os.path.join(resPath,fileName),sep=',')
# behav_coeff = getCoeff(df_sch_mean,'setsize','rt',subjList_final)
#
# # coefficients correlation
# df_corr = pd.DataFrame()
# for cond in cond_label_list:
#     for cp in cp_tag_list:
#         behav_val = behav_coeff.loc[
#             behav_coeff['cond']==cond,'coeff'].values
#         erp_val = erp_coeff.loc[
#             (erp_coeff['cond']==cond)&
#             (erp_coeff['comp']==cp),'coeff'].values
#         coeff_corr = pg.corr(erp_val,behav_val,
#                              method='spearman')
#         coeff_corr['cond'] = [cond]*len(coeff_corr)
#         coeff_corr['comp'] = [cp]*len(coeff_corr)
#         df_corr = pd.concat([df_corr,coeff_corr],
#                             axis=0,ignore_index=True)
# print(df_corr)