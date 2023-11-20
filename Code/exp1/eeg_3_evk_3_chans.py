#!/usr/bin/env python
# -*-coding:utf-8 -*-

# EXP. 3-3.1 (EEG): evoke
# Temporal dynamics of non-target processing vary with
# memory set size in different category conditions
# 2021.Nov.24
# linlin.shang@donders.ru.nl



#%% --- --- --- --- --- --- libraries --- --- --- --- --- ---

from eeg_config import filePath,resPath,allResFigPath,grpFigPath,\
    subjList_final,subjList,subjAllN_final,subjAllN,\
    comp_list,condList,show_flg,evkDataPath,\
    cond_label_list,recog_labels,sizeList,\
    postChans,n_permutations,\
    t_space,p_crit,tmin,tmax,save_fig,set_filepath
import mne
from mne.stats import permutation_cluster_1samp_test,\
    permutation_cluster_test
from mne.stats import f_threshold_mway_rm,f_mway_rm,fdr_correction
import os
import numpy as np
import pandas as pd
import pingouin as pg
import statsmodels.api as sm

from scipy.signal import argrelextrema
from sklearn.linear_model import LinearRegression
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.pyplot import MultipleLocator
import seaborn as sns
import statannot

from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = "all"
model = LinearRegression()



# --- --- --- --- --- --- Set Global Parameters --- --- --- --- --- --- #

df_sch_mean = pd.read_csv(os.path.join(filePath,'AllExpRes','data_allExp_mean.csv'),sep=',')
df_sch_mean = df_sch_mean[df_sch_mean['exp']=='exp3']
df_sch_mean.loc[df_sch_mean['cond']=='within','cond'] = 'w'
df_sch_mean.loc[df_sch_mean['cond']=='between','cond'] = 'b'
df_sft_cate = pd.read_csv(os.path.join(resPath,'exp3_sft_cap_cate.csv'),sep=',')
df_sft_inter = pd.read_csv(os.path.join(resPath,'exp3_sft_cap_inter.csv'),sep=',')
t_val = pg.ttest(df_sft_cate.loc[(df_sft_cate['cond']=='w'),'zscore'],
                 df_sft_cate.loc[(df_sft_cate['cond']=='b'),'zscore'],
                 paired=True,correction=True)
print('--- --- ---')
pg.print_table(t_val)


erp_data = pd.read_csv(os.path.join(resPath,'erp_eeg.csv'), sep=',')
times = erp_data.loc[(erp_data['subj']==1)&
                     (erp_data['type']=='w/1'),'time'].values
erp_recog = erp_data[erp_data['type'].isin(recog_labels)]
erp_recog.reset_index(drop=True,inplace=True)

# similar pattern
picks = ['P5','P6','P7','P8',
         'PO3','PO4','PO7','PO8','PO9','PO10',
         'O1','O2']
pick_tag = 'simi'
simi_chans = 'P5/P6, P7/P8, PO3/PO4, PO7/PO8, PO9/PO10, O1/O2'
fig_pref = os.path.join(allResFigPath,'aov_simi_')

# PO7/8
# picks = ['PO7','PO8']
# pick_tag = 'po78'
# fig_pref = os.path.join(allResFigPath,'aov_po78_')

# # cluster
# clu1 = ['P2','P5','P6','P7','P8',
#         'PO3','PO4','PO7','PO8','PO9','PO10','POz',
#         'O1','O2',
#         'TP8']
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
# picks = clu1
# picks.remove('TP8')
# pick_tag = 'clu_post'
# simi_chans = 'P2, P5/P6, P7/P8, PO3/PO4, PO7/PO8, PO9/PO10, POz, O1/O2'
# fig_pref = os.path.join(allResFigPath,'aov_simi_')


# --- --- --- --- --- --- Sub-Functions --- --- --- --- --- --- #

def get_peak_data(df):
    if df.loc[0,'subj']==12:
        p2_t1 = 0.35
    else:
        p2_t1 = 0.303
    peak_ts,peak_amps,peak_cps = [],[],[]
    for t0,t1,cp in [(0.0,0.2,'p1'),
                     (0.1,0.3,'n1'),
                     (0.19,p2_t1,'p2')]:
        t_win = (df['time']>t0)&(df['time']<t1)
        df_crop = df[t_win].reset_index()

        if cp=='n1':
            amp_great_indx = argrelextrema(
                df_crop[pick_tag].values,
                np.less)[0].tolist()
            extr_indx = df_crop.loc[amp_great_indx,pick_tag].idxmin()
        elif cp=='p1':
            amp_great_indx = argrelextrema(
                df_crop[pick_tag].values,
                np.greater)[0].tolist()
            extr_indx = df_crop.loc[
                amp_great_indx,pick_tag].idxmax()
        else:
            amp_great_indx = argrelextrema(
                df_crop[pick_tag].values,
                np.greater)[0].tolist()
            if amp_great_indx==[]:
                extr_indx = df_crop[pick_tag].idxmax()
            else:
                # extr_indx = df_crop.loc[
                #     amp_great_indx,pick_tag].idxmax()
                extr_indx = df_crop.loc[
                    amp_great_indx,'time'].idxmin()

        t = df_crop.loc[extr_indx,'time']
        amp = df_crop.loc[extr_indx,pick_tag]
        peak_ts.append(t)
        peak_amps.append(amp)
        peak_cps.append(cp)
    return {'cp':peak_cps,'time':peak_ts,'amps':peak_amps}

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
    factor_levels = [2,4]
    effects = 'A:B'
    return f_mway_rm(
        np.array(args).transpose(1,0,2),
        factor_levels=factor_levels,
        effects=effects,return_pvals=False)[0]

def stat_fun_1way(*args):
    factor_levels = [4,1]
    effects = 'A'
    return f_mway_rm(
        np.array(args).transpose(1,0,2),
        factor_levels=factor_levels,
        effects=effects,return_pvals=False)[0]

def clu_permu_cond(erp_data):
    tail = 0
    # pthresh = 0.001
    factor_levels = [2,4]
    effects = 'A:B'
    # f_thresh = f_threshold_mway_rm(
    #     subjAllN_final,factor_levels,effects,p_crit)
    f_thresh = f_threshold_mway_rm(
        subjAllN,factor_levels,effects,p_crit)
    f_obs,clu,clu_p,h0 = permutation_cluster_test(
        erp_data.transpose(1,0,2),
        stat_fun=stat_fun,threshold=f_thresh,
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

    # ax.grid(True)

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
    # plt.grid(linestyle=':')
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
    # plt.grid(linestyle=':')
    fig.tight_layout()

    return fig

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
            model = sm.OLS(y,sm.add_constant(x)).fit()
            pred_value = df_test['setsize'].values
            pred_res = model.predict(sm.add_constant(pred_value))
            dfCoeff.loc[dfCoeff['subj']==n,'coeff'] = model.params[1]
            dfCoeff.loc[dfCoeff['subj']==n,'r2'] = model.rsquared_adj
            dfCoeffNew = pd.concat([dfCoeffNew,dfCoeff],axis=0,ignore_index=True)
            count += 1
    dfCoeffNew.index = range(len(dfCoeffNew))
    return dfCoeffNew



# --- --- --- --- --- --- Main Function --- --- --- --- --- --- #

erp_recog = erp_recog.copy()
erp_recog.loc[:,pick_tag] = erp_recog[picks].mean(axis=1).values
erp_recog.loc[:,['cond','setsize']] = \
    erp_recog['type'].str.split('/',expand=True).values
erp_recog['setsize'] = erp_recog['setsize'].astype('int')
erp_recog_mean = erp_recog.groupby(
    ['time','type'])[picks+[pick_tag]].agg(np.mean).reset_index()
erp_recog_mean['cond'] = erp_recog_mean['type'].str.split(
    '/',expand=True).values[:,0]
erp_recog_mean['setsize'] = erp_recog_mean['type'].str.split(
    '/',expand=True).values[:,1]
erp_recog_mean['setsize'] = erp_recog_mean['setsize'].astype('int')
erp_recogAll_mean = erp_recog.groupby(
    ['time'])[picks+[pick_tag]].agg(np.mean).reset_index()
erp_recogSubj_mean = erp_recog.groupby(
    ['subj','time','cond','setsize'])[
    picks+[pick_tag]].agg(np.mean).reset_index()
erp_recogAllSubj_mean = erp_recog.groupby(
    ['subj','time'])[
    picks+[pick_tag]].agg(np.mean).reset_index()

'''
fname_grand = 'grand_avg_recog_ave.fif'
evk_grand_allChans = mne.combine_evoked(
    mne.read_evokeds(
        os.path.join(evkDataPath,fname_grand)),weights='nave')
evk_grand = evk_grand_allChans.pick(picks).average()
'''



# --- --- --- --- --- --- get peak time

erp_recogAll_mean['subj'] = ['mean']*len(erp_recogAll_mean)
peak_dict = get_peak_data(erp_recogAll_mean)
p_dict = dict()
for n,k in enumerate(peak_dict['cp']):
    p_dict[k] = peak_dict['time'][n]
mpl.rcParams.update({'font.size':18})
fig,ax = plt.subplots(
    1,1,sharex=True,sharey=True,figsize=(9, 6))
ax.plot(erp_recogAll_mean['time'],
        erp_recogAll_mean[pick_tag])
ax.scatter(peak_dict['time'],peak_dict['amps'])
peak_text = 'p1: %(p1)f, n1: %(n1)f, p2: %(p2)f'%p_dict
plt.text(0,-2*(1e-6),peak_text)
figName = os.path.join(set_filepath(
    grpFigPath,'FindPeaks'),
    'peaks_grp_%s.png'%pick_tag)
save_fig(fig,figName)

p1_time_list,n1_time_list,p2_time_list = [],[],[]
peak_dict_subj = dict()
# for subjN in subjList_final:
for subjN in subjList:
    print(subjN)
    subjData_mean = erp_recogAllSubj_mean[
        erp_recogAllSubj_mean['subj']==subjN].copy().reset_index(
        drop=True)
    peak_dict = get_peak_data(subjData_mean)
    p_dict = dict()
    for n,k in enumerate(peak_dict['cp']):
        p_dict[k] = peak_dict['time'][n]
    p1_time_list.append(p_dict['p1'])
    n1_time_list.append(p_dict['n1'])
    p2_time_list.append(p_dict['p2'])

    peak_dict_subj[subjN] = peak_dict

    fig,ax = plt.subplots(
        1,1,sharex=True,sharey=True,figsize=(9,6))
    ax.plot(subjData_mean['time'],
            subjData_mean[pick_tag])
    ax.scatter(peak_dict['time'],peak_dict['amps'])
    peak_text = 'p1: %(p1)f, n1: %(n1)f, p2: %(p2)f'%p_dict
    plt.text(0,-2*(1e-6),peak_text)
    figName = os.path.join(set_filepath(
        grpFigPath,'FindPeaks'),
        'peaks_indv_%s_'%pick_tag)+'%s.png'%subjN
    save_fig(fig,figName)

p1_max,p1_min = max(p1_time_list),min(p1_time_list)
n1_max,n1_min = max(n1_time_list),min(n1_time_list)
p2_max,p2_min = max(p2_time_list),min(p2_time_list)

print(p1_min,p1_max)
print(n1_min,n1_max)
print(p2_min,p2_max)

# bar plot for erp components
cp_df = pd.DataFrame()
for t0,t1,cp_tag in ((0.1,0.16,'P1'),(0.16,0.2,'N1'),(0.2,0.3,'P2')):
    erp_cp = erp_recogAll_mean.loc[
        (erp_recogAll_mean['time']>t0)&
        (erp_recogAll_mean['time']<t1),['time','simi']]
    erp_cp['cp'] = [cp_tag]*len(erp_cp)
    cp_df = pd.concat([cp_df,erp_cp],ignore_index=True,axis=0)

# clrs = ['sun yellow','lightish red','dodger blue']
clrs = sns.color_palette('GnBu_d')
mpl.rcParams.update({'font.size':18})
fig,ax = plt.subplots(figsize=(12,9))
# sns.barplot(data=cp_df,x='cp',y='simi',palette=sns.xkcd_palette(clrs))
sns.barplot(data=cp_df,x='cp',y='simi',palette=clrs)
plt.title('The Average Amplitude of P1, N1, and P2 Components')
# plt.ylim(0.4, 0.85)
plt.xlabel('ERP Component')
plt.ylabel('μV')
# plt.grid(linestyle=':')
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
# ax.legend(loc='best',ncol=3)
figName = os.path.join(allResFigPath,'simi','avg_cp_bar.png')
save_fig(fig,figName)

# barplot for the 3 components

# average ERP
fname_grand = 'grand_avg_recogAllCond_ave.fif'
evks = mne.read_evokeds(os.path.join(evkDataPath,fname_grand))

mpl.rcParams.update({'font.size':19})
fig,ax = plt.subplots(figsize=(12,9))
fig = mne.viz.plot_compare_evokeds(
    {'Averaged ERP':evks},picks=picks,truncate_xaxis=False,
    ylim=dict(eeg=[-3,8]),show=show_flg,combine='mean',
    show_sensors='upper right',title='12 Sensors',legend=False)
# plt.figure(figsize=(20,15))
fig[0].axes[0].axvline(0.1,ls='--',color='grey')
fig[0].axes[0].axvline(0.16,ls='--',color='grey')
fig[0].axes[0].axvline(0.2,ls='--',color='grey')
fig[0].axes[0].axvline(0.3,ls='--',color='grey')
p1_text = 'P1'
n1_text = 'N1'
p2_text = 'P2'
fig[0].axes[0].text(0.115,-2,p1_text,color='grey',fontsize=12,fontweight='bold')
fig[0].axes[0].text(0.16,-2,n1_text,color='grey',fontsize=12,fontweight='bold')
fig[0].axes[0].text(0.232,-2,p2_text,color='grey',fontsize=12,fontweight='bold')
fig[0].axes[0].text(-0.2,8,'1e-6')
# plt.tight_layout()
figName = os.path.join(allResFigPath,'simi','erp_avg.png')
save_fig(fig,figName)

# # --- --- --- --- --- ---
# # ±40 ms anova
# # each subject
# peak_amps,peak_cps,peak_subj = [],[],[]
# erp_cp = pd.DataFrame()
#
# # for subjN in subjList_final:
# for subjN in subjList:
#     subjData = erp_recogSubj_mean[
#         erp_recogSubj_mean['subj']==subjN]
#     peak_dict = peak_dict_subj[subjN]
#
#     for cp,t in zip(comp_list,peak_dict['time']):
#         t0 = t-t_space
#         t1 = t+t_space
#         t_win = (subjData['time']>t0)&(subjData['time']<t1)
#         df_tmp = subjData.loc[t_win,:]
#         df_tmp = df_tmp.copy()
#         df_tmp['cp'] = [cp]*len(df_tmp)
#         erp_cp = pd.concat([erp_cp,df_tmp],
#                            axis=0,ignore_index=True)
#
# erp_cp_mean = erp_cp.groupby(
#     ['subj','cond','setsize','cp'])[
#     picks+[pick_tag]].agg(np.mean).reset_index()
#
# for cp in comp_list:
#     # plot fit
#     evk_cp_avg = erp_cp_mean[(erp_cp_mean['cp']==cp)].groupby(
#         ['cond','setsize'])[pick_tag].agg(np.mean).reset_index()
#     fig = plot_fit(evk_cp_avg,pick_tag,'μV',cp.title())
#     figName = os.path.join(set_filepath(allResFigPath,'%s'%pick_tag),
#                            'fit_%s_40_%s.png'%(pick_tag,cp))
#     save_fig(fig,figName)
#
#     aov = pg.rm_anova(
#         dv=pick_tag,within=['cond','setsize'],subject='subj',
#         data=erp_cp_mean[erp_cp_mean['cp']==cp],detailed=True,
#         effsize='np2')
#     pwc = pg.pairwise_tests(
#         dv=pick_tag,within=['cond','setsize'],subject='subj',
#         data=erp_cp_mean[erp_cp_mean['cp']==cp],
#         padjust='bonf',effsize='hedges')
#     pd.set_option('display.max_columns',None)
#     print(cp)
#     print('--- --- ---')
#     pg.print_table(aov)
#     print('---')
#     # pg.print_table(pwc)
#     print('--- --- ---')


# # --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
# # time-by-time
# df_aov_t = pd.DataFrame()
# for t in times:
#     # t = round(tt,3)
#     aov = pg.rm_anova(
#         dv=pick_tag,within=['cond','setsize'],subject='subj',
#         data=erp_recog[erp_recog['time']==t],
#         detailed=True,effsize='np2')
#     aov['time'] = [t]*len(aov)
#     df_aov_t = pd.concat([df_aov_t,aov],axis=0,ignore_index=True)
# # get significant time points
# # anova
# print('--- *** --- *** --- *** ---')
# sigList = df_aov_t.loc[
#     (df_aov_t['Source']=='cond * setsize')&
#     (df_aov_t['p-GG-corr']<p_crit),'time']
# pd.set_option('display.max_columns',None)
# sig_aov = np.array(sigList)
# grp_aov_start,grp_aov_end = check_sig(sig_aov)
# print(sig_aov)
# print(grp_aov_start)
# print(grp_aov_end)
# # category
# print('--- *** --- *** --- *** ---')
# sigList = df_aov_t.loc[
#     (df_aov_t['Source']=='cond')&
#     (df_aov_t['p-GG-corr']<p_crit),'time']
# pd.set_option('display.max_columns',None)
# print('Category')
# sig_cate = np.array(sigList)
# grp_cate_start,grp_cate_end = check_sig(sig_cate)
# print(sig_cate)
# print(grp_cate_start)
# print(grp_cate_end)
# # set size
# print('--- *** --- *** --- *** ---')
# sigList = df_aov_t.loc[
#     (df_aov_t['Source']=='setsize')&
#     (df_aov_t['p-GG-corr']<p_crit),'time']
# pd.set_option('display.max_columns',None)
# print('Set Size')
# sig_size = np.array(sigList)
# grp_size_start,grp_size_end = check_sig(sig_size)
# print(sig_size)
# print(grp_size_start)
# print(grp_size_end)
# print('--- *** --- *** --- *** ---')
# # Plot
# fig_pref = os.path.join(
#     set_filepath(allResFigPath,'%s'%pick_tag),'aov_tbyt_')
# fig = plotERP(erp_recog_mean,times,grp_cate_start,grp_cate_end,
#               pick_tag,'Category Effect (%s)'%pick_tag)
# figName = fig_pref+'%s_cate.png'%pick_tag
# save_fig(fig,figName)
# fig = plotERP(erp_recog_mean,times,grp_size_start,grp_size_end,
#               pick_tag,'Set Size Effect (%s)'%pick_tag)
# figName = fig_pref+'%s_size.png'
# save_fig(fig,figName)
#
# # --- --- --- --- --- --- --- --- 10-ms bins
# t_step = 0.01
# # t_step = 0.02
# df_aov = pd.DataFrame()
# for t0 in times:
#     t1 = t0+t_step
#     if t1<=times[-1]:
#         erp_stats = erp_recog[
#             (erp_recog['time']<=t1)&
#             (erp_recog['time']>=t0)].groupby(
#             ['subj','cond','setsize'])[pick_tag].agg(np.mean).reset_index()
#         aov = pg.rm_anova(
#             dv=pick_tag,within=['cond','setsize'],subject='subj',
#             data=erp_stats,detailed=True,effsize='np2')
#         pwc = pg.pairwise_tests(
#             dv=pick_tag,within=['cond','setsize'],subject='subj',
#             data=erp_stats,padjust='bonf',effsize='hedges')
#
#         aov['time'] = [t0]*len(aov)
#         df_aov = pd.concat([df_aov,aov],axis=0,ignore_index=True)
#
# sigList = df_aov.loc[(df_aov['Source']=='cond * setsize')&
#                      (df_aov['p-unc']<p_crit),'time']
# print(sigList)
# sigList = df_aov.loc[(df_aov['Source']=='cond * setsize')&
#                      (df_aov['p-GG-corr']<p_crit),'time']
# print(sigList)

# --- --- --- --- --- --- --- --- --- t0-t1 ms
for t0,t1,cp_tag in (
        (0.1,0.3,'all3'),
        (0.1,0.16,'p1'),
        (0.16,0.2,'n1'),
        (0.2,0.3,'p2')):
    print('***')
    print('**')
    print('*')
    print(cp_tag)
    erp_stats = erp_recog[
        (erp_recog['time']<t1)&
        (erp_recog['time']>=t0)].groupby(
        ['subj','cond','setsize'])[pick_tag].agg(np.mean).reset_index()
    erp_stats_both = erp_recog[
        (erp_recog['time']<t1)&
        (erp_recog['time']>=t0)].groupby(
        ['subj','setsize'])[pick_tag].agg(np.mean).reset_index()

    erp_stats_cate = erp_recog[
        (erp_recog['time']<t1) &
        (erp_recog['time']>=t0)].groupby(
        ['subj','cond'])[pick_tag].agg(np.mean).reset_index()

    for cond in cond_label_list:
        wld_corr = pg.corr(erp_stats_cate.loc[erp_stats_cate['cond']==cond,pick_tag],
                           df_sft_cate.loc[(df_sft_cate['cond']=='w'),'zscore'],
                           method='pearson')
        print(cond)
        print('--- --- ---')
        pg.print_table(wld_corr)

    for setsize in sizeList:
        for cond in cond_label_list:
            wld_corr = pg.corr(erp_stats.loc[(erp_stats['cond']==cond)&
                                             (erp_stats['setsize']==setsize),pick_tag],
                               df_sft_cate.loc[(df_sft_cate['cond']=='w'),'zscore'],
                               method='pearson')
            print(setsize,cond)
            print('--- --- ---')
            pg.print_table(wld_corr)


    # log-linear fit
    lm_aic,log_aic = {'w':[],'b':[]},\
                     {'w':[],'b':[]}
    for n in subjList:

        for cond in cond_label_list:
            df_train = erp_stats[
                (erp_stats['subj']==n)&
                (erp_stats['cond']==cond)&
                (erp_stats['setsize']!=sizeList[-1])].copy(
                deep=True).reset_index(drop=True)
            df_test = erp_stats[
                (erp_stats['subj']==n)&
                (erp_stats['cond']==cond)].copy(
                deep=True).reset_index(drop=True)

            # linear
            x = df_train['setsize'].values
            x = x.astype('int')
            y = list(df_train[pick_tag].values)
            model = sm.OLS(y,sm.add_constant(x)).fit()
            pred_value = df_test['setsize'].values
            pred_res = model.predict(sm.add_constant(pred_value))

            erp_stats = erp_stats.copy()
            erp_stats.loc[
                (erp_stats['subj']==n)&
                (erp_stats['cond']==cond),'lm'] = pred_res
            lm_aic[cond].append(model.aic)

            # log2
            x = df_train['setsize'].apply(np.log2).values
            x = x.astype('int')
            y = list(df_train[pick_tag].values)
            model = sm.OLS(y,sm.add_constant(x)).fit()
            pred_value = df_test['setsize'].apply(np.log2).values
            pred_res = model.predict(sm.add_constant(pred_value))

            erp_stats.loc[
                (erp_stats['subj']==n)&
                (erp_stats['cond']==cond),'log'] = pred_res
            log_aic[cond].append(model.aic)

    for cond in cond_label_list:
        print('*** *** *** *** *** ***')
        print('test AIC:')
        print(cond)
        t_val = pg.ttest(lm_aic[cond],log_aic[cond],paired=False,
                         alternative='two-sided',
                         correction='auto')
        pg.print_table(t_val)
        print('Linear AIC: ',np.mean(lm_aic[cond]))
        print('Log AIC: ',np.mean(log_aic[cond]))
        print('')
    for cond in cond_label_list:
        print('*** *** *** *** *** ***')
        print('test observed data vs log vs linear')
        t_val = pg.ttest(erp_stats.loc[(erp_stats['setsize']==sizeList[-1])&
                                       (erp_stats['cond']==cond),'lm'],
                         erp_stats.loc[(erp_stats['setsize']==sizeList[-1])&
                                       (erp_stats['cond']==cond),pick_tag],
                         paired=False,alternative='two-sided',correction='auto')

        print(cond)
        print('observed data vs linear')
        pg.print_table(t_val)

        t_val = pg.ttest(erp_stats.loc[(erp_stats['setsize']==sizeList[-1])&
                                       (erp_stats['cond']==cond),'log'],
                         erp_stats.loc[(erp_stats['setsize']==sizeList[-1])&
                                       (erp_stats['cond']==cond),pick_tag],
                         paired=False,alternative='two-sided',correction='auto')
        print('*** *** *** *** *** ***')
        print(cond)
        print('observed data vs log2')
        pg.print_table(t_val)
        print('*** *** *** *** *** ***')
        print(cond)
        print('observed data vs log2 vs linear')
        print(erp_stats.loc[(erp_stats['setsize']==sizeList[-1])&
                            (erp_stats['cond']==cond),pick_tag].mean(),
              erp_stats.loc[(erp_stats['setsize']==sizeList[-1])&
                            (erp_stats['cond']==cond),'log'].mean(),
              erp_stats.loc[(erp_stats['setsize']==sizeList[-1])&
                            (erp_stats['cond']==cond),'lm'].mean())
        print('')


    # bar plot
    colStyle = ['sandybrown','darkseagreen']
    fig,ax = plt.subplots(figsize=(9,6))
    mean_bar = sns.boxplot(
        x='setsize',y=pick_tag,data=erp_stats,
        hue='cond',hue_order=cond_label_list,
        palette=colStyle)
    statannot.add_stat_annotation(
        mean_bar,data=erp_stats,x='setsize',
        y=pick_tag,hue='cond',
        hue_order=cond_label_list,
        box_pairs=[((1,'w'),(1,'b')),
                   ((2,'w'),(2,'b')),
                   ((4,'w'),(4,'b')),
                   ((8,'w'),(8,'b'))],
        test='t-test_paired',text_format='star',
        loc='inside',verbose=2)
    plt.xlabel('Memory Set Size')
    plt.ylabel('Amplitude (μV)')
    # plt.grid(linestyle='--')
    ax.legend(loc='best',ncol=2)
    title = cp_tag.upper()
    plt.title('%s'%title)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    figName = os.path.join(
        set_filepath(allResFigPath,'%s'%pick_tag),
        'fit_bar_%s_%s.png'%(pick_tag,cp_tag))
    save_fig(fig,figName)

    if cp_tag=='n1':
        # bar plot
        colStyle = ['sandybrown','darkseagreen']
        fig,ax = plt.subplots(figsize=(9,9))
        mean_bar = sns.boxplot(
            x='cond',y=pick_tag,data=erp_stats,
            order=cond_label_list,
            palette=colStyle)
        statannot.add_stat_annotation(
            mean_bar,data=erp_stats,x='cond',
            y=pick_tag,
            order=cond_label_list,
            box_pairs=[('w','b')],
            test='t-test_paired',text_format='star',
            loc='inside',verbose=2)
        plt.xlabel('Memory Set Size')
        plt.ylabel('Amplitude (μV)')
        # plt.grid(linestyle='--')
        ax.legend(loc='best',ncol=2)
        title = cp_tag.upper()
        plt.title('%s'%title)
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        figName = os.path.join(
            set_filepath(allResFigPath,'%s'%pick_tag),
            'fit_bar_main_cate_%s_%s.png'%(pick_tag,cp_tag))
        save_fig(fig,figName)
    elif cp_tag=='p2':
        # bar plot
        colStyle = ['sandybrown','darkseagreen']
        fig,ax = plt.subplots(figsize=(9,9))
        mean_bar = sns.boxplot(
            x='cond',y=pick_tag,data=erp_stats,
            order=cond_label_list,
            palette=colStyle)
        statannot.add_stat_annotation(
            mean_bar,data=erp_stats,x='cond',
            y=pick_tag,
            order=cond_label_list,
            box_pairs=[('w','b')],
            test='t-test_paired',text_format='star',
            loc='inside',verbose=2)
        plt.xlabel('Memory Set Size')
        plt.ylabel('Amplitude (μV)')
        # plt.grid(linestyle='--')
        ax.legend(loc='best',ncol=2)
        title = cp_tag.upper()
        plt.title('%s'%title)
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        figName = os.path.join(
            set_filepath(allResFigPath,'%s'%pick_tag),
            'fit_bar_main_cate_%s_%s.png'%(pick_tag,cp_tag))
        save_fig(fig,figName)

        # bar plot
        colStyle = ['tomato','gold','yellowgreen','cornflowerblue']
        fig,ax = plt.subplots(figsize=(9,6))
        mean_bar = sns.boxplot(
            x='setsize',y=pick_tag,data=erp_stats,
            order=sizeList,
            palette=colStyle)
        statannot.add_stat_annotation(
            mean_bar,data=erp_stats,x='setsize',
            y=pick_tag,
            order=sizeList,
            box_pairs=[(4,8),(2,8),(1,8)],
            test='t-test_paired',text_format='star',
            loc='inside',verbose=2)
        plt.xlabel('Memory Set Size')
        plt.ylabel('Amplitude (μV)')
        # plt.grid(linestyle='--')
        ax.legend(loc='best',ncol=2)
        title = cp_tag.upper()
        plt.title('%s'%title)
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        figName = os.path.join(
            set_filepath(allResFigPath,'%s'%pick_tag),
            'fit_bar_main_size_%s_%s.png'%(pick_tag,cp_tag))
        save_fig(fig,figName)

    # fit
    fig = plot_fit(
        erp_stats.groupby(
            ['cond','setsize'])[
            pick_tag].agg(np.mean).reset_index(),
        pick_tag,'μV',cp_tag.title())
    figName = os.path.join(set_filepath(allResFigPath,'%s'%pick_tag),
                           'fit_%s_%s.png'%(pick_tag,cp_tag))
    save_fig(fig,figName)

    # 2-way repeated anova
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
    pg.print_table(aov)
    print('*** *** ***')
    pg.print_table(pwc1)
    print('*** *** ***')
    pg.print_table(pwc2)


    for n in sizeList:
        t_val = pg.ttest(
            erp_stats.loc[(erp_stats['setsize']==n)&(erp_stats['cond']=='w'),'simi'],
            erp_stats.loc[(erp_stats['setsize']==n)&(erp_stats['cond']=='b'),'simi'],
            paired=True,alternative='two-sided',correction='auto')
        pg.print_table(t_val)

    # 4*2 bar plot
    if cp_tag!='all3':
    # for t0,t1,cp_tag in ((0.1,0.16,'p1'),(0.16,0.2,'n1'),(0.2,0.3,'p2')):
        pv_dict = {'p1':[0.526,0.618,0.706,0.441],
                   'n1':[],'p2':[0.449,0.070,0.014,0.008]}
        erp_barplt_cp_allt = erp_recog[(erp_recog['time']>t0)&(erp_recog['time']<t1)]
        erp_barplt_cp = erp_barplt_cp_allt.groupby(
            ['subj','cond','setsize'])['simi'].agg(np.mean).reset_index()
        erp_barplt_cp['cp'] = [cp_tag]*len(erp_barplt_cp)

        # aov = pg.rm_anova(
        #     dv='simi',within=['cond','setsize'],subject='subj',
        #     data=erp_barplt_cp,detailed=True,effsize='np2')
        # pg.print_table(aov)

        mpl.rcParams.update({'font.size':22})
        fig,ax = plt.subplots(figsize=(12,9))
        # sns.barplot(data=cp_df,x='cp',y='simi',palette=sns.xkcd_palette(clrs))
        cp_bar = sns.barplot(data=erp_barplt_cp_allt,x='setsize',y='simi',
                             hue='cond',hue_order=['w','b'],saturation=0.75,
                             palette='Blues',errorbar='se',
                             capsize=0.1,errcolor='grey')
        # statannot.add_stat_annotation(
        #     cp_bar,data=erp_barplt_cp,x='setsize',y='simi',hue='cond',
        #     box_pairs=[((1,'w'),(1,'b')),((2,'w'),(2,'b')),
        #                ((4,'w'),(4,'b')),((8,'w'),(8,'b'))],
        #     pvalues=pv_dict[cp_tag],test=None,text_format='star',
        #     loc='inside',verbose=2)
        # for n,thisbar in enumerate(cp_bar.patches):
        #     # Set a different hatch for each bar
        #     # thisbar.set_color(colStyle[n])
        #     thisbar.set_edgecolor('white')
        plt.title('%s'%cp_tag.upper())
        # plt.ylim(0.4, 0.85)
        plt.xlabel('Memory Set Size')
        plt.ylabel('μV')
        # plt.grid(linestyle=':')
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        if cp_tag=='p1':
            h,_ = ax.get_legend_handles_labels()
            ax.legend(h,['within','between'],
                      loc='best',ncol=1).set_title('Category')
        else:
            ax.get_legend().remove()
        figName = os.path.join(allResFigPath,'simi','cp_bar_%s.png'%cp_tag)
        save_fig(fig,figName)

        mpl.rcParams.update({'font.size':18})
        fig,ax = plt.subplots(figsize=(12,9))
        # sns.barplot(data=cp_df,x='cp',y='simi',palette=sns.xkcd_palette(clrs))
        cp_bar = sns.barplot(data=erp_barplt_cp,x='setsize',y='simi',
                             hue='cond',hue_order=['w','b'],
                             palette='Blues',errorbar='se',errcolor='grey')
        # statannot.add_stat_annotation(
        #     cp_bar,data=erp_barplt_cp,x='setsize',y='simi',hue='cond',
        #     box_pairs=[((1,'w'),(1,'b')),((2,'w'),(2,'b')),
        #                ((4,'w'),(4,'b')),((8,'w'),(8,'b'))],
        #     pvalues=pv_dict[cp_tag],test=None,text_format='star',
        #     loc='inside',verbose=2)
        # for n,thisbar in enumerate(cp_bar.patches):
        #     # Set a different hatch for each bar
        #     # thisbar.set_color(colStyle[n])
        #     thisbar.set_edgecolor('white')
        plt.title('%s'%cp_tag.upper())
        # plt.ylim(0.4, 0.85)
        plt.xlabel('Memory Set Size')
        plt.ylabel('μV')
        # plt.grid(linestyle=':')
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.legend(loc='best',ncol=1)
        figName = os.path.join(allResFigPath,'simi','cp_bar_sd_%s.png'%cp_tag)
        save_fig(fig,figName)


    erp_coeff = getCoeff(
        erp_stats,'setsize',pick_tag)
    dfCoeffNew = pd.DataFrame(columns=['subj','cond','coeff','r2'])
    count = 0
    lm_aic,log_aic = [],[]
    # for subjN in subjList_final:
    for subjN in subjList:
        dfCoeff = pd.DataFrame()
        dfCoeff.loc[count,'subj'] = subjN
        dfCoeff.loc[count,'cond'] = 'both'
        x = erp_stats_both.loc[(erp_stats_both['subj']==subjN),'setsize'].values
        x = x.astype('int')
        x = np.log2(x)
        y = erp_stats_both.loc[(erp_stats_both['subj']==subjN),pick_tag].values
        model = sm.OLS(y,sm.add_constant(x)).fit()
        dfCoeff.loc[dfCoeff['subj']==subjN,'coeff'] = model.params[1]
        dfCoeff.loc[dfCoeff['subj']==subjN,'r2'] = model.rsquared_adj
        dfCoeffNew = pd.concat([dfCoeffNew,dfCoeff],axis=0,ignore_index=True)
        count += 1

        # fit
        df_train = erp_stats_both[
            (erp_stats_both['subj']==subjN)&
            (erp_stats_both['setsize']!=sizeList[-1])].copy(
            deep=True).reset_index(drop=True)
        df_test = erp_stats_both[
            (erp_stats_both['subj']==subjN)].copy(
            deep=True).reset_index(drop=True)

        # linear
        x = df_train['setsize'].values
        x = x.astype('int')
        y = list(df_train[pick_tag].values)
        model = sm.OLS(y,sm.add_constant(x)).fit()
        pred_value = df_test['setsize'].values
        pred_res = model.predict(sm.add_constant(pred_value))

        erp_stats_both = erp_stats_both.copy()
        erp_stats_both.loc[
            (erp_stats_both['subj']==subjN),'lm'] = pred_res
        lm_aic.append(model.aic)

        # log2
        x = df_train['setsize'].apply(np.log2).values
        x = x.astype('int')
        y = list(df_train[pick_tag].values)
        model = sm.OLS(y,sm.add_constant(x)).fit()
        pred_value = df_test['setsize'].apply(np.log2).values
        pred_res = model.predict(sm.add_constant(pred_value))
        '''
        model.fit(x.reshape(-1,1),y)
        pred_value = df_test['setsize'].apply(np.log2).values
        pred_res = model.predict(pred_value.reshape(-1,1))
        '''
        erp_stats_both.loc[
            (erp_stats_both['subj']==subjN),'log'] = pred_res
        log_aic.append(model.aic)

    dfCoeffNew.index = range(len(dfCoeffNew))
    erp_coeff = pd.concat([erp_coeff,dfCoeffNew],axis=0,ignore_index=True)

    print('fit: use set size 1-4 to predict set size 8 (across category)')
    print('%s: test observed data vs log vs linear'%cp_tag)
    t_val = pg.ttest(
        erp_stats_both.loc[(erp_stats_both['setsize']==sizeList[-1]),'lm'],
        erp_stats_both.loc[(erp_stats_both['setsize']==sizeList[-1]),pick_tag],
        paired=False,alternative='two-sided',correction='auto')
    print('observed data vs linear')
    pg.print_table(t_val)
    t_val = pg.ttest(
        erp_stats_both.loc[(erp_stats_both['setsize']==sizeList[-1]),'log'],
        erp_stats_both.loc[(erp_stats_both['setsize']==sizeList[-1]),pick_tag],
        paired=False,alternative='two-sided',correction='auto')
    print('observed data vs log')
    pg.print_table(t_val)
    print('observed data vs log2 vs linear')
    print(erp_stats_both.loc[(erp_stats_both['setsize']==sizeList[-1]),pick_tag].mean(),
          erp_stats_both.loc[(erp_stats_both['setsize']==sizeList[-1]),'log'].mean(),
          erp_stats_both.loc[(erp_stats_both['setsize']==sizeList[-1]),'lm'].mean())

    print('correlate ERP with behavioural data--- --- --- --- --- --- ***')
    # category
    behav_cate = df_sch_mean.groupby(
        ['subj','cond'])['rt'].agg(np.mean).reset_index()
    erp_cate = erp_stats.groupby(
        ['subj','cond'])['simi'].agg(np.mean).reset_index()
    diff_rt = behav_cate.loc[behav_cate['cond']=='w','rt'].values-\
              behav_cate.loc[behav_cate['cond']=='b','rt'].values
    diff_cp = erp_cate.loc[erp_cate['cond']=='w','simi'].values-\
              erp_cate.loc[erp_cate['cond']=='b','simi'].values
    cate_corr=pg.corr(diff_rt,diff_cp,method='pearson')
    print('pearson')
    pg.print_table(cate_corr)
    cate_corr = pg.corr(diff_rt,diff_cp,method='spearman')
    print('spearman')
    pg.print_table(cate_corr)


    # get behavioural coeffcients
    behav_coeff = getCoeff(df_sch_mean,'setsize','rt')
    df_sch_mean_both = df_sch_mean.groupby(
        ['subj','setsize'])['rt'].agg(np.mean).reset_index()

    dfCoeffNew = pd.DataFrame(columns=['subj','cond','coeff','r2'])
    count = 0
    # for n in subjList_final:
    for n in subjList:
        dfCoeff = pd.DataFrame()
        dfCoeff.loc[count,'subj'] = n
        dfCoeff.loc[count,'cond'] = 'both'
        x = df_sch_mean_both.loc[(df_sch_mean_both['subj']==n),'setsize'].values
        x = x.astype('int')
        x = np.log2(x)
        y = df_sch_mean_both.loc[(df_sch_mean_both['subj']==n),'rt'].values
        model = sm.OLS(y,sm.add_constant(x)).fit()
        dfCoeff.loc[dfCoeff['subj']==n,'coeff'] = model.params[1]
        dfCoeff.loc[dfCoeff['subj']==n,'r2'] = model.rsquared_adj
        dfCoeffNew = pd.concat([dfCoeffNew,dfCoeff],axis=0,ignore_index=True)
        count += 1
    dfCoeffNew.index = range(len(dfCoeffNew))
    behav_coeff = pd.concat([behav_coeff,dfCoeffNew],axis=0,ignore_index=True)

    # coefficients correlation
    print('average the 2 category conditions')
    coeff_corr_df = pd.DataFrame()
    erp_both = erp_coeff[
        erp_coeff['cond']=='both'].reset_index(drop=True)
    behav_both = behav_coeff[
        behav_coeff['cond']=='both'].reset_index(drop=True)
    coeff_corr = pg.corr(erp_both['coeff'],
                         behav_both['coeff'],method='pearson')
    coeff_corr['method'] = ['pearson']*len(coeff_corr)
    coeff_corr_df = pd.concat([coeff_corr_df,coeff_corr],
                              axis=0,ignore_index=True)
    coeff_corr = pg.corr(erp_both['coeff'],
                         behav_both['coeff'],method='spearman')
    coeff_corr['method'] = ['spearman']*len(coeff_corr)
    coeff_corr_df = pd.concat([coeff_corr_df,coeff_corr],
                              axis=0,ignore_index=True)
    pg.print_table(coeff_corr_df)

    corr_data = pd.DataFrame()
    corr_data['erp'] = erp_both['coeff'].values
    corr_data['behav'] = behav_both['coeff'].values
    fig = sns.lmplot(x='behav',y='erp',data=corr_data)
    fig.fig.set_size_inches(9,6)
    plt.title('Slope Coefficients of Behavioural data vs ERP data')
    fig.tight_layout()
    figName = os.path.join(
        set_filepath(allResFigPath,'%s'%pick_tag),
        'corr_%s_%s'%(pick_tag,cp_tag))
    save_fig(fig,figName)

    # minus
    print('b - w')
    minus_corr_df = pd.DataFrame()
    erp_minus = erp_coeff.loc[
                    erp_coeff['cond']=='b','coeff'].values-erp_coeff.loc[
                    erp_coeff['cond']=='w','coeff'].values
    behav_minus = behav_coeff.loc[
                      behav_coeff['cond']=='b','coeff'].values-behav_coeff.loc[
                      behav_coeff['cond']=='w','coeff'].values
    minus_corr = pg.corr(erp_minus,behav_minus,method='pearson')
    minus_corr['method'] = ['pearson']*len(minus_corr)
    minus_corr_df = pd.concat([minus_corr_df,minus_corr],
                              axis=0,ignore_index=True)
    minus_corr = pg.corr(erp_minus,behav_minus,method='spearman')
    minus_corr['method'] = ['spearman']*len(minus_corr)
    minus_corr_df = pd.concat([minus_corr_df,minus_corr],
                              axis=0,ignore_index=True)
    pg.print_table(minus_corr_df)

    corr_data = pd.DataFrame()
    corr_data['erp'] = erp_minus
    corr_data['behav'] = behav_minus
    fig = sns.lmplot(x='behav',y='erp',data=corr_data)
    fig.fig.set_size_inches(9,6)
    plt.title('Slope Coefficients of Behavioural data vs ERP data')
    fig.tight_layout()
    figName = os.path.join(
        set_filepath(allResFigPath,'%s'%pick_tag),
        'corr_minus_%s_%s'%(pick_tag,cp_tag))
    save_fig(fig,figName)

    df_corr = pd.DataFrame()
    for cond in cond_label_list:
        behav_val = behav_coeff.loc[
            behav_coeff['cond']==cond,'coeff'].values
        erp_val = erp_coeff.loc[
            (erp_coeff['cond']==cond),'coeff'].values
        coeff_corr = pg.corr(erp_val,behav_val,
                             method='spearman')
        coeff_corr['cond'] = [cond]*len(coeff_corr)
        coeff_corr['method'] = ['spearman']*len(coeff_corr)
        df_corr = pd.concat([df_corr,coeff_corr],
                            axis=0,ignore_index=True)
        coeff_corr = pg.corr(erp_val,behav_val,
                             method='pearson')
        coeff_corr['cond'] = [cond]*len(coeff_corr)
        coeff_corr['method'] = ['pearson']*len(coeff_corr)
        df_corr = pd.concat([df_corr,coeff_corr],
                            axis=0,ignore_index=True)
    pg.print_table(df_corr)
    # coefficients paired t
    t_val = pg.ttest(
        erp_coeff.loc[(erp_coeff['cond']=='w'),'coeff'],
        erp_coeff.loc[(erp_coeff['cond']=='b'),'coeff'],
        paired=True,correction=True)
    print('--- --- ---')
    pg.print_table(t_val)

# cluster-based anova
t0_clu,t1_clu = 0.1,0.3
erp_recog_t = erp_recog[(erp_recog['time']>=t0_clu)&
                        (erp_recog['time']<t1_clu)].reset_index(drop=True)
# erp_recog_t = erp_recog
t = erp_recog_t.loc[(erp_recog_t['subj']==1)&
                    (erp_recog_t['type']=='w/1'),
                    'time'].values

# plot
# interaction
# erp_arr = np.zeros(
#         [subjAllN_final,len(recog_labels),len(times)])

# erp_arr = np.zeros(
#         [subjAllN_final,len(recog_labels),len(t)])
# for n in range(subjAllN_final):
erp_arr = np.zeros(
    [subjAllN,len(recog_labels),len(t)])
for n in range(subjAllN):
    # subjN = subjList_final[n]
    subjN = subjList[n]
    for k,cond in enumerate(recog_labels):
        erp_arr[n,k,:] = erp_recog_t.loc[
            (erp_recog_t['subj']==subjN)&
            (erp_recog_t['type']==cond),
            pick_tag].values
tail = 0
# pthresh = 0.001
factor_levels = [2,4]
effects = 'A:B'
# f_thresh = f_threshold_mway_rm(
#     subjAllN_final,factor_levels,effects,p_crit)
f_thresh = f_threshold_mway_rm(
    subjAllN,factor_levels,effects,p_crit)
f_obs,clu,clu_p,h0 = permutation_cluster_test(
    erp_arr.transpose(1,0,2),
    stat_fun=stat_fun,threshold=f_thresh,
    tail=tail,n_jobs=None,n_permutations=n_permutations,
    buffer_size=None,out_type='mask')
print(clu)
print(clu_p)
grp_start,grp_end = [],[]
for c,p in zip(clu,clu_p):
    if p < p_crit:
        grp_start.append(t.tolist()[c[0]][0])
        grp_end.append(t.tolist()[c[0]][-2])
print(grp_start,grp_end)

# set size
# erp_arr = np.zeros(
#         [subjAllN_final,len(recog_labels),len(times)])

# erp_arr = np.zeros(
#         [subjAllN_final,len(sizeList),len(t)])
# for n in range(subjAllN_final):
erp_arr = np.zeros(
        [subjAllN,len(sizeList),len(t)])
for n in range(subjAllN):
    # subjN = subjList_final[n]
    subjN = subjList[n]
    for k,cond in enumerate(sizeList):
        erp_arr[n,k,:] = erp_recog_t[
            (erp_recog_t['subj']==subjN)&
            (erp_recog_t['setsize']==cond)
        ].groupby(['time'])[pick_tag].agg(np.mean).values
tail = 0
# pthresh = 0.001
factor_levels = [4,1]
effects = 'A'
# f_thresh = f_threshold_mway_rm(
#     subjAllN_final,factor_levels,effects,p_crit)
f_thresh = f_threshold_mway_rm(
    subjAllN,factor_levels,effects,p_crit)
f_obs,clu,clu_p,h0 = permutation_cluster_test(
    erp_arr.transpose(1,0,2),
    stat_fun=stat_fun_1way,threshold=f_thresh,
    tail=tail,n_jobs=None,n_permutations=n_permutations,
    buffer_size=None,out_type='mask')
print(clu)
print(clu_p)
grp_start_size,grp_end_size = [],[]
for c,p in zip(clu,clu_p):
    if p < p_crit:
        grp_start_size.append(t.tolist()[c[0]][0])
        grp_end_size.append(t.tolist()[c[0]][-2])
print(grp_start_size,grp_end_size)

fig = plotERP(erp_recog_mean,times,grp_start_size,grp_end_size,
              pick_tag,'Set Size Effect (%s)'%pick_tag)
fig_pref = os.path.join(set_filepath(allResFigPath,'%s'%pick_tag),'aov_clu_')
figName = fig_pref+'%s_size.png'%pick_tag
save_fig(fig,figName)

# category
# erp_arr = np.zeros(
#         [subjAllN_final,len(cond_label_list),len(times)])

# erp_arr = np.zeros(
#         [subjAllN_final,len(cond_label_list),len(t)])
# for n in range(subjAllN_final):
erp_arr = np.zeros(
        [subjAllN,len(cond_label_list),len(t)])
for n in range(subjAllN):
    # subjN = subjList_final[n]
    subjN = subjList[n]
    for k,cond in enumerate(cond_label_list):
        erp_arr[n,k,:] = erp_recog_t[
            (erp_recog_t['subj']==subjN)&
            (erp_recog_t['cond']==cond)
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
        grp_start_cate.append(t.tolist()[c[0]][0])
        grp_end_cate.append(t.tolist()[c[0]][-2])
print(grp_start_cate,grp_end_cate)

fig = plotERP(erp_recog_mean,times,grp_start_cate,grp_end_cate,
              pick_tag,'Category Effect (%s)'%pick_tag)
figName = fig_pref+'%s_cate.png'%pick_tag
save_fig(fig,figName)

# plot comparison
# mpl.style.use('default')

# from matplotlib import cm
# map_vir = cm.get_cmap(name='viridis')
# clrs = map_vir(sizeList)
# scale = 1e6
scale = 1
clrs_all_b = sns.color_palette('Blues',n_colors=35)
clrs_all = sns.color_palette('GnBu_d')
clrs = [clrs_all_b[28],clrs_all_b[20],clrs_all[2],clrs_all[0]]

# clrs = [clrs_all[28],clrs_all[24],clrs_all[20],clrs_all[16]]
# clrs = sns.color_palette('Blues_r',2)+sns.color_palette('Oranges',2)
# clrs_all = sns.color_palette('GnBu_d')
# clrs = [clrs_all[3],clrs_all[2],clrs_all[1],clrs_all[0]]
clrList = clrs*2
lineStyList = ['-']*4+['--']*4
x = times
mpl.rcParams.update({'font.size':18})
fig,ax = plt.subplots(1,figsize=(20,12))
for cond,clr,lineSty in zip(recog_labels,clrList,
                            lineStyList):
    print('*')
    print('* %s' % (cond))
    y = erp_recog_mean.loc[erp_recog_mean['type']==cond,pick_tag]
    ax.plot(x,y*scale,color=clr,linestyle=lineSty,
            linewidth=2.5,label=cond,alpha=1)
ymin,ymax = ax.get_ylim()

cate_sig,size_sig = [],[]
text_x = [0.03,0.13]
text_y = [-0.000001,-0.0000015]
text_sig_start = grp_start_cate+grp_start_size
text_sig_end = grp_end_cate+grp_end_size
count = 0
for sig_tag,sig_list in zip(
        ['category','set size'],[cate_sig,size_sig]):
    if sig_tag=='category':
        sig_start,sig_end = grp_start_cate,grp_end_cate
        y_loc = -0.0000002
        sig_clr = 'black'
    else:
        sig_start,sig_end = grp_start_size,grp_end_size
        y_loc = -0.0000004
        sig_clr = 'grey'
    for n in range(len(sig_start)):
        x_sig = np.arange(sig_start[n],sig_end[n],0.004)
        ax.plot(x_sig,[y_loc*scale]*len(x_sig),color=sig_clr,
                linewidth=3,label=sig_tag,alpha=0.5)
        # plt.axhline(y=y_loc,xmin=sig_start,
        #             xmax=sig_end,color=sig_clr,lw=4.5)
    plt.text(text_x[count],text_y[count],
             '%s effect: %.3f-%.3f'%(
                 sig_tag,text_sig_start[count],text_sig_end[count]),
             color=sig_clr,fontsize=15)
    count += 1

# ax.grid(True)
ax.set_xlim(xmin=tmin,xmax=tmax)
ax.set_title('Main Effects (%s)'%simi_chans)
ax.set_xlabel(xlabel='Time (sec)')
ax.set_ylabel(ylabel='μV')
x_major_locator = MultipleLocator(0.1)
ax.xaxis.set_major_locator(x_major_locator)
ax.axvline(0, ls='--',color='k')
ax.axhline(0, ls='--',color='k')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.legend(loc='best',ncol=2,fontsize=15)
# plt.grid(linestyle=':')
fig.tight_layout()
fig_pref = os.path.join(
    set_filepath(allResFigPath,'%s'%pick_tag),'aov_clu_')
figName = fig_pref+'%s_main.png'%pick_tag
save_fig(fig,figName)

# main effects
x = times
mpl.rcParams.update({'font.size':26})
for k in ['cond','setsize']:
    plt_erp_mean = erp_recog_mean.groupby(
        ['time',k])[pick_tag].agg(np.mean).reset_index()
    if k=='cond':
        eff_tag = 'Category'
        eff_ord = ['w','b']
        eff_label = ['within','between']
        plt_erp_mean.rename(columns={'cond':eff_tag},inplace=True)
        sig_start,sig_end = grp_start_cate[0],grp_end_cate[0]
    else:
        eff_tag = 'Memory Set Size'
        eff_ord = [1,2,4,8]
        eff_label = ['size 1','size 2','size 4','size 8']
        plt_erp_mean.rename(columns={'setsize':eff_tag},inplace=True)
        sig_start,sig_end = grp_start_size[0],grp_end_size[0]

    fig,ax = plt.subplots(1,figsize=(20,12))
    if k=='cond':
        sns.lineplot(data=plt_erp_mean,x='time',y=pick_tag,
                     hue=eff_tag,hue_order=eff_ord,
                     style=eff_tag,style_order=eff_ord,
                     palette='Blues',lw=4.5,ax=ax)
    else:
        sns.lineplot(data=plt_erp_mean,x='time',y=pick_tag,
                     hue=eff_tag,hue_order=eff_ord,
                     palette=clrs,lw=4.5,ax=ax)
    ymin,ymax = ax.get_ylim()
    ax.fill_between(
        times,ymin,ymax,
        where=(times>=sig_start)&(times<sig_end),
        color='grey',alpha=0.1)
    plt.text(0.3,-1e-6,'%.3f-%.3f sec'%(
        sig_start,sig_end),color='grey')
    # ax.grid(True)
    ax.set_xlim(xmin=tmin,xmax=tmax)
    ax.set_title('Main Effects (%s)'%eff_tag)
    ax.set_xlabel(xlabel='Time (sec)')
    ax.set_ylabel(ylabel='μV')
    x_major_locator = MultipleLocator(0.1)
    ax.xaxis.set_major_locator(x_major_locator)
    ax.axvline(0,ls='--',color='k')
    ax.axhline(0,ls='--',color='k')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    h,_ = ax.get_legend_handles_labels()
    ax.legend(h,eff_label,loc='best',ncol=1).set_title(eff_tag)
    # plt.grid(linestyle=':')
    fig.tight_layout()
    fig_pref = os.path.join(
        set_filepath(allResFigPath,'%s'%pick_tag),'aov_clu_')
    figName = fig_pref+'%s_main_%s.png'%(pick_tag,k)
    save_fig(fig,figName)
