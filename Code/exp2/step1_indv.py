#!/usr/bin/env python
# -*-coding:utf-8 -*-

# EXP.4 (EEG): configure
# 2023.Mar.13
# linlin.shang@donders.ru.nl



#%% --- --- --- --- --- --- libraries --- --- --- --- --- ---
from exp4_config import epoDataPath,evkDataPath,resPath,indvFigPath,\
    subjList,subjAllN,sizeList,condList,tmin,tmax,\
    show_flg,tag_savefile,save_fig
import mne
import numpy as np
import pandas as pd
import random
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
import os

from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = "all"

print('***')
print('*** 2. INDIVIDUAL LEVEL ***')
print('***')


subjList = [35]
df_epo,df_erp,df_n2pc = pd.DataFrame(),pd.DataFrame(),pd.DataFrame()
pick_chs = ['PO7','PO8']
for subj in subjList:
    fname_epo = 'subj%d_epo.fif'%subj
    fpath = os.path.join(epoDataPath,fname_epo)
    subj_epo = mne.read_epochs(os.path.join(epoDataPath,fname_epo))
    subj_evk_list = []
    mapping = {}

    print('***')
    print('2.1 SUBJECT %d STARTS'%subj)

    for setsize in sizeList:
        for cond in condList:
            tag = cond+'/'+str(setsize)
            subj_evk = subj_epo[tag].average()
            subj_evk.comment = tag
            subj_evk_list.append(subj_evk)

            mapping[tag] = subj_evk

    fname_evk = 'subj%d_ave.fif'%subj
    subj_evk_path = os.path.join(evkDataPath,fname_evk)
    if tag_savefile==1:
        mne.write_evokeds(subj_evk_path,subj_evk_list,overwrite=True)

    for cond,evk in zip(mapping.keys(),mapping.values()):
        chans = evk.ch_names
        times = evk.times

        df_erp_tmp,df_epo_tmp_all = pd.DataFrame(),pd.DataFrame()
        for chan in chans:
            df_erp_tmp[chan] = evk.get_data(picks=chan)[0]

            epo_data = subj_epo[cond].get_data(picks=chan)
            df_epo_tmp_k = pd.DataFrame()
            for k in range(epo_data.shape[0]):
                df_epo_tmp = pd.DataFrame()
                df_epo_tmp[chan] = epo_data[k,0,:]
                df_epo_tmp_k = pd.concat(
                    [df_epo_tmp_k,df_epo_tmp],axis=0,ignore_index=True)
            df_epo_tmp_all[chan] = df_epo_tmp_k[chan]

        df_erp_tmp['time'] = times
        df_erp_tmp['type'] = [cond]*len(times)
        df_erp = pd.concat([df_erp,df_erp_tmp],axis=0,ignore_index=True)

        df_epo_tmp_all['time'] = times.tolist()*epo_data.shape[0]
        df_epo_tmp_all['type'] = [cond]*len(times)*epo_data.shape[0]
        df_epo = pd.concat([df_epo,df_epo_tmp_all],axis=0,ignore_index=True)

    df_erp['subj'] = [subj]*len(df_erp)
    df_epo['subj'] = [subj]*len(df_epo)

    for setsize in sizeList:
        # wt/tw
        df_n2pc_tmp = pd.DataFrame()
        df_n2pc_tmp['contr'] = (mapping['wt/'+str(setsize)].get_data(picks='PO7')[0]+
                                mapping['tw/'+str(setsize)].get_data(picks='PO8')[0])/2
        df_n2pc_tmp['ipsi'] = (mapping['wt/'+str(setsize)].get_data(picks='PO8')[0]+
                               mapping['tw/'+str(setsize)].get_data(picks='PO7')[0])/2
        df_n2pc_tmp['n2pc'] = df_n2pc_tmp['contr'].values-df_n2pc_tmp['ipsi'].values
        df_n2pc_tmp['time'] = times
        df_n2pc_tmp['subj'] = [subj]*len(df_n2pc_tmp)
        df_n2pc_tmp['setsize'] = [setsize]*len(df_n2pc_tmp)
        df_n2pc_tmp['cond'] = ['wt']*len(df_n2pc_tmp)
        df_n2pc = pd.concat([df_n2pc,df_n2pc_tmp],axis=0,ignore_index=True)

        # bt/tb
        df_n2pc_tmp = pd.DataFrame()
        df_n2pc_tmp['contr'] = (mapping['bt/'+str(setsize)].get_data(picks='PO7')[0]+
                                mapping['tb/'+str(setsize)].get_data(picks='PO8')[0])/2
        df_n2pc_tmp['ipsi'] = (mapping['bt/'+str(setsize)].get_data(picks='PO8')[0]+
                               mapping['tb/'+str(setsize)].get_data(picks='PO7')[0])/2
        df_n2pc_tmp['n2pc'] = df_n2pc_tmp['contr'].values-df_n2pc_tmp['ipsi'].values
        df_n2pc_tmp['time'] = times
        df_n2pc_tmp['subj'] = [subj]*len(df_n2pc_tmp)
        df_n2pc_tmp['setsize'] = [setsize]*len(df_n2pc_tmp)
        df_n2pc_tmp['cond'] = ['bt']*len(df_n2pc_tmp)
        df_n2pc = pd.concat([df_n2pc,df_n2pc_tmp],axis=0,ignore_index=True)

        # wb/bw
        df_n2pc_tmp = pd.DataFrame()
        df_n2pc_tmp['contr'] = (mapping['bw/'+str(setsize)].get_data(picks='PO7')[0]+
                                mapping['wb/'+str(setsize)].get_data(picks='PO8')[0])/2
        df_n2pc_tmp['ipsi'] = (mapping['bw/'+str(setsize)].get_data(picks='PO8')[0]+
                               mapping['wb/'+str(setsize)].get_data(picks='PO7')[0])/2
        df_n2pc_tmp['n2pc'] = df_n2pc_tmp['contr'].values-df_n2pc_tmp['ipsi'].values
        df_n2pc_tmp['time'] = times
        df_n2pc_tmp['subj'] = [subj]*len(df_n2pc_tmp)
        df_n2pc_tmp['setsize'] = [setsize]*len(df_n2pc_tmp)
        df_n2pc_tmp['cond'] = ['wb']*len(df_n2pc_tmp)
        df_n2pc = pd.concat([df_n2pc,df_n2pc_tmp],axis=0,ignore_index=True)

        # ww
        h = subj_epo['ww/'+str(setsize)].get_data().shape[0]
        indx = list(range(h))
        random.shuffle(indx)
        ww_contr = subj_epo['ww/'+str(setsize)].get_data(picks=['PO7','PO8'])[indx[0:int(h/2)],:,:]
        ww_ipsi = subj_epo['ww/'+str(setsize)].get_data(picks=['PO7','PO8'])[indx[int(h/2+1):len(indx)-1],:,:]

        df_n2pc_tmp = pd.DataFrame()
        df_n2pc_tmp['contr'] = np.mean(ww_contr,axis=(0,1))
        df_n2pc_tmp['ipsi'] = np.mean(ww_ipsi,axis=(0,1))
        df_n2pc_tmp['n2pc'] = df_n2pc_tmp['contr'].values-df_n2pc_tmp['ipsi'].values
        df_n2pc_tmp['time'] = times
        df_n2pc_tmp['subj'] = [subj]*len(df_n2pc_tmp)
        df_n2pc_tmp['setsize'] = [setsize]*len(df_n2pc_tmp)
        df_n2pc_tmp['cond'] = ['ww']*len(df_n2pc_tmp)
        df_n2pc = pd.concat([df_n2pc,df_n2pc_tmp],axis=0,ignore_index=True)

        # bb
        h = subj_epo['bb/'+str(setsize)].get_data().shape[0]
        indx = list(range(h))
        random.shuffle(indx)
        ww_contr = subj_epo['bb/'+str(setsize)].get_data(picks=['PO7','PO8'])[indx[0:int(h/2)],:,:]
        ww_ipsi = subj_epo['bb/'+str(setsize)].get_data(picks=['PO7','PO8'])[indx[int(h/2+1):len(indx)-1],:,:]

        df_n2pc_tmp = pd.DataFrame()
        df_n2pc_tmp['contr'] = np.mean(ww_contr,axis=(0,1))
        df_n2pc_tmp['ipsi'] = np.mean(ww_ipsi,axis=(0,1))
        df_n2pc_tmp['n2pc'] = df_n2pc_tmp['contr'].values-df_n2pc_tmp['ipsi'].values
        df_n2pc_tmp['time'] = times
        df_n2pc_tmp['subj'] = [subj]*len(df_n2pc_tmp)
        df_n2pc_tmp['setsize'] = [setsize]*len(df_n2pc_tmp)
        df_n2pc_tmp['cond'] = ['bb']*len(df_n2pc_tmp)
        df_n2pc = pd.concat([df_n2pc,df_n2pc_tmp],axis=0,ignore_index=True)

    n2pc_cond = ['wt','bt','wb','ww','bb']

    # PLOT
    # each condition
    # plot PO7/PO8
    scale = 1
    x = times
    chans = ['contr','ipsi']
    lineStys = ['-','--']
    clrs = sns.color_palette('Paired',n_colors=32)
    mpl.rcParams.update({'font.size':16})
    fig,ax = plt.subplots(1,figsize=(12,9))
    count = 0
    for setsize in sizeList:
        for cond in n2pc_cond:
            for chan,lineSty in zip(chans,lineStys):
                y = df_n2pc.loc[(df_n2pc['subj']==subj)&
                                (df_n2pc['setsize']==setsize)&
                                (df_n2pc['cond']==cond),chan].values
                ax.plot(x,y*scale,color=clrs[count],linestyle=lineSty,
                        linewidth=2.5,label=cond+'/'+str(setsize),alpha=1)
            count += 1
    ax.grid(True)
    ax.set_xlim(xmin=tmin,xmax=tmax)
    ax.set_title('PO7/PO8')
    ax.set_xlabel(xlabel='Time (sec)')
    ax.set_ylabel(ylabel='μV')
    ax.axvline(0,ls='--',color='k')
    ax.axhline(0,ls='--',color='k')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.legend(loc='best',ncol=4,fontsize=10)
    plt.grid(linestyle=':')
    fig.tight_layout()
    figName = os.path.join(indvFigPath,'subj%d_PO78.png'%subj)
    save_fig(fig,figName)

    # plot N2pc
    scale = 1
    x = times
    clrs = sns.color_palette('Paired',n_colors=20)
    mpl.rcParams.update({'font.size':16})
    fig,ax = plt.subplots(1,figsize=(12,9))
    count = 0
    for setsize in sizeList:
        for cond in n2pc_cond:
            if cond in n2pc_cond[0:3]:
                lineSty = '-'
            else:
                lineSty = '--'
            y = df_n2pc.loc[(df_n2pc['subj']==subj)&
                            (df_n2pc['setsize']==setsize)&
                            (df_n2pc['cond']==cond),'n2pc'].values
            ax.plot(x,y*scale,color=clrs[count],linestyle=lineSty,
                    linewidth=2.5,label=cond+'/'+str(setsize),alpha=1)
            count += 1
    ax.grid(True)
    ax.set_xlim(xmin=tmin,xmax=tmax)
    ax.set_title('N2pc')
    ax.set_xlabel(xlabel='Time (sec)')
    ax.set_ylabel(ylabel='μV')
    ax.axvline(0,ls='--',color='k')
    ax.axhline(0,ls='--',color='k')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.legend(loc='best',ncol=4,fontsize=10)
    plt.grid(linestyle=':')
    fig.tight_layout()
    figName = os.path.join(indvFigPath,'subj%d_n2pc.png'%subj)
    save_fig(fig,figName)


    # PLOT
    # category
    # PO7/8
    df_n2pc_avg = df_n2pc.groupby(
        ['subj','cond','time'])[['contr','ipsi','n2pc']].agg(np.mean).reset_index()
    scale = 1
    x = times
    chans = ['contr','ipsi']
    lineStys = ['-','--']
    mpl.rcParams.update({'font.size':16})
    fig,ax = plt.subplots(1,figsize=(12,9))
    for cond in n2pc_cond:
        for chan,lineSty in zip(chans,lineStys):
            y = df_n2pc_avg.loc[(df_n2pc_avg['subj']==subj)&
                                (df_n2pc_avg['cond']==cond),chan].values
            ax.plot(x,y*scale,linestyle=lineSty,
                    linewidth=2.5,label=cond+'/'+chan,alpha=1)
    ax.grid(True)
    ax.set_xlim(xmin=tmin,xmax=tmax)
    ax.set_title('PO7/PO8')
    ax.set_xlabel(xlabel='Time (sec)')
    ax.set_ylabel(ylabel='μV')
    ax.axvline(0,ls='--',color='k')
    ax.axhline(0,ls='--',color='k')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.legend(loc='best',ncol=5,fontsize=10)
    plt.grid(linestyle=':')
    fig.tight_layout()
    figName = os.path.join(indvFigPath,'subj%d_cate_PO78.png'%subj)
    save_fig(fig,figName)
    # n2pc
    scale = 1
    x = times
    mpl.rcParams.update({'font.size':16})
    fig,ax = plt.subplots(1,figsize=(12,9))
    for cond in n2pc_cond:
        if cond in ['ww','bb']:
            lineSty = '--'
        else:
            lineSty = '-'
        y = df_n2pc_avg.loc[(df_n2pc_avg['subj']==subj)&
                            (df_n2pc_avg['cond']==cond),'n2pc'].values
        ax.plot(x,y*scale,label=cond,linestyle=lineSty,
                linewidth=2.5,alpha=1)
    ax.grid(True)
    ax.set_xlim(xmin=tmin,xmax=tmax)
    ax.set_title('N2pc')
    ax.set_xlabel(xlabel='Time (sec)')
    ax.set_ylabel(ylabel='μV')
    ax.axvline(0,ls='--',color='k')
    ax.axhline(0,ls='--',color='k')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.legend(loc='best',ncol=5,fontsize=10)
    plt.grid(linestyle=':')
    fig.tight_layout()
    figName = os.path.join(indvFigPath,'subj%d_cate_n2pc.png'%subj)
    save_fig(fig,figName)

    # PLOT
    # setsize
    # plot PO7/PO8
    df_n2pc_avg = df_n2pc.groupby(
        ['subj','setsize','time'])[['contr','ipsi','n2pc']].agg(np.mean).reset_index()
    scale = 1
    x = times
    chans = ['contr','ipsi']
    lineStys = ['-','--']
    clrs = sns.color_palette('Paired',n_colors=10)
    mpl.rcParams.update({'font.size':16})
    fig,ax = plt.subplots(1,figsize=(12,9))
    count = 0
    for cond in sizeList:
        for chan,lineSty in zip(chans,lineStys):
            y = df_n2pc_avg.loc[(df_n2pc_avg['subj']==subj)&
                                (df_n2pc_avg['setsize']==cond),chan].values
            ax.plot(x,y*scale,color=clrs[count],linestyle=lineSty,
                    linewidth=2.5,label=str(cond)+'('+chan+')',alpha=1)
            count += 1
    ax.grid(True)
    ax.set_xlim(xmin=tmin,xmax=tmax)
    ax.set_title('PO7/PO8')
    ax.set_xlabel(xlabel='Time (sec)')
    ax.set_ylabel(ylabel='μV')
    ax.axvline(0,ls='--',color='k')
    ax.axhline(0,ls='--',color='k')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.legend(loc='best',ncol=5,fontsize=10)
    plt.grid(linestyle=':')
    fig.tight_layout()
    figName = os.path.join(indvFigPath,'subj%d_size_PO78.png'%subj)
    save_fig(fig,figName)

    # n2pc
    scale = 1
    x = times
    mpl.rcParams.update({'font.size':16})
    fig,ax = plt.subplots(1,figsize=(12,9))
    for cond in sizeList:
        y = df_n2pc_avg.loc[(df_n2pc_avg['subj']==subj)&
                            (df_n2pc_avg['setsize']==cond),'n2pc'].values
        ax.plot(x,y*scale,label=cond,linestyle=lineSty,
                linewidth=2.5,alpha=1)
    ax.grid(True)
    ax.set_xlim(xmin=tmin,xmax=tmax)
    ax.set_title('N2pc')
    ax.set_xlabel(xlabel='Time (sec)')
    ax.set_ylabel(ylabel='μV')
    ax.axvline(0,ls='--',color='k')
    ax.axhline(0,ls='--',color='k')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.legend(loc='best',ncol=5,fontsize=10)
    plt.grid(linestyle=':')
    fig.tight_layout()
    figName = os.path.join(indvFigPath,'subj%d_size_n2pc.png'%subj)
    save_fig(fig,figName)


    # PLOT
    # wt-bt
    # plot PO7/PO8
    df_n2pc_avg = df_n2pc[(df_n2pc['cond']=='wt')|(df_n2pc['cond']=='bt')].groupby(
        ['subj','cond','time'])[['contr','ipsi','n2pc']].agg(np.mean).reset_index()
    scale = 1
    x = times
    chans = ['contr','ipsi']
    mpl.rcParams.update({'font.size':16})
    fig,ax = plt.subplots(1,figsize=(12,9))
    count = 0
    for cond in ['wt','bt']:
        for chan,lineSty in zip(chans,lineStys):
            y = df_n2pc_avg.loc[(df_n2pc_avg['subj']==subj)&
                                (df_n2pc_avg['cond']==cond),chan].values
            ax.plot(x,y*scale,
                    linewidth=2.5,label=cond+'('+chan+')',alpha=1)
            count += 1
    ax.grid(True)
    ax.set_xlim(xmin=tmin,xmax=tmax)
    ax.set_title('PO7/PO8')
    ax.set_xlabel(xlabel='Time (sec)')
    ax.set_ylabel(ylabel='μV')
    ax.axvline(0,ls='--',color='k')
    ax.axhline(0,ls='--',color='k')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.legend(loc='best',ncol=5,fontsize=10)
    plt.grid(linestyle=':')
    fig.tight_layout()
    figName = os.path.join(indvFigPath,'subj%d_avg_wt_bt_PO78.png'%subj)
    save_fig(fig,figName)

    # plot N2pc
    scale = 1
    x = times
    mpl.rcParams.update({'font.size':16})
    fig,ax = plt.subplots(1,figsize=(12,9))
    count = 0
    for cond in ['wt','bt']:
        y = df_n2pc_avg.loc[(df_n2pc_avg['subj']==subj)&
                            (df_n2pc_avg['cond']==cond),'n2pc'].values
        ax.plot(x,y*scale,
                linewidth=2.5,label=cond,alpha=1)
        count += 1
    ax.grid(True)
    ax.set_xlim(xmin=tmin,xmax=tmax)
    ax.set_title('N2pc')
    ax.set_xlabel(xlabel='Time (sec)')
    ax.set_ylabel(ylabel='μV')
    ax.axvline(0,ls='--',color='k')
    ax.axhline(0,ls='--',color='k')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.legend(loc='best',ncol=4,fontsize=10)
    plt.grid(linestyle=':')
    fig.tight_layout()
    figName = os.path.join(indvFigPath,'subj%d_wt_bt_n2pc.png'%subj)
    save_fig(fig,figName)
    print('Subject %d Finished'%subj)

# save to .csv file
if tag_savefile==1:
    fileName_erp = 'exp4_erp.csv'
    fileName_epo = 'exp4_epo.csv'
    fileName_n2pc = 'exp4_n2pc.csv'
    if (len(subjList)==1)&(subjList[0]==1):
        mode_tag = 'w'
        head_tag = True
    elif subjList[0]!=1:
        mode_tag = 'a'
        head_tag = False

    df_epo.to_csv(
        os.path.join(resPath,fileName_epo),
        mode=mode_tag,header=head_tag,index=False)
    df_erp.to_csv(
        os.path.join(resPath,fileName_erp),
        mode=mode_tag,header=head_tag,index=False)
    df_n2pc.to_csv(
        os.path.join(resPath,fileName_n2pc),
        mode=mode_tag,header=head_tag,index=False)