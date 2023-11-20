#!/usr/bin/env python
# -*-coding:utf-8 -*-

# EXP.4 (EEG): configure
# 2023.Mar.9
# linlin.shang@donders.ru.nl


# --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

show_flg = True
tag_savefile = 1
tag_savefig = 1

# show_flg = False
# tag_savefile = 0
# tag_savefig = 0

# --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
def set_filepath(file_path,*path_names):
    for path_name in path_names:
        file_path = os.path.join(file_path, path_name)
    if not os.path.exists(file_path):
        os.makedirs(file_path)
    return file_path

def dict_to_arr(d):
    return np.array([item for item in d.values()],dtype=object)

def save_fig(fig,figName):
    plt.show(block=show_flg)

    if tag_savefig == 1:
        if isinstance(fig,list):
            if len(fig) == 1:
                fig[0].savefig(
                    figName,bbox_inches='tight',dpi=300)
            else:
                for k in range(len(fig)):
                    figName_new = figName.replace('.png',
                                                  '_%d.png'%(k+1))
                    fig[k].savefig(
                        figName_new,bbox_inches='tight',dpi=300)
        else:
            fig.savefig(figName,
                        bbox_inches='tight',
                        dpi=300)
        # plt.pause(3)
        plt.gcf().clear()
        plt.close('all')


# --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---

# set path
# filePath = 'U:\Documents\DCC\exp4'
filePath = os.path.abspath(os.path.join(os.getcwd(),'..'))
eegRootPath = set_filepath(filePath,'Data','DataEEG')
rawEEGPath = set_filepath(eegRootPath,'RawEEG')
epoDataPath = set_filepath(eegRootPath,'EpoData')
epoDataPath_all = set_filepath(eegRootPath,'EpoData_All')
evkDataPath = set_filepath(eegRootPath,'evkData')
decoDataPath = set_filepath(eegRootPath,'decoData')
decoDataPath_all = set_filepath(eegRootPath,'decoData_All')
resPath = set_filepath(filePath,'Results')
figPath = set_filepath(resPath,'Figs')
behavFigPath = set_filepath(figPath,'behav')
preFigPath = set_filepath(figPath,'step1_Preproc')
preFigPath_all = set_filepath(figPath,'step1_Preproc_All')
indvFigPath = set_filepath(figPath,'step2-1_Indv')
grpFigPath = set_filepath(figPath,'step2-2_Grp')
decoFigPath = set_filepath(figPath,'step3-Deco')
decoFigPath_all = set_filepath(figPath,'step3-Deco_All')
tfrFigPath = set_filepath(figPath,'step4_TFR')

# parameters
sizeList = [1,2,4,8]
cateList = ['Animals','Objects']
condList = ['ww','bb','wb','bw','wt','tw','bt','tb']
cond_list = ['ww','bb','wb','wt','bt']
cond_size_list = [cond+'/'+str(setsize) for cond in condList for setsize in sizeList]
targ_list = [['wt/1','tw/1'],['bt/1','tb/1'],
             ['wt/2','tw/2'],['bt/2','tb/2'],
             ['wt/4','tw/4'],['bt/4','tb/4'],
             ['wt/8','tw/8'],['bt/8','tb/8']]
targ_names = ['wt1','bt1','wt2','bt2','wt4','bt4','wt8','bt8']
targ_plt_ord = [['wt1','bt1','wb1'],['wt2','bt2','wb2'],
                ['wt4','bt4','wb4'],['wt8','bt8','wb8'],]
wb_list = [['wb/1','bw/1'],['wb/2','bw/2'],
           ['wb/4','bw/4'],['wb/8','bw/8']]
wb_names = ['wb1','wb2','wb4','wb8']
mk_a = '1'
mk_tw = '1'
mk_wt = '2'
mk_tb = '3'
mk_bt = '4'
mk_wb = '5'
mk_bw = '6'
mk_ww = '7'
mk_bb = '8'
event_dict_all = {}
l_list,r_list,sameList,diffList = [],[],[],[]
for cate_tag in cateList:
    if cate_tag=='Animals':
        cate = '/a/'
    else:
        cate = '/o/'
    for setsize in sizeList:
        if cate=='/a/':
            event_dict_all['tw'+cate+str(setsize)] = int(mk_a+mk_tw+str(setsize))
            l_list.append(int(mk_a+mk_tw+str(setsize)))
            event_dict_all['wt'+cate+str(setsize)] = int(mk_a+mk_wt+str(setsize))
            r_list.append(int(mk_a+mk_wt+str(setsize)))
            event_dict_all['tb'+cate+str(setsize)] = int(mk_a+mk_tb+str(setsize))
            l_list.append(int(mk_a+mk_tb+str(setsize)))
            event_dict_all['bt'+cate+str(setsize)] = int(mk_a+mk_bt+str(setsize))
            r_list.append(int(mk_a+mk_bt+str(setsize)))
            event_dict_all['wb'+cate+str(setsize)] = int(mk_a+mk_wb+str(setsize))
            l_list.append(int(mk_a+mk_wb+str(setsize)))
            event_dict_all['bw'+cate+str(setsize)] = int(mk_a+mk_bw+str(setsize))
            r_list.append(int(mk_a+mk_bw+str(setsize)))
            event_dict_all['ww'+cate+str(setsize)] = int(mk_a+mk_ww+str(setsize))
            sameList.append(int(mk_a+mk_ww+str(setsize)))
            event_dict_all['bb'+cate+str(setsize)] = int(mk_a+mk_bb+str(setsize))
            diffList.append(int(mk_a+mk_bb+str(setsize)))
        else:
            event_dict_all['tw'+cate+str(setsize)] = int(mk_tw+str(setsize))
            l_list.append(int(mk_tw+str(setsize)))
            event_dict_all['wt'+cate+str(setsize)] = int(mk_wt+str(setsize))
            r_list.append(int(mk_wt+str(setsize)))
            event_dict_all['tb'+cate+str(setsize)] = int(mk_tb+str(setsize))
            l_list.append(int(mk_tb+str(setsize)))
            event_dict_all['bt'+cate+str(setsize)] = int(mk_bt+str(setsize))
            r_list.append(int(mk_bt+str(setsize)))
            event_dict_all['wb'+cate+str(setsize)] = int(mk_wb+str(setsize))
            l_list.append(int(mk_wb+str(setsize)))
            event_dict_all['bw'+cate+str(setsize)] = int(mk_bw+str(setsize))
            r_list.append(int(mk_bw+str(setsize)))
            event_dict_all['ww'+cate+str(setsize)] = int(mk_ww+str(setsize))
            sameList.append(int(mk_ww+str(setsize)))
            event_dict_all['bb'+cate+str(setsize)] = int(mk_bb+str(setsize))
            diffList.append(int(mk_bb+str(setsize)))
event_dict_subCate = {}
rootPath = os.path.join(filePath,'StimList')
for fileName in os.listdir(path=rootPath):
    file = pd.read_csv(os.path.join(rootPath,fileName))
    event_dict_subCate[file.loc[0,'subCate']] = file.loc[0,'mkCate']

p_crit = 0.05
crit_rt = 0.2
crit_sd = 3
crit_acc = 0.70
blockN = 8
trialN = 120
trialN_all = int(trialN*blockN)

# eeg
montageFile = os.path.join(
    eegRootPath,'ActiCAP_64Channel_3Dlayout_EOG.txt')
fileList_EEG = sorted([fileName for fileName in
                       os.listdir(rawEEGPath)
                       if '.vhdr' in fileName])
subjAllN_raw = len(fileList_EEG)
subjList = [n for n in range(1,subjAllN_raw+1)]
outliers = [7,12,18]
subjList = [subj for subj in subjList if subj not in outliers]
subjAllN = len(subjList)
fileList_EEG = ['subj%d.vhdr'%n for n in subjList]

chanN = 60
tmin,tmax = -0.2,0.5
baseline = (tmin,0)
fmin,fmax = 0.1,40
power_line_noise_freq = np.arange(50,200,50)
sampleNum = 250

threshold_muscle = 5
# z-score
freq_muscle = [110,125]
reject_criteria = dict(eeg=200e-6)
ica_num = None
random_num = 97
scoring = 'roc_auc'
n_permutations = 1000
jobN = None
fdN = 10
chance_crit = 0.5
t_space = 0.04

bands = [(1,3.5,'Delta (1-3.5 Hz)'),
         (4,7,'Theta (4-7 Hz)'),
         (8,12,'Alpha (8-12 Hz)'),
         (13,28,'Beta (13-28 Hz)'),
         (30,40,'Gamma (30-40 Hz)')]

badChanDict = {1:[],2:[],3:[],4:[],5:[],
               6:['AF8'],7:[],8:['T8','TP7'],9:['AF4','AF8'],10:['TP10'],
               11:['TP10'],12:[],13:[],14:['TP10'],15:['AF7','Oz'],
               16:[],17:[],18:[],19:['TP10'],20:[],
               21:[],22:[],23:['TP10'],24:['TP10'],25:[],
               26:[],27:[],28:['AF3'],29:['TP10'],30:['AF7','AF8'],
               31:['F2','FT7'],32:[],33:['TP10'],34:[],35:[],
               36:[],37:[]}