#!/usr/bin/env python
# -*-coding:utf-8 -*-

# EXP. 3 (EEG): configure
# Temporal dynamics of non-target processing vary with
# memory set size in different category conditions
# 2022.04.29
# linlin.shang@donders.ru.nl


# --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
import os
import numpy as np
import matplotlib.pyplot as plt

show_flg = True
# show_flg = False
tag_savefile = 1
# tag_savefile = 0
tag_savefig = 1
# tag_savefig = 0

# --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
def set_filepath(file_path,*path_names):
    for path_name in path_names:
        file_path = os.path.join(file_path, path_name)
    if not os.path.exists(file_path):
        os.makedirs(file_path)
    return file_path

def dict_to_arr(d):
    return np.array([item for item in d.values()])

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
# EEG parameters
event_dict_all_mk = {'respMk_recog':96,'delMk_recog':33,
                     'w/a/1':111,'w/a/2':112,'w/a/4':114,'w/a/8':118,
                     'b/a/1':211,'b/a/2':212,'b/a/4':214,'b/a/8':218,
                     'w/o/1':121,'w/o/2':122,'w/o/4':124,'w/o/8':128,
                     'b/o/1':221,'b/o/2':222,'b/o/4':224,'b/o/8':228,
                     'targ/1':91,'targ/2':92,'targ/4':94,'targ/8':98,
                     'respMk_odd':97,'delMk_odd':55,
                     'anim':1,'obj':2,
                     'num':9}
event_dict_all = {'w/a/1':111,'w/a/2':112,'w/a/4':114,'w/a/8':118,
                  'b/a/1':211,'b/a/2':212,'b/a/4':214,'b/a/8':218,
                  'w/o/1':121,'w/o/2':122,'w/o/4':124,'w/o/8':128,
                  'b/o/1':221,'b/o/2':222,'b/o/4':224,'b/o/8':228,
                  'targ/1':91,'targ/2':92,'targ/4':94,'targ/8':98,
                  'anim':1,'obj':2,
                  'num':9}
event_dict_recog = {'w/a/1':111,'w/a/2':112,'w/a/4':114,'w/a/8':118,
                    'b/a/1':211,'b/a/2':212,'b/a/4':214,'b/a/8':218,
                    'w/o/1':121,'w/o/2':122,'w/o/4':124,'w/o/8':128,
                    'b/o/1':221,'b/o/2':222,'b/o/4':224,'b/o/8':228,
                    'targ/1':91,'targ/2':92,'targ/4':94,'targ/8':98}
event_dict_distr = {'w/a/1':111,'w/a/2':112,'w/a/4':114,'w/a/8':118,
                    'b/a/1':211,'b/a/2':212,'b/a/4':214,'b/a/8':218,
                    'w/o/1':121,'w/o/2':122,'w/o/4':124,'w/o/8':128,
                    'b/o/1':221,'b/o/2':222,'b/o/4':224,'b/o/8':228}
event_dict_odd = {'anim': 1,'obj': 2,
                  'num':9}

badChanDict = {1: [],
               2: [],
               3: [],
               4: [],
               5: [],
               6: [],
               7: [],
               8: [],
               9: [],
               10: [],
               11: [],
               12: [],
               13: [],
               14: [],
               15: [],
               16: [],
               17: [],
               18: [],
               19: [],
               20: [],
               21: [],
               22: [],
               23: [],
               24: [],
               25: [],
               26: [],
               27: [],
               28: [],
               29: [],
               30: [],
               31: [],
               32: []}

reset_elect = {'VEOGUP': 'AF3','VEOGDO': 'AF4',
               'AF7': 'F3','AF3': 'VEOGUP',
               'AF4': 'VEOGDO','AF8': 'F4',
               'F7': 'F1','F5': 'FC5',
               'F3': 'AF7','F1': 'F7',
               'Fz': 'FC4','F2': 'F8',
               'F4': 'AF8','F6': 'FC6',
               'F8': 'F2','HEOGL': 'T7',
               'FT7': 'C3','FC5': 'F5',
               'FC3': 'FC2','FC1': 'C1',
               'FCz': 'C2','FC2': 'FC3',
               'FC4': 'Fz','FC6': 'F6',
               'FT8': 'C4','HEOGR': 'T8',
               'T7': 'HEOGL','C5': 'CP1',
               'C3': 'FT7','C1': 'FC1',
               'Cz': 'CPz','C2': 'FCz',
               'C4': 'FT8','C6': 'CP2',
               'T8': 'HEOGR','TP7': 'P3',
               'CP5': 'CP3','CP3': 'CP5',
               'CP1': 'C5','CPz': 'Cz',
               'CP2': 'C6','CP4': 'CP6',
               'CP6': 'CP4','TP8': 'TP10',
               'TP10': 'TP8','P7': 'P5',
               'P5': 'P7','P3': 'TP7',
               'P1': 'Pz','Pz': 'P1',
               'P2': 'P4','P4': 'P2',
               'P6': 'P8','P8': 'P6',
               'PO7': 'PO9','PO3': 'O1',
               'POz': 'Oz','PO4': 'O2',
               'PO8': 'PO10','PO9': 'PO7',
               'O1': 'PO3','Oz': 'POz',
               'O2': 'PO4','PO10': 'PO8'}

postChans = ['O1','O2','Oz',
             'P1','P2','P3','P4','P5','P6','P7','P8',
             'PO10','PO3','PO4','PO7','PO8','PO9','POz','Pz']
frontChans = ['F1','F2','F3','F4','F5','F6','F7','F8','Fz',
              'FC1','FC2','FC3','FC4','FC5','FC6','FCz',
              'FT7','FT8']
fcpChans = ['FC1','FC2','FC3','FC4','FC5','FC6','FCz',
            'C1','C2','C3','C4','C5','C6','Cz',
            'CP1','CP2','CP3','CP4','CP5','CP6','CPz']

odd_label_list = ['anim','obj']
recog_label_list = ['w1','w2','w4','w8','b1','b2','b4','b8']
recog_labels = ['w/1','w/2','w/4','w/8','b/1','b/2','b/4','b/8']
cond_label_list = ['w','b']
size_label_list = [['w/1','b/1'], ['w/2','b/2'],
                   ['w/4','b/4'], ['w/8','b/8']]
size_labels = ['1','2','4','8']

aList = [111,112,114,118,211,212,214,218]
oList = [121,122,124,128,221,222,224,228]
w1 = [111,121]
b1 = [211,221]
w2 = [112,122]
b2 = [212,222]
w4 = [114,124]
b4 = [214,224]
w8 = [118,128]
b8 = [218,228]
sizeList = [1, 2, 4, 8]
cateList = ['Animals', 'Objects']
condList = ['within','between']

bands = [(1,3.5,'Delta (1-3.5 Hz)'),
         (4,7,'Theta (4-7 Hz)'),
         (8,12,'Alpha (8-12 Hz)'),
         (13,28,'Beta (13-28 Hz)'),
         (30,40,'Gamma (30-40 Hz)')]


# --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
# set path
# filePath = 'U:\Documents\DCC\exp3'
filePath = os.path.abspath(os.path.join(os.getcwd(),'..'))
eegRootPath = set_filepath(filePath,'Data','DataEEG')
rawEEGPath = set_filepath(eegRootPath,'RawEEG')
epoDataPath = set_filepath(eegRootPath,'EpoData')
evkDataPath = set_filepath(eegRootPath,'evkData')
decoDataPath = set_filepath(eegRootPath,'decoData')
decoDataPath_full = set_filepath(eegRootPath,'decoData_full')
resPath = set_filepath(filePath,'Results')
figPath = set_filepath(resPath,'Figs')
chkFigPath = set_filepath(figPath,'step0_SensorChk')
preFigPath = set_filepath(figPath,'step1_Preproc')
decoFigPath = set_filepath(figPath,'step2_Decode')
indvFigPath = set_filepath(figPath,'step3-1_IndvLevel')
grpFigPath = set_filepath(figPath,'step3-2_GrpLevel')
glmFigPath = set_filepath(figPath,'step3-2_GrpLevel','GLM')
aovFigPath = set_filepath(figPath,'step3-2_GrpLevel','AOV')
tfrFigPath = set_filepath(figPath,'step4_TFR')
allResFigPath = set_filepath(filePath,'AllExpRes','Figs')



# --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
montageFile = os.path.join(
    eegRootPath,'ActiCAP_64Channel_3Dlayout_EOG.txt')
fileList_EEG = sorted([fileName for fileName in
                       os.listdir(rawEEGPath)
                       if '.vhdr' in fileName])
subjAllN = len(fileList_EEG)
subjList = list(range(1,subjAllN+1))
fileList_EEG = ['subj%d.vhdr'%n for n in subjList]
outliers = [22,24,27,31]
subjList_final = [n for n in subjList if n not in outliers]
subjAllN_final = len(subjList_final)
subj_n1 = [1,4,5,8,14]
subj_p2 = [n for n in subjList_final if n not in subj_n1]

resetList = np.arange(8,33)
chanN = 60
tmin,tmax,tmax_ica = -0.2,0.8,2
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
p_crit = 0.05
p_show = 0.0015
trialN = 810
t_space = 0.04
comp_list = ['p1','n1','p2']

lineSty = {}
clr = {}
for cond,line_s,clr_s in zip(recog_labels,
                             ['solid']*4+['dashed']*4,
                             ['crimson','gold','darkturquoise',
                              'dodgerblue'] * 2):
    lineSty[cond] = line_s
    clr[cond] = clr_s