#!/usr/bin/env python
#-*-coding:utf-8 -*-

from psychopy import monitors,visual,core,event,gui
from PIL import Image
from rusocsci import buttonbox
bb = buttonbox.Buttonbox(port='COM3')

import os,copy,time,random,csv
from math import atan,pi,ceil
import numpy as np
from scipy import stats
import pandas as pd



#
# parameters

# subj info
info = {'ID NO.':'','Age':'','Sex':['F','M']}
infoDlg = gui.DlgFromDict(dictionary=info,title=u'Subject Information',\
order=['ID NO.','Age','Sex'])
if infoDlg.OK == False:
    core.quit()
subj = int(info['ID NO.'])
age = info['Age']
sex = info['Sex']

# marker
mk_start = 1
mk_end = 9
mk_a = '1'

# experimental conditions
sizeList = [1,2,4,8]
cateList = ['Animals','Objects']
condList = ['within','between']
blockN = 8
trialN = 15
imgDur = 0.2
isiDurList = [1.8,2.2]*60
fixDur = 0.5
startKey = 'space'
quitKey = 'escape'
dResp = 'up'
tResp = 'down'
respKeyList = [dResp,tResp,quitKey]
testCrit = 0.8

# windows
imgDeg = 4
visuDeg = 3
fixDeg = 1.75
fontDeg = 1.35
distMon = 57
scrWidCm = 53.5
scrWidPix = 1920
scrWidDeg = (atan((scrWidCm/2.0)/distMon)/pi)*180*2
mon = monitors.Monitor('testMonitor',distance=distMon)
#scrSize = (800,600)
#win = visual.Window(monitor=mon,color=(1,1,1),size=scrSize,fullscr=False,units='deg')
scrSize = (scrWidPix,1080)
win = visual.Window(monitor=mon,color=(1,1,1),size=scrSize,fullscr=True,units='deg')
win.mouseVisible = False
timer = core.Clock()
#
# get path
filePath = os.getcwd()
stimFileDf = pd.read_csv(os.path.join(filePath,'stimList.csv'),sep=',')
# targets
animFileList = stimFileDf['anim'].tolist()
objFileList = stimFileDf['obj'].tolist()



#
# sub-functions

def sendMarkCode(mkCode):
    bb.sendMarker(val=mkCode)
    core.wait(0.002)
    bb.sendMarker(val=0) 

def checkEsc(keyName):
    if keyName==quitKey:
        core.quit()

def iti(dur_time):
    iti_fix = visual.TextStim(win,text='+',height=fixDeg,units='deg',
    pos=(0.0,0.0),color='black',bold=False,italic=False)
    iti_fix.draw()
    win.flip()
    core.wait(dur_time)

def task_start(text_par):
    start_text = visual.TextStim(win,text=text_par,height=fontDeg,units='deg',
    pos=(0.0,0.0),color='black',bold=False,italic=False)
    start_text.draw()
    win.flip()
    core.wait(0.95)

def std_task(targList):
    iti(0.95)
    for img in targList:
        # presenting stimulus
        img_pres = visual.ImageStim(win,image=img,
        pos=(0.0,0.0),size=imgDeg,units='deg')
        img_pres.draw()
        win.flip()
        core.wait(3)
        # iti
        iti(0.95)

def fbkShow(fbkText,clr):
    fbk_text = visual.TextStim(win,text=fbkText,height=fontDeg,
    units='deg',pos=(0.0,0.0),color=clr,bold=False,italic=False)
    fbk_text.draw()
    win.flip()
    core.wait(0.5)

def test_task(testNDf_round):
    iti(0.95)
    testRespList,testRTList = [],[]
    
    testNList = testNDf_round['stimulus'].tolist()
    indxN = 0
    for img in testNList:
        # presenting stimulus
        img_pres = visual.ImageStim(win,image=img,
        pos=(0.0,0.0),size=imgDeg,units='deg')
        img_pres.draw()
        t0 = win.flip()
        respKey = event.waitKeys(keyList=respKeyList,
        timeStamped=True,clearEvents=True)
        
        checkEsc(respKey[0][0])
        respKeyFirst = respKey[0][0].lower()
        respKeyRT = respKey[0][1]-t0
        testRespList.append(respKeyFirst)
        testRTList.append(respKeyRT)
        
        if respKeyFirst==testNDf_round.loc[indxN,'ans']:
            fbkText = u'Correct!'
            clr = 'green'
        else:
            testNDf_round.loc[indxN,'Correct'] = 0
            fbkText = u'Wrong!'
            clr = 'red'
        fbkShow(fbkText,clr)
        indxN += 1
        # iti
        iti(0.45)
        
    testNDf_round['RT'] = testRTList
    testNDf_round['resp'] = testRespList
    
    return testNDf_round



#
# preparation
usedSubCate_dict,usedSubCate_list,usedStim_list = {},[],[]
targDf,testDf,test1Df,test2Df,test3Df,schDf = \
pd.DataFrame(),pd.DataFrame(),pd.DataFrame(),pd.DataFrame(),pd.DataFrame(),pd.DataFrame()
cateList_blk = cateList*int(blockN/len(cateList))
sizeList_blk = [setsize for setsize in sizeList for n in range(int(blockN/len(sizeList)))]
blkOrd = pd.DataFrame({'cate':cateList_blk,'setsize':sizeList_blk})
blkOrd = blkOrd.sample(frac=1).reset_index(drop=True)

cateList_blk = blkOrd['cate'].tolist()
sizeList_blk = blkOrd['setsize'].tolist()
for blkN in range(blockN):
    blk_cate = cateList_blk[blkN]
    setsize = sizeList_blk[blkN]
    
    # preparation
    if blk_cate=='Animals':
        targFileList = animFileList
        distrFileList = objFileList
    else:
        targFileList = objFileList
        distrFileList = animFileList
    
    # get targets
    targFileList_blk = random.sample(targFileList,setsize)
    while list(set(targFileList_blk)&set(usedSubCate_list))!=[]:
        targFileList_blk = random.sample(targFileList,setsize)
    usedSubCate_list += targFileList_blk
    usedSubCate_dict[blkN] = targFileList_blk
    
    targDf_blk,test1Df_blk,test2Df_blk,test3Df_blk = \
    pd.DataFrame(),pd.DataFrame(),pd.DataFrame(),pd.DataFrame()
    for subCateFile in targFileList_blk:
        targCateDf = pd.read_csv(os.path.join(filePath,subCateFile),sep=',')
        targCateDf['blkN'] = [blkN+1]*len(targCateDf)
        targCateDf['blk_cate'] = [blk_cate]*len(targCateDf)
        targCateDf['setsize'] = [setsize]*len(targCateDf)
        
        targDf_temp = targCateDf.sample(n=4).reset_index(drop=True)
        targDf_blk = pd.concat([targDf_blk,targDf_temp.iloc[[0]]],axis=0,ignore_index=True)
        test1Df_blk = pd.concat([test1Df_blk,targDf_temp.iloc[[1]]],axis=0,ignore_index=True)
        test2Df_blk = pd.concat([test2Df_blk,targDf_temp.iloc[[2]]],axis=0,ignore_index=True)
        test3Df_blk = pd.concat([test3Df_blk,targDf_temp.iloc[[3]]],axis=0,ignore_index=True)
        
    targDf_blk['ans'] = [tResp]*len(targDf_blk)
    targDf = pd.concat([targDf,targDf_blk],axis=0,ignore_index=True)
    
    test1Df_blk['blkN'] = [blkN+1]*len(test1Df_blk)
    test1Df_blk['blk_cate'] = [blk_cate]*len(test1Df_blk)
    test1Df_blk['setsize'] = [setsize]*len(test1Df_blk)
    test1Df_blk['ans'] = [dResp]*len(test1Df_blk)
    test1Df = pd.concat([test1Df,test1Df_blk],axis=0,ignore_index=True)
    test2Df_blk['blkN'] = [blkN+1]*len(test2Df_blk)
    test2Df_blk['blk_cate'] = [blk_cate]*len(test2Df_blk)
    test2Df_blk['setsize'] = [setsize]*len(test2Df_blk)
    test2Df_blk['ans'] = [dResp]*len(test2Df_blk)
    test2Df = pd.concat([test2Df,test2Df_blk],axis=0,ignore_index=True)
    test3Df_blk['blkN'] = [blkN+1]*len(test3Df_blk)
    test3Df_blk['blk_cate'] = [blk_cate]*len(test3Df_blk)
    test3Df_blk['setsize'] = [setsize]*len(test3Df_blk)
    test3Df_blk['ans'] = [dResp]*len(test3Df_blk)
    test3Df = pd.concat([test3Df,test3Df_blk],axis=0,ignore_index=True)
    
    testDfList_blk = [test1Df_blk,test2Df_blk,test3Df_blk]
    testDf_blk = pd.concat(testDfList_blk,axis=0,ignore_index=True)
    
    targList = targDf_blk['stimulus'].tolist()
    usedStim_list += targList+testDf_blk['stimulus'].tolist()

for blkN in range(blockN):
    mk_tw = '1'
    mk_wt = '2'
    mk_tb = '3'
    mk_bt = '4'
    mk_wb = '5'
    mk_bw = '6'
    mk_ww = '7'
    mk_bb = '8'
    
    blk_cate = cateList_blk[blkN]
    setsize = sizeList_blk[blkN]
    targCate_list = usedSubCate_dict[blkN]
    
    if blk_cate=='Animals':
        targFileList = animFileList
        distrFileList = objFileList
        mk_tw = int(mk_a+mk_tw+str(setsize))
        mk_wt = int(mk_a+mk_wt+str(setsize))
        mk_tb = int(mk_a+mk_tb+str(setsize))
        mk_bt = int(mk_a+mk_bt+str(setsize))
        mk_wb = int(mk_a+mk_wb+str(setsize))
        mk_bw = int(mk_a+mk_bw+str(setsize))
        mk_ww = int(mk_a+mk_ww+str(setsize))
        mk_bb = int(mk_a+mk_bb+str(setsize))
    else:
        targFileList = objFileList
        distrFileList = animFileList
        mk_tw = int(mk_tw+str(setsize))
        mk_wt = int(mk_wt+str(setsize))
        mk_tb = int(mk_tb+str(setsize))
        mk_bt = int(mk_bt+str(setsize))
        mk_wb = int(mk_wb+str(setsize))
        mk_bw = int(mk_bw+str(setsize))
        mk_ww = int(mk_ww+str(setsize))
        mk_bb = int(mk_bb+str(setsize))
    
    distrDf_w,distrDf_b = pd.DataFrame(),pd.DataFrame()
    for subCateFile in targFileList:
        if subCateFile not in targCate_list:
            distrCateDf = pd.read_csv(os.path.join(filePath,subCateFile),sep=',')
            distrCateDf.drop(
            distrCateDf[distrCateDf['stimulus'].isin(usedStim_list)].index,
            axis=0,inplace=True)
            distrCateDf = distrCateDf.reset_index(drop=True)
            distrDf_w = pd.concat([distrDf_w,distrCateDf],axis=0,ignore_index=True)
    distrDf_w['blkN'] = [blkN+1]*len(distrDf_w)
    distrDf_w['blk_cate'] = [blk_cate]*len(distrDf_w)
    distrDf_w['setsize'] = [setsize]*len(distrDf_w)
    distrDf_w['ans'] = [dResp]*len(distrDf_w)
    for subCateFile in distrFileList:
        distrCateDf = pd.read_csv(os.path.join(filePath,subCateFile),sep=',')
        distrCateDf.drop(
        distrCateDf[distrCateDf['stimulus'].isin(usedStim_list)].index,
        axis=0,inplace=True)
        distrCateDf = distrCateDf.reset_index(drop=True)
        distrDf_b = pd.concat([distrDf_b,distrCateDf],axis=0,ignore_index=True)
    distrDf_b['blkN'] = [blkN+1]*len(distrDf_b)
    distrDf_b['blk_cate'] = [blk_cate]*len(distrDf_b)
    distrDf_b['setsize'] = [setsize]*len(distrDf_b)
    distrDf_b['ans'] = [dResp]*len(distrDf_b)
    
    distrDf_w.rename(columns={'stimulus':'left'},inplace=True)
    distrDf_w = distrDf_w.sample(frac=1).reset_index(drop=True)
    distrDf_b.rename(columns={'stimulus':'left'},inplace=True)
    distrDf_b = distrDf_b.sample(frac=1).reset_index(drop=True)
    
    targDf_blk = targDf[targDf['blkN']==blkN+1]
    targDf_blk.drop(columns=['subCate'],inplace=True)
    targDf_blk.rename(columns={'stimulus':'left'},inplace=True)
    
    schDf_tw_blk,schDf_wt_blk = pd.DataFrame(),pd.DataFrame()
    schDf_tb_blk,schDf_bt_blk = pd.DataFrame(),pd.DataFrame()
    for k in range(trialN):
        for n in range(len(targDf_blk)):
            # t+w
            sch_tw = targDf_blk.sample(n=1).reset_index(drop=True)
            sch_wt = distrDf_w.sample(n=1).reset_index(drop=True)
            sch_wt.drop(columns=['subCate'],inplace=True)
            sch_tw.loc[0,'right'] = sch_wt.loc[0,'left']
            schDf_tw_blk = pd.concat([schDf_tw_blk,sch_tw],axis=0,ignore_index=True)
            # w+t
            sch_tw = targDf_blk.sample(n=1).reset_index(drop=True)
            sch_wt = distrDf_w.sample(n=1).reset_index(drop=True)
            sch_wt.drop(columns=['subCate'],inplace=True)
            sch_wt.loc[0,'right'] = sch_tw.loc[0,'left']
            schDf_wt_blk = pd.concat([schDf_wt_blk,sch_wt],axis=0,ignore_index=True)
            # t+b
            sch_tb = targDf_blk.sample(n=1).reset_index(drop=True)
            sch_bt = distrDf_b.sample(n=1).reset_index(drop=True)
            sch_bt.drop(columns=['subCate'],inplace=True)
            sch_tb.loc[0,'right'] = sch_bt.loc[0,'left']
            schDf_tb_blk = pd.concat([schDf_tb_blk,sch_tb],axis=0,ignore_index=True)
            # b+t
            sch_tb = targDf_blk.sample(n=1).reset_index(drop=True)
            sch_bt = distrDf_b.sample(n=1).reset_index(drop=True)
            sch_bt.drop(columns=['subCate'],inplace=True)
            sch_bt.loc[0,'right'] = sch_tb.loc[0,'left']
            schDf_bt_blk = pd.concat([schDf_bt_blk,sch_bt],axis=0,ignore_index=True)
    
    schDf_tw_blk = schDf_tw_blk.sample(n=trialN).reset_index(drop=True)
    schDf_tw_blk['cond'] = ['tw']*len(schDf_tw_blk)
    schDf_tw_blk['mkCode'] = [mk_tw]*len(schDf_tw_blk)
    schDf_tw_blk['Channel1'] = [0]*len(schDf_tw_blk)
    schDf_tw_blk['Channel2'] = [1]*len(schDf_tw_blk)
    schDf_tw_blk['ans'] = [tResp]*len(schDf_tw_blk)
    schDf_wt_blk = schDf_wt_blk.sample(n=trialN).reset_index(drop=True)
    schDf_wt_blk['cond'] = ['wt']*len(schDf_wt_blk)
    schDf_wt_blk['mkCode'] = [mk_wt]*len(schDf_wt_blk)
    schDf_wt_blk['Channel1'] = [1]*len(schDf_wt_blk)
    schDf_wt_blk['Channel2'] = [0]*len(schDf_wt_blk)
    schDf_wt_blk['ans'] = [tResp]*len(schDf_wt_blk)
    schDf_tb_blk = schDf_tb_blk.sample(n=trialN).reset_index(drop=True)
    schDf_tb_blk['cond'] = ['tb']*len(schDf_tb_blk)
    schDf_tb_blk['mkCode'] = [mk_tb]*len(schDf_tb_blk)
    schDf_tb_blk['Channel1'] = [0]*len(schDf_tb_blk)
    schDf_tb_blk['Channel2'] = [2]*len(schDf_tb_blk)
    schDf_tb_blk['ans'] = [tResp]*len(schDf_tb_blk)
    schDf_bt_blk = schDf_bt_blk.sample(n=trialN).reset_index(drop=True)
    schDf_bt_blk['cond'] = ['bt']*len(schDf_bt_blk)
    schDf_bt_blk['mkCode'] = [mk_bt]*len(schDf_bt_blk)
    schDf_bt_blk['Channel1'] = [2]*len(schDf_bt_blk)
    schDf_bt_blk['Channel2'] = [0]*len(schDf_bt_blk)
    schDf_bt_blk['ans'] = [tResp]*len(schDf_bt_blk)
    schDf = pd.concat([schDf,schDf_tw_blk,schDf_wt_blk,schDf_tb_blk,schDf_bt_blk],
    axis=0,ignore_index=True)
    
    for k in range(trialN):
        # ww
        sch_ww_both = distrDf_w.sample(n=2).reset_index(drop=True)
        while sch_ww_both.loc[0,'subCate'] == sch_ww_both.loc[1,'subCate']:
            sch_ww_both = distrDf_w.sample(n=2).reset_index(drop=True)
        sch_ww_both.drop(columns=['subCate'],inplace=True)
        sch_ww = sch_ww_both.iloc[[0]]
        sch_ww.loc[0,'right'] = sch_ww_both.loc[1,'left']
        sch_ww.loc[0,'cond'] = 'ww'
        sch_ww.loc[0,'mkCode'] = mk_ww
        sch_ww.loc[0,'Channel1'] = 1
        sch_ww.loc[0,'Channel2'] = 1
        # bb
        sch_bb_both = distrDf_b.sample(n=2).reset_index(drop=True)
        while sch_bb_both.loc[0,'subCate'] == sch_bb_both.loc[1,'subCate']:
            sch_bb_both = distrDf_b.sample(n=2).reset_index(drop=True)
        sch_bb_both.drop(columns=['subCate'],inplace=True)
        sch_bb = sch_bb_both.iloc[[0]]
        sch_bb.loc[0,'right'] = sch_bb_both.loc[1,'left']
        sch_bb.loc[0,'cond'] = 'bb'
        sch_bb.loc[0,'mkCode'] = mk_bb
        sch_bb.loc[0,'Channel1'] = 2
        sch_bb.loc[0,'Channel2'] = 2
        
        # wb
        sch_wb = distrDf_w.sample(n=1).reset_index(drop=True)
        sch_wb.drop(columns=['subCate'],inplace=True)
        sch_wb_r = distrDf_b.sample(n=1).reset_index(drop=True)
        sch_wb.loc[0,'right'] = sch_wb_r.loc[0,'left']
        sch_wb.loc[0,'cond'] = 'wb'
        sch_wb.loc[0,'mkCode'] = mk_wb
        sch_wb.loc[0,'Channel1'] = 1
        sch_wb.loc[0,'Channel2'] = 2
        # bw
        sch_bw = distrDf_b.sample(n=1).reset_index(drop=True)
        sch_bw.drop(columns=['subCate'],inplace=True)
        sch_bw_r = distrDf_w.sample(n=1).reset_index(drop=True)
        sch_bw.loc[0,'right'] = sch_bw_r.loc[0,'left']
        sch_bw.loc[0,'cond'] = 'bw'
        sch_bw.loc[0,'mkCode'] = mk_bw
        sch_bw.loc[0,'Channel1'] = 2
        sch_bw.loc[0,'Channel2'] = 1
        schDf = pd.concat([schDf,sch_ww,sch_bb,sch_bw,sch_wb],axis=0,ignore_index=True)
schDf['Correct'] = [1]*120*blockN
schDf['mkCode'] = schDf['mkCode'].astype(int)


#
# instructions
for instr_img in ['instr9.png','exp_start.png']:
    instr = visual.ImageStim(win,image=os.path.join(
    filePath,'Instructions',instr_img),pos=(0.0,0.0))
    instr.draw()
    win.flip()
    instrKey = event.waitKeys(keyList=[startKey,quitKey])
    checkEsc(instrKey[0])


#
# experiment
testDfList = [test1Df,test2Df,test3Df]
for blkN in range(blockN):
    task_start('Block %d/%d'%(int(blkN+1),blockN))
    schDf_blk = schDf[schDf['blkN']==blkN+1]
    schDf_blk = schDf_blk.reset_index(drop=True)
    schDf_blk = schDf_blk.sample(frac=1).reset_index(drop=True)
    random.shuffle(isiDurList)
    schDf_blk['dur'] = isiDurList
    targDf_blk = targDf[targDf['blkN']==blkN+1]
    #
    # 1) memorization
    targList = targDf_blk['stimulus'].tolist()
    random.shuffle(targList)
    task_start('Memorizing')
    std_task(targList)
    #
    # (2) testing
    test_round = 1
    roundN = 3
    accList = []
    new_round = 0
    for n in range(roundN):
        testNDf_all = testDfList[n]
        testNDf = testNDf_all[testNDf_all['blkN']==blkN+1]
        testNDf_round = pd.concat([targDf_blk,testNDf],axis=0,ignore_index=True)
        testNDf_round['round'] = [test_round]*len(testNDf_round)
        testNDf_round['Correct'] = [1]*len(testNDf_round)
        testNDf_round = testNDf_round.sample(frac=1).reset_index(drop=True)
        
        task_start('Testing %d'%(n+1))
        testNDf_round = test_task(testNDf_round)
        
        # storing search data
        if (subj==1)&(blkN==0)&(test_round==1):
            headTag = True
            modeTag = 'w'
        else:
            headTag = False
            modeTag = 'a'
        testNDf_round['Subject'] = [subj]*len(testNDf_round)
        testNDf_round.to_csv(os.path.join(filePath,'Data','DataBehav','test.csv'),
        sep=',',header=headTag,index=False,mode=modeTag)
        test_round += 1

        acc = testNDf_round['Correct'].mean()
        if acc>=testCrit:
            accList.append(acc)
            new_round += 1
            if new_round==2:
                break
        
        while acc<testCrit:
            random.shuffle(targList)
            testNDf_round = testNDf_round.sample(frac=1).reset_index(drop=True)
            testNDf_round['round'] = [test_round]*len(testNDf_round)
            testNDf_round['Correct'] = [1]*len(testNDf_round)
            
            task_start('Re-Memorizing')
            std_task(targList)
            task_start('Re-Testing %d'%(n+1))
            testNDf_round = test_task(testNDf_round)
            # storing search data
            testNDf_round.to_csv(os.path.join(filePath,'Data','DataBehav','test.csv'),
            sep=',',header=False,index=False,mode='a')
            test_round += 1
            
            acc = testNDf_round['Correct'].mean()
    #
    # (3) search
    schRespList,schRTList,t0List = [],[],[]
    task_start('Searching')
    sendMarkCode(mk_start)
    for indxN in range(len(schDf_blk)):
        cateMark = schDf_blk.loc[indxN,'mkCate']
        schMark = schDf_blk.loc[indxN,'mkCode']
        isiDur = schDf_blk.loc[indxN,'dur']
        img_l = schDf_blk.loc[indxN,'left']
        img_l_pres = visual.ImageStim(win,image=img_l,
        pos=(-visuDeg,0.0),size=imgDeg,units='deg')
        img_r = schDf_blk.loc[indxN,'right']
        img_r_pres = visual.ImageStim(win,image=img_r,
        pos=(visuDeg,0.0),size=imgDeg,units='deg')
        fx = visual.TextStim(win,text='+',height=fixDeg,units='deg',
        pos=(0.0,0.0),color='black',bold=False,italic=False)
        isi = visual.TextStim(win,text='+',height=fixDeg,units='deg',
        pos=(0.0,0.0),color='black',bold=False,italic=False)
        # trial starts
        iti(fixDur)
        img_l_pres.draw()
        img_r_pres.draw()
        fx.draw()
        win.flip()
        sendMarkCode(schMark)
        core.wait(imgDur)
        sendMarkCode(cateMark)
        isi.draw()
        t0 = win.flip()
        respKey = event.waitKeys(maxWait=isiDur,keyList=respKeyList,
        timeStamped=True,clearEvents=True)
        if respKey != None:
            checkEsc(respKey[0][0])
            respKeyFirst = respKey[0][0].lower()
            respRT = respKey[0][1]-t0+imgDur
            waitTime = imgDur+isiDur-respRT
        else:
            respKeyFirst = 'None'
            respRT = 0
            waitTime = 0
        core.wait(waitTime)
        
        schRespList.append(respKeyFirst)
        schRTList.append(respRT)
        t0List.append(t0+imgDur)
        
        if respKeyFirst != schDf_blk.loc[indxN,'ans']:
            schDf_blk.loc[indxN,'Correct'] = 0
    sendMarkCode(mk_end)
    
    schDf_blk['Condition'] = ['AND']*len(schDf_blk)
    schDf_blk['RT'] = schRTList
    schDf_blk['resp'] = schRespList
    schDf_blk['onsetT'] = t0List
    schDf_blk['Subject'] = [subj]*len(schDf_blk)
    schDf_blk['age'] = [age]*len(schDf_blk)
    schDf_blk['sex'] = [sex]*len(schDf_blk)
    
    # storing search data
    if (subj==1)&(blkN==0):
        headTag = True
        modeTag = 'w'
    else:
        headTag = False
        modeTag = 'a'
    schDf_blk.to_csv(os.path.join(filePath,'Data','DataBehav','sch.csv'),
    sep=',',header=headTag,index=False,mode=modeTag)
    
    # Block ACC & Break
    schPosACC = schDf_blk.loc[
    (schDf_blk['blkN']==blkN+1)&(schDf_blk['ans']==tResp),'Correct'].mean()
    schRejACC = schDf_blk.loc[
    (schDf_blk['blkN']==blkN+1)&(schDf_blk['ans']==dResp),'Correct'].mean()
    
    showPosACC_num = round(schPosACC*100,2)
    showRejACC_num = round(schRejACC*100,2)
    showPosACC_text = str(showPosACC_num)+'%'
    showRejACC_text = str(showRejACC_num)+'%'
    showACC_text1 = 'You have correctly decided '+showPosACC_text+' of old objects.'
    showACC_text2 = 'You have correctly rejected '+showRejACC_text+' of new objects.'
    
    ACC_text1 = visual.TextStim(win,text=showACC_text1,height=1.5,wrapWidth=30,
    units='deg',pos=(0,7),color='black',bold=False,italic=False)
    ACC_text2 = visual.TextStim(win,text=showACC_text2,height=1.5,wrapWidth=30,
    units='deg',pos=(0,3.5),color='black',bold=False,italic=False)
    break_img = visual.ImageStim(win,
    image=os.path.join(filePath,'Instructions','blk_break.png'),
    pos=(0.0, 0.0))
    ACC_text1.draw()
    ACC_text2.draw()
    break_img.draw()
    win.flip()
    continKey = event.waitKeys(keyList=[startKey,quitKey])
    if continKey != None:
        checkEsc(continKey[0])

gby = visual.ImageStim(win,image=os.path.join(
filePath,'Instructions','gby.png'),pos=(0.0, 0.0))
gby.draw()
win.flip()
event.waitKeys(maxWait=180,keyList=[startKey],timeStamped=True)
