#!/usr/bin/env python
#-*-coding:utf-8 -*-

from psychopy import monitors,visual,core,event,gui
from PIL import Image

import os,copy,time,random,csv
from math import atan,pi,ceil
import numpy as np
from scipy import stats
import pandas as pd


# get path
rootPath = os.getcwd()
filePath = os.path.join(rootPath,'Practice')
pracStudyDf = pd.read_csv(os.path.join(filePath,'PracStudy.csv'),sep=',')
pracTestDf = pd.read_csv(os.path.join(filePath,'PracTest.csv'),sep=',')
pracSchDf = pd.read_csv(os.path.join(filePath,'PracSearch.csv'),sep=',')

cateList = [1,2]
random.shuffle(cateList)
imgDeg = 4
visuDeg = 3
fixDeg = 1.75
fontDeg = 1.35
numDeg = 1.9
fixDur = 0.5
imgDur = 0.2
testCrit = 0.8
trialN = 20
fixDurList = [0.8,1.2]*int(trialN/2)
random.shuffle(fixDurList)
isiDur = 2
startKey = 'space'
quitKey = 'escape'
dResp = 'up'
tResp = 'down'
respKeyList = [tResp,dResp,quitKey]
testCrit = 0.8

distMon = 57
scrWidCm = 53.5
scrWidPix = 1920
scrWidDeg = (atan((scrWidCm/2.0)/distMon)/pi)*180*2
mon = monitors.Monitor('testMonitor',distance=distMon)
scrSize = (scrWidPix,1080)

def checkEsc(keyName):
    if keyName==quitKey:
        core.quit()

def showInstr(win,imgName):
    instr_text = visual.ImageStim(win,image=imgName,pos=(0.0,0.0))
    instr_text.draw()
    win.flip()

def iti(dur_time):
    iti = visual.TextStim(win,text='+',height=fixDeg,
    units='deg',pos=(0.0,0.0),color='black',bold=False,
    italic=False)
    iti.draw()
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
    units='deg',pos=(0.0,0.0),
    color=clr,bold=False,italic=False)
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
        event.clearEvents()
        core.wait(0)
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



# subj infor
info = {'ID NO.':''}
infoDlg = gui.DlgFromDict(dictionary=info,title=u'Subject Information',order=['ID NO.'])
if infoDlg.OK == False:
    core.quit()
subj = int(info['ID NO.'])
win = visual.Window(monitor=mon,color=(1,1,1),size=scrSize,fullscr=True,units='deg')
win.mouseVisible = False

# instructions
for n in range(10):
    img = os.path.join(rootPath,'Instructions','instr%d.png'%n)
    instr = visual.ImageStim(win,image=img,pos=(0.0,0.0))
    instr.draw()
    win.flip()
    if n==3:
        respContinKey = tResp
    elif n==4:
        respContinKey = dResp
    else:
        respContinKey = startKey
    instrKey = event.waitKeys(keyList=[respContinKey,quitKey])
    checkEsc(instrKey[0])

# practice
while True:
    for cate in cateList:
        targDf = pracStudyDf[pracStudyDf['cate']==cate]
        targDf.reset_index(drop=True,inplace=True)
        testingDf = pracTestDf[pracTestDf['cate']==cate]
        testingDf.reset_index(drop=True,inplace=True)
        schDf_blk = pracSchDf[pracSchDf['cate']==cate]
        schDf_blk['fix'] = fixDurList
        schDf_blk = schDf_blk.sample(frac=1).reset_index(drop=True)
        
        # 1) memorization
        targList = targDf['stimulus'].tolist()
        random.shuffle(targList)
        task_start('Memorizing')
        std_task(targList)
        
        # (2) testing
        test_round = 1
        accList = []
        roundN = 3
        new_round = 0
        for n in range(roundN):
            testNDf_round = testingDf[testingDf['round']==n+1]
            testNDf_round['Correct'] = [1]*len(testNDf_round)
            testNDf_round = testNDf_round.sample(frac=1).reset_index(drop=True)
            
            task_start('Testing %d'%(n+1))
            testNDf_round = test_task(testNDf_round)
            
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
                test_round += 1
                
                acc = testNDf_round['Correct'].mean()
        
        # (3) search
        # searching phase
        task_start('Searching')
        for indxN in range(len(schDf_blk)):
            fixDur = schDf_blk.loc[indxN,'fix']
            iti(fixDur)
            img_l = schDf_blk.loc[indxN,'left']
            img_l_pres = visual.ImageStim(win,image=img_l,\
            pos=(-visuDeg,0.0),size=imgDeg,units='deg')
            img_l_pres.draw()
            img_r = schDf_blk.loc[indxN,'right']
            img_r_pres = visual.ImageStim(win,image=img_r,\
            pos=(visuDeg,0.0),size=imgDeg,units='deg')
            img_r_pres.draw()
            fx = visual.TextStim(win,text='+',height=fixDeg,units='deg',\
            pos=(0.0,0.0),color='black',bold=False,italic=False)
            fx.draw()
            win.flip()
            core.wait(imgDur)
            isi = visual.TextStim(win,text='+',height=fixDeg,units='deg',pos=(0.0,0.0),\
            color='black',bold=False,italic=False)
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
            
            if respKeyFirst != schDf_blk.loc[indxN,'ans']:
                schDf_blk.loc[indxN,'Correct'] = 0
        # Block ACC & Break
        schPosACC = schDf_blk.loc[
        (schDf_blk['cate']==cate)&(schDf_blk['ans']==tResp),'Correct'].mean()
        schRejACC = schDf_blk.loc[
        (schDf_blk['cate']==cate)&(schDf_blk['ans']==dResp),'Correct'].mean()
        
        showPosACC_num = round(schPosACC*100,2)
        showRejACC_num = round(schRejACC*100,2)
        showPosACC_text = str(showPosACC_num)+'%'
        showRejACC_text = str(showRejACC_num)+'%'
        showACC_text1 = 'You have correctly decided '+showPosACC_text+' of old objects.'
        showACC_text2 = 'You have correctly rejected '+showRejACC_text+' of new objects.'
        
        ACC_text1 = visual.TextStim(win,text=showACC_text1,height=1.5,wrapWidth=30,
        units='deg',pos=(0,7),color='black',bold=False,italic=False)
        ACC_text2 = visual.TextStim(win,text=showACC_text2,height=1.5,wrapWidth=30,
        units='deg',pos=(0,2),color='black',bold=False,italic=False)
        break_img = visual.ImageStim(win,
        image=os.path.join(rootPath,'Instructions','blk_break.png'),
        pos=(0.0, 0.0))
        ACC_text1.draw()
        ACC_text2.draw()
        break_img.draw()
        win.flip()
        continKey = event.waitKeys(keyList=[startKey,quitKey])
        if continKey != None:
            checkEsc(continKey[0])
    
    # End
    rp_instr = visual.ImageStim(win,
    image=os.path.join(rootPath,'Instructions','rp_prac.png'),pos=(0.0,0.0))
    rp_instr.draw()
    win.flip()
    continKey = event.waitKeys(keyList=[startKey,quitKey])
    if continKey != None:
        checkEsc(continKey[0])




