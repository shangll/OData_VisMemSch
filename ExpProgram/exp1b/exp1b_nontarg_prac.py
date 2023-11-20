#!/usr/bin/env python
#-*-coding:utf-8 -*-

from psychopy import monitors, visual, core, event, gui

import os, time, random, csv
from math import atan, pi
import numpy as np
from scipy import stats
import pandas as pd



# ####################
# par
# ####################

# get path
filePath = os.getcwd()+'/'
pracStudyDf = pd.read_csv(filePath+'Practice/PracStudy.csv',sep=',')
pracTestDf = pd.read_csv(filePath+'Practice/PracTest.csv',sep=',')
pracSchDf = pd.read_csv(filePath+'Practice/PracSearch.csv',sep=',')
oddTaskDf = pd.read_csv(filePath+'Practice/PracOdd.csv',sep=',')

startKey = 'space'
quitKey = 'escape'
corrResp = 'f'
inCorrResp = 'j'
respKeyList = [corrResp,inCorrResp,quitKey]
cateList = [1,2]
random.shuffle(cateList)
imgDeg = 4.9
fixDeg = 1.75
fontDeg = 1.25
numDeg = 1.9
fixDur = 0.5
imgDur = 0.2
testCrit = 0.8
trialN = 20
itiDurList = [1.8,2.3]*int(trialN/2)
random.shuffle(itiDurList)

distMon = 57
scrWidCm = 40.0
scrWidPix = 1920
scrWidDeg = (atan((scrWidCm/2.0)/distMon)/pi)*180*2
mon = monitors.Monitor('testMonitor',distance=distMon)



# ####################
# sub-functions
# ####################

def checkEsc(keyName):
    if keyName==quitKey:
        core.quit()

def showInstr(win,imgName):
    instr_text = visual.ImageStim(win,image=imgName,pos=(0.0, 0.0),size=(1280,800),units='pixels')
    instr_text.draw()
    win.flip()

def taskStart(win,text_par,font_par):
    start_text = visual.TextStim(win,text=text_par,height=font_par,units='deg',
    pos=(0.0,0.0),color='black',bold=False,italic=False)
    start_text.draw()
    win.flip()
    core.wait(0.95)
    
    fix_text = visual.TextStim(win,text=u'+',height=fixDeg,units='deg',pos=(0.0,0.0),
    color='black',bold=False,italic=False)
    fix_text.draw()
    win.flip()
    core.wait(fixDur)

def stdTask(win,targDf,std_text):
    targDf = targDf.reindex(np.random.permutation(targDf.index))
    targDf.reset_index(inplace=True,drop=True)
    
    taskStart(win,std_text, fontDeg)
    
    for targImg in targDf['stimulus']:
        targ_img = visual.ImageStim(win,image=targImg,pos=(0.0, 0.0),size=imgDeg,units='deg')
        targ_img.draw()
        win.flip()
        core.wait(3)
        
        iti = visual.TextStim(win,text='+',height=fixDeg,units='deg',pos=(0.0,0.0),\
        color='black',bold=False,italic=False)
        # iti = visual.ImageStim(win,image=None)
        iti.draw()
        win.flip()
        core.wait(0.95)

def fbkShow(win,fbkText,clr):
    fbk_text = visual.TextStim(win,text=fbkText,height=fontDeg,units='deg',pos=(0.0,0.0),
    color=clr,bold=False,italic=False)
    fbk_text.draw()
    win.flip()
    core.wait(0.5)

def recogTask(win,testingDf,testing_text):
    testingDf = testingDf.reindex(np.random.permutation(testingDf.index))
    testingDf.reset_index(inplace=True,drop=True)
    
    taskStart(win,testing_text, fontDeg)
    
    indxN = 0
    testACC = 0
    for testImg in testingDf['stimulus']:
        test_img = visual.ImageStim(win,image=testImg,pos=(0.0, 0.0),size=imgDeg,units='deg')
        test_img.draw()
        win.flip()
        event.clearEvents()
        core.wait(0)
        respKey = event.waitKeys(keyList=respKeyList)
        checkEsc(respKey[0])
        respKeyFirst = respKey[0].lower()
        
        if respKeyFirst == testingDf.loc[indxN,'testAns']:
            testACC +=1
            fbkText = u'Correct!'
            clr = 'green'
        else:
            fbkText = u'Wrong!'
            clr = 'red'
        fbkShow(win,fbkText,clr)
        indxN += 1
        
        iti = visual.TextStim(win,text='+',height=fixDeg,units='deg',pos=(0.0,0.0),\
        color='black',bold=False,italic=False)
        # iti = visual.ImageStim(win,image=None)
        iti.draw()
        win.flip()
        core.wait(0.95)
    return testACC/4

def showImg(win,ImgName):
    img = visual.ImageStim(win,image=ImgName,pos=(0.0,0.0),size=imgDeg,units='deg')
    instr_text.draw()
    win.flip()

def quitExp():
    quitText = u'Notice that your accuracy is too low. In formal experiment, \
    the program has to be terminated.\n\
    Please press SPACE to continue.'
    quit_text = visual.TextStim(win,text=quitText,height=fontDeg,units='deg',pos=(0.0,0.0),
    color='black',bold=False,italic=False)
    quit_text.draw()
    win.flip()
    quitExpKey = event.waitKeys(keyList=[startKey,quitKey])
    checkEsc(quitExpKey[0])



# ####################
# main function
# ####################

# subj infor
info = {'ID NO.':''}
infoDlg = gui.DlgFromDict(dictionary=info,title=u'Subject Information',order = ['ID NO.'])
if infoDlg.OK == False:
    core.quit()
subj = int(info['ID NO.'])

win = visual.Window(monitor=mon,color=(1,1,1),fullscr=True,units='deg')
win.mouseVisible = False

# instructions
timeList = [float('inf')]*10+[0.5,0.2,1.5]*3+[float('inf')]+[0.5,0.2,1.5]*4+[float('inf')]*2
for instrN in range(35):
    imgName = filePath+'Instructions/instr'+str(instrN)+'.png'
    if instrN in [13,16,19,23,26,29,32]:
        waitTime = 1.5
        rightKeys = respKeyList
    elif instrN == 6:
        rightKeys = [corrResp]
    elif instrN == 7:
        rightKeys = [inCorrResp]
    elif instrN==34:
        imgName = 'Instructions/breakITI_recog.png'
        rightKeys = [startKey,quitKey]
        waitTime = 0
    else:
        waitTime = 0
        rightKeys = [startKey,quitKey]
    showInstr(win,imgName)
    core.wait(waitTime)
    instrKey = event.waitKeys(maxWait=timeList[instrN-1],keyList=rightKeys)
    if instrKey!=None:
        checkEsc(instrKey[0])
    else:
        continue
    while (instrKey[0] not in rightKeys) and (instrN in [6,7,13,16,19,23,26,29,32]):
        showInstr(win,instrN)
        core.wait(waitTime)
        instrKey = event.waitKeys(maxWait=timeList[instrN-1],keyList=rightKeys)
        if instrKey!=None:
            checkEsc(instrKey[0])
    
# practice
while True:
    for cate in cateList:
        targDf = pracStudyDf[pracStudyDf['cate']==cate]
        targDf.reset_index(drop=True,inplace=True)
        testingDf = pracTestDf[pracTestDf['cate']==cate]
        testingDf.reset_index(drop=True,inplace=True)
        schDf = pracSchDf[pracSchDf['cate']==cate]
        schDf['dur'] = itiDurList
        schDf = schDf.sample(frac=1).reset_index(drop=True)
        oddTaskDf['dur'] = itiDurList
        oddTaskDf = oddTaskDf.sample(frac=1).reset_index(drop=True)
        # study phase
        stdTask(win,targDf,'Memorizing')
        # testing phase
        testACCList = []
        inCorrList = []
        for roundN in range(1,4): 
            testingRoundDf = testingDf[testingDf['round']==roundN]
            testing_text='Testing '+str(roundN)
            testACC = recogTask(win,testingRoundDf,testing_text)
            # ACC check
            testACCList.append(testACC)
            if testACC < testCrit:
                inCorrList.append(testACC)
                
            if len(testACCList)==2 and testACCList[0] >= testCrit and testACCList[1] >= testCrit:
                break
            elif len(inCorrList) == 3:
                quitExp()
            while testACC < testCrit:
                stdTask(win,targDf,'Re-Memorizing')
                testACC = recogTask(win,testingRoundDf,testing_text)
                testACCList.append(testACC)
                inCorrList = [x for x in testACCList if x < testCrit]
                if len(inCorrList) == 3:
                    quitExp()
        # searching phase
        sch_start = visual.TextStim(win,text='Searching',height=fontDeg,units='deg',pos=(0.0,0.0),
        color='black',bold=False,italic=False)
        sch_start.draw()
        win.flip()
        core.wait(0.95)
        
        # fixation
        schFix = visual.TextStim(win,text='+',height=fixDeg,units='deg',pos=(0.0,0.0),
        color='black',bold=False,italic=False)
        schFix.draw()
        win.flip()
        core.wait(fixDur)
        
        schPosACC = 0
        schRejACC = 0
        for schN in range(trialN):
            schImg = schDf.loc[schN,'stimulus']
            schITIDur = schDf.loc[schN,'dur']
            # fixation
            # schFix = visual.TextStim(win,text='+',height=fixDeg,units='deg',pos=(0.0,0.0),
            # color='black',bold=False,italic=False)
            # schFix.draw()
            # win.flip()
            # core.wait(fixDur)
            # image presented
            sch_img = visual.ImageStim(win,image=schImg,pos=(0.0, 0.0),size=imgDeg,units='deg')
            sch_img.draw()
            t0 = win.flip()
            core.wait(imgDur)
            # response & ITI
            schITI = visual.TextStim(win,text='+',height=fixDeg,units='deg',pos=(0.0,0.0),\
            color='black',bold=False,italic=False)
            # schITI = visual.ImageStim(win,image=None)
            schITI.draw()
            win.flip()
            event.clearEvents()
            respKey = event.waitKeys(maxWait=schITIDur,keyList=respKeyList,timeStamped=True)
            if respKey != None:
                checkEsc(respKey[0][0])
                respKeyFirst = respKey[0][0].lower()
                respRT = respKey[0][1]-t0
                waitTime = schITIDur-respRT+imgDur
            else:
                respKeyFirst = None
                respRT = 0
                waitTime = 0
            core.wait(waitTime)
            # storing response data
            if (respKeyFirst==schDf.loc[schN,'schAns']):
                if schDf.loc[schN,'schAns']==corrResp:
                    schPosACC +=1
                else:
                    schRejACC +=1
        showPosACC_num = round(schPosACC/4*100,2)
        showRejACC_num = round(schRejACC/(trialN-4)*100,2)
        showPosACC_text = str(showPosACC_num)+'%'
        showRejACC_text = str(showRejACC_num)+'%'
        showACC_text1 = 'You have correctly decided '+showPosACC_text+' of targets.'
        showACC_text2 = 'You have correctly rejected '+showRejACC_text+' of distractors.'
        
        ACC_text1 = visual.TextStim(win,text=showACC_text1,height=fontDeg,units='deg',pos=(0,7),\
        color='black',bold=False,italic=False)
        ACC_text2 = visual.TextStim(win,text=showACC_text2,height=fontDeg,units='deg',pos=(0,3.5),\
        color='black',bold=False,italic=False)
        break_img = visual.ImageStim(win,image=filePath+'Instructions/breakITI.png',pos=(0.0, 0.0))
        ACC_text1.draw()
        ACC_text2.draw()
        break_img.draw()
        win.flip()
        continKey = event.waitKeys(maxWait=180,keyList=[startKey,quitKey])
        checkEsc(continKey[0])

    # oddbal task
    odd_instr = visual.ImageStim(win,image=filePath+'Instructions/breakITI_odd.png',\
    pos=(0.0, 0.0))
    odd_instr.draw()
    win.flip()
    continKey = event.waitKeys(keyList=[startKey,quitKey])
    checkEsc(continKey[0])
    
    # fixation
    oddFix = visual.TextStim(win,text='+',height=fixDeg,units='deg',pos=(0.0,0.0),
    color='black',bold=False,italic=False)
    oddFix.draw()
    win.flip()
    core.wait(fixDur)
    
    oddNumACC = 0
    oddIgnACC = 0
    for oddN in range(trialN):
        oddImg = oddTaskDf.loc[oddN,'stimulus']
        oddITIDur = oddTaskDf.loc[oddN,'dur']
        
        # image presented
        if oddTaskDf.loc[oddN,'cate'] != 'num':
            odd_img = visual.ImageStim(win,image=oddImg,pos=(0.0, 0.0),size=imgDeg,units='deg')
        else:
            odd_img = visual.TextStim(win,text=str(oddImg),height=numDeg,units='deg',pos=(0.0,0.0),
            color='black',bold=False,italic=False)
        odd_img.draw()
        t0 = win.flip()
        core.wait(imgDur)
        # response & ITI
        oddITI = visual.TextStim(win,text='+',height=fixDeg,units='deg',\
        pos=(0.0,0.0),color='black',bold=False,italic=False)
        # oddITI = visual.ImageStim(win,image=None)
        oddITI.draw()
        win.flip()
        respKey = event.waitKeys(maxWait=oddITIDur,keyList=[corrResp,quitKey],timeStamped=True)
        if respKey != None:
            checkEsc(respKey[0][0])
            respKeyFirst = respKey[0][0].lower()
            respRT = respKey[0][1]-t0
            waitTime = oddITIDur-respRT+imgDur
        else:
            respKeyFirst = None
            respRT = 0
            waitTime = 0
        core.wait(waitTime)
        if (respKeyFirst==None) and (oddTaskDf.loc[oddN,'cate']!='num'):
            oddIgnACC +=1
        elif (respKeyFirst!=None) and (oddTaskDf.loc[oddN,'cate']=='num'):
            oddNumACC +=1
    oddNumACC_num = round(oddNumACC/3*100,2)
    oddIgnACC_num = round(oddIgnACC/(trialN-3)*100,2)
    oddNumACC_text = str(oddNumACC_num)+'%'
    oddIgnACC_text = str(oddIgnACC_num)+'%'
    showOddACC_text1 = 'You have correctly decided '+oddNumACC_text+' of numbers.'
    showOddACC_text2 = 'You have correctly ignored '+oddIgnACC_text+' of images.'
    oddACC_text1 = visual.TextStim(win,text=showOddACC_text1,height=fontDeg,
    units='deg',pos=(0,7),color='black',bold=False,italic=False)
    oddACC_text2 = visual.TextStim(win,text=showOddACC_text2,height=fontDeg,
    units='deg',pos=(0.0,3.5),color='black',bold=False,italic=False)
    break_img = visual.ImageStim(win,image=filePath+'Instructions/breakITI.png',pos=(0.0, 0.0))
    oddACC_text1.draw()
    oddACC_text2.draw()
    break_img.draw()
    win.flip()
    continKey = event.waitKeys(maxWait=180,keyList=[startKey,quitKey])
    checkEsc(continKey[0])
        
    # end
    showInstr(win,'Instructions/pracEnd.png')
    continKey = event.waitKeys(keyList=[startKey,quitKey])
    checkEsc(continKey[0])