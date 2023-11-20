#!/usr/bin/env python
#-*-coding:utf-8 -*-

from psychopy import monitors, visual, core, event, gui
from rusocsci import buttonbox

import os, time, random, csv
from math import atan, pi
import numpy as np
from scipy import stats
import pandas as pd



# ####################
# subj info
# ####################

info = {'ID NO.':'','Age':'','Sex':['F','M'],'Handedness':['R','L']}
infoDlg = gui.DlgFromDict(dictionary=info,title=u'Subject Information',\
order = ['ID NO.','Age','Sex','Handedness'])
if infoDlg.OK == False:
    core.quit()

subj = int(info['ID NO.'])
age = info['Age']
sex = info['Sex']
handed = info['Handedness']



# ####################
# par
# ####################

sizeList = [1,2,4,8]
cateList = ['Animals','Objects']

blockN = 8
trialN = 60
oddNumN = 10
oddTrialN = 110
imgDur = 0.2
itiDurList = [1.8,2.3]*int(trialN/2)
random.shuffle(itiDurList)
oddNumList = list(range(10,100))

startKey = 'space'
quitKey = 'escape'
corrResp = 'f'
inCorrResp = 'j'
respKeyList = [corrResp,inCorrResp,quitKey]
testCrit = 0.8

# EEG trigger/mark numbers
bb = buttonbox.Buttonbox(port='COM3')

delMk_recog = 33
delMk_odd = 55
# fixation & response
fixMk_recog = 60
respMk_recog = 96
fixMk_odd = 70
respMk_odd = 97
# search
mk_within1_a = 111
mk_within2_a = 112
mk_within4_a = 114
mk_within8_a = 118
mk_between1_a = 211
mk_between2_a = 212
mk_between4_a = 214
mk_between8_a = 218
mk_within1_o = 121
mk_within2_o = 122
mk_within4_o = 124
mk_within8_o = 128
mk_between1_o = 221
mk_between2_o = 222
mk_between4_o = 224
mk_between8_o = 228
mk_targ1 = 91
mk_targ2 = 92
mk_targ4 = 94
mk_targ8 = 98
# oddball
mk_anim = 1
mk_obj = 2
mk_num = 9

# get path
filePath = os.getcwd()+'/'
stimListDf = pd.read_csv(filePath+'stimList.csv',sep=',')
# anim
animList = stimListDf['anim'].tolist()
targAnimList = random.sample(animList,15)
distrAnimList = [x for x in animList if x not in targAnimList]
random.shuffle(distrAnimList)
# obj
objList = stimListDf['obj'].tolist()
targObjList = random.sample(objList,15)
distrObjList = [x for x in objList if x not in targObjList]
random.shuffle(distrObjList)

#imgSize = (0.25,0.25)
imgDeg = 4.9
fixDeg = 1.75
#fixSize = 0.1
#fontSize = 1
fontDeg = 1.25
numDeg = 1.9

distMon = 57
scrWidCm = 40.0
scrWidPix = 1920
scrWidDeg = (atan((scrWidCm/2.0)/distMon)/pi)*180*2
#deg2pix = int(scrWidPix/scrWidDeg)

mon = monitors.Monitor('testMonitor',distance=distMon)
win = visual.Window(monitor=mon,color=(1,1,1),size=(1920,1080),\
fullscr=True,units='deg')
win.mouseVisible = False
timer = core.Clock()



# ####################
# sub-functions
# ####################

def checkEsc(keyName):
    if keyName==quitKey:
        core.quit()
    
def sendMarkCode(mkCode):
    bb.sendMarker(val=mkCode)
    core.wait(0.002)
    bb.sendMarker(val=0) 

def getLatinSqr(itemList):
    LatinNumRow = list(range(len(itemList)))
    
    LatinSqrList = []
    maxNum = LatinNumRow[-1]
    # row 1
    LatinNumRow.insert(2,maxNum)
    LatinNumRow.pop()
    newItemList = [itemList[LatinNum] for LatinNum in LatinNumRow]
    LatinSqrList.append(newItemList)
    # other rows
    while LatinNumRow[0] != maxNum:
        tempLatinNumRow = []
        for LatinNum in LatinNumRow:
            itemIndx = LatinNum+1
            if itemIndx > maxNum:
                itemIndx = 0
            tempLatinNumRow.append(itemIndx)
        LatinNumRow = tempLatinNumRow
        newItemList = [itemList[LatinNum] for LatinNum in LatinNumRow]
        LatinSqrList.append(newItemList)
    
    return LatinSqrList

def distrChoose(distrSource,oldStimList):
    distrSchDf = pd.DataFrame()
    if distrSource==distrHomoSourceList:
        distrType = 'within'
    else:
        distrType = 'between'
    
    distrCount = 0
    for subCate in distrSource:
        distrCount += 1
        if distrCount <= 9:
            distrNum = 2
        else:
            distrNum = 1
        
        distrDf = pd.read_csv(filePath+subCate,sep=',')
        distrStimDf = distrDf.sample(n=distrNum)
        tempDistrList = distrStimDf['stimulus'].tolist()
        while list(set(tempDistrList).intersection(set(oldStimList))) != []:
            distrStimDf = distrDf.sample(n=distrNum)
            tempDistrList = distrStimDf['stimulus'].tolist()
        oldStimList += tempDistrList
        distrStimDf['cond'] = [distrType]*distrNum
        distrSchDf = pd.concat([distrSchDf,distrStimDf],axis=0,\
        ignore_index=True,sort=False)
    
    return distrSchDf

def taskStart(text_par,font_par):
    start_text = visual.TextStim(win,text=text_par,height=font_par,units='deg',
    pos=(0.0,0.0),color='black',bold=False,italic=False)
    start_text.draw()
    win.flip()
    core.wait(0.95)
    
    fix_text = visual.TextStim(win,text=u'+',height=fixDeg,units='deg',pos=(0.0,0.0),
    color='black',bold=False,italic=False)
    fix_text.draw()
    win.flip()
    core.wait(0.95)

def stdTask(win,targDf,std_text):
    targDf = targDf.reindex(np.random.permutation(targDf.index))
    targDf.reset_index(inplace=True,drop=True)
    
    taskStart(std_text, fontDeg)
    
    for targImg in targDf['stimulus']:
        targ_img = visual.ImageStim(win,image=targImg,pos=(0.0, 0.0),size=imgDeg,\
        units='deg')
        targ_img.draw()
        win.flip()
        core.wait(3)
        iti = visual.TextStim(win,text='+',height=fixDeg,units='deg',pos=(0.0,0.0),\
        color='black',bold=False,italic=False)
        # iti = visual.ImageStim(win,image=None)
        iti.draw()
        win.flip()
        core.wait(0.95)

def fbkShow(fbkText,clr):
    fbk_text = visual.TextStim(win,text=fbkText,height=fontDeg,\
    units='deg',pos=(0.0,0.0),
    color=clr,bold=False,italic=False)
    fbk_text.draw()
    win.flip()
    core.wait(0.5)

def recogTask(win,testingDf,testing_text):
    testRTList = []
    testRespList = []
    
    testingDf = testingDf.reindex(np.random.permutation(testingDf.index))
    testingDf.reset_index(inplace=True,drop=True)
    
    taskStart(testing_text, fontDeg)
    
    indxN = 0
    testACC = 0
    for testImg in testingDf['stimulus']:
        test_img = visual.ImageStim(win,image=testImg,pos=(0.0, 0.0),\
        size=imgDeg,units='deg')
        test_img.draw()
        t0 = win.flip()
        event.clearEvents()
        core.wait(0)
        respKey = event.waitKeys(keyList=respKeyList,timeStamped=True)
        checkEsc(respKey[0][0])
        respKeyFirst = respKey[0][0].lower()
        respKeyRT = respKey[0][1]-t0
        testRespList.append(respKeyFirst)
        testRTList.append(respKeyRT)
        
        if respKeyFirst == testingDf.loc[indxN,'testAns']:
            testACC +=1
            fbkText = u'Correct!'
            clr = 'green'
        else:
            fbkText = u'Wrong!'
            clr = 'red'
        fbkShow(fbkText,clr)
        indxN += 1
        
        iti = visual.TextStim(win,text='+',height=fixDeg,units='deg',pos=(0.0,0.0),\
        color='black',bold=False,italic=False)
        # iti = visual.ImageStim(win,image=None)
        iti.draw()
        win.flip()
        core.wait(0.95)
    
    testingDf['rt'] = testRTList
    testingDf['resp'] = testRespList
    
    return testingDf, testACC/len(testingDf)

def quitExp():
    quitText = u'Your accuracy is too low, so the program has to be terminated.\n\
    The experiment has been finished, and will automatically exist in 10 sec.\n\
    Thank you for your participation, and have a nice day!'
    quit_text = visual.TextStim(win,text=quitText,height=fontDeg,units='deg',\
    pos=(0.0,0.0),color='black',bold=False,italic=False)
    quit_text.draw()
    win.flip()
    core.wait(10)
    win.close()
    core.quit()



# ####################
# main function
# ####################

# sort stimuli
oldStimList = []
oldSubCateList = []
sizeOrder = getLatinSqr(sizeList)
sizeOrder = sizeOrder*8

# preparing for oddball task
oddTaskDf_img = pd.DataFrame()
for oddCate in cateList:
    if oddCate == 'Animals':
        oddStimList = distrAnimList
        oddMkTarg = mk_anim
    else:
        oddStimList = distrObjList
        oddMkTarg = mk_obj
    oddCount = 0
    for subCate in oddStimList:
        oddCount += 1
        if oddCount <= 9:
            oddNum = 1
        else:
            oddNum = 7
        oddDf = pd.read_csv(filePath+subCate,sep=',')
        oddStimDf = oddDf.sample(n=oddNum)
        oddStimDf.reset_index(inplace=True,drop=True)
        tempOddStimList = oddStimDf['stimulus'].tolist()
        oldStimList += tempOddStimList
        oddStimDf['markCode'] = [oddMkTarg]*oddNum
        oddTaskDf_img = pd.concat([oddTaskDf_img,oddStimDf],axis=0,ignore_index=True,sort=False)
    oddTaskDf_img.drop(index=[len(oddTaskDf_img)-1],inplace=True)


# instructions
for instrName in ['formalInstr1.png','formalInstr2.png','breakITI_recog.png']:
    instr_text = visual.ImageStim(win,image=filePath+'Instructions/'+str(instrName),
    pos=(0.0, 0.0),size=(1280,800),units='pixels')
    instr_text.draw()
    win.flip()
    instrKey = event.waitKeys(keyList=[startKey,quitKey])
    checkEsc(instrKey[0])

indxNum = 0
for setsize in sizeOrder[subj-1]:
    random.shuffle(cateList)
    
    for cate in cateList:
        indxNum += 1
        
        if cate == 'Animals':
            targSourceList = targAnimList
            distrHomoSourceList = distrAnimList
            distrHeteSourceList = distrObjList
        else:
            targSourceList = targObjList
            distrHomoSourceList = distrObjList
            distrHeteSourceList = distrAnimList
    
        # select sub-categories
        targSourceList = list(set(targSourceList)-set(oldSubCateList))
        subCateList = random.sample(targSourceList,setsize)
        while list(set(subCateList).intersection(set(oldSubCateList))) != []:
            subCateList = random.sample(targSourceList,setsize)
        oldSubCateList += subCateList
    
        # select study & testing stimuli
        targDf = pd.DataFrame()
        testDf = pd.DataFrame()
        for subCate in subCateList:
            # target stimuli
            targChooseDf = pd.read_csv(filePath+subCate,sep=',')
            targStimDf = targChooseDf.sample().reset_index(drop=True)
            targStimDf['testAns']=corrResp
            tempTarg = targStimDf.loc[0,'stimulus']
            while tempTarg in oldStimList:
                targStimDf = targChooseDf.sample().reset_index(drop=True)
                tempTarg = targStimDf.loc[0,'stimulus']
            oldStimList.append(tempTarg)
            targDf = pd.concat([targDf,targStimDf],axis=0,ignore_index=True,sort=False)
            
            # testing stimuli
            testStimDf = targChooseDf.sample(n=3)
            tempTestList = testStimDf['stimulus'].tolist()
            while list(set(tempTestList).intersection(set(oldStimList))) != []:
                testStimDf = targChooseDf.sample(n=3)
                tempTestList = testStimDf['stimulus'].tolist()
            oldStimList += tempTestList
            testStimDf['round'] = [1,2,3]
            testStimDf['testAns'] = [inCorrResp]*3
            testDf = pd.concat([testDf,testStimDf],axis=0,ignore_index=True,sort=False)
            
        # select search distractors
        if cate == 'Animals':
            if setsize==1:
                mk_code_w = mk_within1_a
                mk_code_b = mk_between1_o
            elif setsize==2:
                mk_code_w = mk_within2_a
                mk_code_b = mk_between2_o
            elif setsize==4:
                mk_code_w = mk_within4_a
                mk_code_b = mk_between4_o
            else:
                mk_code_w = mk_within8_a
                mk_code_b = mk_between8_o
        else:
            if setsize==1:
                mk_code_w = mk_within1_o
                mk_code_b = mk_between1_a
            elif setsize==2:
                mk_code_w = mk_within2_o
                mk_code_b = mk_between2_a
            elif setsize==4:
                mk_code_w = mk_within4_o
                mk_code_b = mk_between4_a
            else:
                mk_code_w = mk_within8_o
                mk_code_b = mk_between8_a
        distrSchDf_w = distrChoose(distrHomoSourceList,oldStimList)
        distrSchDf_w['markCode'] = [mk_code_w]*len( distrSchDf_w)
        distrSchDf_b = distrChoose(distrHeteSourceList,oldStimList)
        distrSchDf_b['markCode'] = [mk_code_b]*len( distrSchDf_b)
        distrSchDf = pd.concat([distrSchDf_w,distrSchDf_b],axis=0,ignore_index=True,sort=False)
        distrSchDf['schAns'] = [inCorrResp]*len(distrSchDf)
        
        targSchDf = targDf.drop(labels='testAns',axis=1,inplace=False)
        targSchDf['schAns'] = [corrResp]*setsize
        targSchDf['cond'] = ['targ']*setsize
        if setsize==1:
            targSchDf = pd.DataFrame(np.repeat(targSchDf.values,12,axis=0),columns=targSchDf.columns)
            targSchDf['markCode'] = [mk_targ1]*len(targSchDf)
        elif setsize==2:
            targSchDf = pd.DataFrame(np.repeat(targSchDf.values,6,axis=0),columns=targSchDf.columns)
            targSchDf['markCode'] = [mk_targ2]*len(targSchDf)
        elif setsize==4:
            targSchDf = pd.DataFrame(np.repeat(targSchDf.values,3,axis=0),columns=targSchDf.columns)
            targSchDf['markCode'] = [mk_targ4]*len(targSchDf)
        elif setsize==8:
            targSchDfNew = targSchDf.sample(n=4).reset_index(drop=True)
            targSchDf = pd.concat([targSchDf,targSchDfNew],axis=0,ignore_index=True,sort=False)
            targSchDf['markCode'] = [mk_targ8]*len(targSchDf)
        
        # search stimuli
        schDf = pd.DataFrame()
        schDf = pd.concat([targSchDf,distrSchDf],axis=0,ignore_index=True,sort=False)
        schDf['dur'] = itiDurList
        schDf['acc'] = [0]*len(schDf)
        schDf['markCodeResp'] = [0]*len(schDf)
        schDf = schDf.sample(frac=1).reset_index(drop=True)
        
        block_start = visual.TextStim(win,text='Block '+str(indxNum)+'/8',
        height=fontDeg,units='deg',pos=(0.0,0.0),color='black',bold=False,italic=False)
        block_start.draw()
        win.flip()
        event.clearEvents()
        core.wait(0.95)
        block_iti = visual.ImageStim(win,image=None)
        block_iti.draw()
        win.flip()
        core.wait(0.5)
        
        # stage 1: study
        stdTask(win, targDf, u'Memorizing')
        
        # stage 2: testing
        testingDfNew = pd.DataFrame()
        testACCList = []
        inCorrList = []
        for roundN in range(1,4):
            testRoundDf = testDf[testDf['round']==roundN]
            targDf['round'] = [roundN]*setsize
            testingDf = pd.concat([targDf,testRoundDf],axis=0,ignore_index=True,sort=False)
            
            testing_text='Testing '+str(roundN)
            testRoundDfNew, testACC = recogTask(win, testingDf,testing_text)
            testingDfNew = pd.concat([testingDfNew,testRoundDfNew],axis=0,ignore_index=True,sort=False)
            
            # ACC check
            testACCList.append(testACC)
            if testACC < testCrit:
                inCorrList.append(testACC)
            
            if len(testACCList)==2 and testACCList[0] >= testCrit and testACCList[1] >= testCrit:
                break
            elif len(inCorrList) == 3:
                quitExp()
            while testACC < testCrit:
                stdTask(win, targDf, u'Re-Memorizing')
                testRoundDfNew,testACC = recogTask(win, testingDf,testing_text)
                testingDfNew = pd.concat([testingDfNew,testRoundDfNew],\
                axis=0,ignore_index=True,sort=False)
                
                testACCList.append(testACC)
                if testACC < testCrit:
                    inCorrList.append(testACC)
                if len(inCorrList) == 3:
                    quitExp()
    
        testingDfNew['setsize'] = [setsize]*len(testingDfNew)
        testingDfNew['subj'] = [subj]*len(testingDfNew)
        testingDfNew['age'] = [age]*len(testingDfNew)
        testingDfNew['sex'] = [sex]*len(testingDfNew)
        testingDfNew['handed'] = [handed]*len(testingDfNew)
        
        # storing testing data
        if (subj==1) & (indxNum==1):
            headTag = True
            modeTag = 'w'
        else:
            headTag = False
            modeTag = 'a'
        testingDfNew.to_csv(filePath+'Data/DataBehav/testing.csv',sep=',',\
        header=headTag,index=False,mode=modeTag)
        
        # stage 3: searching
        schRTList = []
        schRespList = []
        t0List = []
        schPosACC = 0
        schRejACC = 0
        
        sch_start = visual.TextStim(win,text='Searching',height=fontDeg,units='deg',pos=(0.0,0.0),
        color='black',bold=False,italic=False)
        sch_start.draw()
        win.flip()
        core.wait(0.95)
        
        # block start
        # fixation
        fixDur = random.choice([1.8,2.3])
        schFix = visual.TextStim(win,text='+',height=fixDeg,units='deg',pos=(0.0,0.0),
        color='black',bold=False,italic=False)
        schFix.draw()
        win.flip()
        # mark
        sendMarkCode(fixMk_recog)
        core.wait(fixDur)
        
        for schN in range(trialN):
            schImg = schDf.loc[schN,'stimulus']
            schITIDur = schDf.loc[schN,'dur']
            schMark = schDf.loc[schN,'markCode']
            
            # fixation
            # schFix = visual.TextStim(win,text='+',height=fixDeg,units='deg',pos=(0.0,0.0),
            # color='black',bold=False,italic=False)
            # schFix.draw()
            # win.flip()
            # mark
            # sendMarkCode(fixMk_recog)
            # core.wait(fixDur)
            
            # image presented
            sch_img = visual.ImageStim(win,image=schImg,pos=(0.0, 0.0),size=imgDeg,units='deg')
            sch_img.draw()
            t0 = win.flip()
            # mark
            sendMarkCode(schMark)
            core.wait(imgDur)
            
            # response & ITI
            schITI = visual.TextStim(win,text='+',height=fixDeg,units='deg',pos=(0.0,0.0),\
            color='black',bold=False,italic=False)
            schITI.draw()
            win.flip()
            event.clearEvents()
            respKey = event.waitKeys(maxWait=schITIDur,keyList=respKeyList,timeStamped=True)
            if respKey != None:
                checkEsc(respKey[0][0])
                schDf.loc[schN,'markCodeResp'] = respMk_recog
                respKeyFirst = respKey[0][0].lower()
                respRT = respKey[0][1]-t0
                waitTime = schITIDur-respRT+imgDur
            else:
                respKeyFirst = None
                respRT = 0
                waitTime = 0
            core.wait(waitTime)
            
            # storing response data
            schRespList.append(respKeyFirst)
            schRTList.append(respRT)
            t0List.append(t0)
            if (respKeyFirst==schDf.loc[schN,'schAns']):
                schDf.loc[schN,'acc'] = 1
                if schDf.loc[schN,'schAns']==corrResp:
                    schPosACC +=1
                else:
                    schRejACC +=1
                # mark
                sendMarkCode(respMk_recog)
            else:
                # mark
                sendMarkCode(delMk_recog)
        # block end
        
        schDf['onsetT'] = t0List
        schDf['rt'] = schRTList
        schDf['resp'] = schRespList
        schDf['setsize'] = [setsize]*len(schDf)
        schDf['block'] = [cate]*len(schDf)
        schDf['subj'] = [subj]*len(schDf)
        schDf['age'] = [age]*len(schDf)
        schDf['sex'] = [sex]*len(schDf)
        schDf['handed'] = [handed]*len(schDf)
        
        # storing search data
        if (subj==1) & (indxNum==1):
            headTag = True
            modeTag = 'w'
        else:
            headTag = False
            modeTag = 'a'
        schDf.to_csv(filePath+'Data/DataBehav/sch.csv',sep=',',header=headTag,\
        index=False,mode=modeTag)
        
        showPosACC_num = round(schPosACC/(trialN*0.2)*100,2)
        showRejACC_num = round(schRejACC/(trialN*0.8)*100,2)
        showPosACC_text = str(showPosACC_num)+'%'
        showRejACC_text = str(showRejACC_num)+'%'
        showACC_text1 = 'You have correctly decided '+showPosACC_text+' of targets.'
        showACC_text2 = 'You have correctly rejected '+showRejACC_text+' of distractors.'
        
        ACC_text1 = visual.TextStim(win,text=showACC_text1,height=fontDeg,\
        units='deg',pos=(0,7),color='black',bold=False,italic=False)
        ACC_text2 = visual.TextStim(win,text=showACC_text2,height=fontDeg,\
        units='deg',pos=(0,3.5),color='black',bold=False,italic=False)
        break_img = visual.ImageStim(win,image=filePath+'Instructions/breakITI.png',\
        pos=(0.0, 0.0))
        ACC_text1.draw()
        ACC_text2.draw()
        break_img.draw()
        win.flip()
        continKey = event.waitKeys(maxWait=180,keyList=[startKey,quitKey])
        if continKey != None:
            checkEsc(continKey[0])
        
        # oddball task
        oddTaskDf = pd.DataFrame()
        oddNumShowList = random.sample(oddNumList,oddNumN)
        for oddChooseNum in oddNumShowList:
            oddNumList.remove(oddChooseNum)
        
        oddNumDf = pd.DataFrame({'cate':['num']*oddNumN,'subCate':['num']*oddNumN,\
        'stimulus':oddNumShowList,'markCode':[mk_num]*oddNumN})
        oddTaskDf = pd.concat([oddTaskDf_img,oddNumDf],axis=0,\
        ignore_index=True,sort=False)
        oddTaskDf['dur'] = [1.8,2.3]*int(oddTrialN/2)
        oddTaskDf['acc'] = [0]*oddTrialN
        oddTaskDf['markCodeResp'] = [delMk_odd]*oddTrialN
        oddTaskDf = oddTaskDf.sample(frac=1).reset_index(drop=True)
        
        oddRespList = []
        oddRTList = []
        oddT0List = []
        oddNumACC = 0
        oddIgnACC = 0
        if indxNum in [2,4,6]:
            fixDur = random.choice([1.8,2.3])
            
            odd_instr = visual.ImageStim(win,image=filePath+'Instructions/breakITI_odd.png',\
            pos=(0.0, 0.0))
            odd_instr.draw()
            win.flip()
            continKey = event.waitKeys(keyList=[startKey,quitKey])
            checkEsc(continKey[0])
            
            # block start
            # fixation
            oddFix = visual.TextStim(win,text='+',height=fixDeg,units='deg',pos=(0.0,0.0),
            color='black',bold=False,italic=False)
            oddFix.draw()
            win.flip()
            # mark
            sendMarkCode(fixMk_odd)
            core.wait(fixDur)
            
            for oddN in range(oddTrialN):
                oddImg = oddTaskDf.loc[oddN,'stimulus']
                oddITIDur = oddTaskDf.loc[oddN,'dur']
                oddMark = oddTaskDf.loc[oddN,'markCode']
                
                # image presented
                if oddTaskDf.loc[oddN,'cate'] != 'num':
                    odd_img = visual.ImageStim(win,image=oddImg,pos=(0.0,0.0),\
                    size=imgDeg,units='deg')
                else:
                    odd_img = visual.TextStim(win,text=str(oddImg),height=numDeg,\
                    units='deg',pos=(0.0,0.0),color='black',bold=False,italic=False)
                odd_img.draw()
                t0 = win.flip()
                # mark
                sendMarkCode(oddMark)
                core.wait(imgDur)
                
                # response & ITI
                oddITI = visual.TextStim(win,text='+',height=fixDeg,units='deg',\
                pos=(0.0,0.0),color='black',bold=False,italic=False)
                # oddITI = visual.ImageStim(win,image=None)
                oddITI.draw()
                win.flip()
                respKey = event.waitKeys(maxWait=oddITIDur,keyList=[corrResp,quitKey],\
                timeStamped=True)
                if respKey != None:
                    checkEsc(respKey[0][0])
                    oddTaskDf.loc[oddN,'markCodeResp'] = respMk_odd
                    respKeyFirst = respKey[0][0].lower()
                    respRT = respKey[0][1]-t0
                    waitTime = oddITIDur-respRT+imgDur
                else:
                    respKeyFirst = None
                    respRT = 0
                    waitTime = 0
                core.wait(waitTime)
                
                # storing response data
                oddRespList.append(respKeyFirst)
                oddRTList.append(respRT)
                oddT0List.append(t0)
                
                if (respKeyFirst==None) and (oddTaskDf.loc[oddN,'cate']!='num'):
                    oddTaskDf.loc[oddN,'acc'] = 1
                    oddIgnACC +=1
                    # mark
                    sendMarkCode(respMk_odd)
                elif (respKeyFirst!=None) and (oddTaskDf.loc[oddN,'cate']=='num'):
                    oddTaskDf.loc[oddN,'acc'] = 1
                    oddNumACC +=1
                    # mark
                    sendMarkCode(respMk_odd)
                else:
                    # mark
                    sendMarkCode(delMk_odd)
            #block end
            
            oddTaskDf['rt'] = oddRTList
            oddTaskDf['resp'] = oddRespList
            oddTaskDf['onsetT'] = oddT0List
            oddTaskDf['setsize'] = [setsize]*len(oddTaskDf)
            oddTaskDf['subj'] = [subj]*len(oddTaskDf)
            oddTaskDf['age'] = [age]*len(oddTaskDf)
            oddTaskDf['sex'] = [sex]*len(oddTaskDf)
            oddTaskDf['handed'] = [handed]*len(oddTaskDf)
            
            # storing oddball task data
            if (subj==1) & (indxNum==2):
                headTag = True
                modeTag = 'w'
            else:
                headTag = False
                modeTag = 'a'
            oddTaskDf.to_csv(filePath+'Data/DataBehav/odd.csv',sep=',',header=headTag,\
            index=False,mode=modeTag)
            
            oddNumACC_num = round(oddNumACC/(oddNumN)*100,2)
            oddIgnACC_num = round(oddIgnACC/(oddTrialN-oddNumN)*100,2)
            oddNumACC_text = str(oddNumACC_num)+'%'
            oddIgnACC_text = str(oddIgnACC_num)+'%'
            showOddACC_text1 = 'You have correctly decided '+oddNumACC_text+' of numbers.'
            showOddACC_text2 = 'You have correctly ignored '+oddIgnACC_text+' of images.'
            oddACC_text1 = visual.TextStim(win,text=showOddACC_text1,height=fontDeg,
            units='deg',pos=(0,7),color='black',bold=False,italic=False)
            oddACC_text2 = visual.TextStim(win,text=showOddACC_text2,height=fontDeg,
            units='deg',pos=(0.0,3.5),color='black',bold=False,italic=False)
            break_img = visual.ImageStim(win,image=filePath+'Instructions/breakITI.png',\
            pos=(0.0, 0.0))
            
            oddACC_text1.draw()
            oddACC_text2.draw()
            break_img.draw()
            win.flip()
            continKey = event.waitKeys(maxWait=180,keyList=[startKey,quitKey])
            if continKey != None:
                checkEsc(continKey[0])

gby = visual.ImageStim(win,image=filePath+'Instructions/gby.png',pos=(0.0, 0.0))
gby.draw()
win.flip()
event.waitKeys(maxWait=oddITIDur,keyList=[startKey],timeStamped=True)

# ####################
# exp. finish
# ####################

win.close()
core.quit()