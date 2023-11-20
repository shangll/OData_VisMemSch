#!/usr/bin/env python
#-*-coding:utf-8 -*-
import pandas as pd

# EXP.4 :
# 2023.01.19
# linlin.shang@donders.ru.nl

from exp4_config import filePath,set_filepath
import os


rootPath = set_filepath(filePath,'Stimuli')
for dir in os.listdir(path=rootPath):
    dirPath = os.path.join(rootPath,dir)

    for subDir in os.listdir(path=dirPath):
        subDirPath = os.path.join(dirPath,subDir)

        imgList = []
        for img in os.listdir(path=subDirPath):
            if img!='Thumbs.db':
                imgPath = os.path.join(subDirPath,img)
                imgList.append(imgPath)

        fileN = len(imgList)
        fileDict = {'cate':[dir]*fileN,
                    'subCate':[subDir]*fileN,
                    'stimulus':imgList}
        subCateFile = pd.DataFrame(fileDict)
        fileName = 'StimList/'+subDir+'.csv'
        subCateFile.to_csv(
            fileName,sep=',',header=True,index=False,mode='w')



# allFiles = os.walk(imgPath,topdown=False)
# for dirPath,dirNames,fileNames in allFiles:
#     for dirName in dirNames:
#         print(dirName)
#     for fileName in fileNames:
#         print(os.path.join(dirPath,fileName))