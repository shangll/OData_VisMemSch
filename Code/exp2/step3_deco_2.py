#!/usr/bin/env python
#-*-coding:utf-8 -*-

# EXP. 4-3-2 (EEG): Decoding
# Temporal dynamics of non-target processing vary with
# memory set size in different category conditions
# 2022.Jan.18
# linlin.shang@donders.ru.nl


from exp4_config import resPath

import numpy as np
import pandas as pd
import pingouin as pg
import random
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
import os


#


#

fileName = 'deco_data_all.csv'

acc_df = pd.read_csv(os.path.join(resPath,fileName),sep=',')
acc_df.columns = ['chans','pred','type','subj','time','acc']
acc_df['cond'] = acc_df['type'].str[0].values
acc_df['setsize'] = acc_df['type'].str[1].values
for t0,t1 in zip([0.2,0.2,0.26],[0.3,0.26,0.32]):
    n2pc_df = acc_df[(acc_df['time']>=t0)&
                     (acc_df['time']<t1)&
                     (acc_df['chans']=='eeg')&
                     (acc_df['subj']!='mean')].reset_index(drop=True)
    print('wt vs bt')
    aov = pg.rm_anova(
        dv='acc',within=['cond','setsize'],subject='subj',
        data=n2pc_df,detailed=True,effsize='np2')
    pg.print_table(aov)
    print('*** *** ***')

n2pc_data = acc_df[(acc_df['time']>=0.2)&
                   (acc_df['time']<0.3)&
                   (acc_df['chans']=='eeg')&
                   (acc_df['subj']!='mean')].reset_index(drop=True)
n2pc_b_w = n2pc_data[n2pc_data['cond']=='b'].reset_index(drop=True)
n2pc_b_w['acc_w'] = n2pc_data.loc[(n2pc_data['cond']=='w'),'acc'].values
n2pc_b_w['diff'] = n2pc_b_w['acc'].values-n2pc_b_w['acc_w'].values

print('wt vs bt')
aov = pg.rm_anova(
    dv='diff',within='setsize',subject='subj',
    data=n2pc_b_w,detailed=True,effsize='np2')
pg.print_table(aov)
print('*** *** ***')