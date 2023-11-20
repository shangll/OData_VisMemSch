#!/usr/bin/env python
#-*-coding:utf-8 -*-

# EXP. 3 (Behavioural):
# Temporal dynamics of non-target processing vary with
# memory set size in different category conditions
# 2021.Nov.24
# linlin.shang@donders.ru.nl

from eeg_config import resPath,outliers,sizeList,\
    cateList,condList,p_crit,set_filepath,save_fig

import os
from math import log
import random
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from sklearn.linear_model import LinearRegression
import pingouin as pg
import statannot

from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = "all"

model = LinearRegression()



# --- --- --- Set Global Parameters --- --- --- #

trialType = ['targ','within','between']
crit_rt = 0.2
crit_sd = 3
crit_acc = 0.70
blockN = 8
recogTrialN = 120
oddTrialN = 110
colStyle = ['sandybrown','darkseagreen']
dotStyle = ['^','o']



# --- --- ---

def AIC(y_test, y_pred, k, n):
    resid = y_test - y_pred
    SSR = sum(resid ** 2)
    AICValue = 2*k+n*log(float(SSR)/n)
    return AICValue



# --- --- --- 1. Read Files --- --- ---


expResPath = set_filepath('U:\Documents\DCC\exp3','AllExpRes')
figPath = set_filepath(expResPath,'Figs','behav')
filePath = 'U:\Documents\DCC'
#
# exp_tag = 'exp1a'
# subjAllN = 41
# subjList = [n for n in range(1,subjAllN+1)]

# exp_tag = 'exp1b'
# subjAllN = 31
# subjList = [n for n in range(1,subjAllN+1)]

exp_tag = 'exp2'
subjAllN = 33
subjList = [n for n in range(1,subjAllN+1)]
#
# exp_tag = 'exp3'
# subjAllN = 32
# subjList = [n for n in range(1,subjAllN+1)]

if exp_tag!='exp3':
    if exp_tag=='exp1a':
        # title = 'Experiment 1a: LTM (Target-Absent Trials: 80%)'
        title = 'Experiment 1'
        sizeList = [1,2,4,8,16]
    elif exp_tag=='exp1b':
        title = 'Experiment 1b: LTM (Target-Absent Trials: 50%)'
    else:
        title = 'Experiment 2: STM (Target-Absent Trials: 50%)'
    expFile = filePath+'\%s\Results\ExpData\%s_Raw.csv'%(exp_tag,exp_tag)
else:
    expFile = filePath+'\%s\Data\DataBehav\sch.csv'%(exp_tag)
    # title = 'Experiment 3: LTM (Target-Absent Trials: 80%)'
    title = 'Experiment 2'
    # trigger_lag = 0.002

df_sch_raw = pd.read_csv(expFile,sep=',')
df_sch_raw = df_sch_raw[
    df_sch_raw['cond'].isin(condList)].reset_index(drop=True)

pd.set_option('display.max_rows',None)
pd.set_option('display.max_columns',None)

subj_info = pd.DataFrame()
age_list,gender_list,hand_list = [],[],[]
for n in subjList:
    df_subj = df_sch_raw[
        df_sch_raw['subj']==n].copy(deep=True).reset_index(drop=True)
    age_list.append(df_subj.loc[0,'age'])
    gender_list.append(df_subj.loc[0,'gender'])
    hand_list.append(df_subj.loc[0,'handed'])

print('*** *** *** *** *** ***')
print('averge age: %f, sd: %f, range: %d ~ %d'
      %(np.mean(age_list),np.std(age_list),
        min(age_list),max(age_list)))
male_list = list(filter(lambda x:x=='M',gender_list))
print('male: %d, female: %d'%(
    len(male_list),len(gender_list)-len(male_list)))
r_list = list(filter(lambda x:x=='R',hand_list))
print('right-handed: %d'%len(r_list))
print('*** *** *** *** *** ***')

# --- --- --- 2. Exclusion --- --- ---

incorr_points = len(df_sch_raw[df_sch_raw['acc']==0])
df_sch_clean = df_sch_raw.copy(deep=True)

# 2.1 <0.2 sec
df_sch_clean.loc[df_sch_clean['acc']==1,'acc'] = \
    np.where((df_sch_clean.loc[(df_sch_clean['acc']==1),'rt']<crit_rt),
             0,1)
# 2.2 ±3 sd
outRTs = df_sch_clean[df_sch_clean['acc']==1].copy(deep=True).groupby(
    ['subj','cond','setsize','block'])['rt'].transform(
    lambda x:stats.zscore(x))
df_sch_clean.loc[np.where(np.abs(outRTs)>crit_sd)[0],'acc'] = 0


# --- --- ---
out_subjs = []

if exp_tag=='exp3':
    # subjList_final = [n for n in subjList if n not in outliers]
    # out_subjs += outliers
    subjList_final = subjList
    subjAllN_final = len(subjList_final)
    '''
    oddFile = filePath+'\exp3\Data\DataBehav\odd.csv'
    df_odd_raw = pd.read_csv(oddFile,sep=',')
    # deleted index for EEG
    df_del = pd.DataFrame()
    for n in subjList_final:
        for k in range(int(blockN/2)):
            df_sch_subj = df_sch_del.loc[
                (df_sch_del['subj']==n),['subj','acc']].copy()
            df_odd_subj = df_odd_raw.loc[
                (df_odd_raw['subj']==n),['subj','acc']].copy()
            df_sch_subj.reset_index(drop=True,inplace=True)
            df_odd_subj.reset_index(drop=True,inplace=True)

            if k!=3:
                df_del = pd.concat(
                    [df_del,df_sch_subj[recogTrialN*k:recogTrialN*k+recogTrialN],
                     df_odd_subj[oddTrialN*k:oddTrialN*k+oddTrialN]],axis=0,
                    ignore_index=True)
            else:
                df_del = pd.concat(
                    [df_del,df_sch_subj[recogTrialN*k:recogTrialN*k+recogTrialN]],
                    axis=0,ignore_index=True)
    df_del['del_indx'] = list(range(810))*subjAllN
    del_indx = df_del[df_del['acc']==0]
    del_indx.reset_index(drop=True,inplace=True)
    del del_indx['acc']
    
    del_indx.to_csv(os.path.join(resPath,'del_indx.csv'),
                    mode='w',header=True,index=False)
    '''
# --- --- ---

df_sch_del = df_sch_clean.copy(deep=True)
del_acc_indx = df_sch_del[df_sch_del['acc']==0].index.tolist()
df_sch_del.drop(del_acc_indx,axis=0,inplace=True)

df_sch_mean = df_sch_clean.groupby(
    ['subj','setsize','cond'])[
    ['rt','acc']].agg(np.mean).reset_index()

df_sch_mean_del = df_sch_del.groupby(
    ['subj','setsize','cond'])[
    ['rt','acc']].agg(np.mean).reset_index()
df_sch_mean['rt'] = df_sch_mean_del['rt']

# 2.3 subj outliers
# acc
if exp_tag!='exp2':
    acc_check = df_sch_mean.groupby(
        ['subj','setsize','cond'])['acc'].agg(np.mean).reset_index()
else:
    acc_check = df_sch_mean.groupby(
        ['subj','setsize'])['acc'].agg(np.mean).reset_index()
# acc_check = df_sch_mean.groupby(
#     ['subj'])['acc'].agg(np.mean).reset_index()
if exp_tag=='exp2':
    out_subjs_acc = list(set(
        acc_check.loc[acc_check['acc']<0.65,'subj'].tolist()))
    out_subjs_acc += list(set(
        acc_check.loc[(acc_check['acc']<crit_acc)&
                      (acc_check['setsize']!=8),'subj'].tolist()))
else:
    out_subjs_acc = list(set(
        acc_check.loc[acc_check['acc']<crit_acc,'subj'].tolist()))
print('ourliers (ACC):',out_subjs_acc)
# ±3 sd
outRTs_mean = df_sch_mean.copy(deep=True).groupby(
    ['subj','cond','setsize'])['rt'].transform(
    lambda x:stats.zscore(x))
out_subjs_rt = list(set(
    df_sch_mean.loc[np.where(np.abs(outRTs_mean)>crit_sd)[0],
                    'subj'].tolist()))
print('ourliers (RT):',out_subjs_rt)
out_subjs += out_subjs_acc+out_subjs_rt
out_subjs = list(set(out_subjs))
print('ourliers (ACC+RT):',out_subjs)

out_points = len(df_sch_clean[(df_sch_clean['acc']==1)
                            &(df_sch_clean['subj'].isin(out_subjs))])
del_points = len(df_sch_clean[df_sch_clean['acc']==0])-incorr_points
print('RT: delete %.3f%% data points'%(del_points/len(df_sch_raw)*100))
del_points = len(df_sch_clean[df_sch_clean['acc']==0])-incorr_points+out_points
print('Totally delete %.3f%% data points'%(del_points/len(df_sch_raw)*100))


# if exp_tag=='exp1a':
#     df_acc = df_sch_raw[df_sch_raw['subj']!=33].groupby(
#         ['subj','setsize','cond','block'])['acc'].agg(np.mean).reset_index()
#     df_acc_rt = df_sch_del[df_sch_del['subj']!=33].groupby(
#         ['subj','setsize','cond'])[
#         ['rt']].agg(np.mean).reset_index()
#     df_acc['rt'] = df_acc_rt['rt']
#     df_acc.rename(columns={'block':'group'},inplace=True)
#     exp1a_data_path = os.path.join(expResPath,'data_exp1_blk_mean.csv')
#     df_acc.to_csv(exp1a_data_path,mode='w',header=True,index=False)


# --- --- --- 3. Descriptive Statistic --- --- ---

if exp_tag!='exp3':
    subjList_final = [n for n in subjList if n not in out_subjs]
    subjAllN_final = len(subjList_final)
df_sch_mean_del = df_sch_mean[df_sch_mean['subj'].isin(subjList_final)]
df_sch_del = df_sch_del[df_sch_del['subj'].isin(subjList_final)]
df_sch_del = df_sch_del[['subj','age','gender','handed',
                         'block','cond','setsize',
                         'rt','resp','acc']]
df_sch_del['exp'] = [exp_tag]*len(df_sch_del)
clean_data_path = os.path.join(expResPath,'data_allExp.csv')
# if os.path.isfile(clean_data_path):
#     df_sch_del.to_csv(
#         clean_data_path,mode='a',header=False,index=False)
# else:
#     df_sch_del.to_csv(
#         clean_data_path,mode='w',header=True,index=False)


# ACC & RT
for y in ['acc','rt']:
    mpl.rcParams.update({'font.size':18})
    fig,ax = plt.subplots(figsize=(12,9))
    # # sns.set_style("whitegrid")
    # sns.violinplot(x='setsize',y=y,data=df_sch_mean_del,
    #                hue='cond',palette=colStyle,
    #                hue_order=condList,split=True,inner='quartile',
    #                saturation=0.7)
    # sns.stripplot(x='setsize',y=y,data=df_sch_mean_del,
    #               hue='cond',jitter=True,dodge=True,
    #               palette=colStyle,hue_order=condList,alpha=1)
    sns.barplot(data=df_sch_mean_del,x='setsize',y=y,
                hue='cond',hue_order=['within','between'],
                palette='Blues',saturation=0.75,
                errorbar='se',capsize=0.1,errcolor='grey')
    # plt.grid(linestyle=':')
    plt.title('Mean %ss in %s' %(y.upper(),title))
    if y=='acc':
        # plt.ylim(0.7,1.1)
        plt.ylabel('ACC')
    else:
        # plt.ylim(0.1,1.0)
        ax.set_ylim(0,0.7)
        plt.ylabel('RT (sec)')
    plt.xlabel('Memory Set Size')
    # plt.xticks(sizeList)
    # plt.grid(linestyle=':')
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.legend(loc='best',ncol=2,fontsize=15)
    figName = os.path.join(figPath,'descr_%s_%s')%(y,exp_tag)
    save_fig(fig,figName)

# mean data
mean_data = df_sch_clean.groupby(
    ['setsize','cond'])[['rt','acc']].agg(np.mean).reset_index()
mean_data_del = df_sch_del.groupby(
    ['setsize','cond'])[['rt']].agg(np.mean).reset_index()
mean_data['rt'] = mean_data_del['rt']

# print('*** *** *** *** *** ***')
# mean_data['rt'] = mean_data['rt'].apply(lambda x:'%.3f'%x)
# mean_data['acc'] = mean_data['acc'].apply(lambda x:'%.3f'%x)
# print(mean_data)
# print('')

mean_blk = df_sch_clean.groupby(
    ['block','setsize','cond'])[['rt','acc']].agg(np.mean).reset_index()
mean_blk_del = df_sch_del.groupby(
    ['block','setsize','cond'])[['rt']].agg(np.mean).reset_index()
mean_blk['rt'] = mean_blk_del['rt']
print('*** *** *** *** *** ***')
mean_blk['rt'] = mean_blk['rt'].apply(lambda x:'%.3f'%x)
mean_blk['acc'] = mean_blk['acc'].apply(lambda x:'%.3f'%x)
print(mean_blk)
print('')

# --- --- --- 4. Statistics --- --- ---

# ANOVA: interaction
for y in ['acc','rt']:
    aov = pg.rm_anova(dv=y,within=['setsize','cond'],subject='subj',
                      data=df_sch_mean_del,detailed=True,effsize="np2")
    print('*** *** *** *** *** ***')
    print('ANOVA: interaction')
    print(y)
    pg.print_table(aov)

    if aov.loc[
        aov['Source']=='setsize * cond',
        'p-GG-corr'].values<p_crit:
        mean_table = df_sch_mean_del.groupby(
            ['cond','setsize'])[y].agg(
            ['mean','std']).round(3)
        print(mean_table)
    print('')

    # print('*** *** *** *** *** ***')
    # pwc1 = pg.pairwise_tests(dv=y,within=['setsize','cond'],
    #                          subject='subj',data=df_sch_mean_del)
    # pwc2 = pg.pairwise_tests(dv=y,within=['cond','setsize'],
    #                          subject='subj',data=df_sch_mean_del)
    # pg.print_table(pwc1)
    # print('*** *** *** *** *** ***')
    # pg.print_table(pwc2)

# log-linear fit
dfCoeff = pd.DataFrame(
    {'subj':[n for n in subjList_final for k in range(2)],
     'cond':condList*subjAllN_final})
lm_aic,log_aic = {'within':[],'between':[]},{'within':[],'between':[]}
for n in subjList_final:

    for cond in condList:
        df_train = df_sch_mean_del[
            (df_sch_mean_del['subj']==n)&
            (df_sch_mean_del['cond']==cond)&
            (df_sch_mean_del['setsize']!=sizeList[-1])].copy(
            deep=True).reset_index(drop=True)
        df_test = df_sch_mean_del[
            (df_sch_mean_del['subj']==n)&
            (df_sch_mean_del['cond']==cond)].copy(
            deep=True).reset_index(drop=True)

        # linear
        x = df_train['setsize'].values
        y = list(df_train['rt'].values)
        model = sm.OLS(y,sm.add_constant(x)).fit()
        pred_value = df_test['setsize'].values
        pred_res = model.predict(sm.add_constant(pred_value))
        '''
        model.fit(x.reshape(-1,1),y)
        pred_value = df_test['setsize'].values
        pred_res = model.predict(pred_value.reshape(-1,1))
        '''
        df_sch_mean_del = df_sch_mean_del.copy()
        df_sch_mean_del.loc[
            (df_sch_mean_del['subj']==n)&
            (df_sch_mean_del['cond']==cond),'lm'] = pred_res
        lm_aic[cond].append(model.aic)

        # log2
        x = df_train['setsize'].apply(np.log2).values
        y = list(df_train['rt'].values)
        model = sm.OLS(y,sm.add_constant(x)).fit()
        pred_value = df_test['setsize'].apply(np.log2).values
        pred_res = model.predict(sm.add_constant(pred_value))
        '''
        model.fit(x.reshape(-1,1),y)
        pred_value = df_test['setsize'].apply(np.log2).values
        pred_res = model.predict(pred_value.reshape(-1,1))
        '''
        df_sch_mean_del.loc[
            (df_sch_mean_del['subj']==n)&
            (df_sch_mean_del['cond']==cond),'log'] = pred_res
        log_aic[cond].append(model.aic)

        # get slope coefficients
        y_val = df_sch_mean_del.loc[
            (df_sch_mean_del['subj']==n)&
            (df_sch_mean_del['cond']==cond),'rt'].values
        model = sm.OLS(
            y_val,sm.add_constant(np.log2(sizeList))).fit()
        dfCoeff.loc[
            (dfCoeff['subj']==n)&
            (dfCoeff['cond']==cond),'coeff'] = model.params[1]
        dfCoeff.loc[(dfCoeff['subj']==n)&
                    (dfCoeff['cond']==cond),'r2'] = model.rsquared_adj
        '''
        model.fit(np.log2(sizeList).reshape(-1,1),y_val)
        dfCoeff.loc[
            (dfCoeff['subj']==n)&
            (dfCoeff['cond']==cond),'coeff'] = model.coef_
        dfCoeff.loc[(dfCoeff['subj']==n)&
            (dfCoeff['cond']==cond),'r2'] = model.score(
            np.log2(sizeList).reshape(-1,1),y_val)
        '''

for cond in condList:
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


for cond in condList:
    print('*** *** *** *** *** ***')
    print('test observed data vs log vs linear')
    t_val = pg.ttest(df_sch_mean_del.loc[(df_sch_mean_del['setsize']==sizeList[-1])&
                                         (df_sch_mean_del['cond']==cond),'lm'],
                     df_sch_mean_del.loc[(df_sch_mean_del['setsize']==sizeList[-1])&
                                         (df_sch_mean_del['cond']==cond),'rt'],
                     paired=False,alternative='greater',correction='auto')

    print(cond)
    print('observed data vs linear')
    pg.print_table(t_val)

    t_val = pg.ttest(df_sch_mean_del.loc[(df_sch_mean_del['setsize']==sizeList[-1])&
                                         (df_sch_mean_del['cond']==cond),'log'],
                     df_sch_mean_del.loc[(df_sch_mean_del['setsize']==sizeList[-1])&
                                         (df_sch_mean_del['cond']==cond),'rt'],
                     paired=False,alternative='two-sided',correction='auto')
    print('*** *** *** *** *** ***')
    print(cond)
    print('observed data vs log2')
    pg.print_table(t_val)
    print('*** *** *** *** *** ***')
    print(cond)
    print('observed data vs log2 vs linear')
    print(df_sch_mean_del.loc[(df_sch_mean_del['setsize']==sizeList[-1])&
                                         (df_sch_mean_del['cond']==cond),'rt'].mean(),
          df_sch_mean_del.loc[(df_sch_mean_del['setsize']==sizeList[-1])&
                              (df_sch_mean_del['cond']==cond),'log'].mean(),
          df_sch_mean_del.loc[(df_sch_mean_del['setsize']==sizeList[-1])&
                              (df_sch_mean_del['cond']==cond),'lm'].mean())
    print('')

print('*** *** *** *** *** ***')
print('test slope coefficients')
t_val = pg.ttest(dfCoeff.loc[(dfCoeff['cond']=='within'),'coeff'],
                 dfCoeff.loc[(dfCoeff['cond']=='between'),'coeff'],
                 paired=True,alternative='two-sided',
                 correction='auto')
print('within vs between')
pg.print_table(t_val)
print('')


df_sch_mean_del['exp'] = [exp_tag]*len(df_sch_mean_del)
mean_data_path = os.path.join(
    expResPath,'data_allExp_mean.csv')

'''
# save mean data file
if os.path.isfile(mean_data_path):
    df_sch_mean_del.to_csv(
        mean_data_path,mode='a',header=False,index=False)
else:
    df_sch_mean_del.to_csv(
        mean_data_path,mode='w',header=True,index=False)
'''

mean_data_pred = df_sch_mean_del.groupby(
    ['setsize','cond'])[['lm','log']].agg(np.mean).reset_index()
lm_df = mean_data_pred[['setsize','cond','lm']]
lm_df.rename(columns={'lm':'rt'},inplace=True)
lm_df['Fit'] = ['linear']*len(lm_df)
log_df = mean_data_pred[['setsize','cond','log']].reset_index(drop=True)
log_df.rename(columns={'log':'rt'},inplace=True)
log_df['Fit'] = ['log2']*len(log_df)
pred_plt = pd.concat([lm_df,log_df],axis=0,ignore_index=True)
pred_plt.rename(columns={'cond':'Category'},inplace=True)
mean_data.rename(columns={'cond':'Category'},inplace=True)
text_labels = []
for cate_tag in condList:
    text_label = mean_data.loc[
        (pred_plt['Category']==cate_tag),'rt'].tolist()[-1]
    text_labels.append(text_label)
    for pred_tag in ['linear','log2']:
        text_label = pred_plt.loc[
            (pred_plt['Category']==cate_tag)&
            (pred_plt['Fit']==pred_tag),'rt'].tolist()[-1]
        text_labels.append(text_label)
if exp_tag=='exp3':
    text_x = 8.35
    text_y = [0.54,0.57,0.53,0.493,0.55,0.503]
elif exp_tag=='exp1a':
    text_x = 16.35
    text_y = [0.602,0.701,0.623,0.49,0.54,0.524]
clrs = ['grey','lightskyblue','lightskyblue',
        'k','dodgerblue','dodgerblue']
# plot
# fit
markers = {'within':'S','between':'X'}
mpl.rcParams.update({'font.size':18})
fig,ax = plt.subplots(figsize=(12,9))
sns.lineplot(data=pred_plt,x='setsize',y='rt',
             hue='Category',hue_order=['within','between'],
             style='Fit',style_order=['log2','linear'],
             markers=True,palette='Blues',ax=ax)
sns.scatterplot(data=mean_data,x='setsize',y='rt',hue='Category',
                style='Category',style_order=['within','between'],
                palette={'within':'grey','between':'black'},
                ax=ax,legend=False)
for label,y,clr in zip(text_labels,text_y,clrs):
    num = round(label,3)
    plt.text(text_x,y,str(num),ha='center',
             va='bottom',color=clr,fontsize=15)

plt.title('%s'%title)
plt.ylim(0.4,0.75)
plt.xlabel('Memory Set Size')
plt.ylabel('RT (sec)')
plt.xticks(sizeList)
# plt.grid(linestyle=':')
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.legend(loc='best',ncol=2,fontsize=15)
figName = os.path.join(figPath,'fit_%s')%(exp_tag)
save_fig(fig,figName)


# labels = 'linear'
# if exp_tag=='exp1a':
#     text_loc = [0.675,0.565]
#     y_gap = 1
# else:
#     y_gap = 0
# for k,cond in enumerate(condList):
#     y = mean_data_pred.loc[mean_data_pred['cond']==cond,'lm'].values
#     plt.plot(sizeList,y,color=colStyle[k],linestyle='--',
#              marker=dotStyle[k],alpha=0.5,label=labels+' ('+cond+')')
#     if (cond=='within') and (exp_tag=='exp3'):
#         plt.text(7.5,0.58,'%.3f'%(y[len(y)-1]),color=colStyle[k])
#     elif (cond=='between') and (exp_tag=='exp3'):
#         plt.text(7.5,0.545,'%.3f'%(y[len(y)-1]),color=colStyle[k])
#     else:
#         plt.text(sizeList[-1]-y_gap,text_loc[k],'%.3f'%(y[len(y)-1]),color=colStyle[k])
# labels = 'log2'
# if exp_tag=='exp1a':
#     text_loc = [0.63,0.53]
# for k,cond in enumerate(condList):
#     y = mean_data_pred.loc[mean_data_pred['cond']==cond,'log'].values
#     plt.plot(sizeList,y,color=colStyle[k],linestyle='-',
#              marker=dotStyle[k],alpha=0.8,label=labels+' ('+cond+')')
#     if (cond=='within') and (exp_tag=='exp3'):
#         plt.text(8.02,0.519,'%.3f'%(y[len(y)-1]),color=colStyle[k])
#     elif (cond=='between') and (exp_tag=='exp3'):
#         plt.text(8.02,0.503,'%.3f'%(y[len(y)-1]),color=colStyle[k])
#     else:
#         plt.text(sizeList[-1]-y_gap,text_loc[k],'%.3f'%(y[len(y)-1]),color=colStyle[k])
# labels = 'observed'
# if exp_tag=='exp1a':
#     text_loc = [0.6,0.49]
# for k,cond in enumerate(condList):
#     y = mean_data.loc[mean_data['cond']==cond,'rt'].values
#     plt.scatter(sizeList,y,marker=dotStyle[k],color='k',label=labels+' ('+cond+')')
#     if (cond=='within') and (exp_tag=='exp3'):
#         plt.text(8.02,0.53,'%.3f'%(y[len(y)-1]),color='k')
#     elif (cond=='between') and (exp_tag=='exp3'):
#         plt.text(8.02,0.485,'%.3f'%(y[len(y)-1]),color='k')
#     else:
#         plt.text(sizeList[-1]-y_gap,text_loc[k],'%.3f'%(y[len(y)-1]),color='k')
# # y_major_locator = plt.MultipleLocator(0.1)
# plt.title('%s: Linear/Log-Linear(log2) Fit'%title)
# plt.ylim(0.4,0.75)
# plt.xlabel('Memory Set Size')
# plt.ylabel('RT (sec)')
# plt.xticks(sizeList)
# plt.grid(linestyle=':')
# ax.spines['right'].set_visible(False)
# ax.spines['top'].set_visible(False)
# ax.legend(loc='best',ncol=3,fontsize=15)
# figName = os.path.join(figPath,'fit_%s')%(exp_tag)
# save_fig(fig,figName)



