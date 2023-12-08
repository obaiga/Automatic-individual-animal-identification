#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 29 01:56:58 2022

modify on 01/17/2023

modify on 03/07/2023 for I_nn satsify condition

@author: obaiga
"""

# In[Packs]
import numpy as np 
from os.path import join
import os
import copy
# import time
from sklearn import metrics
# import warnings
# warnings.filterwarnings("ignore")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
# import random
# dpath = 'C:\\Users\\95316\\code1'
# dpath = 'C:\\Users\\SHF\\code1'
dpath= '/Users/obaiga/Jupyter/Python-Research'
os.chdir(join(dpath,'test'))

import utils

new_db = 'ds_160'

query_flag = 'vsmany_'
fg_flag = 'fg'
data_mof = '_diag'
used_nonzero = False
sq_flag = True

# In[Load]
db_dir = join(dpath,new_db)
hsres = utils.HotspotterRes(db_dir)
hsres.res_dir = join(hsres.db_dir,'results')
hsres.data_dir = join(hsres.res_dir,'data')
utils.CheckDir(hsres.res_dir)
utils.CheckDir(hsres.data_dir)
Lis_Chip2ID,Lis_Img,Lis_chipNo,Lis_ID = hsres.load_info()

cluster_gt,label_gt,nonzero_flag =\
    hsres.load_cluster_gt(query_flag,fg_flag,used_nonzero)
# idxNonConnect = None

nsample = hsres.nsample
ncluster = hsres.k_gt

idxNonConnect,_ = utils.load_idxNonConnect(hsres,query_flag,fg_flag)   

scoreAry = hsres.load_score(('ImgScore_%s%s%s%s.xlsx')%
                              (query_flag[2:],fg_flag,nonzero_flag,data_mof))

scoreAry = copy.copy(hsres.scoreAry)
if sq_flag:
    scoreAry = scoreAry**2
    
# In[Load scorematrix]

if scoreAry is not None:
    gt_Lis_SC,gt_SCavg,gt_T2,gt_Lis_SCcls,gt_SCinfo = \
        utils.SilhouetteScore(hsres,cluster_gt,sq_flag=sq_flag,scoreAry=scoreAry)
    
    centrd_gt,centroid_ssum_gt = utils.centroid(hsres,cluster_gt)
    TWCV_gt = np.sum(centroid_ssum_gt)
    
    print('groundtruth:Avg_SC:%.4f;nonindiv SC:%.4f,T2:%d'%
          (np.mean(gt_Lis_SC),gt_SCavg,gt_T2))  
    
    idx_iso = np.where(gt_SCinfo[:,1] == -1)[0]  
    ans = gt_Lis_SC[idx_iso]
    print('indiv SC:%.4f'%(np.mean(ans)))
    
# In[Load TrueID distribution]
CountInfo = np.array(hsres.CountInfo)
barInfo = []
Lis_Count = []
sumNum = 0
true_k = len(hsres.Lis_ID)
print('true_k=%d'%true_k)
Lis_Ratio = []
for ians in CountInfo:
    if ians[0]>5:
        sumNum += ians[1]
    else:
        barInfo.append(str(ians[0]))
        Lis_Ratio.append(ians[1]/true_k*100)
        Lis_Count.append(ians[1])
Lis_Ratio.append(sumNum/true_k*100)
Lis_Count.append(sumNum)
barInfo.append('6 - 17')

# In[Distribution about TrueID]
'''
Distribution about TrueID
'''
labelsize = 14
fontsize = 12
fig,ax = plt.subplots(figsize=(10,5))
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)


# plt.subplot(1,1,1)
bar_plot = plt.bar(barInfo, Lis_Ratio, color = 'forestgreen',alpha=0.6,edgecolor='black',width=0.5)
# plt.grid(axis='y') 


for idx,rect in enumerate(bar_plot):
    height = rect.get_height()
    plt.text(rect.get_x() + rect.get_width()/2., 1*height,
            np.round(Lis_Ratio[idx],1),
            ha='center', va='bottom', rotation=0,fontsize=fontsize)
    
plt.text(3,50,'total images: %d'%nsample,fontsize=labelsize)
plt.text(3,46,'leopards: %d'%hsres.k_gt,fontsize=labelsize)
plt.ylabel('percent of total leopard IDs',fontsize=labelsize)
plt.xlabel('number of images per true leopard ID (individual leopard)',fontsize=labelsize)
plt.xticks(fontsize=fontsize)
plt.yticks(fontsize=fontsize)
plt.title(r'Distribution of leopards in the ${Panthera}$ dataset',fontsize=labelsize+2)
ax.yaxis.set_major_formatter(ticker.PercentFormatter(xmax=100))

plt.show()

# In[Plot clustering progress]
'''
Comparing accuracy between adaptive k++ and k++ (before verify)
'''
if 1:
    times = 20000
    record = 8002
    
    k = ncluster
    name = '%d_%d_%s_%s_k%d.npz'\
        %(record,times,fg_flag,query_flag[2:-1],k)
    data = np.load(join(hsres.data_dir,name),allow_pickle=True)
    print(list(data.keys()))
    
    note = data['note']
    nsample = data['nsample']
    Lis_upditer = data['Lis_upditer']  ### what iterations are updated?
    Lis_numconvg = data['Lis_numconvg']   ### total iterations when convergence
    Lis_SC = data['Lis_SC']
    Lis_RI = data['Lis_RI']
    Lis_FMS = data['Lis_FMS']
    Lis_MI= data['Lis_MI']
   
 
#### In[Distribution]
num = len(Lis_numconvg)
sums = 0
Lis_bstSC = []
Lis_bstmear = []
Lis_bstidx = []
Lis_mear = copy.copy(Lis_MI)
Lis_bstmear = []
for i,num in enumerate(Lis_numconvg):
    
    maxidx = np.argmax(Lis_SC[sums:sums+num])
    Lis_bstSC.append(Lis_SC[maxidx+sums])
    Lis_bstidx.append(maxidx+sums)
    Lis_bstmear.append(Lis_mear[maxidx+sums])
    sums += Lis_numconvg[i]
    

#### In[Load npz (benchmark)]
if 1:
    times = 20000
    record0 = 8001
    ##### benchmark k++, repeat 200*100 executions
    k = ncluster
    
    name = '%d_%d_%s_%s_k%d.npz'\
        %(record0,times,fg_flag,query_flag[2:-1],k)
    data0 = np.load(join(hsres.data_dir,name),allow_pickle=True)
    print(list(data0.keys()))
    
    note0 = data0['note']
    nsample = data0['nsample']
    Lis_upditer0 = data0['Lis_upditer']  ### what iterations are updated?
    Lis_numconvg0 = data0['Lis_numconvg']   ### total iterations when convergence
    Lis_SC0 = data0['Lis_SC']
    Lis_RI0 = data0['Lis_RI']
    Lis_FMS0 = data0['Lis_FMS']
    Lis_MI0= data0['Lis_MI']
    
#### In[Distribution (benchmark)]
num = len(Lis_numconvg0)
sums = 0
Lis_bstSC0 = []
Lis_bstmear0 = []
Lis_bstidx0 = []
Lis_mear0 = copy.copy(Lis_MI0)
Lis_bstmear0 = []
for i,num in enumerate(Lis_numconvg0):
    maxidx = np.argmax(Lis_SC0[sums:sums+num])
    Lis_bstSC0.append(Lis_SC0[maxidx+sums])
    Lis_bstidx0.append(maxidx+sums)
    Lis_bstmear0.append(Lis_mear0[maxidx+sums])
    sums += Lis_numconvg0[i]

#### In[Plot clustering progress]
'''
Plot clustering progress for adaptive k++
'''
labelsize = 14
fontsize = 12

##### the middle one 
Lis_bstSC_sort = np.argsort(Lis_bstSC)
cache =int(len(Lis_bstSC)/2)
idx = Lis_bstSC_sort[cache+1]
# idx = np.argmax(Lis_bstSC)
# Lis_bstSC_sort0 = np.argsort(Lis_bstSC0)
# i = Lis_bstSC_sort0[int(len(Lis_bstSC0)/2)]

##### the best one 
# idx0 = np.argmax(Lis_bstSC0)
Lis_bstSC0_sort = np.argsort(Lis_bstSC0)
idx0 = Lis_bstSC0_sort[int(len(Lis_bstSC0)/2)]


fig,ax1 = plt.subplots(figsize=[10,5])
ax1.spines['top'].set_visible(False)
ax1.spines['right'].set_visible(False)

Lis_idx = [np.sum(Lis_numconvg[:idx]),np.sum(Lis_numconvg[:idx+1])]
maxidx = Lis_bstidx[idx]
x = np.arange(Lis_numconvg[idx])
y = Lis_SC[Lis_idx[0]:Lis_idx[1]]
y_mear = Lis_MI[Lis_idx[0]:Lis_idx[1]]


plt.plot(x,y_mear,'o',label = r'adjusted mutual info',c='forestgreen')
plt.plot(x,y,'o',label = r'mean silhouette score, $\bar{s}$',c='darkorange')

bstidx = np.argmax(y)
plt.plot([0,x[-1]],np.ones(2)*y[bstidx],'--',c='r')
plt.plot([0,x[-1]],np.ones(2)*y_mear[bstidx],'--',c='r')
plt.plot(bstidx,y[bstidx],'o',mfc='none',ms=15,color='r')
plt.plot(bstidx,y_mear[bstidx],'o',mfc='none',ms=15,color='r')
cont = r'$\mathcal{C}^*$ for adaptive $k$++'
plt.text(bstidx,y[bstidx]+0.01,cont,fontsize=labelsize)
plt.text(bstidx,y_mear[bstidx]+0.008,cont,fontsize=labelsize)

y0 = Lis_bstSC0[idx0]
y0_mear = Lis_bstmear0[idx0]

plt.plot([0,x[-1]],np.ones(2)*y0,'--',c='blue')
plt.plot([0,x[-1]],np.ones(2)*y0_mear,'--',c='blue')

cont = r'$\mathcal{C}^*$ for $k$++'
plt.text(22,y0_mear-0.02,cont,fontsize=labelsize)
plt.text(22,y0-0.02,cont,fontsize=labelsize)

plt.xlabel('iteration',fontsize=labelsize)
plt.ylabel('measures',fontsize=labelsize)
plt.legend(loc='lower right',fontsize=labelsize,
           fancybox=True, framealpha=0.5)
plt.xticks(fontsize=fontsize)
plt.yticks(fontsize=fontsize)
plt.xlim([-1,x[-1]+1])
plt.ylim([np.min(y)-0.05,0.915])
cont = r'Learning progress for adaptive $k$++ ($k=K=%d$)'%ncluster
plt.title(cont,fontsize=labelsize+2 )

# In[Plot distribution 100 repeated (before & after verify)]

'''
Load verification step 
'''
times = 20000
record = 8006  ##### method 0, adaptive k++, fixed k value
record0 = 8005  ##### method 0, k++ (benchmark)

name = '%d_%d_verify.npz'%(record,times)
data = np.load(join(hsres.data_dir,name),allow_pickle=True)

print(data['note'])
# print(list(data.keys()))

Lis_SC_upd = data['Lis_SC_upd']
Lis_SC_upd = np.mean(Lis_SC_upd,axis=1)
Lis_SC_bst = data['Lis_SC_bst']
Lis_SC_bst = np.mean(Lis_SC_bst,axis=1)
Lis_mear_upd = data['Lis_MI_upd']
Lis_mear_bst = data['Lis_MI_bst']
Lis_ncluster_bst = data['Lis_ncluster_bst']
Lis_ncluster_upd = data['Lis_ncluster_upd']

name = '%d_%d_verify.npz'%(record0,times)
data0 = np.load(join(hsres.data_dir,name),allow_pickle=True)

Lis_SC_upd0 = data0['Lis_SC_upd']
Lis_SC_upd0 = np.mean(Lis_SC_upd0,axis=1)
Lis_SC_bst0 = data0['Lis_SC_bst']
Lis_SC_bst0 = np.mean(Lis_SC_bst0,axis=1)
Lis_mear_upd0 = data0['Lis_MI_upd']
Lis_mear_bst0 = data0['Lis_MI_bst']
Lis_ncluster_bst0 = data0['Lis_ncluster_bst']
Lis_ncluster_upd0 = data0['Lis_ncluster_upd']


'''plot twins figure left: silhouette right: measurement, like adjusted mutual info'''

meas_name = 'adjusted MI'
labelsize = 14
fontsize = 12


colorLis = ['#2ca02c', '#ff7f0e', '#1f77b4', '#e377c2',
            '#8c564b','#17becf','#bcbd22','#d62728',  
            '#9467bd', '#7f7f7f',]

fig,(ax1,ax2) = plt.subplots(1,2,figsize=[10,5])
# fig,ax2 = plt.subplots(figsize=[10,5])
# ax2.set_xlabel('mutual     info',fontsize=labelsize,loc='bottom')
# ax1.set_xlabel('adjusted ',fontsize=labelsize,loc='top')
# hide the spines between ax and ax2
ax1.spines['top'].set_visible(False)
ax1.spines['right'].set_visible(False)

ax2.spines['left'].set_visible(False)
ax2.spines['top'].set_visible(False)
ax2.spines['right'].set_visible(False)
ax2.get_yaxis().set_visible(False)

d = .5  # proportion of vertical to horizontal extent of the slanted line
kwargs = dict(marker=[(-d, -1), (d, 1)], markersize=12,
              linestyle="none", color='k', mec='k', mew=1, clip_on=False)
ax1.plot([1,], [0, ], transform=ax1.transAxes, **kwargs)
ax2.plot([0,], [0, ], transform=ax2.transAxes, **kwargs)

# ax2.text(0.89,-0.025,r'adjusted mutual info',fontsize=labelsize)
# ax2.set_xticks(np.arange(0.8,0.9,0.03))

##### measurement
y0 = copy.copy(Lis_mear_bst0)
weights0 = np.ones(len(y0)) / len(y0)
n0, bins0, patches0 = ax1.hist(y0,bins=10, weights=weights0,alpha=0.6,edgecolor='black', \
                              hatch='-',rwidth=0.85,color=colorLis[0],label=r'$k$++')
n0, bins0, patches0 = ax2.hist(y0,bins=10, weights=weights0,alpha=0.6,edgecolor='black', \
                          hatch='-',rwidth=0.85,color=colorLis[0],label=r'$k$++')


y = copy.copy(Lis_mear_bst)
weights1 = np.ones(len(y)) / len(y)
n1, bins1, patches1 = ax1.hist(y, bins=10,weights=weights1,alpha=0.6, edgecolor='black',\
                              hatch='\\',rwidth=0.85,color=colorLis[1],label=r'adaptive $k$++')
n1, bins1, patches1 = ax2.hist(y, bins=10,weights=weights1,alpha=0.6, edgecolor='black',\
                          hatch='\\',rwidth=0.85,color=colorLis[1],label=r'adaptive $k$++')

y0 = copy.copy(Lis_mear_upd0)
weights0 = np.ones(len(y0)) / len(y0)
n0, bins0, patches0 = ax1.hist(y0,bins=10, weights=weights0,alpha=0.6,edgecolor='black', \
                              hatch='.',rwidth=0.85,color=colorLis[2],label=r'$k$++ & verification')
n0, bins0, patches0 = ax2.hist(y0,bins=10, weights=weights0,alpha=0.6,edgecolor='black', \
                              hatch='.',rwidth=0.85,color=colorLis[2],label=r'$k$++ & verification')

    
y = copy.copy(Lis_mear_upd)
weights1 = np.ones(len(y)) / len(y)
n1, bins1, patches1 = ax1.hist(y, bins=10,weights=weights1,alpha=0.6, edgecolor='black',\
                              hatch='/',rwidth=0.85,color=colorLis[7],label=r'adaptive $k$++ & verification')
n1, bins1, patches1 = ax2.hist(y, bins=10,weights=weights1,alpha=0.6, edgecolor='black',\
                          hatch='/',rwidth=0.85,color=colorLis[7],label=r'adaptive $k$++ & verification')

plt.ylim(0,0.3)
ax1.set_xlim([0.825,0.885])
ax2.set_xlim([0.943,0.97])
ax1.set_xticks(np.arange(0.83,0.885,0.015))
ax2.set_xticks(np.arange(0.945,0.97,0.01))
ax1.tick_params(axis='both', labelsize=fontsize)
ax2.tick_params(axis='both', labelsize=fontsize)

#get handles and labels
handles, labels = plt.gca().get_legend_handles_labels()

#specify order of items in legend
order = [0,1,2,3]
###add legend to plot
ax2.legend([handles[jj] for jj in order],[labels[jj] for jj in order],\
            loc='upper center',fontsize=fontsize,fancybox=True,framealpha=0.5,
             bbox_to_anchor=(-0.2, 0.425, 0.55, 0.6)) 
    
ax1.yaxis.set_major_formatter(ticker.PercentFormatter(xmax=1,decimals=0))

# plt.xticks(fontsize=fontsize)
ax1.set_ylim(0,0.3)
ax1.set_yticks(np.arange(0,0.35,0.05))
ax1.set_ylabel('percent of total executions',fontsize=labelsize)
# ax2.set_xlabel('mutual     info',fontsize=labelsize,loc='left')
# ax1.set_xlabel('adjusted ',fontsize=labelsize,loc='right')
content =r'Distribution of clustering accuracies from 100 executions'
plt.suptitle(content,fontsize=labelsize+2)
plt.text(0.93, -0.04, "adjusted mutual info",fontsize=labelsize)







# In[Distribution about correct-clusters (before & after verify)]
 
'''
Percent of correct clusters for different cluster sizes

modify on 01/18/23, changed variable names

'''
k = ncluster

colorLis = ['#2ca02c', '#ff7f0e', '#1f77b4', '#e377c2',
            '#8c564b','#17becf','#bcbd22','#d62728',  
            '#9467bd', '#7f7f7f',]

Lis_pos = [-0.2,0.2,0,0.2]
# Lis_pos = [-0.2,0.2,0,0.4]
Lis_color = [colorLis[0],colorLis[2],colorLis[1],colorLis[7]]
Lis_hatch = ['-','.','\\','/']
Lis_note = [r'$k$++',r'$k$++ & verification',r'adaptive $k$++',r'adpative $k$++ & verification']
p=0
fig,ax1 = plt.subplots(figsize=[10,6])
ax1.spines['top'].set_visible(False)
ax1.spines['right'].set_visible(False)

labelsize = 14
fontsize = 12

# records = [8005,8006]
# records = [8003,8004]
records = [8001,8002]
# record = 8002  ##### method 0, adaptive k++, fixed k value
# record0 = 8020  ##### method 0, k++ (benchmark)

times = 20000

for j,record in enumerate(records):
    name = '%d_%d_verify.npz'%(record,times)
    data = np.load(join(hsres.data_dir,name),allow_pickle=True)
    Lis_bstSC = np.mean(data['Lis_SC_bst'],axis=1)
    ##### the middle one 
    Lis_bstSC_sort = np.argsort(Lis_bstSC)
    cache = int(len(Lis_bstSC)/2)
    idx = Lis_bstSC_sort[cache+1]
    
    print('the mean silhouette score, before:%.3f, non:%.3f \n after:%.3f,non:%.3f'
          %(Lis_bstSC[idx],data['Lis_SCnon_bst'][idx],
            np.mean(data['Lis_SC_upd'][idx,:]),data['Lis_SCnon_upd'][idx]))
    print('adjusted mutual info, before:%.3f, after:%.3f'%(data['Lis_MI_bst'][idx],data['Lis_MI_upd'][idx]))
    for i in [0,1]:
        
        if i == 0:
            # print('before')
            predcls = data['Lis_cls_bst'][idx]
            predcls_pt = data['Lis_clspt_bst'][idx]
            
            
        elif i==1:
            # print('after')
            predcls = data['Lis_cls_upd'][idx]
            predcls_pt = data['Lis_clspt_upd'][idx]
            
        res_fullcorrect,res_partial,res_wrong,_,_,_ =\
            utils.correct_denom_pred(hsres,predcls)
            
        Lis_Count_correct = []
        Lis_Ratio_correct = []
        Lis_Countimg_correct = []
        
        sumNum = 0
        for ii,ians in enumerate(res_fullcorrect):
            if ii>4:
                sumNum += ians
            else:
                Lis_Count_correct.append(ians)
                Lis_Ratio_correct.append(ians/Lis_Count[ii]*100)
                
            Lis_Countimg_correct.append(ians*(ii+1))
            
        Lis_Ratio_correct.append(sumNum/Lis_Count[-1]*100)
        Lis_Count_correct.append(sumNum)
        
        sum_correctID = np.sum(Lis_Count_correct)
        sum_correctImg = np.sum(Lis_Countimg_correct)
        
        print(Lis_Count_correct)
        print('predk:%d, correct num:%d, partial correct num:%d, wrong num:%d'
              %(len(predcls),sum_correctID,len(res_partial),len(res_wrong)))
        print('correct image num:%d,ratio/n=%.3f'%(sum_correctImg,sum_correctImg/nsample))
        
        x = np.arange(1,7)
        if p!=1:
            bar_plot = plt.bar(x+Lis_pos[p], Lis_Ratio_correct,width=0.12,\
                                label=Lis_note[p],alpha=0.6,color=Lis_color[p],\
                                  hatch=Lis_hatch[p], edgecolor='black')
            
            for ii,rect in enumerate(bar_plot):
                height = rect.get_height()
                plt.text(rect.get_x() + rect.get_width()/2., 1*height,
                        int(np.round(Lis_Ratio_correct[ii],0)),
                        ha='center', va='bottom', rotation=0,fontsize=fontsize)
        p += 1
        
plt.xticks(x,barInfo,fontsize=fontsize)
plt.yticks(np.arange(0,95,15),fontsize=fontsize)
plt.ylim([0,102])

plt.ylabel('percent of clusters',fontsize=labelsize)
plt.xlabel('number of images per cluster (predict leopard ID)',fontsize=labelsize)
         
#get handles and labels
handles, labels = plt.gca().get_legend_handles_labels()

# specify order of items in legend
# order = [0,2,1,3]
order = [0,1,2]
#add legend to plot
plt.legend([handles[ii] for ii in order],[labels[ii] for ii in order],\
            loc='lower left',fontsize=labelsize-1,
            bbox_to_anchor=(0, 0.01, 0.55, 0.6)) 
    
content = r'Improvement in correct clusters'
plt.title(content,fontsize=labelsize+2)
ax1.yaxis.set_major_formatter(ticker.PercentFormatter(xmax=100))
plt.show()


# In[Print accuracy]
k = ncluster

records = [8250,8251]
# records = [8220,8202]
# records = [8210]
# records = [8020,8002
# record = 8002  ##### method 0, adaptive k++, fixed k value
# record0 = 8020  ##### method 0, k++ (benchmark)

for j,record in enumerate(records):
    name = '%d_verify.npz'%(record)
    data = np.load(join(hsres.data_dir,name),allow_pickle=True)
    print(data['note'])
    Lis_bstSC = np.mean(data['Lis_SC_bst'],axis=1)
    ##### the middle one 
    Lis_bstSC_sort = np.argsort(Lis_bstSC)
    idx = Lis_bstSC_sort[int(len(Lis_bstSC)/2)]
    
    print('the mean silhouette score, before:%.3f, non:%.3f \n after:%.3f,non:%.3f'
          %(Lis_bstSC[idx],data['Lis_SCnon_bst'][idx],
            np.mean(data['Lis_SC_upd'][idx,:]),data['Lis_SCnon_upd'][idx]))
    print('adjusted mutual info, before:%.3f, after:%.3f'%(data['Lis_MI_bst'][idx],data['Lis_MI_upd'][idx]))
    for i in [0,1]:
        
        if i == 0:
            print('before')
            predcls = data['Lis_cls_bst'][idx]
            predcls_pt = data['Lis_clspt_bst'][idx]
            
            
        elif i==1:
            print('after')
            predcls = data['Lis_cls_upd'][idx]
            predcls_pt = data['Lis_clspt_upd'][idx]
            
        ########## show detailed results
        MI = metrics.adjusted_mutual_info_score(label_gt, predcls_pt)
        print('MI:%.3f'%MI)
        pred_Lis_SC,pred_SCavg,_,_,pred_SCinfo = utils.SilhouetteScore(hsres,predcls,sq_flag=sq_flag)
        idx_iso = np.where(pred_SCinfo[:,1] == -1)[0]  
        ans = pred_Lis_SC[idx_iso]
        
        print('Avg_SC:%.3f;nonindiv SC:%.3f,indiv SC:%.3f,num:%d'
              %(np.mean(pred_Lis_SC),pred_SCavg,np.mean(ans),len(ans)))  
        ######### -------------------------------
            
        res_fullcorrect,res_partial,res_wrong,_,_,_ =\
            utils.correct_denom_pred(hsres,predcls)
            
        Lis_Count_correct = []
        Lis_Ratio_correct = []
        Lis_Countimg_correct = []
        
        sumNum = 0
        for ii,ians in enumerate(res_fullcorrect):
            if ii>4:
                sumNum += ians
            else:
                Lis_Count_correct.append(ians)
                Lis_Ratio_correct.append(ians/Lis_Count[ii]*100)
                
            Lis_Countimg_correct.append(ians*(ii+1))
            
        Lis_Ratio_correct.append(sumNum/Lis_Count[-1]*100)
        Lis_Count_correct.append(sumNum)
        
        sum_correctID = np.sum(Lis_Count_correct)
        sum_correctImg = np.sum(Lis_Countimg_correct)

        print('predk:%d, correct num:%d, partial correct num:%d, wrong num:%d'
              %(len(predcls),sum_correctID,len(res_partial),len(res_wrong)))
        print('correct image num:%d,ratio/n=%.3f'%(sum_correctImg,sum_correctImg/nsample))
        


# In[Test for Plot distribution 100 repeated (before & after verify)]

'''
For all experiments,
Load verification step  
'''
recordLis = [8020,8002,8005,8006,8007,8008,8010,8009]
recordLis = [8220,8202,8005,8006,8007,8208,8210,8209]
labelLis = [r'k++','adaptive k++, mtd.0','mtd.1','mtd.2','mtd.3','mtd.4','mtd.5','mtd.6']

colorLis = ['#2ca02c', '#ff7f0e', '#1f77b4', '#e377c2',
            '#8c564b','#17becf','#bcbd22','#d62728',  
            '#9467bd', '#7f7f7f',]
Lis = np.arange(len(recordLis))
# Lis = [0,1,2,3,4,]
Lis = [1,5,6,7]
# record = 8002  ##### method 0, adaptive k++, fixed k value
# record = 8005  ##### method 1, adaptive k++, fixed k value
# record = 8006  ##### method 2, adaptive k++, fixed k value
# record = 8007  ##### method 3, adaptive k++, fixed k value
# record0 = 8020  ##### method 0, k++ (benchmark)

meas_name = 'adjusted MI'
labelsize = 14
fontsize = 12

fig,ax2 = plt.subplots(figsize=[10,5])

# ax2 = ax1.twinx()
ax2.spines['top'].set_visible(False)
ax2.spines['right'].set_visible(False)

for ii in Lis:
    record = recordLis[ii]
    name = '%d_verify.npz'%(record)
    data = np.load(join(hsres.data_dir,name),allow_pickle=True)
    
    print(data['note'])
    
    Lis_SC_upd = data['Lis_SC_upd']
    Lis_SC_upd = np.mean(Lis_SC_upd,axis=1)
    Lis_SC_bst = data['Lis_SC_bst']
    Lis_SC_bst = np.mean(Lis_SC_bst,axis=1)
    Lis_mear_upd = data['Lis_MI_upd']
    Lis_mear_bst = data['Lis_MI_bst']
    

    if ii == 0 or ii ==1:
        alpha=1
    else:
        alpha=0.6
        
    # y = copy.copy(Lis_mear_bst)
    # weights1 = np.ones(len(y)) / len(y)
    # n1, bins1, patches1 = ax2.hist(y, bins=10,weights=weights1,alpha=alpha, edgecolor='black',\
    #                               rwidth=0.85,color=colorLis[ii],label=labelLis[ii])
    y = copy.copy(Lis_mear_upd)
    weights1 = np.ones(len(y)) / len(y)
    n1, bins1, patches1 = ax2.hist(y, bins=10,weights=weights1,alpha=alpha, edgecolor='black',\
                                  hatch='/',rwidth=0.85,color=colorLis[ii],label=labelLis[ii])
        
plt.ylabel('percent of 100 experiments',fontsize=labelsize)
plt.xticks(fontsize=fontsize)
plt.ylim(0,0.3)
plt.xlim(0.91,0.955)
plt.yticks(np.arange(0,0.35,0.05),fontsize=fontsize)
plt.xlabel('adjusted mutual information score',fontsize=labelsize)

#get handles and labels
handles, labels = plt.gca().get_legend_handles_labels()

#specify order of items in legend
order = np.arange(len(Lis))

#add legend to plot
ax2.legend([handles[jj] for jj in order],[labels[jj] for jj in order],\
            loc='upper center',fontsize=labelsize-1,fancybox=True,framealpha=0.5,
            bbox_to_anchor=(0.0, 0.48, 0.5, 0.5)) 
    
    # bbox_to_anchor=(0.4, 0.48, 0.5, 0.5)
content ='Distribution of clustering accuracies in 100 repeated experiments'
ax2.set_title(content,fontsize=labelsize+2)
ax2.yaxis.set_major_formatter(ticker.PercentFormatter(xmax=1,decimals=0))



# In[Test for Distribution about correct-clusters (before & after verify)]
 
'''
Percent of correct clusters for different cluster sizes

modify on 01/18/23, changed variable names

'''
k = ncluster

colorLis = ['#2ca02c', '#ff7f0e', '#1f77b4', '#e377c2',
            '#8c564b','#17becf','#bcbd22','#d62728',  
            '#9467bd', '#7f7f7f',]

fig,ax1 = plt.subplots(figsize=[15,5])
ax1.spines['top'].set_visible(False)
ax1.spines['right'].set_visible(False)

labelsize = 14
fontsize = 12

recordLis = [8220,8202,8005,8006,8007,8208,8210,8209]
# recordLis = [8020,8002,8005,8006,8007,8008,8010,8009]
labelLis = [r'k++','adaptive k++, mtd.0','mtd.1','mtd.2','mtd.3','mtd.4','mtd.5','mtd.6']
Lis = [0,1,5,6,7]

# recordLis = [8010,8002,8005,8006,8007]
# labelLis = [r'k++','adaptive k++, mtd.0','mtd.1','mtd.2','mtd.3']

Lis_hatch = []
for ii in range(len(Lis)):
    Lis_hatch.append(None)
    Lis_hatch.append('/')
    

mins = -0.2*5
maxs = 0.2*4
Lis_pos = []
for ii in range(len(recordLis)):
    Lis_pos.append(mins+0.2*ii)
    Lis_pos.append(0+0.2*ii)
    
p=0
for j in Lis:
    record = recordLis[j]
    for i in [0,1]:
        name = '%d_verify.npz'%(record)
        data = np.load(join(hsres.data_dir,name),allow_pickle=True)
        
        Lis_bstSC = np.mean(data['Lis_SC_bst'],axis=1)
        ##### the middle one 
        Lis_bstSC_sort = np.argsort(Lis_bstSC)
        idx = Lis_bstSC_sort[int(len(Lis_bstSC)/2)]
        
        if i == 0:
            predcls = data['Lis_cls_bst'][idx]
            predcls_pt = data['Lis_clspt_bst'][idx]
            
            
        elif i==1:
            predcls = data['Lis_cls_upd'][idx]
            predcls_pt = data['Lis_clspt_upd'][idx]

            
        # res_fullcorrect,res_partial,res_wrong,_,_,_ =\
        #     utils.correct_denom_gt(hsres,predcls_pt,predcls)
        res_fullcorrect,res_partial,res_wrong,_,_,_ =\
            utils.correct_denom_pred(hsres,predcls)
        
        Lis_Count_correct = []
        Lis_Ratio_correct = []
        Lis_Countimg_correct = []
        
        sumNum = 0
        for ii,ians in enumerate(res_fullcorrect):
            if ii>4:
                sumNum += ians
            else:
                Lis_Count_correct.append(ians)
                Lis_Ratio_correct.append(ians/Lis_Count[ii]*100)
                
            Lis_Countimg_correct.append(ians*(ii+1))
            
        Lis_Ratio_correct.append(sumNum/Lis_Count[-1]*100)
        Lis_Count_correct.append(sumNum)
        
        sum_correctID = np.sum(Lis_Count_correct)
        sum_correctImg = np.sum(Lis_Countimg_correct)
        
        x = np.linspace(1, 12, num=6)
        # x = np.arange(1,17,3)
         
        if i==1:
            bar_plot = plt.bar(x+Lis_pos[p], Lis_Ratio_correct,width=0.12,\
                                alpha=0.6,color=colorLis[j],\
                                  hatch=Lis_hatch[p], edgecolor='black')
        else:
            bar_plot = plt.bar(x+Lis_pos[p], Lis_Ratio_correct,width=0.12,\
                                label=labelLis[j],alpha=0.6,color=colorLis[j],\
                                  hatch=Lis_hatch[p], edgecolor='black')
        p += 1
        
        for ii,rect in enumerate(bar_plot):
            height = rect.get_height()
            plt.text(rect.get_x() + rect.get_width()/2., 1*height,
                    int(np.round(Lis_Ratio_correct[ii],0)),
                    ha='center', va='bottom', rotation=0,fontsize=fontsize-1)
        
plt.xticks(x,barInfo,fontsize=fontsize)
plt.yticks(np.arange(0,95,15),fontsize=fontsize)
plt.ylim([0,102])

plt.ylabel('percent of correct-clusters',fontsize=labelsize)
plt.xlabel('number of images per predict leopard ID (cluster)',fontsize=labelsize)
         
#get handles and labels
handles, labels = plt.gca().get_legend_handles_labels()

#specify order of items in legend
order = np.arange(len(Lis))

#add legend to plot
plt.legend([handles[ii] for ii in order],[labels[ii] for ii in order],\
            loc='lower left',fontsize=labelsize-1) 
    
content = 'Improvement in clustering accuracy'
plt.title(content,fontsize=labelsize+2)
ax1.yaxis.set_major_formatter(ticker.PercentFormatter(xmax=100))
plt.show()


# In[Test for Distribution about correct-clusters (before & after verify)]
 
'''
Percent of correct clusters for different cluster sizes
show independently: before verification OR after verification
modify on 01/31/23, changed variable names

'''

##### TrueID distribution-----------
CountInfo = np.array(hsres.CountInfo)
barInfo = []
Lis_Count = []
sumNum = 0
true_k = len(hsres.Lis_ID)
# print('true_k=%d'%true_k)
Lis_Ratio = []
for ians in CountInfo:
    if ians[0]>5:
        sumNum += ians[1]
    else:
        barInfo.append(str(ians[0]))
        Lis_Ratio.append(ians[1]/true_k*100)
        Lis_Count.append(ians[1])
Lis_Ratio.append(sumNum/true_k*100)
Lis_Count.append(sumNum)
barInfo.append('6 - 17')
####----------------------------

k = ncluster

colorLis = ['#2ca02c', '#ff7f0e', '#1f77b4', '#e377c2',
            '#8c564b','#17becf','#bcbd22','#d62728',  
            '#9467bd', '#7f7f7f',]

recordLis = [8020,8002,8005,8006,8007,8008,8010,8009]
labelLis = [r'k++','adaptive k++, mtd.0','mtd.1','mtd.2','mtd.3','mtd.4','mtd.5','mtd.6']

Lis = np.arange(len(recordLis))
Lis = [0,1,2,3,4,5,6]
# Lis = [0,1,5,6]

fig,ax1 = plt.subplots(figsize=[15,5])
ax1.spines['top'].set_visible(False)
ax1.spines['right'].set_visible(False)

labelsize = 14
fontsize = 12


Lis_hatch = []
for ii in range(len(recordLis)):
    Lis_hatch.append(None)
    Lis_hatch.append('/')
    

mins = -0.1*len(Lis)
Lis_pos = []
for ii in range(len(Lis)):
    Lis_pos.append(mins+0.2*ii)
    
idx = 0
# p=0
for j in Lis:
    record = recordLis[j]
    name = '%d_verify.npz'%(record)
    data = np.load(join(hsres.data_dir,name),allow_pickle=True)
    Lis_bstSC = np.mean(data['Lis_SC_bst'],axis=1)
    ##### the middle one 
    Lis_bstSC_sort = np.argsort(Lis_bstSC)
    idx = Lis_bstSC_sort[int(len(Lis_bstSC)/2)]
    
    
    predcls = data['Lis_cls_bst'][idx]
    predcls_pt = data['Lis_clspt_bst'][idx]
    hatch = None
        
    # predcls = data['Lis_cls_upd'][idx]
    # predcls_pt = data['Lis_clspt_upd'][idx]
    # hatch = '/'

    
    res_fullcorrect,res_partial,res_wrong,_,_,_ =\
        utils.correct_denom_pred(hsres,predcls)
    
    Lis_Count_correct = []
    Lis_Ratio_correct = []
    Lis_Countimg_correct = []
    
    sumNum = 0
    for ii,ians in enumerate(res_fullcorrect):
        if ii>4:
            sumNum += ians
        else:
            Lis_Count_correct.append(ians)
            Lis_Ratio_correct.append(ians/Lis_Count[ii]*100)
            
        Lis_Countimg_correct.append(ians*(ii+1))
        
    Lis_Ratio_correct.append(sumNum/Lis_Count[-1]*100)
    Lis_Count_correct.append(sumNum)
    
    sum_correctID = np.sum(Lis_Count_correct)
    sum_correctImg = np.sum(Lis_Countimg_correct)
    
    x = np.linspace(1, 9, num=6)
    # x = np.arange(1,17,3)
     
    bar_plot = plt.bar(x+Lis_pos[j], Lis_Ratio_correct,width=0.12,\
                        alpha=0.6,color=colorLis[j],\
                          hatch=hatch, edgecolor='black',label=labelLis[j])
            
    
    for ii,rect in enumerate(bar_plot):
        height = rect.get_height()
        plt.text(rect.get_x() + rect.get_width()/2., 1*height,
                int(np.round(Lis_Ratio_correct[ii],0)),
                ha='center', va='bottom', rotation=0,fontsize=fontsize)
        
plt.xticks(x,barInfo,fontsize=fontsize)
plt.yticks(np.arange(0,95,15),fontsize=fontsize)
plt.ylim([0,102])

plt.ylabel('percent of correct-clusters',fontsize=labelsize)
plt.xlabel('number of images per predict leopard ID (cluster)',fontsize=labelsize)
         
#get handles and labels
handles, labels = plt.gca().get_legend_handles_labels()

#specify order of items in legend

#add legend to plot
plt.legend([handles[ii] for ii in order],[labels[ii] for ii in order],\
            loc='lower left',fontsize=labelsize-1) 
    
content = 'Improvement in clustering accuracy'
plt.title(content,fontsize=labelsize+2)
ax1.yaxis.set_major_formatter(ticker.PercentFormatter(xmax=100))
plt.show()






