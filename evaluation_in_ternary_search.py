#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 29 17:31:00 2022

Modify on 01/18/23

@author: obaiga
"""

# In[Packs]
import numpy as np 
from os.path import join
import os
import copy
from sklearn import metrics
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from mpl_toolkits.axes_grid.inset_locator import (inset_axes, InsetPosition,mark_inset)

# code_dir = 'C:/Users/95316/seal'
code_dir = '/Users/obaiga/github/Automatic-individual-animal-identification/'
os.chdir(code_dir)

import utils

# dpath = '/Users/obaiga/Jupyter/Python-Research/SealID/'
# new_db = 'seal_hotspotter'

dpath = '/Users/obaiga/Jupyter/Python-Research/Africaleopard'
# # new_db = 'non_iso'
new_db = 'snow leopard'

# dpath = '/Users/obaiga/Jupyter/Python-Research/'
# new_db = 'ds_160'

query_flag = 'vsmany_'
fg_flag = 'fg'   #### only containing animal body without background
# fg_flag = ''
# data_mof = '_mean'    #### modify similarity score matrix '_mean' or '' or '_diag'
# data_mof = ''
data_mof = '_diag'

sq_flag = True    #### True: the square of a similarity score

# In[Load]
db_dir = join(dpath,new_db)
hsres = utils.HotspotterRes(db_dir)
hsres.res_dir = join(hsres.db_dir,'results')
hsres.data_dir = join(hsres.res_dir,'data')
utils.CheckDir(hsres.res_dir)
utils.CheckDir(hsres.data_dir)
Lis_Chip2ID,Lis_Img,Lis_chipNo,Lis_ID = hsres.load_info()

cluster_gt,label_gt = hsres.load_cluster_gt(query_flag,fg_flag)

nsample = hsres.nsample
ncluster = hsres.k_gt
   
scoreAry = hsres.load_score(('ImgScore_%s%s%s.xlsx')%(query_flag[2:],fg_flag,data_mof))

scoreAry = copy.copy(hsres.scoreAry)
if sq_flag:
    scoreAry = scoreAry**2 
    
# In[Plot TrueID distribution]
if 0:
    countInfo = []
    for icluster in cluster_gt:
        countInfo.append(len(icluster))
    
    fontsize = 12
    fig,(ax1) = plt.subplots(1,1,figsize=[10,5])
    y = copy.copy(countInfo)
    weights1 = np.ones(len(y)) / len(y)
    n1, bins1, patches1 = ax1.hist(y, bins=10,weights=weights1,alpha=0.6, \
                                  hatch='\\',rwidth=0.85)
    
    # Add text on top of each bin
    for count, b, patch in zip(n1, bins1, patches1):
        # Get the center of each bin
        bin_center = (b + bins1[np.where(bins1 == b )[0][0] + 1]) / 2
        # Add text annotation
        plt.text(bin_center, count, str(int(count*ncluster))+' IDs', ha='center', va='bottom')     
        
    plt.xticks(np.array(bins1,dtype=np.int32))
    plt.xlabel('number of images per true seal ID (individual seal)',fontsize=fontsize)
    plt.ylabel('percent of total seal IDs',fontsize=fontsize)
    plt.title('Distribution of seal in the dataset, images:%d,seals:%d'%(nsample,ncluster),
              fontsize=fontsize+2)

# In[Plot TrueID distribution]
'''
only for Africa leopard 
'''
if 1:
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
        
    plt.ylabel('percent of total leopard IDs',fontsize=labelsize)
    plt.xlabel('number of images per true leopard ID (individual leopard)',fontsize=labelsize)
    plt.xticks(fontsize=fontsize)
    plt.yticks(fontsize=fontsize)
    plt.title(r'Distribution of leopards in the ${Panthera}$ dataset',fontsize=labelsize+2)
    ax.yaxis.set_major_formatter(ticker.PercentFormatter(xmax=100))
    
    # plt.text(3,50,'total images: %d'%nsample,fontsize=labelsize)
    # plt.text(3,46,'leopards: %d'%hsres.k_gt,fontsize=labelsize)
    plt.text(3,30,'total images: %d'%nsample,fontsize=labelsize)
    plt.text(3,26,'leopards: %d'%hsres.k_gt,fontsize=labelsize)
    
    plt.show()


# In[Load scorematrix]

if scoreAry is not None:
    Lis_SC_gt,SCavg_gt,Lis_SCcls_gt,SC_info_gt = \
        utils.SilhouetteScore(hsres,cluster_gt,sq_flag=sq_flag,scoreAry=scoreAry)
    
    centrd_gt,centroid_ssum_gt = utils.centroid(hsres,cluster_gt)
    TWCV_gt = np.sum(centroid_ssum_gt)
    
    print('groundtruth:Avg_SC:%.4f;nonindiv SC:%.4f'
          %(np.mean(Lis_SC_gt),SCavg_gt))  
    
    idx_iso = np.where(SC_info_gt[:,1] == -1)[0]  
    if len(idx_iso) > 0:
        ans = Lis_SC_gt[idx_iso]
        print('indiv SC:%.4f,num:%d'%(np.mean(ans),len(ans))) 
    else:
        print('no isolated images') 
    

# In[plot figure from verified record]
'''
title: determine the best clustering in ternary search
x-axis: k value; y-axis: measures
'''

times = 200
record = 1002   ##### adaptive k++
k = hsres.k_gt

name = '%d_%d_verify.npz'%(record,times)
data = np.load(join(hsres.data_dir,name),allow_pickle=True)
print(list(data.keys()))

Lis_ncluster = data['Lis_ncluster_bst']
Lis_ncluster_upd = data['Lis_ncluster_upd']
Lis_SC_upd = data['Lis_SC_upd']
Lis_SC_bst = data['Lis_SC_bst']

Lis_mear_upd = data['Lis_MI_upd']
Lis_bstmear = data['Lis_MI_bst']


##----------choose the best index------------------
Lis_bstSC = np.mean(Lis_SC_bst,axis=1)
x = np.array(Lis_ncluster)
    
bstidx = np.argmax(Lis_bstSC)
xbst = Lis_ncluster[bstidx]
ybst = Lis_bstSC[bstidx]
ybst_mear = Lis_bstmear[bstidx]


y = copy.copy(Lis_bstSC)
y_mear = copy.copy(Lis_bstmear)

xsort_idx = np.argsort(x)
x = np.sort(x)
y = copy.copy(np.array(Lis_bstSC)[xsort_idx.astype(int)])
y_mear = copy.copy(np.array(Lis_bstmear)[xsort_idx.astype(int)])

x_uni = np.unique(x)
y_uni = []
y_mear_uni = []
for xi in x_uni:
    ans = np.where(x==xi)[0]
    bst_xi = ans[ np.argmax(y[ans])]
    y_uni.append( y[bst_xi])
    y_mear_uni.append(y_mear[bst_xi])
    

#### In[plot figure: determine an optimal clustering]
##----------plot-------------------
labelsize = 20
fontsize = 15
markersize = 10

fig,ax1 = plt.subplots(figsize=[15,10])

ax1.spines['top'].set_visible(False)
ax1.spines['right'].set_visible(False)

plt.plot(x_uni,y_mear_uni,'o--',markersize=markersize,
         c='forestgreen',label=r'adjusted mutual info')
plt.plot(x_uni,y_uni,'o--',markersize=markersize,
         c='darkorange',label=r'mean silhouette score, $\bar{s}^*(k)$')

# plt.plot(x,y_mear,'o--',markersize=markersize,c='forestgreen',label=r'adjusted mutual info')
ax1.plot(xbst,ybst_mear,'o',
            markerfacecolor='none',markersize=markersize+9,color='r')
content = '(%d,%.3f)'%(xbst,ybst_mear)
ax1.text(xbst+10,ybst_mear+0.006,content,fontsize=fontsize)

# plt.plot(x,y,'o--',markersize=markersize,c='darkorange',label=r'best mean silhouette score, $\bar{s}^*$')
ax1.plot(xbst,ybst,'o',
            markerfacecolor='none',markersize=markersize+9,color='r')
content = '(%d,%.3f)'%(xbst,ybst)
ax1.text(xbst+10,ybst-0.002,content,fontsize=fontsize)

plt.legend(fontsize=labelsize)
### loc='upper left'

xlabel = np.unique(Lis_ncluster[:10])
# cache = np.array([519, 1037,  346,  576,  807,  724,  883,  678, 708,769,749] )
# xticks = np.unique(np.concatenate([cache]))

plt.xticks(xlabel,rotation=-65,fontsize=fontsize)

# plt.yticks(np.arange(0.4,0.95,0.05),fontsize=fontsize)

ax1.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.2f'))
# plt.xlim(1,nsample)
plt.xlabel(r'$k$ value',fontsize=labelsize)
plt.ylabel(r'measures',fontsize=labelsize)

plt.title(r'Determine the best clustering $\mathcal{C}^*$',fontsize=labelsize+2)

plt.xticks(fontsize=fontsize)
plt.yticks(fontsize=fontsize)

plt.ylim([0.1,0.93])

'''
plot a sub-figure in the figure
'''
# # Create a set of inset Axes: these should fill the bounding box allocated to
# # them.
# ax2 = plt.axes([0,0,1,1])
# # ax2.spines['top'].set_visible(False)
# # ax2.spines['right'].set_visible(False)
# ax2.patch.set_alpha(0.01)

# # Manually set the position and relative size of the inset axes within ax1
# ip = InsetPosition(ax1, [0.3,0.06,0.5,0.45])
# ax2.set_axes_locator(ip)
# # Mark the region corresponding to the inset axes on ax1 and draw lines
# # in grey linking the two axes.
# mark_inset(ax1, ax2, loc1=2, loc2=1, fc="none", ec='0.5')
# # The data: only display for low temperature in the inset figure.

# '''
# plot adjusted mutual information score with top-15 mean silhouette scores
# '''
# total = 10
# # Lis_idx = np.argsort(Lis_bstSC)[::-1][:total]
# # ax2.plot(Lis_ncluster[Lis_idx], np.array(Lis_bstmear)[Lis_idx], 
# #          'o', c='forestgreen', mew=2, alpha=0.8,markersize=markersize)

# Lis_idx = np.argsort(y_uni)[::-1][:total]
# ax2.plot(x_uni[Lis_idx], np.array(y_mear_uni)[Lis_idx], 
#           'o', c='forestgreen', mew=2, alpha=0.8,markersize=markersize)


# content = '(%d,%.3f)'%(xbst,ybst_mear)
# ax2.text(xbst+1.5,ybst_mear-0.0005,content,fontsize=fontsize)
# ax2.plot(xbst,ybst_mear,'o',
#             markerfacecolor='none',markersize=markersize+9,c='r')

# ax2.set_yticks(np.arange(0.875,0.9,0.0125))
# ax2.set_ylim(0.8745,0.9,0.0125)
# ax2.set_xticks(np.arange(720,775,10))

'''save the figure'''
# plt.savefig(join(hsres.db_dir,'demo.png'), format='png', dpi=100, transparent=True)

# In[Plot distribution (after verify)]
'''
Load verification step 
'''
record = 1001  ##### adaptive k++,
# record = 1001  ##### k++ (benchmark)
times = 200

name = '%d_%d_verify.npz'%(record,times)
data = np.load(join(hsres.data_dir,name),allow_pickle=True)

Lis_ncluster_bst = data['Lis_ncluster_bst']
Lis_ncluster_upd = data['Lis_ncluster_upd']
Lis_SC_upd = data['Lis_SC_upd']
Lis_SC_bst = data['Lis_SC_bst']

Lis_mear_upd = data['Lis_MI_upd']
Lis_mear_bst = data['Lis_MI_bst']

Lis_SC_upd = np.mean(Lis_SC_upd,axis=1)
Lis_SC_bst = np.mean(Lis_SC_bst,axis=1)

labelsize = 14
fontsize = 12

total = 15
Lis_idx = np.argsort(Lis_SC_bst)[::-1][:total]
bstidx = 0

fig,(ax1,ax2) = plt.subplots(2,1,figsize=[10,5])

# ax1.set_ylabel('mutual     info',fontsize=labelsize,loc='bottom')
# ax2.set_ylabel('adjusted ',fontsize=labelsize,loc='top')

# hide the spines between ax and ax2
ax1.spines['top'].set_visible(False)
ax1.spines['right'].set_visible(False)
ax1.spines['bottom'].set_visible(False)
ax1.get_xaxis().set_visible(False)

ax2.spines['top'].set_visible(False)
ax2.spines['right'].set_visible(False)
ax2.xaxis.tick_bottom()

# ax1.spines['right'].set_visible(False)

xupd = copy.copy(Lis_ncluster_upd[Lis_idx])
yupd = copy.copy(Lis_mear_upd[Lis_idx])
x = copy.copy(Lis_ncluster_bst[Lis_idx])
y = copy.copy(Lis_mear_bst[Lis_idx])

ax1.plot(xupd,yupd,'o',markersize=7,alpha=1,c='forestgreen',label='after verification')
ax1.plot(x,y,'o',markersize=7,alpha=1,c='darkorange',label='before verification')

ax2.plot(xupd,yupd,'o',markersize=7,alpha=1,c='forestgreen',label='after verification')
ax2.plot(x,y,'o',markersize=7,alpha=1,c='darkorange',label='before verification')

# ax1.set_yticks(np.arange(0.94,0.966,0.01))
# ax2.set_yticks(np.arange(0.88,0.905,0.01))

d = .5  # proportion of vertical to horizontal extent of the slanted line
kwargs = dict(marker=[(-1, -d), (1, d)], markersize=12,
              linestyle="none", color='k', mec='k', mew=1, clip_on=False)
ax1.plot([0,], [0, ], transform=ax1.transAxes, **kwargs)
ax2.plot([0,], [1, ], transform=ax2.transAxes, **kwargs)

content = '(%d,%.3f)'%(x[bstidx],y[bstidx])
ax2.text(x[bstidx]+1,y[bstidx],content,fontsize=labelsize)
ax2.plot(x[bstidx],y[bstidx],'o',markerfacecolor='none',
          markersize=17,c='r')
content = '(%d,%.3f)'%(xupd[bstidx],yupd[bstidx])
ax1.text(xupd[bstidx]-1.5,yupd[bstidx]-0.005,content,fontsize=labelsize)
ax1.plot(xupd[bstidx],yupd[bstidx],'o',markerfacecolor='none',
          markersize=17,c='r')

content = r'Clustering accuracy after verification'
plt.suptitle(content,fontsize=labelsize+2)
ax1.legend(fontsize=labelsize)

plt.xlabel(r'$k$ value',fontsize=labelsize)

# ax1.set_xlim([705,770])
# ax2.set_xlim([705,770])
ax1.tick_params(axis='both', labelsize=fontsize)
ax2.tick_params(axis='both', labelsize=fontsize)


plt.text(285,0.7,'adjusted mutual info',rotation=90,fontsize=labelsize)
ax1.set_ylim([0.83,0.9])
ax2.set_ylim([0.66,0.72])

plt.show()


# plt.xticks(fontsize=fontsize)
# plt.yticks(fontsize=fontsize)
# # plt.grid()
# plt.xlim([705,800])

# # plt.ylim([0.875,0.965])
# plt.ylim([0.85,0.965])


# In[table for performance]
dpath = '/Users/obaiga/Jupyter/Python-Research/Africaleopard'
# # new_db = 'non_iso'
new_db = 'bright50'

dpath = '/Users/obaiga/Jupyter/Python-Research'
new_db = 'ds_160'

# record = 1002
# record = 8201  ##### adaptive k++
record = 8201  ##### method 0, k++ (benchmark)
times = 200

db_dir = join(dpath,new_db)
hsres = utils.HotspotterRes(db_dir)
hsres.res_dir = join(hsres.db_dir,'results')
hsres.data_dir = join(hsres.res_dir,'data')
Lis_Chip2ID,Lis_Img,Lis_chipNo,Lis_ID = hsres.load_info()
cluster_gt,label_gt = hsres.load_cluster_gt(query_flag,fg_flag)
nsample = hsres.nsample
ncluster = hsres.k_gt



name = '%d_%d_verify.npz'%(record,times)
data = np.load(join(hsres.data_dir,name),allow_pickle=True)
# print(list(data.keys()))
Lis_ncluster_bst = data['Lis_ncluster_bst']

print(data['note'])
Lis_SC_bst = data['Lis_SC_bst']
Lis_bstSC = np.mean(Lis_SC_bst,axis=1)

idx = np.argmax(Lis_bstSC)

print('the mean silhouette score, before:%.3f, non:%.3f \n after:%.3f,non:%.3f'
      %(Lis_bstSC[idx],data['Lis_SCnon_bst'][idx],
        np.mean(data['Lis_SC_upd'][idx,:]),data['Lis_SCnon_upd'][idx]))
print('adjusted mutual info, before:%.3f, after:%.3f'%(data['Lis_MI_bst'][idx],data['Lis_MI_upd'][idx]))

for i in [0,1]:
    
    if i == 0:
        predcls = data['Lis_cls_bst'][idx]
        predcls_pt = data['Lis_clspt_bst'][idx]
        SC_bst = np.mean(data['Lis_SC_bst'][idx])
        mear_bst = data['Lis_MI_bst'][idx]
        print('\nbefore verify')
        
    elif i==1:
        predcls = data['Lis_cls_upd'][idx]
        predcls_pt = data['Lis_clspt_upd'][idx]
        SC_bst = np.mean(data['Lis_SC_upd'][idx])
        mear_bst = data['Lis_MI_upd'][idx]
        print('\nafter verify')
    
    MI = metrics.adjusted_mutual_info_score(label_gt, predcls_pt)
    print('MI:%.3f'%MI)
        
    res_fullcorrect,res_partial,res_wrong,res_dtl_correct,res_dtl_partial,res_dtl_wrong =\
        utils.correct_denom_pred(hsres,predcls)
    
    correct_clusters = np.sum(res_fullcorrect)
    sum_correctImg = 0
    for inum,icorrect in enumerate(res_fullcorrect):
        inum += 1
        sum_correctImg += inum * icorrect
    
    print('predk:%d, correct num:%d, partial correct num:%d, wrong num:%d'
          %(len(predcls),sum(res_fullcorrect),len(res_partial),len(res_wrong)))
    print('correct image num:%d,ratio/n=%.3f'%(sum_correctImg,sum_correctImg/nsample))
    print('partial correct image:%d,wrong image:%d'%(np.sum(np.array(res_partial)[:,0]),
                                                 np.sum(np.array(res_wrong)[:,0])))


# In[Load from original record]

    times = 200
    # record = 8102
    # record = 8202   ##### adaptive k++, mtd.0
    record = 8201   ##### benchmark k++, mtd.0
    record = 1005
    k = hsres.k_gt
    
    name = '%d_%d_%s_%s_k%d.npz'\
        %(record,times,fg_flag,query_flag[2:-1],k)
    data = np.load(join(hsres.data_dir,name),allow_pickle=True)
    print(list(data.keys()))
    print(data['SC_gt'])
    
    note = data['note']
    nsample = data['nsample']
    Lis_upditer = data['Lis_upditer']  ### what iterations are updated?
    Lis_numconvg = data['Lis_numconvg']   ### total iterations when convergence
    Lis_SC = data['Lis_SC']
    
    ##### check the silhouette score 
    # idx = 0 
    # predcls = data['Lis_cls'][idx]
    # pred_Lis_SC,_,_,_ = utils.SilhouetteScore(hsres,predcls,sq_flag=sq_flag)
    # print(Lis_SC[idx],np.mean(pred_Lis_SC))
    ####---------------------------
    
    # Lis_RI = data['Lis_RI']
    # Lis_FMS = data['Lis_FMS']
    Lis_MI= data['Lis_MI']
    Lis_ncluster = data['Lis_ncluster']
    # Lis_iters = data['Lis_iters']
    
    ###### In[Distribution]
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
        Lis_bstidx.append(maxidx)
        Lis_bstmear.append(Lis_mear[maxidx+sums])
        sums += Lis_numconvg[i]
    
    ##----------choose the best index------------------
    
    x = np.array(Lis_ncluster)
        
    bstidx = np.argmax(Lis_bstSC)
    
    predcls = data['Lis_cls'][bstidx]
    xbst = Lis_ncluster[bstidx]
    ybst = Lis_bstSC[bstidx]
    ybst_mear = Lis_bstmear[bstidx]
    
    
    y = copy.copy(Lis_bstSC)
    y_mear = copy.copy(Lis_bstmear)
    
    xsort_idx = np.argsort(x)
    x = np.sort(x)
    y = copy.copy(np.array(Lis_bstSC)[xsort_idx.astype(int)])
    y_mear = copy.copy(np.array(Lis_bstmear)[xsort_idx.astype(int)])
    
    print('ncluster:%d'%(xbst))
    print('the mean silhouette score, before:%.3f, non:%.3f'%(ybst,data['Lis_SCnon'][bstidx]))
          
    print('adjusted mutual info, before:%.3f'%(ybst_mear))

    
    res_fullcorrect,res_partial,res_wrong,res_dtl_correct,res_dtl_partial,res_dtl_wrong =\
        utils.correct_denom_pred(hsres,predcls)
    
    correct_clusters = np.sum(res_fullcorrect)
    sum_correctImg = 0
    for inum,icorrect in enumerate(res_fullcorrect):
        inum += 1
        sum_correctImg += inum * icorrect
    
    print('predk:%d, correct num:%d, partial correct num:%d, wrong num:%d'
          %(len(predcls),sum(res_fullcorrect),len(res_partial),len(res_wrong)))
    print('correct image num:%d,ratio/n=%.3f'%(sum_correctImg,sum_correctImg/nsample))
    print('partial correct image:%d,wrong image:%d'%(np.sum(np.array(res_partial)[:,0]),
                                                 np.sum(np.array(res_wrong)[:,0])))
    