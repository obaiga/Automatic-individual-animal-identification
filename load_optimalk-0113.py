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
    Lis_SC_gt,SCavg_gt,T2_gt,Lis_SCcls_gt,SC_info_gt = \
        utils.SilhouetteScore(hsres,cluster_gt,sq_flag=sq_flag,scoreAry=scoreAry)
    
    centrd_gt,centroid_ssum_gt = utils.centroid(hsres,cluster_gt)
    TWCV_gt = np.sum(centroid_ssum_gt)
    
    print('groundtruth:Avg_SC:%.4f;nonindiv SC:%.4f,T2:%d'
          %(np.mean(Lis_SC_gt),SCavg_gt,T2_gt))  
    idx_iso = np.where(SC_info_gt[:,1] == -1)[0]  
    ans = Lis_SC_gt[idx_iso]
    print('indiv SC:%.4f,num:%d'%(np.mean(ans),len(ans))) 
    
# In[Load npz (benchmark)]
times = 200
record = 8102   ####
# record = 8105
k = hsres.k_gt

name = '%d_%d_%s_%s_k%d.npz'\
    %(record,times,fg_flag,query_flag[2:-1],k)
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
Lis_ncluster0 = data0['Lis_ncluster']
# Lis_iters = data['Lis_iters']

#### In[Distribution]
sums = 0
Lis_bstSC0 = []
Lis_bstmear0 = []
Lis_bstidx0 = []
Lis_mear0 = copy.copy(Lis_MI0)
Lis_bstmear0 = []
for i,num in enumerate(Lis_numconvg0):
    maxidx = np.argmax(Lis_SC0[sums:sums+num])
    Lis_bstSC0.append(Lis_SC0[maxidx+sums])
    Lis_bstidx0.append(maxidx)
    Lis_bstmear0.append(Lis_mear0[maxidx+sums])
    sums += Lis_numconvg0[i]

# In[Load from original record]
if 0:
    times = 200
    # record = 8102
    # record = 8202   ##### adaptive k++, mtd.0
    record = 8201   ##### benchmark k++, mtd.0
    k = hsres.k_gt
    k = 678
    
    name = '%d_%d_%s_%s_k%d.npz'\
        %(record,times,fg_flag,query_flag[2:-1],k)
    data = np.load(join(hsres.data_dir,name),allow_pickle=True)
    print(list(data.keys()))
    print(data['SC_gt'])
    
    # Lis_T2 = data['Lis_T2']
    # Lis_T2_mean = []
    # Lis_ncluster = data['Lis_ncluster']
    # Lis_numconvg = data['Lis_numconvg']   ### total iterations when convergence
    
    # sums = 0
    # for i, inum in enumerate(Lis_numconvg):
    #     Lis_T2_mean.append(np.mean(Lis_T2[sums:sums+inum]))
    #     sums += Lis_numconvg[i]
    
    # show_det = np.column_stack((Lis_ncluster,Lis_T2_mean))
    
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
    xbst = Lis_ncluster[bstidx]
    ybst = Lis_bstSC[bstidx]
    ybst_mear = Lis_bstmear[bstidx]
    
    
    y = copy.copy(Lis_bstSC)
    y_mear = copy.copy(Lis_bstmear)
    
    xsort_idx = np.argsort(x)
    x = np.sort(x)
    y = copy.copy(np.array(Lis_bstSC)[xsort_idx.astype(int)])
    y_mear = copy.copy(np.array(Lis_bstmear)[xsort_idx.astype(int)])
    
# In[Load from verified record]

times = 200
record = 8202   ##### adaptive k++, mtd.0
# record = 8201   ##### benchmark k++, mtd.0
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

try:
    Lis_T2_bst = data['Lis_T2_bst']
    Lis_T2_upd = data['Lis_T2_upd']
except Exception:
    Lis_T2_bst = None
    Lis_T2_upd = None


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

plt.legend(fontsize=labelsize,loc='upper left')
# xlabel = np.unique(Lis_ncluster)
cache = np.array([519, 1037,  346,  576,  807,  724,  883,  678, 708,769,749] )
xticks = np.unique(np.concatenate([cache]))

plt.xticks(xticks,rotation=-65,fontsize=fontsize)
plt.yticks(np.arange(0.4,0.95,0.05),fontsize=fontsize)
plt.ylim([0.4,0.93])
ax1.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.2f'))
# plt.xlim(1,nsample)
plt.xlabel(r'$k$ value',fontsize=labelsize)
plt.ylabel(r'measures',fontsize=labelsize)

plt.title(r'Determine the best clustering $\mathcal{C}^*$',fontsize=labelsize+2)


# Create a set of inset Axes: these should fill the bounding box allocated to
# them.
ax2 = plt.axes([0,0,1,1])
# ax2.spines['top'].set_visible(False)
# ax2.spines['right'].set_visible(False)
ax2.patch.set_alpha(0.01)

# Manually set the position and relative size of the inset axes within ax1
ip = InsetPosition(ax1, [0.3,0.06,0.5,0.45])
ax2.set_axes_locator(ip)
# Mark the region corresponding to the inset axes on ax1 and draw lines
# in grey linking the two axes.
mark_inset(ax1, ax2, loc1=2, loc2=1, fc="none", ec='0.5')
# The data: only display for low temperature in the inset figure.

'''
plot adjusted mutual information score with top-15 mean silhouette scores
'''
total = 10
# Lis_idx = np.argsort(Lis_bstSC)[::-1][:total]
# ax2.plot(Lis_ncluster[Lis_idx], np.array(Lis_bstmear)[Lis_idx], 
#          'o', c='forestgreen', mew=2, alpha=0.8,markersize=markersize)

Lis_idx = np.argsort(y_uni)[::-1][:total]
ax2.plot(x_uni[Lis_idx], np.array(y_mear_uni)[Lis_idx], 
         'o', c='forestgreen', mew=2, alpha=0.8,markersize=markersize)


content = '(%d,%.3f)'%(xbst,ybst_mear)
ax2.text(xbst+1.5,ybst_mear-0.0005,content,fontsize=fontsize)
ax2.plot(xbst,ybst_mear,'o',
            markerfacecolor='none',markersize=markersize+9,c='r')
ax2.set_yticks(np.arange(0.875,0.9,0.0125))
ax2.set_ylim(0.8745,0.9,0.0125)
ax2.set_xticks(np.arange(720,775,10))

plt.xticks(fontsize=fontsize)
plt.yticks(fontsize=fontsize)

# plt.savefig(join(hsres.db_dir,'demo.png'), format='png', dpi=100, transparent=True)

# In[do verify]
'''
Run verification step 
Redirect the file 'verify-0117.py'
'''

# In[Plot distribution (after verify)]
'''
Load verification step 
'''
record = 8202  ##### method 0, adaptive k++,
# record = 8201  ##### method 0, k++ (benchmark)
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

ax1.set_ylabel('mutual     info',fontsize=labelsize,loc='bottom')
ax2.set_ylabel('adjusted ',fontsize=labelsize,loc='top')
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

ax1.set_yticks(np.arange(0.94,0.966,0.01))
ax1.set_ylim([0.94,0.965])
ax2.set_ylim([0.875,0.90])
ax2.set_yticks(np.arange(0.88,0.905,0.01))

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
ax1.legend(loc='right',fontsize=labelsize)

plt.xlabel(r'$k$ value',fontsize=labelsize)

ax1.set_xlim([705,770])
ax2.set_xlim([705,770])
ax1.tick_params(axis='both', labelsize=fontsize)
ax2.tick_params(axis='both', labelsize=fontsize)
plt.show()


# plt.xticks(fontsize=fontsize)
# plt.yticks(fontsize=fontsize)
# # plt.grid()
# plt.xlim([705,800])

# # plt.ylim([0.875,0.965])
# plt.ylim([0.85,0.965])



# In[table for performance]
record = 8202  ##### method 0, adaptive k++, fixed k value
# record = 8201  ##### method 0, k++ (benchmark)
times = 200

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

for i in [1]:
    
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
        
    pred_Lis_SC,pred_SCavg,_,_,pred_SCinfo = utils.SilhouetteScore(hsres,predcls,sq_flag=sq_flag)
    idx_iso = np.where(pred_SCinfo[:,1] == -1)[0]  
    ans = pred_Lis_SC[idx_iso]
    
    print('Avg_SC:%.3f;nonindiv SC:%.3f,indiv SC:%.3f,num:%d'
          %(np.mean(pred_Lis_SC),pred_SCavg,np.mean(ans),len(ans)))  
    
    # res_fullcorrect,res_partial,res_wrong,gt_dtl_correct,gt_dtl_partial,gt_dtl_wrong =\
    #     utils.correct_denom_gt(hsres,predcls_pt,predcls)
        
    res_fullcorrect,res_partial,res_wrong,res_dtl_correct,res_dtl_partial,res_dtl_wrong =\
        utils.correct_denom_pred(hsres,predcls)
    
    correct_clusters = np.sum(res_fullcorrect)
    sum_correctImg = 0
    for inum,icorrect in enumerate(res_fullcorrect):
        inum += 1
        sum_correctImg += inum * icorrect
        
    # print('k=%d, SC=%.3f,MI=%.3f \n (Denominator: gt) full clusters:%.1f, images:%.1f,partially correct:%.1f, wrong:%.1f'\
    #       %(ncluster,SC_bst,mear_bst,correct_clusters/hsres.k_gt*100,correct_imgs/nsample*100,
    #         len(res_partial)/hsres.k_gt*100,len(res_wrong)/hsres.k_gt*100))
    
    print('predk:%d, correct num:%d, partial correct num:%d, wrong num:%d'
          %(len(predcls),sum(res_fullcorrect),len(res_partial),len(res_wrong)))
    print('correct image num:%d,ratio/n=%.3f'%(sum_correctImg,sum_correctImg/nsample))
    print('partial correct image:%d,wrong image:%d'%(np.sum(np.array(res_partial)[:,0]),
                                                 np.sum(np.array(res_wrong)[:,0])))

#%%
hsres.plot_dir = join(hsres.db_dir,'plot')
save_dir = join(hsres.plot_dir,str(record))
# save_folder = 'wrong_res'
save_folder = 'partial'
# save_folder = 'correct'
# res_req = res_dtl_partial

# req = res_wrong
req = res_partial
# req = res_dtl_correct

for (inum,idx) in req[:30]:
    print(idx)
    reqLis = predcls[idx]
    
# for reqLis in req[:30]:   #### only for res_dtl_correct
    
    # predID = idx
    # hsres.plot_res_pred(reqLis,predID,save_folder,save_dir=save_dir,trueidx_flag=False)
    trueID = hsres.Lis_Chip2ID[reqLis[0]]
    reqLis = np.where(hsres.Lis_Chip2ID==trueID)[0]
    hsres.plot_res_true(reqLis,predcls_pt,save_folder,save_dir=save_dir,specImg=None,trueidx_flag=False)



# In[plot figure: determine an optimal clustering]
'''
plot all methods
'''
labelsize = 14
fontsize = 12

fig,ax1 = plt.subplots(figsize=[11,6.5])
ax1.spines['top'].set_visible(False)
ax1.spines['right'].set_visible(False)


# Create a set of inset Axes: these should fill the bounding box allocated to
# them.
ax2 = plt.axes([0,0,1,1])
ax2.spines['top'].set_visible(False)
# ax2.spines['right'].set_visible(False)
ax2.patch.set_alpha(0.01)

# Manually set the position and relative size of the inset axes within ax1
ip = InsetPosition(ax1, [0.3,0.08,0.5,0.45])
ax2.set_axes_locator(ip)
# Mark the region corresponding to the inset axes on ax1 and draw lines
# in grey linking the two axes.
mark_inset(ax1, ax2, loc1=2, loc2=4, fc="none", ec='0.5')
# The data: only display for low temperature in the inset figure.
    
    


times = 200
Lis_record = [8103,8104,8105,8106,8107,8108,8110,8109]
Lis_color = ['#2ca02c', '#ff7f0e', '#1f77b4', '#e377c2',
            '#8c564b','#17becf','#bcbd22','#d62728',  
            '#9467bd', '#7f7f7f',]
Lis_label = [r'k++',r'adaptive k++, mtd.0','mtd.1','mtd.2','mtd.3','mtd.4','mtd.5','mtd.6']
Lis = np.arange(len(Lis_record))
Lis = [0,1,5,6,7]
# record = 8104   ##### adaptive k++, mtd.0
# record = 8103   ##### benchmark k++, mtd.0
# record = 8108   ##### adaptive k++, mtd.4
# record = 8110   ##### adaptive k++, mtd.5

k = hsres.k_gt
for iidx in Lis:
    record = Lis_record[iidx]
    name = '%d_%d_%s_%s_k%d.npz'\
        %(record,times,fg_flag,query_flag[2:-1],k)
    data = np.load(join(hsres.data_dir,name),allow_pickle=True)
    # print(list(data.keys()))
    print(data['SC_gt'])
    
    note = data['note']
    nsample = data['nsample']
    Lis_upditer = data['Lis_upditer']  ### what iterations are updated?
    Lis_numconvg = data['Lis_numconvg']   ### total iterations when convergence
    Lis_SC = data['Lis_SC']
    Lis_RI = data['Lis_RI']
    Lis_FMS = data['Lis_FMS']
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
    xbst = Lis_ncluster[bstidx]
    ybst = Lis_bstSC[bstidx]
    ybst_mear = Lis_bstmear[bstidx]
    
    
    y = copy.copy(Lis_bstSC)
    y_mear = copy.copy(Lis_bstmear)
    
    xsort_idx = np.argsort(x)
    x = np.sort(x)
    y = copy.copy(np.array(Lis_bstSC)[xsort_idx.astype(int)])
    y_mear = copy.copy(np.array(Lis_bstmear)[xsort_idx.astype(int)])
    
    ##----------plot------------------
    
    ax1.plot(x,y_mear,'o-',markersize=6,c=Lis_color[iidx],label=Lis_label[iidx])
    
    # plt.plot(x,y,'o--',markersize=6,c=Lis_color[iidx])
    
    ax1.plot(xbst,ybst_mear,'o',
                markerfacecolor='none',markersize=15,color='r')
    # ax1.plot(xbst,ybst,'o',
    #             markerfacecolor='none',markersize=15,color='r')
    
    ax1.legend(fontsize=labelsize,loc='upper left')
    
    # xlabel = np.unique(Lis_ncluster)
    xticks = np.unique(np.concatenate([Lis_ncluster[:10]]))
    
    # plt.xticks(xticks,rotation=65,fontsize=fontsize)
    ax1.set_xticks(np.arange(350,1100,50),rotation=65,fontsize=fontsize)
    ax1.set_yticks(np.arange(0.4,0.95,0.10),fontsize=fontsize)
    # plt.yticks(np.arange(0.4,0.95,0.10),fontsize=fontsize)
    ax1.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.2f'))
    # plt.xlim(1,nsample)
    ax1.set_xlabel(r'$k$ value',fontsize=labelsize)
    ax1.set_ylabel(r'measures',fontsize=labelsize)
    ax1.set_title(r'Determine an optimal clustering for adaptive $k$++',fontsize=labelsize+2)

    
    '''
    small window: 
        plot adjusted mutual information score with top-15 mean silhouette scores
    '''
    total = 15
    Lis_idx = np.argsort(Lis_bstSC)[::-1][:total]
    ax2.plot(Lis_ncluster[Lis_idx], np.array(Lis_bstmear)[Lis_idx], 'o', c=Lis_color[iidx], mew=2, alpha=0.8)
    
    # content = '(%d,%.3f)'%(xmax,ymax_mear)
    # ax2.text(xmax+1,ymax_mear-0.005,content,fontsize=fontsize)
    # ax2.plot(xmax,ymax_mear,'o',
    #             markerfacecolor='none',markersize=15,color='r')
    
    
    content = '(%d,%.3f)'%(xbst,ybst_mear)
    ax2.text(xbst+1,ybst_mear-0.002,content,fontsize=fontsize)
    ax2.plot(xbst,ybst_mear,'o',
                markerfacecolor='none',markersize=15,c='r')
    
    plt.xticks(fontsize=fontsize)
    plt.yticks(fontsize=fontsize)




# In[Plot distribution (after verify)]
'''
Load verification step 
'''
labelsize = 14
fontsize = 12
total = 15   #### show clustering results with top-15 mean silhouette scores

fig,ax1 = plt.subplots(figsize=[10,5])
ax1.spines['top'].set_visible(False)
ax1.spines['right'].set_visible(False)

Lis_record = [8120,8104,8105,8106,8107,8108,8110,8109]
# Lis_record = [8320,8304,8105,8106,8107,8308,8310,8309]
Lis_color = ['#2ca02c', '#ff7f0e', '#1f77b4', '#e377c2',
            '#8c564b','#17becf','#bcbd22','#d62728',  
            '#9467bd', '#7f7f7f',]
Lis_label = [r'k++',r'adaptive k++, mtd.0','mtd.1','mtd.2','mtd.3','mtd.4','mtd.5','mtd.6']

Lis = np.arange(len(Lis_record))
Lis = [0,1,5,6,7]

for iidx in Lis:
    record = Lis_record[iidx]
    name = '%d_verify.npz'%(record)
    data = np.load(join(hsres.data_dir,name),allow_pickle=True)
    
    Lis_ncluster_bst = data['Lis_ncluster_bst']
    Lis_ncluster_upd = data['Lis_ncluster_upd']
    Lis_SC_upd = data['Lis_SC_upd']
    Lis_SC_bst = data['Lis_SC_bst']
    
    Lis_mear_upd = data['Lis_MI_upd']
    Lis_mear_bst = data['Lis_MI_bst']
    
    Lis_SC_upd = np.mean(Lis_SC_upd,axis=1)
    Lis_SC_bst = np.mean(Lis_SC_bst,axis=1)
    
    
    Lis_idx = np.argsort(Lis_SC_bst)[::-1][:total]
    bstidx = 0
    
    xupd = copy.copy(Lis_ncluster_upd[Lis_idx])
    yupd = copy.copy(Lis_mear_upd[Lis_idx])
    plt.plot(xupd,yupd,'o',markersize=7,alpha=1,c=Lis_color[iidx],label=Lis_label[iidx])
    
    x = copy.copy(Lis_ncluster_bst[Lis_idx])
    y = copy.copy(Lis_mear_bst[Lis_idx])
    # plt.plot(x,y,'o',markersize=7,alpha=1,c='darkorange',label='before verification')
    plt.plot(x,y,'o',markersize=7,alpha=1,c=Lis_color[iidx],mfc='none')
    content = '(%d,%.3f)'%(x[bstidx],y[bstidx])
    plt.text(x[bstidx]-1.5,y[bstidx],content,fontsize=labelsize)
    plt.plot(x[bstidx],y[bstidx],'o',markerfacecolor='none',
              markersize=17,c='r')
    content = '(%d,%.3f)'%(xupd[bstidx],yupd[bstidx])
    plt.text(xupd[bstidx]-1.5,yupd[bstidx],content,fontsize=labelsize)
    plt.plot(xupd[bstidx],yupd[bstidx],'o',markerfacecolor='none',
              markersize=17,c='r')
    
plt.legend(loc='upper right',fontsize=labelsize,bbox_to_anchor=(1.2, 1.))
plt.ylabel('adjusted mutual info',fontsize=labelsize)
plt.xlabel(r'$k$ value',fontsize=labelsize)
plt.xticks(fontsize=fontsize)
plt.yticks(fontsize=fontsize)
plt.ylim(0.845,0.96)
# plt.grid()

content = r'Clustering accuracy after verification for adaptive $k$++'
plt.title(content,fontsize=labelsize+2)




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


# Lis_record = [8120,8104,8105,8106,8017,8108,8110,8109]
Lis_record = [8320,8304,8105,8106,8107,8308,8310,8309]
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
for ii in range(len(Lis_record)):
    Lis_pos.append(mins+0.2*ii)
    Lis_pos.append(0+0.2*ii)
    
p=0
for j in Lis:
    record = Lis_record[j]
    print(labelLis[j])
    name = '%d_verify.npz'%(record)
    data = np.load(join(hsres.data_dir,name),allow_pickle=True)
    
    Lis_bstSC = np.mean(data['Lis_SC_bst'],axis=1)
    ##### the best one 
    idx = np.argmax(Lis_bstSC)
    # print('the mean silhouette score, before:%.3f, non:%.3f \n after:%.3f,non:%.3f'
    #       %(Lis_bstSC[idx],data['Lis_SCnon_bst'][idx],
    #         np.mean(data['Lis_SC_upd'][idx,:]),data['Lis_SCnon_upd'][idx]))
    print('adjusted mutual info, before:%.3f, after:%.3f'%(data['Lis_MI_bst'][idx],data['Lis_MI_upd'][idx]))
    
    
    for i in [0,1]:
        
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

# In[plot detailed results for all case: in terms of pred]
if 0:
    record = 8104
    hsres.plot_dir = join(hsres.db_dir,'plot')
    save_dir = join(hsres.plot_dir,str(record))
    save_folder = 'wrong_res'
    res_req = res_dtl_wrong
    
    # save_folder = 'partial'
    # res_req = res_dtl_partial
    
    for reqLis in res_req:
        ID = predcls_pt[reqLis[0]]
        hsres.plot_res(reqLis,ID,save_folder,save_dir=save_dir,trueidx_flag=False)
        
# In[plot detailed results for all case: in terms of gt]
if 0:
    record = 8104
    hsres.plot_dir = join(hsres.db_dir,'plot')
    save_dir = join(hsres.plot_dir,str(record))
    
    # save_folder = 'partial_res'
    # res_req = res_dtl_partial
    
    save_folder = 'correct_res'
    res_req = res_dtl_correct
    
    for reqLis in res_req:
        cache_gt = np.where(hsres.Lis_Chip2ID == hsres.Lis_Chip2ID[reqLis[0]])[0]
        if len(cache_gt) > len(reqLis):
            others = list(set(list(cache_gt)) - set(list(reqLis)))
            ors_prdID = predcls_pt[others]
            
            ors_prdID_ele = np.where(predcls_pt == ors_prdID)[0]
            if len(ors_prdID_ele) == len(others):
                req = [reqLis,others]
        
                hsres.plot_res_block(req,predcls_pt,save_folder,save_dir=save_dir,trueidx_flag=False)
        elif len(cache_gt) == len(reqLis):
            hsres.plot_res_block([reqLis],predcls_pt,save_folder,save_dir=save_dir,trueidx_flag=False)
