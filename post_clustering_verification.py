#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep  1 11:21:22 2022

@author: obaiga
"""

# In[Packs]
import numpy as np 
from os.path import join
import os
import copy
from sklearn import metrics

# code_dir = '/Users/obaiga/github/Automatic-individual-animal-identification/'
code_dir = 'C:/Users/95316/seal'
os.chdir(code_dir)

import utils

# dpath = '/Users/obaiga/Jupyter/Python-Research/SealID/'
dpath = 'C:/Users/95316/seal'
new_db = 'seal_hotspotter'

query_flag = 'vsmany_'
# fg_flag = 'fg'   #### only containing animal body without background
fg_flag = ''
# data_mof = '_mean'    #### modify similarity score matrix '_mean' or '' or '_diag'
# data_mof = ''
data_mof = '_diag'

sq_flag = True

print('data format:%s'%data_mof[1:])   
#### '_diag': the diagonal value is the sum of the simailrity scores for an image

# In[Load]
db_dir = join(dpath,new_db)
hsres = utils.HotspotterRes(db_dir)
hsres.res_dir = join(hsres.db_dir,'results')

hsres.data_dir = join(hsres.res_dir,'data')

utils.CheckDir(hsres.res_dir)
Lis_Chip2ID,Lis_Img,Lis_chipNo,Lis_ID = hsres.load_info()

cluster_gt,label_gt =hsres.load_cluster_gt(query_flag,fg_flag)

nsample = hsres.nsample
ncluster = hsres.k_gt
label_gt = hsres.label_gt
    
print('nsample:%d, ncluster:%d'%(nsample,ncluster))

scoreAry = hsres.load_score(('ImgScore_%s%s%s.xlsx')\
                            %(query_flag[2:],fg_flag,data_mof))

scoreAry = copy.copy(hsres.scoreAry)
if sq_flag:
    scoreAry = scoreAry**2
    
# In[calculate ground truth silhouette]
if scoreAry is not None:
    Lis_SC_gt,SCavg_gt,Lis_SCcls_gt,SC_info_gt = \
        utils.SilhouetteScore(hsres,cluster_gt,sq_flag=sq_flag,scoreAry=scoreAry)
    
    centrd_gt,centroid_ssum_gt = utils.centroid(hsres,cluster_gt)
    TWCV_gt = np.sum(centroid_ssum_gt)
    
    print('groundtruth:Avg_SC:%.4f;nonindiv SC:%.4f'%
          (np.mean(Lis_SC_gt),SCavg_gt))   
    
    idx_iso = np.where(SC_info_gt[:,1] == -1)[0]  
    if len(idx_iso) > 0:
        ans = Lis_SC_gt[idx_iso]
        print('indiv SC:%.4f,num:%d'%(np.mean(ans),len(ans))) 
    else:
        print('no isolated images')

# In[load clustering data]
'''
Run post-clustering verification step 
'''
if 1:

    times = 200
    record = 1002
    k = hsres.k_gt

    name = '%d_%d_%s_%s_k%d.npz'\
        %(record,times,fg_flag,query_flag[2:-1],k)
      
try:
    data = np.load(join(hsres.data_dir,name),allow_pickle=True)
    print(list(data.keys()))
    print(data['note'])
    
    Lis_numconvg = data['Lis_numconvg']
    Lis_SC = data['Lis_SC']
    Lis_MI = data['Lis_MI']
    num = len(Lis_numconvg)
    sums = 0
    Lis_bstSC = []
    Lis_bstidx = []
    # Lis_ncluster = []
    for i,num in enumerate(Lis_numconvg):
        
        maxidx = np.argmax(Lis_SC[sums:sums+num])
        Lis_bstSC.append(Lis_SC[maxidx+sums])
        Lis_bstidx.append(maxidx+sums)
        
        # Lis_ncluster.append(len(Lis_cls[maxidx+sums]))
        
        sums += Lis_numconvg[i]
        
    # ans = np.column_stack((Lis_ncluster,Lis_bstSC))
    
except Exception:
    print('Did not load a data')     
      

# In[Execute verify]
Lis_idx = []
Lis_ncluster_bst = []
Lis_MI_bst = []
Lis_SC_bst = []
Lis_SCnon_bst = []
Lis_cls_bst = []
Lis_clspt_bst = []
Lis_ncluster_upd = []
Lis_MI_upd = []
Lis_SC_upd = []
Lis_SCnon_upd = []
Lis_cls_upd = []
Lis_clspt_upd = []

Lis_reapt = []


cache = copy.copy(Lis_bstidx)

bstidx = np.argmax(Lis_bstSC)

cache = Lis_bstidx[bstidx:bstidx+1]

Lis_cls = data['Lis_cls'][cache]

#%%
for i,idx in enumerate(cache):

    MI_req = Lis_MI[idx]
    
    predcls,predcls_pt,CountInfo_pred = utils.load_info_pred\
        (hsres,Lis_cls[i],nsample)
    
    pred_SCs,pred_SCavgnon,_,pred_SCinfo =\
        utils.SilhouetteScore(hsres,predcls,scoreAry=scoreAry)
    nsample = len(pred_SCs)
    
    if (np.mean(pred_SCs)-np.mean(Lis_SC[idx])) < 1e-6:
        print('i:%d,idx:%d,correct'%(i,idx))
    else:
        print('i:%d,idx:%d,wrong,re-cal:%.3f,sc:%.3f '
              %(i,idx,np.mean(pred_SCs),np.mean(Lis_SC[idx])))
 
    ####-----modify on 03/07/23
    bst_cls = copy.copy(predcls)
    bst_SCs = copy.copy(pred_SCs)
    bst_SCinfo = copy.copy(pred_SCinfo)
    bst_SCavgnon = copy.copy(pred_SCavgnon)
    bst_pt = copy.copy(predcls_pt)
    bst_SCmean = np.mean(pred_SCs)
    
    prev_cls = copy.deepcopy(predcls)
    prev_SClis = copy.deepcopy(pred_SCs)
    prev_SCinfo = copy.deepcopy(pred_SCinfo)
    prev_SCavgnon = copy.deepcopy(pred_SCavgnon)
    prev_pt = copy.deepcopy(bst_pt)
    
    ireapt = 0
    while (1):
        
        clsI_cache,res_checkI = utils.merge_clusters\
            (scoreAry,prev_cls,prev_pt,prev_SClis,prev_SCinfo)
            
        clsI,clsI_pt,CountInfo_clsI = utils.load_info_pred\
            (hsres,clsI_cache,nsample)
            
        pred_SCs_clsI,pred_SCavgnon_clsI,_,pred_SCinfo_clsI = \
            utils.SilhouetteScore(hsres,clsI,scoreAry=scoreAry)
    
        #--------------------------------
        prev_cls = copy.deepcopy(clsI)
        prev_SCinfo = copy.deepcopy(pred_SCinfo_clsI)
        prev_SClis = copy.deepcopy(pred_SCs_clsI)
        prev_pt = copy.deepcopy(clsI_pt)


        clsII_cache,res_checkII = utils.lowSC_solve\
            (prev_cls,prev_SClis,prev_SCinfo,prev_SCavgnon)
            
        clsII,clsII_pt,CountInfo_clsII = utils.load_info_pred\
            (hsres,clsII_cache,nsample)
            
        pred_SCs_clsII,pred_SCavgnon_clsII,_,pred_SCinfo_clsII = \
            utils.SilhouetteScore(hsres,clsII,scoreAry=scoreAry)
            
        ans = np.mean(pred_SCs_clsII)
        if  ans >= bst_SCmean:
            
            bst_SCmean = ans
            prev_pt = copy.deepcopy(clsII_pt)
            prev_cls = copy.deepcopy(clsII)
            prev_SClis = copy.copy(pred_SCs_clsII)
            prev_SCinfo = copy.copy(pred_SCinfo_clsII)
            prev_SCavgnon = copy.copy(pred_SCavgnon_clsII)
            
            bst_cls = copy.deepcopy(clsII)
            bst_pt = copy.copy(clsII_pt)
            bst_SCs = copy.copy(pred_SCs_clsII)
            bst_SCinfo = copy.copy(pred_SCinfo_clsII)
            bst_SCavgnon = copy.copy(pred_SCavgnon_clsII)
            
            ireapt+=1
            
        else:
            break
        
    if (len(cache)  == 1) or (idx == Lis_bstidx[bstidx]):
        ####---------  show detailed results  --------------
        MI = metrics.adjusted_mutual_info_score(label_gt, bst_pt)
        print('After verification, MI:%.3f'%MI)
            
        pred_Lis_SC,pred_SCavg,_,pred_SCinfo =\
            utils.SilhouetteScore(hsres,bst_cls,sq_flag=sq_flag,scoreAry=scoreAry)
        idx_iso = np.where(pred_SCinfo[:,1] == -1)[0]  
        ans = pred_Lis_SC[idx_iso]
        
        print('Avg_SC:%.3f;nonindiv SC:%.3f,indiv SC:%.3f,num:%d'
              %(np.mean(pred_Lis_SC),pred_SCavg,np.mean(ans),len(ans)))  
            
        res_fullcorrect,res_partial,res_wrong,res_dtl_correct,res_dtl_partial,res_dtl_wrong =\
            utils.correct_denom_pred(hsres,bst_cls)
        
        correct_clusters = np.sum(res_fullcorrect)
        sum_correctImg = 0
        for inum,icorrect in enumerate(res_fullcorrect):
            inum += 1
            sum_correctImg += inum * icorrect
        
        print('predk:%d, correct num:%d, partial correct num:%d, wrong num:%d'
              %(len(bst_cls),sum(res_fullcorrect),len(res_partial),len(res_wrong)))
        print('correct image num:%d,ratio/n=%.3f'%(sum_correctImg,sum_correctImg/nsample))
        print('partial correct image:%d,wrong image:%d'%(np.sum(np.array(res_partial)[:,0]),
                                                 np.sum(np.array(res_wrong)[:,0])))
        ####-------------------------------------------
          
    info = 'verification \n'
    info += 'cluster num:%d, after %d repeated, num: %d \n'%\
          (len(predcls),ireapt,len(bst_cls))
    info += 'meanSC:%.3f; after verify, meanSC:%.3f \n'%\
          (np.mean(pred_SCs),bst_SCmean)
    info +='meanSC for nonindivi-clusters:%.3f; after verify:%.3f \n'%\
          (pred_SCavgnon,bst_SCavgnon)
    
    MI = metrics.adjusted_mutual_info_score(label_gt, bst_pt)

    info +='MI:%.3f; after verify, MI:%.3f \n'%\
          (MI_req,MI)

    print(info)
    
    Lis_reapt.append(ireapt)
    Lis_idx.append(idx)
    Lis_ncluster_bst.append(len(predcls))
    Lis_MI_bst.append(MI_req)
    Lis_SC_bst.append(pred_SCs)
    Lis_SCnon_bst.append(pred_SCavgnon)
    Lis_cls_bst.append(predcls)
    Lis_clspt_bst.append(predcls_pt)
    
    Lis_ncluster_upd.append(len(bst_cls))
    Lis_MI_upd.append(MI)
    Lis_SC_upd.append(bst_SCs)
    Lis_SCnon_upd.append(bst_SCavgnon)
    Lis_cls_upd.append(bst_cls)
    Lis_clspt_upd.append(bst_pt)
        
if len(cache) == len(Lis_bstidx):
    print('save file')
    name = '%d_%d_verify.npz'%(record,times)
    
    np.savez(join(hsres.data_dir,name),
              Lis_idx=Lis_idx,Lis_ncluster_bst=Lis_ncluster_bst,Lis_MI_bst=Lis_MI_bst,
              Lis_SC_bst=Lis_SC_bst,Lis_SCnon_bst=Lis_SCnon_bst,Lis_cls_bst=Lis_cls_bst,
              Lis_clspt_bst=Lis_clspt_bst,Lis_reapt = Lis_reapt,
              Lis_ncluster_upd=Lis_ncluster_upd,Lis_MI_upd=Lis_MI_upd,
              Lis_SC_upd=Lis_SC_upd,Lis_SCnon_upd=Lis_SCnon_upd,Lis_cls_upd=Lis_cls_upd,
              Lis_clspt_upd=Lis_clspt_upd,note=data['note'],nsample=data['nsample'])  


# In[Show the detailed results]

# record = 1002

name = '%d_verify.npz'%(record)
data = np.load(join(hsres.data_dir,name),allow_pickle=True)
# print(list(data.keys()))
print(data['note'])

Lis_SC_upd = data['Lis_SC_upd']
Lis_SC_upd = np.mean(Lis_SC_upd,axis=1)
Lis_SC_bst = data['Lis_SC_bst']
Lis_SC_bst = np.mean(Lis_SC_bst,axis=1)
Lis_mear_upd = data['Lis_MI_upd']
Lis_mear_bst = data['Lis_MI_bst']
Lis_SCnon_bst = data['Lis_SCnon_bst']
Lis_SCnon_upd = data['Lis_SCnon_upd']
k_bst = data['Lis_ncluster_bst']
k_upd = data['Lis_ncluster_upd']


idx = np.argmax(Lis_SC_bst)

print('before verification: k=%d, SC=%.3f, SCnon=%.3f, AdjustMI=%.3f'
      %(k_bst[idx],Lis_SC_bst[idx],Lis_SCnon_bst[idx],Lis_mear_bst[idx]))
print('after verification: k=%d, SC=%.3f, SCnon=%.3f, AdjustMI=%.3f'
      %(k_upd[idx],Lis_SC_upd[idx],Lis_SCnon_upd[idx],Lis_mear_upd[idx]))