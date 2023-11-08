#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep  1 11:21:22 2022

Modify on 01/17/2023

modify on 03/07/2023 
    for repeat verification step until cannot improve    
    
modify on 03/14/2023
    merge two parts (optimal k & fixed k ) to one     
    
### comment: these chipID do not have any matched descriptors with other images
  89,  155,  267,  618,  662,  672,  752,  838, 1075, 1505, 1532
@author: obaiga
"""

# In[Packs]
import numpy as np 
from os.path import join
import os
import copy
from sklearn import metrics

# dpath = 'C:\\Users\\95316\\code1'
# dpath = 'C:\\Users\\SHF\\code1'
dpath= '/Users/obaiga/Jupyter/Python-Research'
os.chdir(join(dpath,'test'))

import utils

new_db = 'ds_160'
query_flag = 'vsmany_'
fg_flag = 'fg'
data_mof = '_diag'
sq_flag = True
used_nonzero = False
# print('used_nonzero:%s'%used_nonzero)

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
    
nsample = hsres.nsample
ncluster = hsres.k_gt

scoreAry = hsres.load_score(('ImgScore_%s%s%s%s.xlsx')%
                              (query_flag[2:],fg_flag,nonzero_flag,data_mof))


# if sq_flag:
#     scoreAry = copy.copy(hsres.scoreAry)
#     scoreAry = scoreAry**2
#     ##### scoreAry[np.arange(nsample),np.arange(nsample)] = 0
#     scoreAry[np.arange(nsample),np.arange(nsample)] = np.sum(scoreAry,axis=1)

if sq_flag:
    scoreAry = copy.copy(hsres.scoreAry)
    scoreAry = scoreAry**2
    
# In[Load scorematrix]

if scoreAry is not None:
    Lis_SC_gt,SCavg_gt,T2_gt,Lis_SCcls_gt,SC_info_gt = \
        utils.SilhouetteScore(hsres,cluster_gt,sq_flag=sq_flag,scoreAry=scoreAry)
    
    centrd_gt,centroid_ssum_gt = utils.centroid(hsres,cluster_gt)
    TWCV_gt = np.sum(centroid_ssum_gt)
    
    print('groundtruth:Avg_SC:%.4f;nonindiv SC:%.4f,T2:%d'
          %(np.mean(Lis_SC_gt),SCavg_gt,np.sqrt(T2_gt))) 
    
    idx_iso = np.where(SC_info_gt[:,1] == -1)[0]  
    ans = Lis_SC_gt[idx_iso]
    print('indiv SC:%.4f,num:%d'%(np.mean(ans),len(ans))) 

# In[load data from fixed k]
'''
Run verification step 
for fixed k value 
'''
if 1:
    # times = 20000
    # record = 8001     ### for benchmark k++, 200iterations * 100 repeated times
    # record = 8002     ### for adaptive k++
    # k = ncluster
    
    times = 200
    # record = 8202
    k = hsres.k_gt
    record = 8201 
    ### for benchmark k++, determine the best clustering by ternary search
    # k =678
    
    
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
Lis_T2_bst = []
Lis_T2_upd = []


cache = copy.copy(Lis_bstidx)

# Lis_bstSC_sort = np.argsort(Lis_bstSC)
# bstidx = Lis_bstSC_sort[int(len(Lis_bstSC)/2)+1]

bstidx = np.argmax(Lis_bstSC)

cache = Lis_bstidx[bstidx:bstidx+1]

Lis_cls = data['Lis_cls'][cache]

for i,idx in enumerate(cache):
    # print(i,idx)
    # RI_req = data['Lis_RI'][idx]
    MI_req = Lis_MI[idx]
    # FMS_req = data['Lis_FMS'][idx]
    predcls,predcls_pt,CountInfo_pred = utils.load_info_pred\
        (hsres,Lis_cls[i],nsample)
    
    pred_SCs,pred_SCavgnon,T2,_,pred_SCinfo =\
        utils.SilhouetteScore(hsres,predcls,scoreAry=scoreAry)
    nsample = len(pred_SCs)
    Lis_T2_bst.append(T2)
    
    if (np.mean(pred_SCs)-np.mean(Lis_SC[idx])) < 1e-6:
        print('i:%d,idx:%d,correct,T2:%d'%(i,idx,np.sqrt(T2)))
    else:
        print('i:%d,idx:%d,wrong,re-cal:%.3f,sc:%.3f '
              %(i,idx,np.mean(pred_SCs),np.mean(Lis_SC[idx])))
    # if (len(cache)  == 1) or (idx == Lis_bstidx[bstidx]):
    #     MI = metrics.adjusted_mutual_info_score(label_gt, predcls_pt)
    #     print('before verification, MI:%.3f'%MI)
            
    #     pred_Lis_SC,pred_SCavg,T2,_,pred_SCinfo =\
    #         utils.SilhouetteScore(hsres,predcls,sq_flag=sq_flag,scoreAry=scoreAry)
    #     idx_iso = np.where(pred_SCinfo[:,1] == -1)[0]  
    #     ans = pred_Lis_SC[idx_iso]
        
    #     print('Avg_SC:%.3f;nonindiv SC:%.3f,indiv SC:%.3f,num:%d, T2:%d'
    #           %(np.mean(pred_Lis_SC),pred_SCavg,np.mean(ans),len(ans),np.sqrt(T2)))  
            
    #     res_fullcorrect,res_partial,res_wrong,res_dtl_correct,res_dtl_partial,res_dtl_wrong =\
    #         utils.correct_denom_pred(hsres,predcls)
        
    #     correct_clusters = np.sum(res_fullcorrect)
    #     sum_correctImg = 0
    #     for inum,icorrect in enumerate(res_fullcorrect):
    #         inum += 1
    #         sum_correctImg += inum * icorrect
    #     print('predk:%d, correct num:%d, partial correct num:%d, wrong num:%d'
    #           %(len(predcls),sum(res_fullcorrect),len(res_partial),len(res_wrong)))
    #     print('correct image num:%d,ratio/n=%.3f'%(sum_correctImg,sum_correctImg/nsample))
    #     print('partial correct image:%d,wrong image:%d'%(np.sum(np.array(res_partial)[:,0]),
    #                                                      np.sum(np.array(res_wrong)[:,0])))
        # print(np.mean(pred_SCs),np.mean(Lis_SC[idx]))
 
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
    # while (ireapt<3):
    while (1):
        
        clsI_cache,res_checkI = utils.merge_clusters\
            (scoreAry,prev_cls,prev_pt,prev_SClis,prev_SCinfo)
        
        # clsI_cache,res_checkI = utils.lowSC_solve\
        #     (prev_cls,prev_SClis,prev_SCinfo,prev_SCavgnon)
            
        clsI,clsI_pt,CountInfo_clsI = utils.load_info_pred\
            (hsres,clsI_cache,nsample)
            
        pred_SCs_clsI,pred_SCavgnon_clsI,T2,_,pred_SCinfo_clsI = \
            utils.SilhouetteScore(hsres,clsI,scoreAry=scoreAry)
        
        # RI_I = metrics.adjusted_rand_score(label_gt, clsI_pt)
        # MI_I = metrics.adjusted_mutual_info_score(label_gt, clsI_pt)
        # FMS_I = metrics.fowlkes_mallows_score(label_gt, clsI_pt) 
        #--------------------------------
        prev_cls = copy.deepcopy(clsI)
        prev_SCinfo = copy.deepcopy(pred_SCinfo_clsI)
        prev_SClis = copy.deepcopy(pred_SCs_clsI)
        prev_pt = copy.deepcopy(clsI_pt)
        
        clsII_cache,res_checkII = utils.lowSC_solve\
            (prev_cls,prev_SClis,prev_SCinfo,prev_SCavgnon)
        
        # clsII_cache,res_checkII = utils.merge_clusters\
        #     (scoreAry,prev_cls,prev_pt,prev_SClis,prev_SCinfo)
            
        clsII,clsII_pt,CountInfo_clsII = utils.load_info_pred\
            (hsres,clsII_cache,nsample)
            
        pred_SCs_clsII,pred_SCavgnon_clsII,T2,_,pred_SCinfo_clsII = \
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
            
        pred_Lis_SC,pred_SCavg,T2,_,pred_SCinfo =\
            utils.SilhouetteScore(hsres,bst_cls,sq_flag=sq_flag,scoreAry=scoreAry)
        idx_iso = np.where(pred_SCinfo[:,1] == -1)[0]  
        ans = pred_Lis_SC[idx_iso]
        
        print('Avg_SC:%.3f;nonindiv SC:%.3f,indiv SC:%.3f,num:%d, T2:%d'
              %(np.mean(pred_Lis_SC),pred_SCavg,np.mean(ans),len(ans),np.sqrt(T2)))  
            
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
        
    #------show details------------
    # info = 'verification \n'
    # info += 'cluster num:%d, after step I, num: %d, after step II, num: %d \n'%\
    #       (len(predcls),len(clsI),len(clsII))
    # info += 'meanSC:%.3f; after setp I, meanSC:%.3f;after step II, meanSC:%.3f \n'%\
    #       (np.mean(pred_SCs),np.mean(pred_SCs_clsI),np.mean(pred_SCs_clsII))
    # info +='meanSC for nonindivi-clusters:%.3f;after step I, meanSC:%.3f; after step II, meanSC:%.3f \n'%\
    #       (pred_SCavgnon,pred_SCavgnon_clsI,pred_SCavgnon_clsII)
          
    info = 'verification \n'
    info += 'cluster num:%d, after %d repeated, num: %d \n'%\
          (len(predcls),ireapt,len(bst_cls))
    info += 'meanSC:%.3f; after verify, meanSC:%.3f \n'%\
          (np.mean(pred_SCs),bst_SCmean)
    info +='meanSC for nonindivi-clusters:%.3f; after verify:%.3f \n'%\
          (pred_SCavgnon,bst_SCavgnon)
    
    # RI = metrics.adjusted_rand_score(label_gt, clsII_pt)
    MI = metrics.adjusted_mutual_info_score(label_gt, bst_pt)
    # MI = metrics.adjusted_mutual_info_score(label_gt, clsII_pt)
    # FMS = metrics.fowlkes_mallows_score(label_gt, clsII_pt) 
    
    # info +='RI:%.3f; after step I, RI:%.3f; after step II, RI:%.3f \n'%\
    #       (RI_req,RI_I,RI)
    info +='MI:%.3f; after verify, MI:%.3f \n'%\
          (MI_req,MI)
    # info +='FMS:%.3f; after step I, FMS:%.3f; after step II, FMS:%.3f \n'%\
    #       (FMS_req,FMS_I,FMS)
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
              Lis_clspt_upd=Lis_clspt_upd,note=data['note'],nsample=data['nsample'],
              Lis_T2_bst=Lis_T2_bst, Lis_T2_upd=Lis_T2_upd)  

#%%


# In[Show the detailed results]

# record = 8002  ##### old silhouette score, adaptive k++, fixed k value
record = 8005  ##### new silhouette score, adaptive k++, fixed k value
# record = 8001  ##### new silhouette score, k++ (benchmark)
# record = 8007  ##### old silhouette score, k++ (benchmark)

# record = 8101  ##### k++ (benchmark)

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