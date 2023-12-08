#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 28 12:39:42 2023

@author: obaiga
"""

# In[Packs]
import numpy as np 
from os.path import join
import os
import copy
from sklearn import metrics
import matplotlib.pyplot as plt
import pandas as pd 

code_dir = '/Users/obaiga/github/Automatic-individual-animal-identification/'
os.chdir(code_dir)

import utils

# record = 1001
# mtd = 'k++'
record = 1002
mtd = 'adaptive k++'

record = 1005
mtd = 'spectral 3nn'
# record = 8201   ### k-medoids++
# record = 8202    ### adaptive k-medoids++

dpath = '/Users/obaiga/Jupyter/Python-Research/'
new_db = 'ds_160'


# dpath = '/Users/obaiga/Jupyter/Python-Research/Africaleopard'
# new_db = 'non_iso'

data_name = new_db
query_flag = 'vsmany_'
fg_flag = 'fg'   #### only containing animal body without background
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

# scoreAry = hsres.load_score(('ImgScore_%s%s%s.xlsx')%(query_flag[2:],fg_flag,data_mof))

# scoreAry = copy.copy(hsres.scoreAry)
# if sq_flag:
#     scoreAry = scoreAry**2 
    
# In[Load scorematrix]
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

times = 200

# record = 8202 #### adaptive k-medoids++

k = hsres.k_gt

name = '%d_%d_%s_%s_k%d.npz'\
    %(record,times,fg_flag,query_flag[2:-1],k)
      
try:
    data = np.load(join(hsres.data_dir,name),allow_pickle=True)
    # print(list(data.keys()))
    print('groundtruth:Avg_SC:%.4f;nonindiv SC:%.4f'%
      (data['SC_gt'],data['SCnon_gt']))
    
    print(record,data['note'])
    
    Lis_numconvg = data['Lis_numconvg']
    Lis_SC = data['Lis_SC']
    Lis_MI = data['Lis_MI']
    num = len(Lis_numconvg)
    sums = 0
    Lis_bstSC = []
    Lis_bstidx = []

    for i,num in enumerate(Lis_numconvg):
        
        maxidx = np.argmax(Lis_SC[sums:sums+num])
        Lis_bstSC.append(Lis_SC[maxidx+sums])
        Lis_bstidx.append(maxidx+sums)
        
        sums += Lis_numconvg[i]
    
    bstidx = np.argmax(Lis_bstSC)
    bst = Lis_bstidx[bstidx:bstidx+1]
    print('adjusted mutual info:%.3f,sc:%.3f'%(Lis_MI[bst],np.mean(Lis_SC[bst])))

except Exception:
    print('Did not load a data')   
    
    


# In[Load verify data]
try:

    name = '%d_%d_verify.npz'%(record,times)
    data = np.load(join(hsres.data_dir,name),allow_pickle=True)
    # print(list(data.keys()))
    
    Lis_ncluster = data['Lis_ncluster_bst']
    Lis_ncluster_upd = data['Lis_ncluster_upd']
    Lis_SC_upd = data['Lis_SC_upd']
    Lis_SC_bst = data['Lis_SC_bst']
    
    Lis_mear_upd = data['Lis_MI_upd']
    Lis_bstmear = data['Lis_MI_bst']
    
    
    ##----------choose the best index------------------
    Lis_bstSC = np.mean(Lis_SC_bst,axis=1)
    bstidx = np.argmax(Lis_bstSC)
    print('before verifiy, adjusted mutual info:%.3f,sc:%.3f'
          %(Lis_bstmear[bstidx],np.mean(Lis_SC_bst[bstidx]),))
    print('after verifiy, adjusted mutual info:%.3f,sc:%.3f'
          %(Lis_mear_upd[bstidx],np.mean(Lis_SC_upd[bstidx])))

except Exception:
    print('Did not load a data')   

#%%

name = '%d_%d_verify.npz'%(record,times)
data = np.load(join(hsres.data_dir,name),allow_pickle=True)

Lis_SC_bst = data['Lis_SC_bst']
Lis_bstSC = np.mean(Lis_SC_bst,axis=1)

idx = np.argmax(Lis_bstSC)

for i in [0,1]:
    
    if i == 0:
        mtd_fo = ''
        predcls = data['Lis_cls_bst'][idx]
        predcls_pt = data['Lis_clspt_bst'][idx]
        SC_bst = np.mean(data['Lis_SC_bst'][idx])
        SC_non_bst = data['Lis_SCnon_bst'][idx]
        mear_bst = data['Lis_MI_bst'][idx]
        # print('\nbefore verify')
        
    elif i==1:
        mtd_fo = '&verify'
        predcls = data['Lis_cls_upd'][idx]
        predcls_pt = data['Lis_clspt_upd'][idx]
        SC_bst = np.mean(data['Lis_SC_upd'][idx])
        SC_non_bst = data['Lis_SCnon_upd'][idx]
        mear_bst = data['Lis_MI_upd'][idx]
        # print('\nafter verify')
        
    res_fullcorrect,res_partial,res_wrong,res_dtl_correct,res_dtl_partial,res_dtl_wrong =\
        utils.correct_denom_pred(hsres,predcls)
    
    correct_clusters = np.sum(res_fullcorrect)
    sum_correctImg = 0
    for inum,icorrect in enumerate(res_fullcorrect):
        inum += 1
        sum_correctImg += inum * icorrect
    
    ncluster = len(predcls)
    corr_cls = sum(res_fullcorrect)
    part_cls = len(res_partial)
    incor_cls = len(res_wrong)
    corr_imgs = sum_correctImg
    part_imgs = np.sum(np.array(res_partial)[:,0])
    incor_imgs = np.sum(np.array(res_wrong)[:,0])
        
    file_name = 'summary_result.xlsx'
    file_path = '/Users/obaiga/Jupyter/Python-Research/Africaleopard/'
    df = pd.read_excel(join(file_path,file_name))
    new_row = {"data_name":data_name,
               "mtd_name":mtd+mtd_fo,
               "ncluster":ncluster,
               "correct":corr_cls,
               "partial":part_cls,
               "incorrect":incor_cls,
               "nimages":nsample,
               "corr_imgs":corr_imgs,
               "part_imgs":part_imgs,
               "incor_imgs":incor_imgs,
               "mutual_info":np.round(mear_bst,3),
               "silhouette":np.round(SC_bst,3),
               "s_noniso":np.round(SC_non_bst,3),
               "cor_cls_ratio":np.round(corr_cls/ncluster,3),
               "part_cls_ratio":np.round(part_cls/ncluster,3),
               "incor_cls_ratio":np.round(incor_cls/ncluster,3),
               "cor_img_ratio":np.round(corr_imgs/nsample,3),
               "part_img_ratio":np.round(part_imgs/nsample,3),
               "incor_img_ratio":np.round(incor_imgs/nsample,3)}
    
    df = df.append(new_row, ignore_index=True)
    df.to_excel(join(file_path,file_name), index=False)
    
    
# In[write table columns]
# import pandas as pd
# column_names = ["data_name", "mtd_name", "mutual_info","silhouette","s_noniso",
#                 "ncluster","correct","partial","incorrect",
#                 "cor_cls_ratio","part_cls_ratio","incor_cls_ratio",
#                 "nimages","corr_imgs","part_imgs","incor_imgs",
#                 "cor_img_ratio","part_img_ratio","incor_img_ratio"]
# df = pd.DataFrame(columns=column_names)
# file_name = 'summary_result.xlsx'
# file_path = '/Users/obaiga/Jupyter/Python-Research/Africaleopard/'
# df.to_excel(join(file_path,file_name), index=False)

    
#%%

# cls_lis = data['Lis_cls'][bst][0]
    
# clspt_lis = data['Lis_predcls_pt'][bst]

# clspt_lis =  list(clspt_lis)[0]

res_fullcorrect,res_partial,res_wrong,res_dtl_correct,res_dtl_partial,res_dtl_wrong =\
        utils.correct_denom_pred(hsres,cls_lis)
    
MI = metrics.adjusted_mutual_info_score(label_gt, clspt_lis)

correct_clusters = np.sum(res_fullcorrect)
sum_correctImg = 0
for inum,icorrect in enumerate(res_fullcorrect):
    inum += 1
    sum_correctImg += inum * icorrect

ncluster = len(cls_lis)
corr_cls = sum(res_fullcorrect)
part_cls = len(res_partial)
incor_cls = len(res_wrong)
corr_imgs = sum_correctImg
part_imgs = np.sum(np.array(res_partial)[:,0])
incor_imgs = np.sum(np.array(res_wrong)[:,0])

file_name = 'summary_result.xlsx'
file_path = '/Users/obaiga/Jupyter/Python-Research/Africaleopard/'
df = pd.read_excel(join(file_path,file_name))
new_row = {"data_name":data_name,
           "mtd_name":mtd,
           "ncluster":ncluster,
           "correct":corr_cls,
           "partial":part_cls,
           "incorrect":incor_cls,
           "nimages":nsample,
           "corr_imgs":corr_imgs,
           "part_imgs":part_imgs,
           "incor_imgs":incor_imgs,
           "mutual_info":np.round(MI,3),
           "cor_cls_ratio":np.round(corr_cls/ncluster,3),
           "part_cls_ratio":np.round(part_cls/ncluster,3),
           "incor_cls_ratio":np.round(incor_cls/ncluster,3),
           "cor_img_ratio":np.round(corr_imgs/nsample,3),
           "part_img_ratio":np.round(part_imgs/nsample,3),
           "incor_img_ratio":np.round(incor_imgs/nsample,3)}

df = df.append(new_row, ignore_index=True)
df.to_excel(join(file_path,file_name), index=False)