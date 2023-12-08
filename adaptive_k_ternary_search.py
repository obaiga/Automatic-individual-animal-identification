#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May  4 16:29:22 2022

Modify on 11/06/23

Adaptive k-medoids++ clustering algorithm in a ternary search to determine the best clustering

@author: obaiga
"""

# In[Packs]
import numpy as np 
from os.path import join
import os
import copy
import time
from sklearn import metrics
import warnings
warnings.filterwarnings("ignore")
import random

# code_dir = 'C:/Users/95316/seal'
code_dir = '/Users/obaiga/github/Automatic-individual-animal-identification/'
os.chdir(code_dir)

import utils

# dpath = 'C:/Users/95316/seal'
dpath = '/Users/obaiga/Jupyter/Python-Research/Africaleopard/'
new_db = 'snow leopard'
# new_db = 'non_iso'

query_flag = 'vsmany_'
fg_flag = 'fg'   #### only containing animal body without background
# fg_flag = ''
# data_mof = '_mean'    #### modify similarity score matrix '_mean' or '' or '_diag'
# data_mof = ''
data_mof = '_diag'

sq_flag = True    #### True: the square of a similarity score
clsmtd = '++'   #### k-medoids++
# clsmtd = 'origin'

scfeed = False   ##### whether using adaptive k++ (True) or k++ (False)
# scfeed = True

# used_nonzero = True
print('data format:%s'%data_mof[1:])   
#### '_diag': the diagonal value is the sum of the simailrity scores for an image

# In[Load]
db_dir = join(dpath,new_db)
hsres = utils.HotspotterRes(db_dir)
hsres.res_dir = join(hsres.db_dir,'results')
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
        

    
# In[Kcentroid]:
# =============================================================================
#             Adaptive k-medoids++ clustering 
# =============================================================================
# warnings.filterwarnings("ignore")
updateprob_bstSC = True  #### True: update weight factor only when best silhouette score updates

record = 1003

ConvgTimes = 50
learnsTimes = 500
fixedTimes = 1
totalTimes = int(learnsTimes*fixedTimes)

hsres.data_dir = join(hsres.res_dir,'data')
utils.CheckDir(hsres.data_dir)

# ===============================
#  Noted information
# ===============================  
print('record:%d,%s,k_gt=%d,nsample=%d,TotalTimes=%d'%\
      (record,query_flag[:-1],ncluster,nsample,totalTimes))

if scfeed:
    others='Convergence itertaions = %d, otherwise running iteration=%d; ternary search'\
        % (ConvgTimes,learnsTimes)
else:
    others = 'ternary search'
    
note = utils.prints\
    (hsres,clsmtd,data_mof,scfeed,fg_flag,sq_flag,updateprob_bstSC,others=others)
print(note)
# print('set_thred=%d'%set_thres)

#%%
# ===============================
#  Noted information
# =============================== 
# start_time = time.process_time()

Lis_initprob = []
Lis_cls = []
Lis_centrd = []
Lis_seed = []
Lis_TWCV = []
Lis_RI = []
Lis_MI = []
Lis_FMS = []
Lis_SC = []
Lis_SCnon = []
Lis_spendt = []
Lis_iters = []
# Lis_countseed = []
Lis_predcls_pt = []
Lis_learnsTimes = []
Lis_numconvg = []
Lis_upditer = []
Lis_ncluster = []
Lis_SCbst = []

##---------------
##    Optimal k
##---------------
leftp = 1
rightp = nsample
iters = 500

start = time.perf_counter()

while (1):
    if rightp - leftp <= 1:
        break
    rag = np.round((rightp - leftp)/3)
    m1 = leftp + rag
    m2 = rightp - rag
    print('m1=%d, m2=%d, left=%d, right=%d'%(m1,m2,leftp,rightp))
    
    warnings.filterwarnings("error")
    SC_comp = []
    for nclusterPot in [m1,m2]:
        nclusterPot = int(nclusterPot)
        # itimes = 0
        # while(itimes<fixedTimes):
        #     itimes += 1
        ilearn = 0
        # ===============================
        # Initialization k-centroid 
        # ===============================
        init_prob = np.ones(nsample)
        # countseed = np.zeros(nsample)
        SC_bst = 0
        SCnon_bst = 0
        iSCbst_convg = 0
        init_prob_bst = np.ones(nsample)
        upditer = []   
            
        ###  learnsTimes: Maximum iteration 
        #### & ConvgTimes:convergence times after find a clustering with best SC
        while((ilearn<learnsTimes) & (iSCbst_convg<ConvgTimes)):
            ilearn += 1
            
            if not scfeed:
                '''keep probability as equal'''
                init_prob = np.ones(nsample)
                
                
            if clsmtd == hsres.mtdplus:
                '''
                k-means++ initialization
                '''
                seeds = []
                centrd_pool = np.arange(nsample)
                try:
                    icentrd = random.choices(centrd_pool,weights=init_prob)[0]
                except Exception as e:
                    print(e)
                    print('Interrupted times:%d'%ilearn)
                seeds.append(icentrd)
                
                for ii in range(nclusterPot-1):
                    centrd_pool = np.delete(centrd_pool,np.argwhere(centrd_pool==icentrd))
                    scores_candi = scoreAry[np.ix_(centrd_pool,seeds)]
                    scores_max = np.max(scores_candi,axis=1)  ## closest neighbor
                
                    scores_max[scores_max==0] = 1e-6
                    prob_weight = 1/scores_max 
                    try:
                        if scfeed:
                            '''adaptive k++ (update init_prob based on previous clustering performance)'''
                            prob_weight *= init_prob[centrd_pool]
                            
                        icentrd = random.choices(centrd_pool,weights=prob_weight)[0]
                    except Exception as e:
                        print(e)
                        print('Interrupted times:%d'%(ilearn))
                    
                    seeds.append(icentrd)
                    
            elif clsmtd == hsres.mtdori:
                '''
                k-means initialization
                '''
                centrd_pool = np.arange(nsample)
                try:
                    init_norm = init_prob/np.sum(init_prob)
                    seeds = np.random.choice(centrd_pool,nclusterPot,replace=False, p=init_norm)
                except Exception as e:
                    print (e)
                    print('Interrupted times:%d'%(ilearn))
                        
            seeds = np.array(seeds)
            centrd_upd = list(copy.copy(seeds))
        
            iiter = 0
            flag_not_convergence = False
            while(1):
            # while(iiter < iters):
                iiter += 1
                if iiter == iters-1:
                    print('Cannot convergence')
                    flag_not_convergence = True
                    
                # ========================================
                #  Assignment
                # ========================================
                cls_upd = []
                point = list(np.arange(nsample))
                point = list(set(point).difference(set(centrd_upd)))
                point = np.array(point)
                score_pt2med = hsres.scoreAry[np.ix_(point,centrd_upd)]
                score_bst = np.max(score_pt2med,axis=1)
                idx_max = np.argmax(score_pt2med,axis=1)
                ans = np.where(score_bst<1)[0]
                if len(ans) > 0:
                    for ians in ans:
                        randcentrd = random.randint(0,nclusterPot-1)
                        idx_max[ians] = randcentrd
                    # print('point no nearest centroid,that is score=0,idx:%d'%int(len(ans)))
                    
                for ii in range(nclusterPot):
                    cache = np.where(idx_max==ii)[0]
                    if len(cache)>0:
                        cls_upd.append(np.append(centrd_upd[ii],point[cache]))
                    else:
                        cls_upd.append(np.array([centrd_upd[ii]]))
                # ===========================================
                #  Re-centroid
                # ===========================================
                centrd_new,centrd_ssum = utils.centroid(hsres,cls_upd)      
                
                # ==========================================
                #  Convergence               
                # ==========================================
                diff = list(set(centrd_new) - set(centrd_upd))
                if (len(diff)==0) or (flag_not_convergence == True): 
                    centrd_upd = np.array(centrd_upd)
                    # print('predicted_centroid:',new_centroid_)

                    #-------------------------------------------------------
                    '''calculate measurement'''
                    ### calculate silhouette score
                    pred_SCs,pred_SCavgnon,pred_SCcls,_ =\
                        utils.SilhouetteScore(hsres,cls_upd,scoreAry=scoreAry)
                    # pred_SCs,pred_SCavgnon,pred_T2,pred_SCcls,_ =\
                    #     utils.SilhouetteScore(hsres,cls_upd)
                    
                    ## assign all points with a cluster ID
                    predcls_pt = utils.cls2Lispt(hsres,cls_upd)
                    
                    RI = metrics.adjusted_rand_score(label_gt, predcls_pt)
                    MI = metrics.adjusted_mutual_info_score(label_gt, predcls_pt)
                    FMS = metrics.fowlkes_mallows_score(label_gt, predcls_pt)
                    
                    Lis_TWCV.append( int(np.sum(centrd_ssum)))
                    Lis_RI.append(RI)
                    Lis_MI.append( MI)
                    Lis_FMS.append( FMS)
                    pred_SCavg = np.mean(pred_SCs)
                    Lis_SC.append(pred_SCavg)
                    Lis_SCnon.append(pred_SCavgnon)
                    
                    #-------------------------------------------------------
                    '''record all information'''
                    Lis_predcls_pt.append(predcls_pt)

                    Lis_cls.append(cls_upd)
                    Lis_centrd.append(centrd_upd)
                    Lis_seed.append(seeds)
                    Lis_iters.append(iiter)
                    Lis_initprob.append(copy.copy(init_prob))
                    
                    # countseed[seeds] += 1
                    #-------------------------------------
                    if scfeed:
                        '''
                        adaptive algorithm
                        calculate the probability for an image as seed medoid 
                        '''
                        pred_centrd = np.zeros(nsample)
                        pred_centrd[centrd_upd] = pred_SCcls
                        # pred_centrd[seeds] = pred_SCcls
                        pred_centrd += 1  ## range from [-1,1] to [0,2]
                        try:
                            # init_prob[seeds] = init_prob[seeds] * pred_centrd[seeds]
                            init_prob[centrd_upd] = init_prob[centrd_upd] * pred_centrd[centrd_upd]
                        except Exception as e:
                            print (e)
                            print('Interrupted times:%d'%(ilearn))
                            break
                        ###########---------addition--------
                        if updateprob_bstSC:
                            '''update medoid probability only in the best clustering'''
                            if pred_SCavg > SC_bst:
                                upditer.append(ilearn)
                                iSCbst_convg = 0
                                SC_bst = copy.copy(pred_SCavg)
                                MI_bst = copy.copy(MI)
                                SCnon_bst = pred_SCavgnon
                                init_prob_bst = copy.copy(init_prob) 
                            else:
                                init_prob = copy.copy(init_prob_bst)
                                iSCbst_convg += 1
                    
                        ###########-------------------------   
                    break
                else:   
                    centrd_upd = copy.copy(centrd_new)
        
        end = time.perf_counter()        ## performance counter for benchmarking (better)
        localtime = time.asctime(time.localtime(time.time()) )
        print(localtime)
        print('k_pred=%d,spend time:%.2f mins,range=%d '
              %(nclusterPot,(end-start)/60,rag))
        
        if scfeed:
            '''adaptive algorithm'''
            print('bstSC=%.3f,bstSCnon=%.3f;bst_MI=%.3f;iteration=%d'%(SC_bst,SCnon_bst,MI_bst,upditer[-1]))
        else:
            i = int(len(Lis_SC)/learnsTimes)
            idx = np.argmax(Lis_SC[(i-1)*learnsTimes:learnsTimes*i])
            bstidx = int((i-1)*learnsTimes+idx)
            SC_bst = Lis_SC[bstidx]
            MI_bst = Lis_MI[bstidx]
            SCnon_bst = Lis_SCnon[bstidx]
            print('SC_bst=%.3f,bstSCnon=%.3f;bst_MI=%.3f;bstidx=%d'%(SC_bst,SCnon_bst,MI_bst,bstidx))
        
        SC_comp.append(SC_bst)
        Lis_numconvg.append(ilearn)
        Lis_upditer.append(upditer)
        # Lis_countseed.append(countseed)        
        Lis_ncluster.append(nclusterPot)
        Lis_SCbst.append(SC_bst)


        
    if SC_comp[0] < SC_comp[1]:
        leftp = m1
        rightp = rightp
    elif SC_comp[0] > SC_comp[1]:
        leftp = leftp
        rightp = m2
    elif SC_comp[0] == SC_comp[1]:
        leftp = m1
        rightp = m2
        
    print('m1_SC=%.3f,m2_SC=%.3f,leftp=%d,rightp=%d'%(SC_comp[0],SC_comp[1],leftp,rightp))
        
    # =============================================================================
    #         SAVE Outputs
    # =============================================================================
    warnings.filterwarnings("ignore")
    np.savez((join(hsres.data_dir,'%d_%d_%s_%s_k%d'
                   %(record,totalTimes,fg_flag,query_flag[2:-1],ncluster))),
             Lis_TWCV=Lis_TWCV,Lis_RI=Lis_RI,Lis_MI=Lis_MI,Lis_FMS=Lis_FMS,
             Lis_SC=Lis_SC,Lis_SCnon=Lis_SCnon,Lis_cls=Lis_cls,
             Lis_centrd=Lis_centrd,Lis_seed=Lis_seed,
             SC_gt=np.mean(Lis_SC_gt),SCnon_gt=SCavg_gt,TWCV_gt=TWCV_gt,
             Lis_initprob=Lis_initprob,Lis_iters=Lis_iters, 
             Lis_spendt=Lis_spendt,Lis_predcls_pt=Lis_predcls_pt,
             Lis_numconvg=Lis_numconvg,Lis_upditer=Lis_upditer,note=note,nsample=nsample,
             Lis_SCbst=Lis_SCbst,Lis_ncluster=Lis_ncluster,) 

#%%
warnings.filterwarnings("ignore")
# =============================================================================
#         SAVE Outputs
# =============================================================================
warnings.filterwarnings("ignore")
np.savez((join(hsres.data_dir,'%d_%d_%s_%s_k%d'
               %(record,totalTimes,fg_flag,query_flag[2:-1],ncluster))),
         Lis_TWCV=Lis_TWCV,Lis_RI=Lis_RI,Lis_MI=Lis_MI,Lis_FMS=Lis_FMS,
         Lis_SC=Lis_SC,Lis_SCnon=Lis_SCnon,Lis_cls=Lis_cls,
         Lis_centrd=Lis_centrd,Lis_seed=Lis_seed,
         SC_gt=np.mean(Lis_SC_gt),SCnon_gt=SCavg_gt,TWCV_gt=TWCV_gt,
         Lis_initprob=Lis_initprob,Lis_iters=Lis_iters, 
         Lis_spendt=Lis_spendt,Lis_predcls_pt=Lis_predcls_pt,
         Lis_numconvg=Lis_numconvg,Lis_upditer=Lis_upditer,note=note,nsample=nsample,
         Lis_SCbst=Lis_SCbst,Lis_ncluster=Lis_ncluster,) 