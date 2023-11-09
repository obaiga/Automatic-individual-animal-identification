#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May  4 16:29:22 2022

Modified on 01/12/23

Modify on 03/06/23
The best matched image for an image requires its similarity score no less than the mean 

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
dpath = '/Users/obaiga/Jupyter/Python-Research/SealID/'
new_db = 'seal_hotspotter'

query_flag = 'vsmany_'
# fg_flag = 'fg'   #### only containing animal body without background
fg_flag = ''
# data_mof = '_mean'    #### modify similarity score matrix '_mean' or '' or '_diag'
# data_mof = ''
data_mof = '_diag'

sq_flag = True    #### True: the square of a similarity score
clsmtd = '++'   #### adaptive k++
# clsmtd = 'origin'

print('data format:%s'%data_mof[1:])   
#### '_diag': the diagonal value is the sum of the simailrity scores for an image

# In[Load]
db_dir = join(dpath,new_db)
hsres = utils.HotspotterRes(db_dir)
hsres.res_dir = join(hsres.db_dir,'results')
utils.CheckDir(hsres.res_dir)
Lis_Chip2ID,Lis_Img,Lis_chipNo,Lis_ID = hsres.load_info()

cluster_gt,label_non_gt = hsres.load_cluster_gt(query_flag,fg_flag)
idxNonConnect = None

nsample = hsres.nsample
ncluster = hsres.k_gt
label_gt = hsres.label_gt
    
print('nsample:%d, ncluster:%d'%(nsample,ncluster))

scoreAry = hsres.load_score(('ImgScore_%s%s%s.xlsx')%
                              (query_flag[2:],fg_flag,data_mof))

scoreAry = copy.copy(hsres.scoreAry)
if sq_flag:
    scoreAry = scoreAry**2
# In[calculate ground truth silhouette]

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
    
    
# In[Kcentroid]:
# =============================================================================
#             K-medoids++ clustering 
# =============================================================================
# warnings.filterwarnings("ignore")
record = 2002
repeatTimes = 10

learnsTimes = 500

if clsmtd == hsres.mtdori:
    ConvgTimes = learnsTimes
elif clsmtd == hsres.mtdplus:
    ConvgTimes = 200
    
totalTimes = int(learnsTimes*repeatTimes)

hsres.data_dir = join(hsres.res_dir,'data')
utils.CheckDir(hsres.data_dir)

##### for debug (please do not change)
scfeed = True   ##### whether using adaptive k++ (True) or k++ (False)
updateprob_bstSC = True  #### True: update weight factor only when best silhouette score updates
#######

# ===============================
#  Noted information
# ===============================  
print('record:%d,%s,k_gt=%d,nsample=%d,repeat alg=%d'%\
      (record,query_flag[:-1],ncluster,nsample,repeatTimes))
if scfeed & (clsmtd == hsres.mtdplus):
    others='Convergence times = %d, otherwise running iteration=%d; update cluster medoids '\
        % (ConvgTimes,learnsTimes)
else:
    others = 'repeating running=%d' %(learnsTimes)
    
note = utils.prints\
    (hsres,clsmtd,data_mof,scfeed,fg_flag,sq_flag,updateprob_bstSC,others=others)
print(note)
# print('set_thred=%d'%set_thres)

# ===============================
#  Noted information
# =============================== 
start = time.perf_counter()
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

#%%
warnings.filterwarnings("ignore")

itimes = 0
while(itimes<repeatTimes):
    warnings.filterwarnings("error")
    itimes += 1
    ilearn = 0
    # ===============================
    # Initialization k-centroid 
    # ===============================
    init_prob = np.ones(nsample)
    # countseed = np.zeros(nsample)
    SC_bst = -1
    # SCnon_bst = 0
    iSCbst_convg = 0
    init_prob_bst = np.ones(nsample)
    upditer = []   
    
    while((ilearn<learnsTimes) & (iSCbst_convg<ConvgTimes) ):
        
        ilearn += 1
        ####### judge whether using adaptive clustering algorithm
        if not scfeed:
            ##### if not, judge whether the probability is equl to 1
            if init_prob.all() == 1:
                pass
            else:
                print('init_prob WRONG!')
                break
        #######---------------------------------------------------
        ####### determine clusteirng initilization method: k-means++ or k-means 
        if clsmtd == hsres.mtdplus:
            '''
            k-means++ initialization
            '''
            seeds = []
            centrd_pool = np.arange(nsample)
            ##### the first initial cluster medoid
            try:
                icentrd = random.choices(centrd_pool,weights=init_prob)[0]
            except Exception as e:
                print(e)
                print('Interrupted times:%d'%itimes)
            seeds.append(icentrd)
            
            for ii in range(ncluster-1):
                centrd_pool = np.delete(centrd_pool,np.argwhere(centrd_pool==icentrd))
                scores_candi = scoreAry[np.ix_(centrd_pool,seeds)]
                scores_max = np.max(scores_candi,axis=1)  ## closest medoids
                scores_max[scores_max==0] = 1e-6
                prob_weight = 1/scores_max       ### descending 
                try:
                    ### judge whether using weight factor
                    if scfeed:
                        prob_weight *= init_prob[centrd_pool]
                        
                    icentrd = random.choices(centrd_pool,weights=prob_weight)[0]
                except Exception as e:
                    print(e)
                    print('Interrupted times:%d'%(itimes))
                
                seeds.append(icentrd)
        #######---------------------------------------------------    
        elif clsmtd == hsres.mtdori:
            '''
            k-means initialization
            '''
            centrd_pool = np.arange(nsample)
            try:
                init_norm = init_prob/np.sum(init_prob)
                seeds = np.random.choice(centrd_pool,ncluster,replace=False, p=init_norm)
            except Exception as e:
                print (e)
                print('Interrupted times:%d'%(itimes))
        #######---------------------------------------------------
        seeds = np.array(seeds)
        centrd_upd = list(copy.copy(seeds))
    
        iiter = 0
        while(1):
        # while(iiter < iters):
            iiter += 1
            # if iiter == iters-1:
            #     print('Cannot convergence')
            # ========================================
            #  Assignment
            # ========================================
            cls_upd = []
            point = list(np.arange(nsample))   ### non-medoid points
            point = list(set(point).difference(set(centrd_upd)))
            point = np.array(point)
            
            score_pt2med = hsres.scoreAry[np.ix_(point,centrd_upd)]
            score_bst = np.max(score_pt2med,axis=1)
            idx_bst = np.argmax(score_pt2med,axis=1)
            
            ### the similarity score for the image with the assigned medoid < 1
            ans = np.where(score_bst<1)[0]
            if len(ans) > 0:   
                for ians in ans:
                    randcentrd = random.randint(0,ncluster-1)
                    ### the best medoid is randomly picked 
                    idx_bst[ians] = randcentrd
                # print('point no nearest centroid,that is score=0,idx:%d'%int(len(ans)))
            
            #### assign all non-medoid to the nearest medoids, construct to a cluster
            for ii in range(ncluster):
                cache = np.where(idx_bst==ii)[0]
                if len(cache)>0:
                    cls_upd.append(np.append(centrd_upd[ii],point[cache]))
                else:
                    ### isolated clusters
                    cls_upd.append(np.array([centrd_upd[ii]]))
                    
            # ===========================================
            #  Re-centroid
            # ===========================================
            centrd_new,centrd_ssum = utils.centroid(hsres,cls_upd)      
            
            # ==========================================
            #  Convergence               
            # ==========================================
            diff = list(set(centrd_new) - set(centrd_upd))
            if (len(diff)==0): 
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
                if clsmtd == hsres.mtdplus:
                    '''
                    calculate weight factor for an image as seed medoid 
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
                        print('Interrupted times:%d'%(itimes))
                        break
                    ####### judge whether only update when an optimal clusteirng 
                    
                    if pred_SCavg > SC_bst:
                        SC_bst = copy.copy(pred_SCavg)
                        SCnon_bst = pred_SCavgnon
                        MI_bst = copy.copy(MI)    
                        if updateprob_bstSC:
                            upditer.append(ilearn)
                            iSCbst_convg = 0
                            init_prob_bst = copy.copy(init_prob) 
                    elif (pred_SCavg <= SC_bst) & updateprob_bstSC:
                        init_prob = copy.copy(init_prob_bst)
                        iSCbst_convg += 1
                    #-------------------------   
                break
            else:   
                centrd_upd = copy.copy(centrd_new)
    
    Lis_numconvg.append(ilearn)
    Lis_upditer.append(upditer)
    # Lis_countseed.append(countseed)
    #####---------------print info-------------------------------
    end = time.perf_counter()        ## performance counter for benchmarking (better)
    Lis_spendt.append(end-start)
    localtime = time.asctime( time.localtime(time.time()) )
    print(localtime)
    print('itimes=%d,convergence time=%d, spend time:%.2f mins, k_pred=%d'
          %(itimes,ilearn,(end-start)/60,len(cls_upd)))
    if clsmtd == hsres.mtdplus :
        print('bstSC=%.3f,bstSCnon=%.3f;MI_bst=%.3f,iteration=%d'
              %(SC_bst,SCnon_bst,MI_bst,upditer[-1]))
    else:
        i = int(len(Lis_SC)/learnsTimes)
        idx = np.argmax(Lis_SC[(i-1)*learnsTimes:learnsTimes*i])
        bstidx = int((i-1)*learnsTimes+idx)
        SC_bst = Lis_SC[bstidx]
        SCnon_bst = Lis_SCnon[bstidx]
        MI_bst = Lis_MI[bstidx]
        print('SC_bst=%.3f,bstSCnon=%.3f;MI_bst=%.3f,bstidx=%d'
              %(SC_bst,SCnon_bst,MI_bst,bstidx))
        
        # print('bstSC=%.3f,bstSCnon=%.3f;'%(SC_bst,SCnon_bst))
    #####-------------------------------------------------------
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
             Lis_numconvg=Lis_numconvg,Lis_upditer=Lis_upditer,note=note,nsample=nsample)    
    

#%%
# =============================================================================
#         SAVE Outputs
# =============================================================================
np.savez((join(hsres.data_dir,'%d_%d_%s_%s_k%d'
               %(record,totalTimes,fg_flag,query_flag[2:-1],ncluster))),
         Lis_TWCV=Lis_TWCV,Lis_RI=Lis_RI,Lis_MI=Lis_MI,Lis_FMS=Lis_FMS,
         Lis_SC=Lis_SC,Lis_SCnon=Lis_SCnon,Lis_cls=Lis_cls,
         Lis_centrd=Lis_centrd,Lis_seed=Lis_seed,
         SC_gt=np.mean(Lis_SC_gt),SCnon_gt=SCavg_gt,TWCV_gt=TWCV_gt,
         Lis_initprob=Lis_initprob,Lis_iters=Lis_iters, 
         Lis_spendt=Lis_spendt,Lis_predcls_pt=Lis_predcls_pt,
         Lis_numconvg=Lis_numconvg,Lis_upditer=Lis_upditer,note=note,nsample=nsample)   