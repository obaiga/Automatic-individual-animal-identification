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

code_dir = 'C:/Users/95316/test_dataset'
code_dir = 'C:/Users/SHF/test_dataset'
code_dir = '/Users/obaiga/github/Automatic-individual-animal-identification/'
os.chdir(code_dir)

import utils

# dpath = 'C:/Users/95316/test_dataset'
# dpath = 'C:/Users/SHF/test_dataset'
dpath = '/Users/obaiga/Jupyter/Python-Research/Africaleopard'
new_db = 'snow leopard'
new+
# new_db = 'seal_hotspotter'

query_flag = 'vsmany_'

fg_flag = 'fg'   #### only containing animal body without background
# fg_flag = ''

# data_mof = '_mean'    #### modify similarity score matrix '_mean' or '' or '_diag'
data_mof = ''
# data_mof = '_diag'

sq_flag = True    #### True: the square of a similarity score

scfeed = False   ##### whether using adaptive k++ (True) or k++ (False)
scfeed = True

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


# In[k-nearest neighbor]

scoreArymean = np.zeros([nsample,nsample])
for icol in np.arange(nsample):
    for jrow in np.arange(icol,nsample):
        scoreArymean[icol,jrow] = (scoreAry[icol,jrow] + scoreAry[icol,jrow])/2
        scoreArymean[jrow,icol] = (scoreAry[icol,jrow] + scoreAry[icol,jrow])/2

kneighbor = 3
kneighbor_flag = True

scoreArySort = np.zeros([nsample,nsample])

if kneighbor_flag:
    print(kneighbor)
    sortidx = np.argsort(scoreArymean,axis=1)
    sort = np.sort(scoreArymean,axis=1)
    top = np.arange(-1,-1-kneighbor,-1)
    topidx = sortidx[:,top]
    # idx = np.arange(nsample)
    for i in range(nsample):
        scoreArySort[i,topidx[i]] = scoreArymean[i,topidx[i]]
        scoreArySort[topidx[i],i] = scoreArySort[i,topidx[i]]
else:
    scoreArySort = scoreArymean

    
# In[SVD]
ans = (scoreArySort==scoreArySort.T).all()
print('The simialrity score matrix is symmetric? %s'%ans)

D = np.sum(scoreArySort, axis=1)
L = np.diag(D) - scoreArySort
ans = (L==L.T).all()
print('The Laplacian matrix is symmetric? %s'%ans)

x, V = np.linalg.eig(L)
x = zip(x, range(len(x)))
x = sorted(x, key=lambda x:x[0])  ## ascending
H = np.vstack([V[:,i] for (v, i) in x[:]]).T


# In[Kcentroid]:
# =============================================================================
#             Adaptive k-medoids++ clustering 
# =============================================================================
# warnings.filterwarnings("ignore")
record = 1011

ConvgTimes = 50
learnsTimes = 200
fixedTimes = 1
totalTimes = int(learnsTimes*fixedTimes)

hsres.data_dir = join(hsres.res_dir,'data')
utils.CheckDir(hsres.data_dir)

# ===============================
#  Noted information
# ===============================  
print('record:%d,%s,k_gt=%d,nsample=%d,TotalTimes=%d'%\
      (record,query_flag[:-1],ncluster,nsample,totalTimes))
note = ''
if scfeed:
    note += 'adaptive '
    
if kneighbor_flag:
    note += 'spectral clustering %d-nn nearest neighbor'%kneighbor
else:
    note += 'spectral clustering'

if scfeed:
    note += 'Convergence itertaions = %d, otherwise running iteration=%d;'\
        % (ConvgTimes,learnsTimes)

print(note)

#%%
from sklearn.cluster._kmeans import _kmeans_single_elkan
from sklearn.metrics.pairwise import _euclidean_distances
from sklearn.utils._openmp_helpers import _openmp_effective_n_threads
from sklearn.cluster import KMeans
import scipy.sparse as sp
from sklearn.utils.extmath import row_norms

verbose = 0
tol = 1e-4 
copy_x = True

nevector = 100
H_opt = H[:,:nevector]

# H_opt = copy.copy(H)

mtd = KMeans()

X = mtd._validate_data(
    H_opt,
    accept_sparse="csr",
    dtype=[np.float64, np.float32],
    order="C",
    copy=copy_x,
    accept_large_sparse=False,
)
# nsample, nfeature = X.shape

# sample_weight = _check_sample_weight(sample_weight, X, dtype=X.dtype)
sample_weight = np.full(nsample, 1, dtype=X.dtype)
n_threads = _openmp_effective_n_threads()

# subtract of mean of x for more accurate distance computations
if not sp.issparse(X):
    X_mean = X.mean(axis=0)
    # The copy was already done above
    X -= X_mean

# precompute squared norms of data points
x_squared_norms = row_norms(X, squared=True)
'''
Row-wise (squared) Euclidean norm of X.
Equivalent to np.sqrt((X * X).sum(axis=1)) or (X * X).sum(axis=1)
'''
dist_sq_all  = _euclidean_distances(
    X, X, X_norm_squared=x_squared_norms,Y_norm_squared=x_squared_norms, squared=True
)

Lis_SC_gt,SCavg_gt,Lis_SCcls_gt,_ =\
    utils.SilhouetteScore(hsres,cluster_gt,scoreAry=dist_sq_all,mtd='Euclidean')
    
print('groundtruth:Avg_SC:%.4f;nonindiv SC:%.4f'%
      (np.mean(Lis_SC_gt),SCavg_gt))   

idx_iso = np.where(SC_info_gt[:,1] == -1)[0]  
if len(idx_iso) > 0:
    ans = Lis_SC_gt[idx_iso]
    print('indiv SC:%.4f,num:%d'%(np.mean(ans),len(ans))) 
else:
    print('no isolated images')

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
updateprob_bstSC = True
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
        # Initialization k-centroid++ 
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
            if not scfeed:
                '''keep probability as equal'''
                init_prob = np.ones(nsample)
            ilearn += 1
            #-------------------------------------------------------
            #-------------spectral clustering
            #-------------------------------------------------------
            ###-------------------
            # Initialize centers
            ###-------------------
            centers = np.empty((nclusterPot, nevector), dtype=X.dtype)
            '''
            empty, unlike zeros, does not set the array values to zero, 
            and may therefore be marginally faste
            '''
            # Pick first center randomly and track index of point
            centrd_pool = np.arange(nsample)
            '''
            k-means++ initialization
            '''
            best_candidate = random.choices(centrd_pool,weights=init_prob)[0]
            seeds = np.full(nclusterPot, -1, dtype=int)
            
            if sp.issparse(X):
                centers[0] = X[best_candidate].toarray()
            else:
                centers[0] = X[best_candidate]
            seeds[0] = best_candidate
        
            # Initialize list of closest distances and calculate current potential
            # closest_dist_sq = _euclidean_distances(
            #     centers[0, np.newaxis], X, Y_norm_squared=x_squared_norms, squared=True
            # )
            closest_dist_sq = dist_sq_all[seeds[0],np.newaxis]
            # centers2all_dist_sq = copy.copy(closest_dist_sq)
            '''
            np.newaxis: insert new axis, output-1*n_features
            function: _euclidean_distances  output-1*n_samples
            dist(x, y) = sqrt(dot(x, x) - 2 * dot(x, y) + dot(y, y))
            X = centers[0, np.newaxis]
            Y = X (n_samples * n_features)
            XX = row_norms(X, squared=True)[:, np.newaxis] output-1*1
            YY = x_squared_norms.reshape(1, -1)  output-1*n_samples    
            '''
            
            # Pick the remaining n_clusters-1 points
            for c in range(1, nclusterPot):
                
                centrd_pool = np.delete(centrd_pool,np.argwhere(centrd_pool==best_candidate))
                weights = closest_dist_sq[0,centrd_pool]*init_prob[centrd_pool]
                
                best_candidate = random.choices(centrd_pool,weights=weights)[0]
                #####choose a center candidate by sampling with probability proportional 
                ##### to the squared distance to the closest existing center
                if sp.issparse(X):
                    centers[c] = X[best_candidate].toarray()
                else:
                    centers[c] = X[best_candidate]
                seeds[c] = best_candidate
                # newcenter_dist_sq = _euclidean_distances(
                #     centers[c, np.newaxis], X, Y_norm_squared=x_squared_norms, squared=True
                # )
                # centers2all_dist_sq = np.vstack((centers2all_dist_sq,newcenter_dist_sq))
                centers2all_dist_sq = dist_sq_all[seeds[:c+1]]
                closest_dist_sq = np.min(centers2all_dist_sq,axis=0)[np.newaxis,:]
        
            if sp.issparse(centers):
                centers = centers.toarray()
            # return centers
            centers_init = copy.copy(centers)   ### add by obaiga
    
            ###-------------------
            # Update centers
            ###-------------------
            ###run a k-means once
            predcls_pt, inertia, centers, n_iter_ = _kmeans_single_elkan(
                X,
                sample_weight,
                centers_init,
                max_iter=iters,
                verbose=verbose,
                tol=tol,
                x_squared_norms=x_squared_norms,
                n_threads=n_threads,
            )
            predcls_pt_final = predcls_pt
            ans = np.unique(predcls_pt)
            
            cls_upd = []
            for ii in ans:
                cache = np.where(predcls_pt == ii)[0]
                cls_upd.append(cache)
            utils.check_predcls_pt(predcls_pt_final)
            #-------------------------------------------------------
            #####         measurement
            #-------------------------------------------------------
            '''calculate measurement'''
            ### calculate silhouette score
            pred_SCs,pred_SCavgnon,pred_SCcls,_ =\
                    utils.SilhouetteScore(hsres,cls_upd,scoreAry=dist_sq_all,mtd='Euclidean')
            
            ## assign all points with a cluster ID
            predcls_pt = utils.cls2Lispt(hsres,cls_upd)
            
            RI = metrics.adjusted_rand_score(label_gt, predcls_pt)
            MI = metrics.adjusted_mutual_info_score(label_gt, predcls_pt)
            FMS = metrics.fowlkes_mallows_score(label_gt, predcls_pt)
            
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
            Lis_seed.append(seeds)
            Lis_initprob.append(copy.copy(init_prob))
            
            #-------------------------------------
            if scfeed:
                '''
                adaptive algorithm
                calculate the probability for an image as seed medoid 
                '''
                pred_centrd = np.zeros(nsample)
                # pred_centrd[centrd_upd] = pred_SCcls
                
                pred_centrd[seeds] = pred_SCcls
                pred_centrd += 1  ## range from [-1,1] to [0,2]
                try:
                    init_prob[seeds] = init_prob[seeds] * pred_centrd[seeds]
                    # init_prob[centrd_upd] = init_prob[centrd_upd] * pred_centrd[centrd_upd]
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