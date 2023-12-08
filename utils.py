#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May  4 16:49:58 2022

Update merge_clusters function & lowSC_solve function
on Feb 3, 2023

@author: obaiga
"""
from os.path import join,exists
from os import makedirs
from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
import skimage.io 
from collections import Counter
import copy 
from xlwt import Workbook
#%%
'''
function: merge_clusters
modified on 02/10/23
'''
def merge_clusters(scoreAry,prev_cls,prev_pt,SClis,SC_info,mtd='similarity'):
    ## SC_info: store clsidx,a,b1,b_clsidx,b2,b2_clsidx,bstScore,bstidx
    
    bstMtIdx = np.array(SC_info[:,-1],dtype=np.int32)  
    #### bstMtIdx: the best-match image index for an image
    
    # bstClsIdx = SC_info[bstMtIdx,0]
    
    targ_cls = copy.deepcopy(prev_cls) 
    # prev_cls = np.array(prev_cls)
    
    cache_mergeClsIdx = np.ones([1,2])*-1
    # cache_mergeClsIdx = []
    
    targ_pt = copy.deepcopy(prev_pt)
    
    for ii,icluster in enumerate(targ_cls):
        icluster = np.array(icluster,dtype=np.int32)
        if len(icluster) > 1:
        # if (len(icluster)>1) & (ii not in cache_mergeClsIdx[:,1]):
            ###### non-isolated clusters & the cluster did not be merged
            # ibstClsIdx = bstClsIdx[icluster]
            ###### the index of the best matched images might be changed due to merged clusters
            ibstClsIdx = targ_pt[bstMtIdx[icluster]]
            # ibstMtscore = SC_info[icluster,-2]
            ibstCommCls = Counter(ibstClsIdx).most_common()[0]
            
            ##### find the cluster index of best match images for images in a cluster
            ##### the highest number of the same cluster > half of image number
            if ibstCommCls[1] > (len(icluster)/2):
                # print(ii)
                bidx_prev = SC_info[icluster,3]
                bidx_new = np.array([])
                for ibidx in bidx_prev:
                    ele = prev_cls[int(ibidx)]
                    bidx_new = np.concatenate([bidx_new,[targ_pt[ele[0].astype(np.int32)]]])
                
                # Commbidx = Counter(bidx_new).most_common()[0]
                Commbidx = Counter(bidx_prev).most_common()[0]
                merge_clsIdx = int(Commbidx[0])
            
                ##### find the nearest neighbor cluster index for images in a cluster
                ##### the highest number of the same cluster > half of image number
                if (merge_clsIdx == ibstCommCls[0]) & (Commbidx[1] > len(icluster)/2):
                    # print(ii)
                    ######## check whether the merge cluster is a empty cluster or not 
                    ###### (already merged)
                        while len(targ_cls[merge_clsIdx])==0 :
                            # print(ii,icluster,prev_cls[merge_clsIdx])
                            ans = np.where(cache_mergeClsIdx==merge_clsIdx)
                            for jdx,jpos in enumerate(ans[1]):
                                if jpos == 1:    #### means the merged cluster
                            # print(merge_clsIdx,ans)
                            # print(merge_clsIdx,cache_mergeClsIdx[ans[0][0],:])
                                    merge_clsIdx = int(cache_mergeClsIdx[ans[0][jdx],0])
                            # print(merge_clsIdx,targ_cls[merge_clsIdx])
            
                        potential_cls = np.concatenate([icluster,targ_cls[merge_clsIdx]],
                                                       dtype=np.int32)
                        SCbef = SClis[potential_cls]
                        # SCbef1 = SClis[icluster]
                        # SCbef2 = SClis[targ_cls[merge_clsIdx]]
                        
                        # SCmeanbef = np.mean(SCbef)
                        b1 = SC_info[potential_cls,2]
                        b1idx = SC_info[potential_cls,3]
                        b2 = SC_info[potential_cls,4]
                        iscore = scoreAry[potential_cls,:]
                        
                        score_intra = iscore[:,potential_cls] 
                        self_ele = np.diagonal(score_intra)   ## return spexified diagonals
                        ai_new = (np.sum(score_intra,axis=1)-self_ele)/(len(potential_cls)-1)
                        num = len(potential_cls)
                        potential_pt = np.arange(num)
                        b2_as_b1 = np.concatenate([np.where(b1idx == merge_clsIdx)[0],\
                                                    np.where(b1idx == ii)[0]])
                        b1_as_b1 = np.setdiff1d(potential_pt, b2_as_b1)
                        bi_new = np.zeros(num)
                        bi_new[b2_as_b1] = b2[b2_as_b1]
                        bi_new[b1_as_b1] = b1[b1_as_b1]
                        if mtd == 'similarity':
                            SCaft = (ai_new - bi_new ) / (np.max([ai_new,bi_new],axis=0)+1e-6)
                        elif mtd == 'Euclidean':
                            SCaft = (bi_new -  ai_new) / (np.max([ai_new,bi_new],axis=0)+1e-6)
                        if np.mean(SCaft) >= np.mean(SCbef):
                        # if (np.mean(SCaft) >= np.mean(SCbef1)) & (np.mean(SCaft) >= np.mean(SCbef2)):
                            cache_mergeClsIdx = np.vstack([cache_mergeClsIdx,[ii,merge_clsIdx]])
                            targ_cls[ii] = potential_cls
                            targ_cls[merge_clsIdx] = []
                            targ_pt[merge_clsIdx] = ii
                        
    return targ_cls,cache_mergeClsIdx


'''
modify on 03/07/2023
'''
def lowSC_solve(prev_cls,SC_pt,SC_info,thred,mtd='similarity'):  
    ## SC_info: containing clsidx,a,b1,b_clsidx,b2,b2_clsidx,bstScore,bstidx
    # SC_pt = prev_SClis
    # thred =  prev_SCavgnon
    # SC_info = prev_SCinfo
    
    targ_cls = copy.deepcopy(prev_cls)
    lowSC_idx = list(np.where(SC_pt<0)[0])
    ##### include isolated images
    res_check = []
    ##### res_check: sc_new,SC_old,info_label,iidx,aClsIdx,b1ClsIdx,b2ClsIdx
    lowSC_idxupd = copy.deepcopy(lowSC_idx)
    
    if len(lowSC_idx) > 0:

        #####--------------------------
        ######## modify on 11/09/2023
        while (1):
            if len(lowSC_idxupd) == 0:
                break
            
            flag = False   
            ##### determine whether the outlier statfies the requirment (in default: false)
            iidx = int(lowSC_idxupd[0])
            # print(iidx)
            info = SC_info[iidx]
            a = info[1]
            b1 = info[2]
            b2 = info[4]
            aClsIdx = int(info[0])
            b1ClsIdx = int(info[3])
            b2ClsIdx = int(info[5])
            bstidx = int(info[7])
            #####--------------------------
            ######## modify on 12/05/2023    
            if mtd=='similarity':
                snew = (b1-np.max([a,b2]))/(b1+1e-6)
            elif mtd == 'Euclidean':
                snew = (np.max([a,b2])-b1)/np.max([a,b2])
        
            bst_idxcls = SC_info[bstidx,0]
            bst_score = info[6]     
            
            ans  = np.where(lowSC_idx == bstidx)[0]
            if len(ans) > 0:
                print('check: its nearest neighbor is outlier too, %d %d'
                      %(iidx,bstidx))
            
            if (snew >= thred) & (bst_idxcls == b1ClsIdx):
                flag = True
                ans  = np.where(lowSC_idx == bstidx)[0]
                if len(ans) > 0:
                    print('its nearest neighbor is outlier too and is merged so removed from outlier list, %d %d'
                          %(iidx,bstidx))
                    lowSC_idxupd.remove(bstidx)
            if flag == True:
                #### merge to the nearest neighbor cluster
                targ_cls[b1ClsIdx].append(iidx)
                targ_cls[aClsIdx].remove(iidx)
                res_check.append( [snew,SC_pt[iidx],'nearest',\
                                    iidx,aClsIdx,b1ClsIdx,b2ClsIdx])
                    
            elif a!=-1:      ##### non-isolated images
                ### become isolated
                targ_cls.append([iidx])
                targ_cls[aClsIdx].remove(iidx)
                res_check.append([snew,SC_pt[iidx],'indepent',\
                                    iidx,aClsIdx,b1ClsIdx,b2ClsIdx])
            else:   
                #### maintain isolated for isolated images
                res_check.append( [None,SC_pt[iidx],'none',\
                                        iidx,aClsIdx,b1ClsIdx,b2ClsIdx])

            lowSC_idxupd.remove(iidx)
            
    return targ_cls,res_check


def add_0cls(hsres,cls_upd):
    for inon in hsres.idxNonConnect:
        cls_upd.append([inon])
    return cls_upd

def check_predcls_pt(predcls_pt):
    weight = len(predcls_pt)
    ans = np.where(predcls_pt<0)[0]
    if len(ans)>0:
        print('Not correct:%d/%d'%(weight-len(ans),weight))

def create_0Lispt(hsres,ncluster):
    weight = len(hsres.Lis_Chip2ID)
    predcls_pt = np.ones(weight,dtype=np.int32)*-1
    for j,idxNon in enumerate(hsres.idxNonConnect):
        predcls_pt[idxNon] = j+ncluster
    return predcls_pt

def cls2Lispt(hsres,cls_upd):
    # ncluster = len(cls_upd)
    weight = len(hsres.Lis_Chip2ID)
    predcls_pt = np.ones(weight,dtype=np.int32)*-1
    for i,icluster in enumerate(cls_upd):
        predcls_pt[hsres.Lis_idx[icluster]] = i
    check_predcls_pt(predcls_pt)
    return predcls_pt

def correct_denom_gt(hsres,label_pred,cluster_pred,topnum=None):
    '''
    Compared to a clustering result, 
    in a given ground-truth
    what predicted clusters are correct, partial correct, or wrong?
    correct means full-match, a cluster has all items from a label, without irrelative item
    partial correct means: a cluster has partial items from a label, without irrelative item
    otherwise, it is wrong
    '''
    if topnum is None:
        topnum = hsres.k_gt
    # print('Shows topnum:%d'%topnum)
    BiggestNumPerID = np.max(np.array(hsres.CountInfo)[:,0])
    res_correct = np.zeros(BiggestNumPerID)
    res_partial = []
    res_wrong = []
    detail_correct = [] 
    detail_partial = []
    detail_wrong = []
    ### create a matrix 1x the image number for the largest ID
    for i in range(topnum):
        ans = hsres.cluster_gt[i]
        num_gt = len(ans)
        # iID = hsres.Lis_ID[i]
        Lis_IDpred = label_pred[ans]
        IDsCount = np.array(Counter(Lis_IDpred).most_common())
        if (len(IDsCount)==1):
            ### the elements from the trueID only corresponding to 1 predicted cluster
            iclusterpred = cluster_pred[int(IDsCount[0,0])]
            num_pred = len(iclusterpred)
            # if (num_pred == num_gt) & (iclusterpred.all()==ans.all()):
            if (num_pred == num_gt) :
                ### for this predict cluster,len(pred_cls_eles)=len(true_cls_eles)
                # print(num,IDsCount)
                res_correct[num_gt-1] += 1
                detail_correct.append(ans)
                # print(i)
            else:
                res_partial.append([len(ans),i])
                detail_partial.append(ans)
        else:
            res_wrong.append([len(ans),i])
            detail_wrong.append(ans)
                
    return res_correct,res_partial,res_wrong,detail_correct,detail_partial,detail_wrong

def correct_denom_pred(hsres,cluster_pred,topnum=None):
    '''
    Compared to the ground-truth,
    in a given clustering result
    what predicted clusters are correct, partial correct, or wrong?
    correct means full-match, a cluster has all items from a label, without irrelative item
    partial correct means: a cluster has partial items from a label, without irrelative item
    otherwise, it is wrong
    '''
    topnum = None
    if topnum is None:
        topnum = len(cluster_pred)
    # print('Shows topnum:%d'%topnum)

    cluster_pred,_,CountInfo_pred = load_info_pred\
        (hsres,cluster_pred,hsres.nsample)

    BiggestNumPerID = np.max(np.array(CountInfo_pred)[:,0])
    res_correct = np.zeros(BiggestNumPerID)
    res_wrong = []
    res_partial = []
    detail_correct = []
    detail_partial = []
    detail_wrong = []
    ### create a matrix 1x the image number for the largest ID
    for i in range(topnum):
        ans = cluster_pred[i]
        num_pred = len(ans)
        # iID = hsres.Lis_ID[i]
        Lis_IDgt = hsres.label_gt[ans]
        IDsCount = np.array(Counter(Lis_IDgt).most_common())
        if (len(IDsCount)==1):
            # print(i)
            ### the elements from the trueID only corresponding to 1 predicted cluster
            iclustergt = hsres.cluster_gt[int(IDsCount[0,0])]
            num_gt = len(iclustergt)
            # if (num_pred == num_gt) & (iclusterpred.all()==ans.all()):
            if (num_pred == num_gt) :
                ### for this predict cluster,len(pred_cls_eles)=len(true_cls_eles)
                # print(num,IDsCount)
                res_correct[num_pred-1] += 1
                detail_correct.append(ans)
                # print(i)
            else:
                res_partial.append([len(ans),i])
                detail_partial.append(ans)
        else:
            res_wrong.append([len(ans),i])
            detail_wrong.append(ans)
    return res_correct,res_partial,res_wrong,detail_correct,detail_partial,detail_wrong

def load_info_pred(hsres,Lis_clspred,nsample):
    if nsample<len(hsres.Lis_Chip2ID):
        print('special case')
        hsres.cache_dir = join(hsres.db_dir,'cache')
        CheckDir(hsres.cache_dir)
        name ='idxNonConnect.text'
        path = join(hsres.cache_dir,name)
        if CheckFile(path):
            ans = LoadCache(path)
            idxNonConnect = np.array(ans,dtype=np.int32)
            path = join(hsres.cache_dir,'nsample.text')
            ans = LoadCache(path)
            nsample = int(ans[0])
            
            Lis_idx = np.arange(nsample)
            Lis_idx = np.delete(Lis_idx,idxNonConnect)
            ncluster = len(Lis_clspred)-len(idxNonConnect)
    else:
        Lis_idx = np.arange(nsample)
        ncluster = len(Lis_clspred)
        
    labelcaches = np.ones(nsample)*-1
    
    for i,icluster in enumerate(Lis_clspred):
        if i<ncluster:
            labelcaches[Lis_idx[icluster]] = i
        else:
            labelcaches[icluster] = i
        
    ans = np.where(labelcaches<0)[0]
    if len(ans)>0:
        print('Not correct:%d/%d'%(nsample-len(ans),nsample))
        
    IDs = Counter(labelcaches).most_common()
    Counts = []
    clusters = []
    labels = np.zeros(nsample)
    i = 0
    for iIDname,iCount in IDs:
        Counts.append(iCount)
        # IDnames.append(iIDname)   ### record predict leopard ID name (more to less)
        ans = np.where(labelcaches==iIDname)[0]
        clusters.append(list(ans))
        labels[ans] = i
        i += 1
    CountInfo = Counter(Counts).most_common()
    
    return clusters,labels,CountInfo


def load_idxNonConnect(hsres,query_flag,fg_flag):
    hsres.cache_dir = join(hsres.db_dir,'cache')
    CheckDir(hsres.cache_dir)
    name ='idxNonConnect.text'
    path = join(hsres.cache_dir,name)
    if CheckFile(path):
        ans = LoadCache(path)
        idxNonConnect = np.array(ans,dtype=np.int32)
        path = join(hsres.cache_dir,'nsample.text')
        ans = LoadCache(path)
        nsample = int(ans[0])
    else:
        print('No file:%s \n Working on it......'%name)
        name = ('ImgScore_%s%s%s%s.xlsx')%(query_flag[2:],fg_flag,'','')
        scoreAry = hsres.load_score(name)
        if scoreAry is not None:
            D = np.sum(scoreAry, axis=1)
            idxNonConnect = np.where(D==0)[0]
            nsample = len(hsres.Lis_Chip2ID)
            WriteCache(hsres.cache_dir,'idxNonConnect', idxNonConnect)
            WriteCache(hsres.cache_dir,'nsample', nsample)
            name = ('ImgScore_%s%s%s%s.xlsx')%(query_flag[2:],fg_flag,'_non0','')
            path = join(hsres.res_dir,name)
            if not CheckFile(path):
                Lis_idx = np.arange(nsample)
                Lis_idx = np.delete(Lis_idx,idxNonConnect)
                print('No file: %s \n Working on it .......'\
                      %name)
                scoreAryNew = scoreAry[np.ix_(Lis_idx,Lis_idx)]
                save_score(scoreAryNew,path)
            else:
                print('no file:%s'%name)
                
    return idxNonConnect,nsample

def save_excel(Lis_data,Lis_name,sheetName,path):
    
    table = Workbook()
    sheet1 = table.add_sheet(sheetName)
    for ii in range(len(Lis_name)):
        sheet1.write(0,ii,Lis_name[ii])
    
    for ii in range(len(Lis_data)):
        idata = Lis_data[ii]
        for jj in range(len(Lis_name)):
            # print(ii,jj)
            sheet1.write(ii+1,jj,str(idata[jj])) 
    table.save(path)

def prints(hsres,clsmtd,data_mof,scfeed,fg_flag,sq_flag,updateprob_bstSC,others=None):
    
    if clsmtd == hsres.mtdori:
        note = 'k-medoid (random);'
    elif clsmtd == hsres.mtdplus:
        note = 'k-medoid++;'
    if data_mof == '':
        note += 'data no change'
    else:
        note += '%s data'%(data_mof[1:])
    note += '(%s);'%fg_flag
    if (scfeed) & (clsmtd == hsres.mtdplus):
        note += 'feedback with SC (update cluster medoids);'
        if updateprob_bstSC == True:
            note += 'only maintain inital probability with the hightest SC value;'
        else:
            note += 'normally increase inital probability'
    else:
        note += 'no feedback;uniform probability;'
    if sq_flag:
        note += 'squared similarity score;'
    else:
        note += 'similarity score;'
    if others is not None:
        note += others
        
    return note

def save_score(data,path):
    df = pd.DataFrame(data)
    df.to_excel(path, index=False, header=False)

def Modify_scoreAry(scoreAry,flag):
    nsample = len(scoreAry)
    if flag == 'diag':
        D = np.sum(scoreAry,axis=1)
        idxs = np.arange(nsample)
        scoreAryNew = copy.copy(scoreAry)
        scoreAryNew[idxs,idxs] = D
    elif flag == 'mean':
        scoreAryNew = np.zeros([nsample,nsample])
        for i in range(nsample):
            for j in range(i,nsample):
                if i != j:
                    mean = (scoreAry[i,j]+scoreAry[j,i])/2
                    scoreAryNew[i,j] = mean
                    scoreAryNew[j,i] = mean
    return scoreAryNew

def CheckDir(path):
    if not exists(path):
        makedirs(path)

def CheckFile(path):
    return exists(path)
        
def WriteCache(path,name,data):
    
    if not hasattr(data, '__iter__'):
        data = [data]
    CheckDir(path)
    path = join(path,'%s.text'%str(name))
    with open(path, 'w') as f:
        for iidx in data:
            f.write(str(iidx))
            f.write(' ')
    print('[utils] Write cache file:%s'%str(name))
            
def LoadCache(path):
    print('[utils] Load cache file:%s'%path.split('/')[-1])
    with open(path) as f:
        contents = f.readlines()
        ans = contents[0].split()
    return ans

def SilhouetteScore(hsres,clusters,sq_flag = True,\
                    SC_0=0.5,epsilon=1e-6,scoreAry=None,mtd='similarity'):
    '''
    mtd='similarity' or 'Euclidean'
    If mtd == 'similarity':
        Calculate Silhouette coefficient
        For an image I_i, SC = (a-b1)/(max[a,b1])  if |C|>1
        a = cohesion
        b1 = separation
        SC_pot = -(b1-b2)/b1 if |Ci|==1
        an individual cluster merges to the nearest cluster
        if satisfy SC_pot <= -thred;  thred = mean(SCi) for all |Ci| >1
            the closest similar image in the nearest cluster;
            
       SC_info =[ clsidx,a,b1,b_clsidx,b2,b2_clsidx,bstScore,bstidx]s
    '''
    # mtd='Euclidean'
    if scoreAry is None:
        scoreAry = copy.copy(hsres.scoreAry)
        if scoreAry is not None:
            if sq_flag:
                scoreAry = scoreAry**2
            else:
                print('No scoreArray!')
    
    nsample = len(scoreAry)
    SC_info = np.ones([nsample,8])*-1    
        ## store clsidx,a,b1,b_clsidx,b2,b2_clsidx,bstScore,bstidx
    indiv_info = []
    k = len(clusters)
    SCsum = 0
    total = 0
    Lis_SC = np.zeros(nsample)
    Lis_SCcls = np.zeros(k)
        
    for idxcls,icluster in enumerate(clusters):

        numi = len(icluster)
        b = np.zeros([numi,k-1])   
        scorecls = scoreAry[icluster,:]
        scoreIntra = scorecls[:,icluster]
        scoreDiag = np.diagonal(scoreIntra)   ## return spexified diagonals
        bidx = np.zeros([numi,k-1])
        
        idx = 0
        for jj in range(k):  ## calculate b
            if idxcls != jj:
                jcluster = clusters[jj]
                scoreInter = scorecls[:,jcluster]
                numj = len(jcluster)
                b[:,idx] = np.sum(scoreInter,axis=1)/numj  ## axis=1 by rows
                bidx[:,idx] = jj
                idx += 1
        bsort = np.sort(b,axis=1)
        sort = np.argsort(b,axis=1)
        bsortidx = np.zeros([numi,k-1])
        for i in range(numi):
            bsortidx[i,:] = bidx[i,sort[i,:]]
        
        if mtd == 'similarity':        
            b1pos = -1  #### biggest score
            b2pos = -2
        elif mtd =='Euclidean':
            b1pos = 0  ### smallest distance
            b2pos = 1
        
        b1 = bsort[:,b1pos]    ## by rows
        b1idx = bsortidx[:,b1pos]
        b2 = bsort[:,b2pos]
        b2idx = bsortidx[:,b2pos]
        SC_info[icluster,0] = idxcls
        SC_info[icluster,2] = b1
        SC_info[icluster,3] = b1idx
        SC_info[icluster,4] = b2
        SC_info[icluster,5] = b2idx
        if mtd == 'similarity': 
            #####-----modify on 03/07/23
            scorecls[:,icluster] = 0
            # scorecls[np.arange(len(icluster)),icluster] = 0
            bstscore = np.max(scorecls,axis=1)
            bstidx = np.argmax(scorecls,axis=1)
            
        elif mtd == 'Euclidean':
            bstscore = np.sort(scorecls,axis=1)[:,1]
            bstidx = np.argsort(scorecls,axis=1)[:,1]
            
        SC_info[icluster,6] = bstscore
        SC_info[icluster,7] = bstidx
        
        if numi > 1:
            a = (np.sum(scoreIntra,axis=1)-scoreDiag)/(numi-1)
            # a = (np.sum(scoreIntra,axis=1)/2)/(numi-1)
            SC_info[icluster,1] = a
            if mtd == 'similarity':
                s = (a - b1) / (np.max([a,b1],axis=0) + epsilon)
            elif mtd == 'Euclidean':
                s = (b1 - a) /(np.max([a,b1],axis=0)+epsilon)
            Lis_SC[icluster] = s  ## the save order follow by cluster order 
            Lis_SCcls[idxcls] = np.mean(s) 
            ### the mean of Silhouette for images in the cluster
            SCsum += np.sum(s)
            total += len(s)
        else:
            indiv_info.append([idxcls,icluster[0],bstidx,bstscore,\
                               b1,b2,b1idx,b2idx])
            
    SCavg = SCsum/(nsample-len(indiv_info))
        
    for idxcls,idximg,bstidx,bstscore,b1,b2,b1idx,b2idx in indiv_info:
        bst_idxcls = SC_info[bstidx,0]   ## cluster idx for the best similar image
        
        ###### modify on 01/12/23 by Cheng 
        Lis_SC[idximg] = SC_0
        Lis_SCcls[idxcls] = SC_0 
        # if (b2 != 0) & (bst_idxcls == b1idx):
        # if (bstscore != 0) & (bst_idxcls == b1idx):
        if (bst_idxcls == b1idx):
            if mtd == 'similarity':
                sc = (b1 - b2) / (b1+epsilon)
            elif mtd == 'Euclidean':
                sc = (b2 - b1) / b2
            # print(idximg,np.sqrt(bstscore),sc)
                
            if sc >= SCavg:
                # print(idximg,np.sqrt(bstscore),sc)
                Lis_SC[idximg] = -sc
                Lis_SCcls[idxcls] = -sc 
        #-------------------------------------        
    return Lis_SC,SCavg,Lis_SCcls,SC_info

def centroid(hsres,clusters):
    # clusters = copy.copy(cluster_gt)
    centroid_ssum = []
    centroid = []

    for idx,icluster in enumerate(clusters):
        
        scoreIntra = hsres.scoreAry[np.ix_(icluster,icluster)]
        max_sum = np.sum(scoreIntra,axis=1)    ## by rows
        centroid.append(icluster[np.argmax(max_sum)])
        centroid_ssum.append(np.max(max_sum))
        
    return centroid,centroid_ssum

#%%
class HotspotterRes(object):
    def __init__(hsres, db_dir=None):
        if db_dir is not None:
            hsres.db_dir = db_dir
        hsres.res_dir = None
        hsres.plot_dir = None
        hsres.cache_dir = None
        hsres.data_dir = None
        
        hsres.scoreAry = None
        hsres.Lis_Chip2ID = None
        hsres.Lis_Img = None
        hsres.Lis_chipNo = None
        hsres.Lis_ID = None
        hsres.CountInfo = None
        hsres.Lis_idx = None
        hsres.nsample = 0 
        # hsres.used_nonzero = False
        hsres.cluster_gt = None
        hsres.label_gt = None
        hsres.k_gt = 0
        hsres.cluster_non0 = None
        hsres.label_non0 = None
        hsres.CountInfo_non0 = None
        hsres.mtdplus = '++'
        hsres.mtdori = 'origin'
        
        hsres.idxNonConnect = None
        
        hsres.chip_dir = join(hsres.db_dir,'_hsdb','computed','chips')
        hsres.chipname = 'cid%d_CHIP(sz750).png'
        
    def load_score(hsres,name):
        path = join(hsres.res_dir,name)
        if exists(path):
            sheet = pd.read_excel(path,header=None)
            hsres.scoreAry = sheet.values
            return hsres.scoreAry
        else:
            print('Cannot load score: %s!'%name)
            print('Working on .......')
            data_mof = name.split('_')[-1][:-5]
            newname = name[:-(1+len(data_mof)+5)]+'.xlsx'
            newpath = join(hsres.res_dir,newname)
                    
            if exists(newpath):
                sheet = pd.read_excel(newpath,header=None)
                scoreAry = sheet.values
                hsres.scoreAry = Modify_scoreAry(scoreAry,flag=data_mof)
                save_score(hsres.scoreAry,path)
                return hsres.scoreAry
            else:
                print('Cannot load score: %s does not exist!'%path)
                return None
        
    def load_info(hsres,name = 'table.csv'):
        table_dir = join(hsres.db_dir,name)
        table = pd.read_csv(table_dir,skipinitialspace=True)

        hsres.Lis_Chip2ID = np.array(table['Name'])
        hsres.Lis_Img = np.array(table['Image'])
        hsres.Lis_chipNo = np.array(table['#   ChipID'])
        
        IDs = Counter(hsres.Lis_Chip2ID).most_common()

        Counts = []
        IDnames= []
        cluster_gt = []
        label_gt = np.zeros(len(hsres.Lis_Img))
        i = 0
        for iIDname,iCount in IDs:
            Counts.append(iCount)
            IDnames.append(iIDname)   ### record predict leopard ID name (more to less)
            ans = np.where(hsres.Lis_Chip2ID==iIDname)[0]
            cluster_gt.append(ans)
            label_gt[ans] = i
            i += 1
        hsres.CountInfo = Counter(Counts).most_common()
        hsres.cluster_gt = cluster_gt
        hsres.label_gt = label_gt
        print('ground truth:\n',hsres.CountInfo)
        
        hsres.Lis_ID = np.array(IDnames)
        
        return(hsres.Lis_Chip2ID,hsres.Lis_Img,hsres.Lis_chipNo,hsres.Lis_ID)
            
    def cluster_non0_info(hsres):
        if hsres.Lis_idx is not None:
            Chip2ID_non0 = hsres.Lis_Chip2ID[hsres.Lis_idx]
            IDs = Counter(Chip2ID_non0).most_common()
            Counts = []
            cluster_non0 = []
            label_non0 = np.ones(len(hsres.Lis_idx))*-1
            i = 0
            for iIDname,iCount in IDs:
                Counts.append(iCount)
                ans = np.where(Chip2ID_non0==iIDname)[0]
                cluster_non0.append(ans)
                label_non0[ans] = i
                i += 1
            hsres.CountInfo_non0 = Counter(Counts).most_common()
            hsres.cluster_non0 = cluster_non0
            hsres.label_non0 = label_non0
            print('connected ground truth:\n',hsres.CountInfo_non0)
            return(hsres.cluster_non0,hsres.label_non0,hsres.CountInfo_non0)
        
    
    def load_cluster_gt(hsres,query_flag,fg_flag):

        hsres.nsample = len(hsres.Lis_Img)
        hsres.Lis_idx = np.arange(hsres.nsample)
        cluster_gt = hsres.cluster_gt
        label_gt = hsres.label_gt
           
        hsres.k_gt = len(cluster_gt)
        
        # print('nsample:%d'%hsres.nsample)
        
        return cluster_gt,label_gt
    '''
    PLOT
    '''
    def plot_comp(hsres,idx1,idx2,save_folder,save_dir=None,trueID=False):
        save_flag = True
        if save_dir is None:
            save_dir = hsres.plot_dir
            if save_dir is None:
                print('No save directory')
                save_flag = False
        if save_flag == True:
            save_plot_dir = join(save_dir,save_folder)
            CheckDir(save_plot_dir) 
            fig = plt.figure(figsize=(20,16),dpi=100)
            for ii,iidx in enumerate([idx1,idx2]):
                if trueID:
                    trueidx = iidx
                else:
                    trueidx = hsres.Lis_chipID[iidx]
                ichip = hsres.Lis_chipNo[trueidx]
                image = skimage.io.imread(join(hsres.chip_dir,hsres.chipname%ichip))
                fig.add_subplot(2,1,ii+1)
                plt.imshow(image.astype(np.uint8))
                plt.axis('off')  
                plt.title('chip-%d-%s \n %s'%\
                          (ichip,hsres.Lis_Chip2ID[trueidx],hsres.Lis_Img[trueidx]),\
                              fontsize=24)
            fig.tight_layout()
            fig = plt.gcf()
            
            # save_path = join(save_plot_dir,'No.'+str(idx)+'-Num-'+str(num)+'-'+ID_name_lis[iname]+'.JPG')
            save_path = join(save_plot_dir,'No.%s.JPG'%(str(idx1)))
            fig.savefig(save_path,dpi=100,transparent=True,bbox_inches ='tight')
            plt.close()
            
            
    def plot_res_pred(hsres,reqLis,predID,save_folder,save_dir=None,specImg=None,trueidx_flag=False):
        save_flag = True
        if save_dir is None:
            save_dir = hsres.plot_dir
            if save_dir is None:
                print('No save directory')
                save_flag = False
        if save_flag == True:
            column = 4
            save_plot_dir = join(save_dir,save_folder)
            CheckDir(save_plot_dir) 
            
            num = len(reqLis)
            ###---------------------PLOT-------------
            row = int(np.ceil(num/column))
            fig = plt.figure(figsize=(35,6*row),dpi=100)
            ####-------------------
            inum = 1
            # print(reqLis)
            for iidx in list(reqLis):
                if trueidx_flag:
                    trueidx = iidx
                else:
                    trueidx = hsres.Lis_idx[iidx]
                # iname = img_name_lis[lis[trueidx]]
                # image = skimage.io.imread(join(db_dir,'images',iname))
                ichip = hsres.Lis_chipNo[trueidx]
                # print(trueidx)
                image = skimage.io.imread(join(hsres.chip_dir,hsres.chipname%ichip))
                fig.add_subplot(row,column,inum)
                plt.imshow(image.astype(np.uint8))
                plt.axis('off')
                
                if specImg is not None:
                    if not hasattr(specImg, '__iter__'):
                        specImg = [specImg]
                    if iidx in specImg:     
                        plt.title('chip-%d-%s'%(ichip,hsres.Lis_Chip2ID[trueidx]),fontsize=24,color='r')
                    else:
                        plt.title('chip-%d-%s'%(ichip,hsres.Lis_Chip2ID[trueidx]),fontsize=24)
                else:
                    plt.title('chip-%d-%s'%(ichip,hsres.Lis_Chip2ID[trueidx]),fontsize=24)
                
                true_ID = hsres.Lis_Chip2ID[trueidx]
                cache = np.where(hsres.Lis_Chip2ID == true_ID)[0]
                plt.title('chip-%d-%s-truenum-%d'%(ichip,true_ID,len(cache)),fontsize=24)
                inum+=1
            
            fig.tight_layout()
            fig = plt.gcf()
            
            save_path = join(save_plot_dir,'pred-%d-Num-%d.JPG'%(predID,num))
            fig.savefig(save_path,dpi=100,transparent=True,bbox_inches ='tight')
            plt.close()
            
    def plot_res_true(hsres,reqLis,predcls_pt,save_folder,save_dir=None,specImg=None,trueidx_flag=False):
        ##### predcls_pt: a ID list about predict clustering result  (n*1)
        save_flag = True
        if save_dir is None:
            save_dir = hsres.plot_dir
            if save_dir is None:
                print('No save directory')
                save_flag = False
        if save_flag == True:
            
            
            column = 4
            save_plot_dir = join(save_dir,save_folder)
            CheckDir(save_plot_dir) 
            
            # num = 0
            # for ireq in reqLis:
            #     num = num+len(ireq)
            
            num = len(reqLis)
            ###---------------------PLOT-------------
            row = int(np.ceil(num/column))
            fig = plt.figure(figsize=(35,6*row),dpi=100)
            ####-------------------
            inum = 1
            # print(reqLis)
            # for ireq in reqLis:
                # for iidx in ireq:
            for iidx in reqLis:
                if trueidx_flag:
                    trueidx = iidx
                else:
                    trueidx = hsres.Lis_idx[iidx]
                # iname = img_name_lis[lis[trueidx]]
                # image = skimage.io.imread(join(db_dir,'images',iname))
                ichip = hsres.Lis_chipNo[trueidx]
                # print(trueidx)
                image = skimage.io.imread(join(hsres.chip_dir,hsres.chipname%ichip))
                fig.add_subplot(row,column,inum)
                plt.imshow(image.astype(np.uint8))
                # plt.plot(np.arange(0,100),np.arange(0,100),)
                plt.axis('off')
                
                prdID = int(predcls_pt[trueidx])
                prdID_ele = np.where(predcls_pt == int(prdID))[0]
                if specImg is not None:
                    if not hasattr(specImg, '__iter__'):
                        specImg = [specImg]
                    if iidx in specImg:     
                        plt.title('chip-%d-predID-%d-num-%d'%(ichip,prdID,len(prdID_ele)),fontsize=24,color='r')
                    else:
                        plt.title('chip-%d-predID-%d-num-%d'%(ichip,prdID,len(prdID_ele)),fontsize=24)
                else:
                    plt.title('chip-%d-predID-%d-num-%d'%(ichip,prdID,len(prdID_ele)),fontsize=24)
                
                # true_ID = hsres.Lis_Chip2ID[trueidx]
                # cache = np.where(hsres.Lis_Chip2ID == true_ID)[0]
                # plt.title('chip-%d-%s-truenum-%d'%(ichip,true_ID,len(cache)),fontsize=24)
                inum+=1
            
            fig.tight_layout()
            fig = plt.gcf()
            
            save_path = join(save_plot_dir,'trueID-%s-Num-%d.JPG'%(hsres.Lis_Chip2ID[trueidx],num))
            fig.savefig(save_path,dpi=100,transparent=True,bbox_inches ='tight')
            plt.close()