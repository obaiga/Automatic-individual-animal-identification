#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 14 00:08:32 2022

Modify on 02/06/2023
@author: obaiga
"""

# In[Packs]
import numpy as np 
from os.path import join
import os
# import copy
# from sklearn import metrics
import networkx as nx
import matplotlib.pyplot as plt
# import matplotlib.ticker as ticker
# from mpl_toolkits.axes_grid.inset_locator import (inset_axes, InsetPosition,mark_inset)

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
hsres.plot_dir = join(hsres.db_dir,'plot')

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

# In[Load scorematrix]

if scoreAry is not None:
    Lis_SC_gt,SCavg_gt,_,Lis_SCcls_gt,_ = utils.SilhouetteScore(hsres,cluster_gt,sq_flag=sq_flag)
    
    centrd_gt,centroid_ssum_gt = utils.centroid(hsres,cluster_gt)
    TWCV_gt = np.sum(centroid_ssum_gt)
    
    print('groundtruth:Avg_SC:%.4f;nonindiv SC:%.4f'%(np.mean(Lis_SC_gt),SCavg_gt))  
        
        
# In[Load a verified document]
# record = 8002  ### adaptive k++
# record = 8001    ### k++ (benchmark)
# times = 20000
record = 8202  ### adaptive k++
# record = 8201   ### k++ (benchmark)
times = 200 
name = '%d_%d_verify.npz'%(record,times)
data = np.load(join(hsres.data_dir,name),allow_pickle=True)
# print(list(data.keys()))
Lis_ncluster_bst = data['Lis_ncluster_bst']

print(data['note'])
Lis_SC_bst = data['Lis_SC_bst']
Lis_bstSC = np.mean(Lis_SC_bst,axis=1)

Lis_bstSC_sort = np.argsort(Lis_bstSC)
# cache =int(len(Lis_bstSC)/2)
# idx = Lis_bstSC_sort[cache+1]

idx = np.argmax(Lis_bstSC)

for i in [1]:       
    ##### choice: 0 means before verifictaion; 1 means after verification
    
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
        
    ncluster = len(predcls)
    gt_correct,gt_existmore,gt_split,gt_dtl_correct,gt_dtl_partial,gt_dtl_wrong =\
        utils.correct_denom_gt(hsres,predcls_pt,predcls)
        
    correct_clusters = np.sum(gt_correct)
    correct_imgs = 0
    for inum,icorrect in enumerate(gt_correct):
        inum += 1
        correct_imgs += inum * icorrect
    
    res_correct,res_existmore,res_split,res_dtl_correct,res_dtl_partial,res_dtl_wrong =\
        utils.correct_denom_pred(hsres,predcls)
    correct_clusters = np.sum(res_correct)
    
    pred_Lis_SC,pred_SCavg,T2,_,pred_SCinfo =\
            utils.SilhouetteScore(hsres,predcls,sq_flag=sq_flag)
    ####### SC_info =[ clsidx,a,b1,b_clsidx,b2,b2_clsidx,bstScore,bstidx]s
    print(('SC:%.3f')%(np.mean(pred_Lis_SC)))
    

#%%
if 0:
    times = 200
    record = 8202
    k = hsres.k_gt
    name = '%d_%d_%s_%s_k%d.npz'\
            %(record,times,fg_flag,query_flag[2:-1],k)
    # record = 8201 
    ### for benchmark k++, determine the best clustering by ternary search
        
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

#%%
if 0:
    bstidx = Lis_bstidx[np.argmax(Lis_bstSC)]
    
    Lis_seed = data['Lis_seed'][bstidx]
    Lis_initprob = data['Lis_initprob'][bstidx]
    Lis_centrd = data['Lis_centrd'][bstidx]
    Lis_predcls_pt = data['Lis_predcls_pt'][bstidx]
    Lis_cls = data['Lis_cls'][bstidx]

### Lis_chipNo[Lis_centrd[reqIDs]]
### Lis_chipNo[Lis_seed[reqIDs]]
# In[]
##### req: Male_1_1_left Num-17      780
##### req: Male_9_4_left Num-15      797
#### req: Male_2_63_Right Num-14     30 
    
# reqchipid = 780
reqchipid = 30
# reqchipid = 797
req = np.where(hsres.Lis_Chip2ID == hsres.Lis_Chip2ID[reqchipid])[0]
print(Lis_chipNo[req])
# reqIdx = np.where(hsres.Lis_Chip2ID==hsres.Lis_ID[2])[0]
# reqIdx2IDs = predcls_pt[req]
reqIdx2IDs = Lis_predcls_pt[req]
reqIDs = np.unique(reqIdx2IDs)
print(reqIDs)
for iID in reqIDs:
    # ans = np.where(predcls_pt == iID)[0]
    ans = np.where(Lis_predcls_pt == iID)[0]
    # print(Lis_chipNo[ans])
    print(ans)

print(len(req))

#%%
a = np.array([797, 803, 804, 805, 808])-1   #### 5
b = np.array([798, 799, 800, 801, 802, 806, 807, 810, 811])-1   #### 49
c = np.array([809])-1    #### 543

pred_SCinfo[a,-1]

a = np.array([ 779,  780,  783,  784,  785,  786,  787,  789,  791,  792,  793,  794, 1405])
b = np.array([782, 788, 790, 795])
c = np.array([781])

# In[plot detailed results]
if 0:
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
            
            # ors_prdID_ele = np.where(predcls_pt == ors_prdID)[0]
            # if len(ors_prdID_ele) == len(others):
            req = [reqLis,others]
        
            hsres.plot_res_block(req,predcls_pt,save_folder,save_dir=save_dir,trueidx_flag=False)
        elif len(cache_gt) == len(reqLis):
            hsres.plot_res_block([reqLis],predcls_pt,save_folder,save_dir=save_dir,trueidx_flag=False)

#%%
# from networkx.drawing.nx_agraph import graphviz_layout
from numpy.linalg import svd
reqchipid = 807
req = np.where(hsres.Lis_Chip2ID == hsres.Lis_Chip2ID[reqchipid])[0]
# cache2 = []


### chipNo.   hsres.Lis_chipNo
req_score = hsres.scoreAry[np.ix_(req,req)]

u,s,v = svd(req_score)
# idx = eigenvalues.argsort()[::-1]   
# eigenvalues = eigenvalues[idx]
# eigenvectors = eigenvectors[:,idx]
for i in range(len(s)):
    u[:,i] = u[:,i]* s[i]


plt.scatter(u[:,0],u[:,1])
for i in range(len(s)):
    plt.text(u[i,0],u[i,1],str(i+1))
# plt.ylim([-1,1])
# plt.xlim([-2,0])
#%%
'''
full correct
'''
hsres.plot_dir = join(hsres.db_dir,'plot')

label = dict({1:'8',2:'13',3:'16',4:'19',5:'20',
         6:'17',7:'7',8:'10',9:'11',10:'14',
         11:'21',12:'9',13:'12',14:'18',15:'15'})
reqchipid = 807
req = np.where(hsres.Lis_Chip2ID == hsres.Lis_Chip2ID[reqchipid])[0]

### chipNo.   hsres.Lis_chipNo
req_score = hsres.scoreAry[np.ix_(req,req)]

cache = np.arange(len(req))
req_score[cache,cache] = 0

ans = []
for i in np.arange(len(req)):
    for j in np.arange(len(req)):
        if req_score[i,j] >0:
            value = np.int32((req_score[i,j]+req_score[j,i])/2)
            ans.append((i+1,j+1,value))
        
fig,ax1 = plt.subplots(1, figsize=(20, 20), dpi=80)
ax1.spines['top'].set_visible(False)
ax1.spines['right'].set_visible(False)
ax1.spines['left'].set_visible(False)
ax1.spines['bottom'].set_visible(False)

G = nx.Graph()   # or DiGraph, MultiGraph, MultiDiGraph, etc
G.add_weighted_edges_from(ans)
# G = nx.from_numpy_matrix(req_score)

elarge = [(u, v) for (u, v, d) in G.edges(data=True) if d["weight"] >= 10000]
emid = [(u, v) for (u, v, d) in G.edges(data=True) if (1000 <d["weight"] < 10000)]
esmall = [(u, v) for (u, v, d) in G.edges(data=True) if d["weight"] <= 1000]
# pos = nx.spring_layout(G)  # positions for all nodes - seed for reproducibility

pos = dict({1:[-3.770133751593343518e-01, 2.759966482948069677e-01],
        2:[12.635168433167104002e-02,  10.643479040531839658e-02],
        3:[4.515139630646249902e-01,  -6.600797443422670718e-02],
        4:[1.069091624635725990e-01,  -1.827453501795529844e-01],
        5:[3.594126018015566526e-01,  -1.516987789513395968e-01],
        6:[-2.253922620192841064e-01, -2.705679279814620475e-01],
        7:[-7.179862189665923988e-02, 5.654276877820356928e-01],
        8:[-3.005506497299094693e-01, 3.504526348477631836e-02],
        10:[-2.459820939713588950e-01, -1.532979224879535696e-01],
        9:[-8.895072294058203877e-02, 12.504526348477631836e-02],
        11:[-1.621184849674875694e-01, -0.700000000000000000e+00],
        12:[1.496123943761470086e-02, 3.667079321025714544e-01],
        13:[5.20877670833195294e-01, 5.247397784021811562e-01],
        14:[-9.562119492355435069e-02, -2.826645164035045954e-01],
        15:[1.161909874258675729e-01, -5.095217532941232613e-02]})

# edges
nx.draw_networkx_edges(G, pos, edgelist=elarge, width=2,edge_color="forestgreen")

nx.draw_networkx_edges(G, pos, edgelist=emid, width=1,alpha=0.5, edge_color="forestgreen",)

nx.draw_networkx_edges(
    G, pos, edgelist=esmall, width=1, alpha=0.5, edge_color="forestgreen", style="dashed"
)

# nodes
nx.draw_networkx_nodes(G, pos, node_size=1000,node_color='darkorange')
# node labels
nx.draw_networkx_labels(G, pos, font_size=20,labels=label)


plt.savefig(join(hsres.plot_dir,'demo.png'), format='png', dpi=100, transparent=True)
plt.show()


# In[]

label = dict({1:'30',2:'28',3:'38',4:'24',5:'31',
         6:'37',7:'32',8:'35',9:'33',10:'23',
         11:'29',12:'25',13:'36',14:'34',15:'27',
         16:'26',17:'22'})
reqchipid = 791
req = np.where(hsres.Lis_Chip2ID == hsres.Lis_Chip2ID[reqchipid])[0]
# req = np.delete(req,2)

### chipNo.   hsres.Lis_chipNo
req_score = hsres.scoreAry[np.ix_(req,req)]
cache = np.arange(len(req))
req_score[cache,cache] = 0

ans = []
for i in np.arange(len(req)):
    for j in np.arange(len(req)):
        if req_score[i,j]>0:
            ans.append((i+1,j+1,np.int32((req_score[i,j]+req_score[j,i])/2)))



fig,ax1 = plt.subplots(1, figsize=(20, 20), dpi=80)
ax1.spines['top'].set_visible(False)
ax1.spines['right'].set_visible(False)
ax1.spines['left'].set_visible(False)
ax1.spines['bottom'].set_visible(False)

G = nx.Graph()   # or DiGraph, MultiGraph, MultiDiGraph, etc
G.add_weighted_edges_from(ans)


elarge = [(u, v) for (u, v, d) in G.edges(data=True) if d["weight"] >= 10000]
emid = [(u, v) for (u, v, d) in G.edges(data=True) if (1000 <d["weight"] < 10000)]
esmall = [(u, v) for (u, v, d) in G.edges(data=True) if d["weight"] <= 1000]
# pos = nx.spring_layout(G)  # positions for all nodes - seed for reproducibility

pos = dict({1: [-0.17946434, -0.03310053],
  2: [0.40960678, 0.10708885],
  3:[ 0.52257012, -0.5        ],
  4: [0.06605535, 0.43530605],
  5: [0.11569981, 0.02976887],
  6: [-0.08522285, -0.23345919],
  7: [-0.10327258, -0.10335004],
  8: [ 0.09464555, -0.15334772],
  9: [ 0.03779725, -0.11498403],
  10: [-0.34875445,  0.35247289],
  11: [-0.2650478 , -0.02387748],
  12:[0.32780086, 0.32901134],
  13:[ 0.02257012, -0.35       ],
  14:[ 0.1723369 , -0.09104693],
  15:[0.37725211, 0.10863342],
  16:[-0.1266249 ,  0.09000531],
  17:[-0.51537783,  0.30087918]
  })

# edges
nx.draw_networkx_edges(G, pos, edgelist=elarge, width=2,edge_color="forestgreen")

nx.draw_networkx_edges(G, pos, edgelist=emid, width=1,alpha=0.5, edge_color="forestgreen",)

nx.draw_networkx_edges(
    G, pos, edgelist=esmall, width=1, alpha=0.5, edge_color="forestgreen", style="dashed"
)

# nodes
nx.draw_networkx_nodes(G, pos, node_size=500,node_color='darkorange')
# node labels
nx.draw_networkx_labels(G, pos, font_size=20,labels=label)


plt.savefig(join(hsres.plot_dir,'demo.png'), format='png', dpi=100, transparent=True)
plt.show()


#%%

G = nx.Graph()   # or DiGraph, MultiGraph, MultiDiGraph, etc
G.add_weighted_edges_from([1,2,2],[1,3,1])
# pos = nx.spring_layout(G)  # positions for all nodes - seed for reproducibility

nx.draw(G) 
plt.show()


#%%
req = np.array( [30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 43])-1


label = dict({1:'24',2:'26',3:'30',4:'34',6:'29',
         5:'28',7:'31',8:'22',9:'27',10:'32',
         11:'23',12:'33',13:'25'})

### chipNo.   hsres.Lis_chipNo
req_score = hsres.scoreAry[np.ix_(req,req)]
cache = np.arange(len(req))
req_score[cache,cache] = 0

ans = []
for i in np.arange(len(req)):
    for j in np.arange(len(req)):
        if req_score[i,j]>0:
            ans.append((i+1,j+1,np.int32((req_score[i,j]+req_score[j,i])/2)))


fig,ax1 = plt.subplots(1, figsize=(20, 20), dpi=80)
ax1.spines['top'].set_visible(False)
ax1.spines['right'].set_visible(False)
ax1.spines['left'].set_visible(False)
ax1.spines['bottom'].set_visible(False)

G = nx.Graph()   # or DiGraph, MultiGraph, MultiDiGraph, etc
G.add_weighted_edges_from(ans)


elarge = [(u, v) for (u, v, d) in G.edges(data=True) if d["weight"] >= 10000]
emid = [(u, v) for (u, v, d) in G.edges(data=True) if (1000 <d["weight"] < 10000)]
esmall = [(u, v) for (u, v, d) in G.edges(data=True) if d["weight"] <= 1000]

# pos = nx.spring_layout(G)  # positions for all nodes - seed for reproducibility


pos = dict ({
    
    1:[-3.155626567812681027e-01,2.425570505801188104e-01],
    2:[-4.510615619381617702e-01,2.445889345144678095e-01],
    3:[-3.155626567812681027e-01,-0.500000000000000000e+00],
    4:[1.741564711805027965e-01,-2.213599853629866443e-01],
    6:[-9.298998418990112036e-02,-7.746216179476402008e-02],
    5:[1.151399034635477298e-01,7.334036770980555883e-02],
    7:[2.350386580776236300e-01,10.998729548000874101e-02],
    8:[-4.157587661456506600e-01,3.414326422739172862e-01],
    9:[-2.023352379119031097e-01,1.334366756206592508e-02],
    10:[4.506413919991847372e-01, 5.124119060020420474e-02],
    11:[1.741578537838374408e-01,4.299433948454409737e-01],
    12:[3.008606477625769715e-01,-1.901077197812116037e-01],
    13:[-1.185518221062743238e-01,2.124953233729621194e-01],
    
    })

# edges
nx.draw_networkx_edges(G, pos, edgelist=elarge, width=2,edge_color="forestgreen")

nx.draw_networkx_edges(G, pos, edgelist=emid, width=1,alpha=0.5, edge_color="forestgreen",)

nx.draw_networkx_edges(
    G, pos, edgelist=esmall, width=1, alpha=0.5, edge_color="forestgreen", style="dashed"
)

# nodes
nx.draw_networkx_nodes(G, pos, node_size=1000,node_color='darkorange')
# node labels
nx.draw_networkx_labels(G, pos, font_size=22,labels=label)



plt.savefig(join(hsres.plot_dir,'demo.png'), format='png', dpi=100, transparent=True)
plt.show()