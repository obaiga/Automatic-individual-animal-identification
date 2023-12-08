#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 28 15:39:11 2023

@author: obaiga
"""
# In[packs]
import numpy as np 
from os.path import join
import os
import copy
from sklearn import metrics
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import pandas as pd 

file_name = 'summary_result.xlsx'
file_path = '/Users/obaiga/Jupyter/Python-Research/Africaleopard/'
df = pd.read_excel(join(file_path,file_name))

dataname_lis = [0,10,20,30,40,50,60,70]
# column_names = ["data_name", "mtd_name", "mutual_info","silhouette","s_noniso",
#                 "ncluster","correct","partial","incorrect",
#                 "cor_cls_ratio","part_cls_ratio","incor_cls_ratio",
#                 "nimages","corr_imgs","part_imgs","incor_imgs",
#                 "cor_img_ratio","part_img_ratio","incor_img_ratio"]
data = df.to_numpy()


idx_lis = [0,8,8+4,8+4*2,8+4*3,8+4*4,8+4*5,8+4*6]
idx_lis = np.array(idx_lis,dtype=np.int32)

mutual_lis = np.array(data[idx_lis,2],dtype=np.float16)
# sc_lis = data[idx_lis,3]
ncluster_lis = np.array(data[idx_lis,5],dtype=np.int8)
corrcls_rat_lis = np.array(data[idx_lis,9],dtype=np.float16)
partcls_rat_lis = np.array(data[idx_lis,10],dtype=np.float16)
incorcls_rat_lis = np.array(data[idx_lis,11],dtype=np.float16)

corrimgs_rat_lis = np.array(data[idx_lis,16],dtype=np.float16)
partimgs_rat_lis = np.array(data[idx_lis,17],dtype=np.float16)
incorimgs_rat_lis = np.array(data[idx_lis,18],dtype=np.float16)

#%%
##----------plot-------------------
# Lis_color = ['#2ca02c', '#ff7f0e', '#1f77b4', '#e377c2',
#             '#8c564b','#17becf','#bcbd22','#d62728',  
#             '#9467bd', '#7f7f7f']


labelsize = 20
fontsize = 15
markersize = 10

fig,ax1 = plt.subplots(figsize=[15,10])

ax1.spines['top'].set_visible(False)
ax1.spines['right'].set_visible(False)

# order_lis = [0,2]  ### k++, adaptive k++
# title_content = 'Comparison for clustering algorithms in different brightness'

order_lis = [1,3]  ####k++&verify, adaptive k++&verify
title_content = 'Comparison for clustering algorithms followed by verification in different brightness'

for i,ii in enumerate(order_lis):
    req_lis = idx_lis+ii
    # corr_lis = np.array(data[req_lis,9],dtype=np.float16)   ### cluster ratio
    # part_lis = np.array(data[req_lis,10],dtype=np.float16)
    # incor_lis = np.array(data[req_lis,11],dtype=np.float16)
    # ylabel = 'ratio of total clusters'
    
    corr_lis = np.array(data[req_lis,16],dtype=np.float16)   #### image ratio
    part_lis = np.array(data[req_lis,17],dtype=np.float16)
    incor_lis = np.array(data[req_lis,18],dtype=np.float16)
    ylabel = 'ratio of total images'
    
    if i == 0:
        style = 'o--'
        name = 'k++'
    else:
        style = 'o-'
        name = 'adaptive k++'
    plt.plot(dataname_lis,corr_lis,style,markersize=markersize,
             c='forestgreen',label=r'correct in '+name)
    
    plt.plot(dataname_lis,part_lis,style,markersize=markersize,
              c='darkorange',label=r'partial correct in '+name)
    
    plt.plot(dataname_lis,incor_lis,style,markersize=markersize,
              c='#1f77b4',label=r'incorrect in '+name)

mtd_text = [Line2D([0],[0], color='black', lw=2,linestyle='-', label=r'adaptive $k$++'),
               Line2D([0],[0], color='black', lw=2,linestyle='--', label=r'$k$++')]

plt.gca().add_artist(plt.legend(handles=mtd_text, fontsize=labelsize,loc='center'))


legend_text = [Line2D([0],[0], color='#2ca02c', markersize=10,marker='o', label='correct'),
                Line2D([0],[0], color='#ff7f0e', markersize=10,marker='o', label='partial correct'),
                Line2D([0],[0], color='#1f77b4', markersize=10,marker='o', label='incorrect'),]
plt.legend(handles=legend_text,fontsize=labelsize,loc='center left')


plt.yticks(np.arange(0,1,0.1),fontsize=fontsize)
plt.xticks(fontsize=fontsize)

plt.ylim([0,0.92])

plt.ylabel(ylabel,fontsize=labelsize)

plt.xlabel(r'brightness factor (%)',fontsize=labelsize)
plt.title(title_content,fontsize=labelsize+2)

ax1.yaxis.grid(True, linestyle='--', linewidth=0.5, color='gray')

plt.show()
# %%
##----------plot-------------------
# idx_lis = np.array([0,8,8+4,8+4*2,8+4*3,8+4*4])
Lis_color = ['#2ca02c', '#ff7f0e', '#1f77b4', '#e377c2',
            '#8c564b','#17becf','#bcbd22','#d62728',  
            '#9467bd', '#7f7f7f']
labelsize = 20
fontsize = 15
markersize = 10

fig,ax1 = plt.subplots(figsize=[15,10])

ax1.spines['top'].set_visible(False)
# ax2 = ax1.twinx()
# ax2.spines['top'].set_visible(False)

order_lis = [0,2,1,3]  ### k++, adaptive k++
# order_lis = [1,3]  ####k++&verify, adaptive k++&verify

inum = 0
for i,ii in enumerate(order_lis):
    req_lis = idx_lis+ii
    mutual_lis = np.array(data[req_lis,2],dtype=np.float16)
    ncluster_lis = np.array(data[req_lis,5],dtype=np.int32)
    # print(ncluster_lis)
    if (ii == 0) or (ii == 1):
        color = '#2ca02c'
        if ii == 0:
            name = 'k++'
            style = 'o--'
            
        else: 
            name = 'k++&verify'
            style = 'o-'
    else:
        color = '#ff7f0e'
        if ii == 2:
            name = 'adaptive k++'
            style = 'o--'
        else:
            name = 'adaptive k++&verify'
            style = 'o-'
    
    ax1.plot(dataname_lis,mutual_lis,style,markersize=markersize,
             c=color,label=name)
    inum+=1
    
    # ax2.plot(dataname_lis,ncluster_lis,style,markersize=markersize,
    #          c='darkorange',label=name)

#get handles and labels
handles, labels = plt.gca().get_legend_handles_labels()

#specify order of items in legend
order = [3,2,1,0]
###add legend to plot
ax1.legend([handles[jj] for jj in order],[labels[jj] for jj in order],
           fontsize=labelsize) 

plt.yticks(np.arange(0.79,0.98,0.025),fontsize=fontsize)
plt.xticks(fontsize=fontsize)

plt.yticks(fontsize=fontsize)    
ax1.set_xlabel('brightness factor (%)',fontsize=labelsize)
ax1.set_ylabel('mutual information score',fontsize=labelsize)

ax1.yaxis.grid(True, linestyle='--', linewidth=0.5, color='gray')

plt.title('Algorithm accuracy with brightness change',fontsize=labelsize+2)