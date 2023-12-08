#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 12 13:26:24 2023

change brightness for an image dataset

@author: obaiga
"""
# In[]
from PIL import Image, ImageEnhance
from os.path import join,exists
from os import makedirs
import glob
import random
import numpy as np 
import pandas as pd
import copy

# In[]
# Load the image
imgs_dir = '/Users/obaiga/Jupyter/Python-Research/ds_160/images-db/'
imgs_lis = glob.glob(join(imgs_dir,'*.JPG'))
nsample = len(imgs_lis)
# img_idx = np.arange(nsample)
# random.shuffle(img_idx)
img_idx = copy.copy(brightness_rand_ord)
# In[Show exmaple]
if 0:
    image = Image.open(imgs_lis[img_idx[0]])
    
    # Adjust brightness for overexposure (>1) or underexposure (<1)
    # Change this value to control the level of exposure
    enhancer = ImageEnhance.Brightness(image)
    adjusted_image = enhancer.enhance(2)  # Use >1 for overexposure, <1 for underexposure
    
    # Save or display the result
    adjusted_image.show()
    # image.show()
    
# In[enhance brightness]
enhance_lis = np.ones(nsample) 

value_lis = [0.4,1.6]
for i,ivalue in enumerate(value_lis):
    enhance_lis[i*155*3:(i+1)*155*3] = ivalue
    
#%%
db_name = 'bright60'
save_dir = join('/Users/obaiga/Jupyter/Python-Research/Africaleopard',db_name,'img_db')

if not exists(save_dir):
    makedirs(save_dir)

for ii,randi in enumerate(img_idx):
    image = Image.open(imgs_lis[randi])
    imgs_name = imgs_lis[randi][len(imgs_dir):]
    # Adjust brightness for overexposure (>1) or underexposure (<1)
    # Change this value to control the level of exposure
    enhancer = ImageEnhance.Brightness(image)
    adjusted_image = enhancer.enhance(enhance_lis[ii])  # Use >1 for overexposure, <1 for underexposure
    
    # Display the result
    # adjusted_image.show()
    
    # Save the adjusted image
    adjusted_image.save(join(save_dir,imgs_name), 'JPEG')
    

# In[save_table]

sort_enhance_lis = np.ones(nsample)
for j,iidx in enumerate(img_idx):
    sort_enhance_lis[iidx] = enhance_lis[j]

table_dir = join('/Users/obaiga/Jupyter/Python-Research/Africaleopard','table.csv')
table = pd.read_csv(table_dir,skipinitialspace=True)
# print(table.columns.tolist())  ### print table column name
Lis_chipNo = table['#   ChipID']
Lis_img = table['Image']

data = {'#   ChipID':Lis_chipNo,'Image':Lis_img,'Brightness':sort_enhance_lis}
df = pd.DataFrame(data)
df.to_csv(join('/Users/obaiga/Jupyter/Python-Research/Africaleopard',db_name,'bright.csv'), index=False)