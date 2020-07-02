#!/usr/bin/env python
# coding: utf-8

# In[6]:


import os
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import imshow
from keras.models import load_model


# # Parameters

# In[7]:


model_path = r'C:\Downloads\model.h5'
dataset_path = r'C:\Downloads\Dataset'


# In[8]:


model = load_model(model_path)


# # Load Images and Super Sample

# In[10]:


paths = []
for r, d, f in os.walk(dataset_path):
    for file in f:
        if '.png' in file or 'jpg' in file:
            paths.append(os.path.join(r, file))
count = 0
for path in paths:
    #select image
    img = Image.open(path)

    #create plot
    f, axarr = plt.subplots(1,3,figsize=(15,15),gridspec_kw={'width_ratios': [1,4,4]})
    axarr[0].set_xlabel('Original Image (64x64)', fontsize=10)
    axarr[1].set_xlabel('Interpolated Image (256x256)', fontsize=10)
    axarr[2].set_xlabel('Super Sampled Image (256x256)', fontsize=10)

    #original image
    x = img.resize((64,64))
    #interpolated (resized) image
    y = x.resize((256,256))
    #plotting first two images
    x = np.array(x)
    y = np.array(y)
    axarr[0].imshow(x)
    axarr[1].imshow(y)
    #plotting super sampled image
    x = x.reshape(1,64,64,3)/256
    result = np.array(model.predict_on_batch(x))*256
    result = result.reshape(256,256,3)
    result = result.astype(int)
    axarr[2].imshow(result)
    f.savefig('frame_%d.png' % count)
    count = count + 1

