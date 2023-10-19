#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 15 15:12:19 2022

@author: chkaul
"""


#Imports
import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import random
import time
from tqdm.auto import tqdm
import glob
import imageio
import natsort
import cv2
from matplotlib import gridspec
from skimage.metrics import structural_similarity
from sklearn.metrics import mean_squared_error
from tqdm.auto import tqdm
from IPython import display

import tensorflow as tf
tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(log_device_placement=True))

print(tf.__version__)

import importlib
sys.path.insert(1,'../src')
import utils
import radar_utils as ru
import nn_utils as nn
importlib.reload(utils)
importlib.reload(ru)
importlib.reload(nn)



#Load metadata
samples_per_chirp = 128 #number of samples
chirps_per_burst = 16
T_c = (1/2000) #up chirp time = 1/chirp rate
B = 1e9 #bandwidth = 61GHz - 59Ghz
c = 3e8 
f_tx = 60e9 #tranmission frequency = 60GHz
wavelength = c/f_tx
spacing = 2.5e-3 #spacing between receiver antennas in m

#Load range and frequency samples corresponding to this metadata
rang = ru.range_freqsamples()
vel = ru.velocity_freqsamples()


# #Load the crd-image pairs
crd = np.load("/home/chkaul/Desktop/soli/01_data/crd_data_resampled_no_declutter.npy", allow_pickle=True)
rp = np.abs(np.load("/home/chkaul/Desktop/soli/01_data/rp_data_resampled.npy", allow_pickle=True))
images = np.load("/home/chkaul/Desktop/soli/01_data/images_resized.npy", allow_pickle=True).astype('float32')
iq = np.load("/home/chkaul/Desktop/soli/01_data/iq_data_resampled.npy", allow_pickle=True)

crd = crd.reshape(-1,3,64,16)
rp = rp.reshape(-1,3,64)
iq = iq.reshape(-1,3,128)
images = images.reshape(-1,54,96)

aoa = ru.get_aoa_data(crd)



importlib.reload(nn)
data_prep = nn.Data_prep(crd,rp,iq,images)
input_format = r'iq'
#ard-aoa-aoa input format.
#x_train,x_val,x_test,y_train,y_val,y_test = data_prep.ard_aoa_aoa_data_prep(0,10*900,12*900,14*900,thresh=0.25,windowed=True,window=15,dataset_interval=900,step=3) #i keep the data prep separate so i don't forget steps when i change things
#I/q input format.
x_train,x_val,x_test,y_train,y_val,y_test = data_prep.iq_data_prep(0,10*900,12*900,14*900,windowed=False,window=15,dataset_interval=900,step=3) #i keep the data prep separate so i don't forget steps when i change things

#RP input formatting
# x_train = rp[0:10*900][:,2].reshape(-1,1,64)
# x_val = rp[10*900:12*900][:,2].reshape(-1,1,64)
# x_test = rp[12*900:14*900][:,2].reshape(-1,1,64)
# for i in range(len(x_train)):
#     x_train[i] = x_train[i] - np.min(x_train[i])
#     x_train[i] = x_train[i]/np.max(x_train[i])
# for i in range(len(x_val)):
#     x_val[i] = x_val[i] - np.min(x_val[i])
#     x_val[i] = x_val[i]/np.max(x_val[i])
# for i in range(len(x_test)):
#     x_test[i] = x_test[i] - np.min(x_test[i])
#     x_test[i] = x_test[i]/np.max(x_test[i])
# x_train[:,:,0:3] = 0    
# x_val[:,:,0:3] = 0  
# x_test[:,:,0:3] = 0  


max_depth = data_prep.max_depth

print('Max depth in image dataset:',max_depth,'m\n',
      x_train.shape,'\n',
      x_val.shape,'\n',
      x_test.shape,'\n',
      y_train.shape,'\n',
      y_val.shape,'\n',
      y_test.shape)



#Check input output pair
plt.figure(figsize=(8,3))
plt.subplot(121)
plt.plot(x_train[0,0].astype('float32'))
#plt.colorbar()
plt.subplot(122)
plt.imshow(y_train[0,0])
plt.colorbar()
plt.show()



#Plot an image just to check
thresh = 0.5
importlib.reload(utils)
utils.plot_rp_crd_aoa_image_data(rp = rp,
                              crd=crd,
                              aoa=aoa,
                              images=images,
                              co=50,
                              thresh=thresh,
                              rang=rang,
                              vel=vel,
                              max_depth=max_depth)


#Load network architecture
importlib.reload(nn)
architecture = r'cnn_v1'
date = time.strftime('%Y%m%d') 
print(date)
date='20210528'
model = nn.cnn_v1(input_shape = x_train[0].shape)
adam = tf.keras.optimizers.Adam()

# def ssim_loss(y_true, y_pred):
#     return tf.reduce_mean(-tf.image.ssim(y_true, y_pred, 1.0))

optim = tf.keras.optimizers.Adam(lr=1e-4)

model.compile(loss='mse',optimizer=optim)    
model.summary()
mycallback = tf.keras.callbacks.EarlyStopping(monitor='val_loss',
                                            patience=20)


history = model.fit(x = x_train,
                    y = y_train,
                    batch_size = 50,
                    epochs = 100, 
                    validation_data = (x_val, y_val),
                    shuffle=True,
                    callbacks=[mycallback])



fig=plt.figure(figsize=(5,3))
plt.title('Training history')
plt.plot(history.history['loss'],label='training loss')
plt.plot(history.history['val_loss'],label='val set loss')
plt.xlabel('Epoch')
plt.ylabel('SSIM')
plt.yscale('log')
plt.legend()    