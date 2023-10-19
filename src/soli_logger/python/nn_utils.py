# -*- coding: utf-8 -*-
"""
Created on Tue Apr 20 17:01:19 2021

@author: Valentin Kapitany
"""
import numpy as np
import tensorflow as tf
import tensorflow.keras.backend as K
import tensorflow.keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, Conv3D, MaxPooling1D, MaxPooling3D, Conv1D, Input, \
    BatchNormalization, Conv2DTranspose, UpSampling3D, UpSampling2D, UpSampling1D, ZeroPadding1D, Reshape, LeakyReLU
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.regularizers import l2, l1, l1_l2

import os
filePath = __file__
absFilePath = os.path.abspath(__file__)
path, filename = os.path.split(absFilePath)
os.chdir(path)

import radar_utils as ru
import p2go_radar_utils as pu
from skimage.util.shape import view_as_blocks, view_as_windows

#%%

def reset_keras(verbose=False):
    sess = tf.compat.v1.keras.backend.get_session()
    tf.compat.v1.keras.backend.clear_session()
    sess.close()
    tf.compat.v1.sess = tf.compat.v1.keras.backend.get_session()


#     if verbose==True:
#         print(gc.collect()) # if it's done something you should see a number being outputted
#     else: 
#         pass

    # use the same config as you used to create the session
    config = tf.compat.v1.ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = 1
    config.gpu_options.visible_device_list = "0"
    tf.compat.v1.keras.backend.set_session(tf.compat.v1.Session(config=config))

#%% data preparation

class Data_prep(object):
    def __init__(self,crd,rp,iq,images,which='soli'):
        self.crd = crd
        self.rp = rp
        self.iq = iq
        self.images = images
        if which == 'soli':
            self.aoa = ru.get_aoa_data(self.crd)
        elif which =='p2go':
            self.aoa = pu.get_aoa_data(self.crd)
        self.max_depth = np.max(self.images)
        self.images = self.images/self.max_depth #normalise images
        
    def rpx3_data_prep(self,train,val,test):
        rp_max_0 = np.max(np.abs(self.rp[:,0]))
        rp_max_1 = np.max(np.abs(self.rp[:,1]))
        rp_max_2 = np.max(np.abs(self.rp[:,2]))
        self.rp[:,0] = self.rp[:,0]/rp_max_0
        self.rp[:,1] = self.rp[:,1]/rp_max_1
        self.rp[:,2] = self.rp[:,2]/rp_max_2
        #print('ch 0 max:',np.unravel_index(rp[:,0].argmax(), rp[:,0].shape))
        #print('ch 1 max:',np.unravel_index(rp[:,1].argmax(), rp[:,1].shape))
        #print('ch 2 max:',np.unravel_index(rp[:,2].argmax(), rp[:,2].shape))
        x_train = self.rp[:train]
        x_test = self.rp[val:test] 
        x_val = self.rp[train:val] 
        
        y_train = self.images[:train].reshape(-1, 1, self.images[0].shape[0], self.images[0].shape[1])
        y_test = self.images[val:test].reshape(-1, 1, self.images[0].shape[0], self.images[0].shape[1])
        y_val = self.images[train:val].reshape(-1, 1, self.images[0].shape[0], self.images[0].shape[1])
    
        return x_train,x_val,x_test,y_train,y_val,y_test
    
    def iq_data_prep(self,train_start,train,val,test,windowed=False,window=15,dataset_interval=0,step=1):
        iq = self.iq.copy()
        for i in range(len(iq)):
            for j in range(iq.shape[1]):
                iq[i,j] = iq[i,j] - np.min(iq[i,j])
                iq[i,j] = iq[i,j]/np.max(iq[i,j])
        x_train = iq[train_start:train].reshape(-1,3,128)
        x_val = iq[train:val].reshape(-1,3,128)
        x_test = iq[val:test].reshape(-1,3,128)
        
        y_data = self.images.reshape(-1, 1, self.images[0].shape[0], self.images[0].shape[1])
        y_train = y_data[train_start:train]
        y_val = y_data[train:val]
        y_test = y_data[val:test]
        
        if windowed==True:
            
            if dataset_interval==0:
                #Organise into blocks/windows for video processing
                
                x_train = view_as_windows(x_train,(window,3,x_train[0,0].shape[0]),step=(step,1,1))
                x_train = np.squeeze(x_train)
                x_train = np.transpose(x_train, (0,2,1,3))
                
                x_val = view_as_windows(x_val,(window,3,x_val[0,0].shape[0]),step=(step,1,1,1))
                x_val = np.squeeze(x_val)
                x_val = np.transpose(x_val, (0,2,1,3))
                
                x_test = view_as_blocks(x_test,(window,3,x_test[0,0].shape[0]))
                x_test = np.squeeze(x_test)
                x_test = np.transpose(x_test, (0,2,1,3))
                
                
                y_train = view_as_windows(y_train,(window,1,y_data[0,0].shape[0], y_data[0,0].shape[1]),step=(step,1,1,1))
                y_train = np.squeeze(y_train)
                y_train = y_train.reshape(-1,1,window,y_data[0,0].shape[0],y_data[0,0].shape[1])
                
                y_val = view_as_windows(y_val,(window,1,y_data[0,0].shape[0], y_data[0,0].shape[1]),step=(step,1,1,1))
                y_val = np.squeeze(y_val)
                y_val = y_val.reshape(-1,1,window,y_data[0,0].shape[0],y_data[0,0].shape[1])
                
                y_test = view_as_blocks(y_test,(window,1,y_data[0,0].shape[0], y_data[0,0].shape[1]))
                y_test = np.squeeze(y_test)
                y_test = y_test.reshape(-1,1,window,y_data[0,0].shape[0],y_data[0,0].shape[1])
            else:
                X_train = []
                X_val = []
                X_test = []
                
                Y_train = []
                Y_val= []
                Y_test = []
                
                for i in range(int(len(x_train)/dataset_interval)):
                    x_tr = view_as_windows(x_train[i*dataset_interval:(i+1)*dataset_interval],(window,3,x_train[0,0].shape[0]),step=(step,1,1))
                    x_tr = np.squeeze(x_tr)
                    X_train.append(np.transpose(x_tr, (0,2,1,3)))
                
                for i in range(int(len(x_val)/dataset_interval)):
                    x_va = view_as_windows(x_val[i*dataset_interval:(i+1)*dataset_interval],(window,3,x_val[0,0].shape[0]),step=(step,1,1))
                    x_va = np.squeeze(x_va)
                    X_val.append(np.transpose(x_va, (0,2,1,3)))
                
                for i in range(int(len(x_test)/dataset_interval)):
                    x_te = view_as_blocks(x_test[i*dataset_interval:(i+1)*dataset_interval],(window,3,x_test[0,0].shape[0]))
                    x_te = np.squeeze(x_te)
                    X_test.append(np.transpose(x_te, (0,2,1,3)))
                
                for i in range(int(len(y_train)/dataset_interval)):
                    y_tr = view_as_windows(y_train[i*dataset_interval:(i+1)*dataset_interval],(window,1,y_data[0,0].shape[0], y_data[0,0].shape[1]),step=(step,1,1,1))
                    y_tr = np.squeeze(y_tr)
                    Y_train.append(y_tr.reshape(-1,1,window,y_data[0,0].shape[0],y_data[0,0].shape[1]))
                
                for i in range(int(len(y_val)/dataset_interval)):
                    y_va = view_as_windows(y_val[i*dataset_interval:(i+1)*dataset_interval],(window,1,y_data[0,0].shape[0], y_data[0,0].shape[1]),step=(step,1,1,1))
                    y_va = np.squeeze(y_va)
                    Y_val.append(y_va.reshape(-1,1,window,y_data[0,0].shape[0],y_data[0,0].shape[1]))
                    
                for i in range(int(len(y_test)/dataset_interval)):
                    y_te = view_as_blocks(y_test[i*dataset_interval:(i+1)*dataset_interval],(window,1,y_data[0,0].shape[0], y_data[0,0].shape[1]))
                    y_te = np.squeeze(y_te)
                    Y_test.append(y_te.reshape(-1,1,window,y_data[0,0].shape[0],y_data[0,0].shape[1]))
                x_train = np.asarray(X_train)
                x_train = x_train.reshape(-1,x_train.shape[2],x_train.shape[3],x_train.shape[4])
                x_val = np.asarray(X_val)
                x_val = x_val.reshape(-1,x_val.shape[2],x_val.shape[3],x_val.shape[4])
                x_test = np.asarray(X_test)
                x_test = x_test.reshape(-1,x_test.shape[2],x_test.shape[3],x_test.shape[4])
                y_train = np.asarray(Y_train)
                y_train = y_train.reshape(-1,y_train.shape[2],y_train.shape[3],y_train.shape[4],y_train.shape[5])
                y_val = np.asarray(Y_val)
                y_val = y_val.reshape(-1,y_val.shape[2],y_val.shape[3],y_val.shape[4],y_val.shape[5])
                y_test = np.asarray(Y_test)
                y_test = y_test.reshape(-1,y_test.shape[2],y_test.shape[3],y_test.shape[4],y_test.shape[5])
        
        
        
        return x_train,x_val,x_test,y_train,y_val,y_test
    
    def iq_data_prep_p2go(self,train_start,train,val,test,windowed=False,window=15,dataset_interval=0,step=1):
        I = np.real(self.iq.copy())
        print(I.shape)
        Q = np.imag(self.iq.copy())
        for i in range(len(I)):
            for j in range(I.shape[1]):
                I[i,j] = I[i,j] - np.min(I[i,j])
                I[i,j] = I[i,j]/(max(np.max(I[i,j]),1e-5))
                Q[i,j] = Q[i,j] - np.min(Q[i,j])
                Q[i,j] = Q[i,j]/(max(np.max(Q[i,j]),1e-5))
        iq = np.concatenate([I,Q],axis=1)
        x_train = iq[train_start:train].reshape(-1,4,128)
        x_val = iq[train:val].reshape(-1,4,128)
        x_test = iq[val:test].reshape(-1,4,128)        
        y_data = self.images.reshape(-1, 1, self.images[0].shape[0], self.images[0].shape[1])
        y_train = y_data[train_start:train]
        y_val = y_data[train:val]
        y_test = y_data[val:test]
        
        if windowed==True:
            
            if dataset_interval==0:
                #Organise into blocks/windows for video processing
                
                x_train = view_as_windows(x_train,(window,2,x_train[0,0].shape[0]),step=(step,1,1))
                x_train = np.squeeze(x_train)
                x_train = np.transpose(x_train, (0,2,1,3))
                
                x_val = view_as_windows(x_val,(window,2,x_val[0,0].shape[0]),step=(step,1,1,1))
                x_val = np.squeeze(x_val)
                x_val = np.transpose(x_val, (0,2,1,3))
                
                x_test = view_as_blocks(x_test,(window,2,x_test[0,0].shape[0]))
                x_test = np.squeeze(x_test)
                x_test = np.transpose(x_test, (0,2,1,3))
                
                
                y_train = view_as_windows(y_train,(window,1,y_data[0,0].shape[0], y_data[0,0].shape[1]),step=(step,1,1,1))
                y_train = np.squeeze(y_train)
                y_train = y_train.reshape(-1,1,window,y_data[0,0].shape[0],y_data[0,0].shape[1])
                
                y_val = view_as_windows(y_val,(window,1,y_data[0,0].shape[0], y_data[0,0].shape[1]),step=(step,1,1,1))
                y_val = np.squeeze(y_val)
                y_val = y_val.reshape(-1,1,window,y_data[0,0].shape[0],y_data[0,0].shape[1])
                
                y_test = view_as_blocks(y_test,(window,1,y_data[0,0].shape[0], y_data[0,0].shape[1]))
                y_test = np.squeeze(y_test)
                y_test = y_test.reshape(-1,1,window,y_data[0,0].shape[0],y_data[0,0].shape[1])
            else:
                X_train = []
                X_val = []
                X_test = []
                
                Y_train = []
                Y_val= []
                Y_test = []
                
                for i in range(int(len(x_train)/dataset_interval)):
                    x_tr = view_as_windows(x_train[i*dataset_interval:(i+1)*dataset_interval],(window,2,x_train[0,0].shape[0]),step=(step,1,1))
                    x_tr = np.squeeze(x_tr)
                    X_train.append(np.transpose(x_tr, (0,2,1,3)))
                
                for i in range(int(len(x_val)/dataset_interval)):
                    x_va = view_as_windows(x_val[i*dataset_interval:(i+1)*dataset_interval],(window,2,x_val[0,0].shape[0]),step=(step,1,1))
                    x_va = np.squeeze(x_va)
                    X_val.append(np.transpose(x_va, (0,2,1,3)))
                
                for i in range(int(len(x_test)/dataset_interval)):
                    x_te = view_as_blocks(x_test[i*dataset_interval:(i+1)*dataset_interval],(window,2,x_test[0,0].shape[0]))
                    x_te = np.squeeze(x_te)
                    X_test.append(np.transpose(x_te, (0,2,1,3)))
                
                for i in range(int(len(y_train)/dataset_interval)):
                    y_tr = view_as_windows(y_train[i*dataset_interval:(i+1)*dataset_interval],(window,1,y_data[0,0].shape[0], y_data[0,0].shape[1]),step=(step,1,1,1))
                    y_tr = np.squeeze(y_tr)
                    Y_train.append(y_tr.reshape(-1,1,window,y_data[0,0].shape[0],y_data[0,0].shape[1]))
                
                for i in range(int(len(y_val)/dataset_interval)):
                    y_va = view_as_windows(y_val[i*dataset_interval:(i+1)*dataset_interval],(window,1,y_data[0,0].shape[0], y_data[0,0].shape[1]),step=(step,1,1,1))
                    y_va = np.squeeze(y_va)
                    Y_val.append(y_va.reshape(-1,1,window,y_data[0,0].shape[0],y_data[0,0].shape[1]))
                    
                for i in range(int(len(y_test)/dataset_interval)):
                    y_te = view_as_blocks(y_test[i*dataset_interval:(i+1)*dataset_interval],(window,1,y_data[0,0].shape[0], y_data[0,0].shape[1]))
                    y_te = np.squeeze(y_te)
                    Y_test.append(y_te.reshape(-1,1,window,y_data[0,0].shape[0],y_data[0,0].shape[1]))
                x_train = np.asarray(X_train)
                x_train = x_train.reshape(-1,x_train.shape[2],x_train.shape[3],x_train.shape[4])
                x_val = np.asarray(X_val)
                x_val = x_val.reshape(-1,x_val.shape[2],x_val.shape[3],x_val.shape[4])
                x_test = np.asarray(X_test)
                x_test = x_test.reshape(-1,x_test.shape[2],x_test.shape[3],x_test.shape[4])
                y_train = np.asarray(Y_train)
                y_train = y_train.reshape(-1,y_train.shape[2],y_train.shape[3],y_train.shape[4],y_train.shape[5])
                y_val = np.asarray(Y_val)
                y_val = y_val.reshape(-1,y_val.shape[2],y_val.shape[3],y_val.shape[4],y_val.shape[5])
                y_test = np.asarray(Y_test)
                y_test = y_test.reshape(-1,y_test.shape[2],y_test.shape[3],y_test.shape[4],y_test.shape[5])
        
        
        
        return x_train,x_val,x_test,y_train,y_val,y_test
    def ardx3_data_prep(self,train,val,test):
        crd_max_0 = np.max(np.abs(self.crd[:,0]))
        crd_max_1 = np.max(np.abs(self.crd[:,1]))
        crd_max_2 = np.max(np.abs(self.crd[:,2]))
        self.crd[:,0] = self.crd[:,0]/crd_max_0
        self.crd[:,1] = self.crd[:,1]/crd_max_1
        self.crd[:,2] = self.crd[:,2]/crd_max_2
        
        x_data = np.abs(self.crd)
        y_data = self.images.reshape(-1, 1, self.images[0].shape[0], self.images[0].shape[1])
        
        x_train  = x_data[:train]
        x_val = x_data[train:val]
        x_test = x_data[val:test]
        
        y_train = y_data[:train]
        y_val = y_data[train:val]
        y_test = y_data[val:test]
        
        return x_train,x_val,x_test,y_train,y_val,y_test
    
    def ard_aoa_aoa_data_prep(self,train_start,train,val,test,thresh=1,windowed=False,window=15,dataset_interval=0,step=1):
        """Test set = view_as_blocks"""
        x_data = np.zeros((len(self.crd),3,self.crd[0,0].shape[0],self.crd[0,0].shape[1]))
        x_data[:,0,:,:] = np.mean(np.abs(self.crd[:,:,:,:]),axis=1) #mean of ard
        x_data[:,1,:,:] = self.aoa[:,0,:,:]
        x_data[:,2,:,:] = self.aoa[:,1,:,:]
        #Manually set closest 3 bins to 0, for some reason they're populated by noise and ruins thresholding
        x_data[:,:,0:3,:] = 0
        
        #Threshold
        for i in range(len(x_data)):
            x_data[i,0][x_data[i,0]<np.max(x_data[i,0]*thresh)] = 0
            x_data[i,1][x_data[i,0]<np.max(x_data[i,0]*thresh)] = 0
            x_data[i,2][x_data[i,0]<np.max(x_data[i,0]*thresh)] = 0
            
            x_data[i,0] =  x_data[i,0]/np.max(x_data[i,0])
            #x_data[i,1] =  x_data[i,1]/np.max(x_data[i,1])# angle bound between -pi/2 and pi/2, don't normalise
            #x_data[i,2] =  x_data[i,2]/np.max(x_data[i,2])# angle bound between -pi/2 and pi/2, don't normalise
        

            
            
        y_data = self.images.reshape(-1, 1, self.images[0].shape[0], self.images[0].shape[1])
        
        #Train val test split
        x_train  = x_data[train_start:train]
        x_val = x_data[train:val]
        x_test = x_data[val:test]
        
        y_train = y_data[train_start:train]
        y_val = y_data[train:val]
        y_test = y_data[val:test]
        
        if windowed==True:
            
            if dataset_interval==0:
                #Organise into blocks/windows for video processing
                
                x_train = view_as_windows(x_train,(window,3,x_data[0,0].shape[0],x_data[0,0].shape[1]),step=(step,1,1,1))
                x_train = np.squeeze(x_train)
                x_train = np.transpose(x_train, (0,2,1,3,4))
                
                x_val = view_as_windows(x_val,(window,3,x_data[0,0].shape[0],x_data[0,0].shape[1]),step=(step,1,1,1))
                x_val = np.squeeze(x_val)
                x_val = np.transpose(x_val, (0,2,1,3,4))
                
                x_test = view_as_blocks(x_test,(window,3,x_data[0,0].shape[0],x_data[0,0].shape[1]))
                x_test = np.squeeze(x_test)
                x_test = np.transpose(x_test, (0,2,1,3,4))
                
                
                y_train = view_as_windows(y_train,(window,1,y_data[0,0].shape[0], y_data[0,0].shape[1]),step=(step,1,1,1))
                y_train = np.squeeze(y_train)
                y_train = y_train.reshape(-1,1,window,y_data[0,0].shape[0],y_data[0,0].shape[1])
                
                y_val = view_as_windows(y_val,(window,1,y_data[0,0].shape[0], y_data[0,0].shape[1]),step=(step,1,1,1))
                y_val = np.squeeze(y_val)
                y_val = y_val.reshape(-1,1,window,y_data[0,0].shape[0],y_data[0,0].shape[1])
                
                y_test = view_as_blocks(y_test,(window,1,y_data[0,0].shape[0], y_data[0,0].shape[1]))
                y_test = np.squeeze(y_test)
                y_test = y_test.reshape(-1,1,window,y_data[0,0].shape[0],y_data[0,0].shape[1])
            else:
                X_train = []
                X_val = []
                X_test = []
                
                Y_train = []
                Y_val= []
                Y_test = []
                
                for i in range(int(len(x_train)/dataset_interval)):
                    x_tr = view_as_windows(x_train[i*dataset_interval:(i+1)*dataset_interval],(window,3,x_data[0,0].shape[0],x_data[0,0].shape[1]),step=(step,1,1,1))
                    x_tr = np.squeeze(x_tr)
                    X_train.append(np.transpose(x_tr, (0,2,1,3,4)))
                
                for i in range(int(len(x_val)/dataset_interval)):
                    x_va = view_as_windows(x_val[i*dataset_interval:(i+1)*dataset_interval],(window,3,x_data[0,0].shape[0],x_data[0,0].shape[1]),step=(step,1,1,1))
                    x_va = np.squeeze(x_va)
                    X_val.append(np.transpose(x_va, (0,2,1,3,4)))
                
                for i in range(int(len(x_test)/dataset_interval)):
                    x_te = view_as_blocks(x_test[i*dataset_interval:(i+1)*dataset_interval],(window,3,x_data[0,0].shape[0],x_data[0,0].shape[1]))
                    x_te = np.squeeze(x_te)
                    X_test.append(np.transpose(x_te, (0,2,1,3,4)))
                
                for i in range(int(len(y_train)/dataset_interval)):
                    y_tr = view_as_windows(y_train[i*dataset_interval:(i+1)*dataset_interval],(window,1,y_data[0,0].shape[0], y_data[0,0].shape[1]),step=(step,1,1,1))
                    y_tr = np.squeeze(y_tr)
                    Y_train.append(y_tr.reshape(-1,1,window,y_data[0,0].shape[0],y_data[0,0].shape[1]))
                
                for i in range(int(len(y_val)/dataset_interval)):
                    y_va = view_as_windows(y_val[i*dataset_interval:(i+1)*dataset_interval],(window,1,y_data[0,0].shape[0], y_data[0,0].shape[1]),step=(step,1,1,1))
                    y_va = np.squeeze(y_va)
                    Y_val.append(y_va.reshape(-1,1,window,y_data[0,0].shape[0],y_data[0,0].shape[1]))
                    
                for i in range(int(len(y_test)/dataset_interval)):
                    y_te = view_as_blocks(y_test[i*dataset_interval:(i+1)*dataset_interval],(window,1,y_data[0,0].shape[0], y_data[0,0].shape[1]))
                    y_te = np.squeeze(y_te)
                    Y_test.append(y_te.reshape(-1,1,window,y_data[0,0].shape[0],y_data[0,0].shape[1]))
                x_train = np.asarray(X_train)
                x_train = x_train.reshape(-1,x_train.shape[2],x_train.shape[3],x_train.shape[4],x_train.shape[5])
                x_val = np.asarray(X_val)
                x_val = x_val.reshape(-1,x_val.shape[2],x_val.shape[3],x_val.shape[4],x_val.shape[5])
                x_test = np.asarray(X_test)
                x_test = x_test.reshape(-1,x_test.shape[2],x_test.shape[3],x_test.shape[4],x_test.shape[5])
                y_train = np.asarray(Y_train)
                y_train = y_train.reshape(-1,y_train.shape[2],y_train.shape[3],y_train.shape[4],y_train.shape[5])
                y_val = np.asarray(Y_val)
                y_val = y_val.reshape(-1,y_val.shape[2],y_val.shape[3],y_val.shape[4],y_val.shape[5])
                y_test = np.asarray(Y_test)
                y_test = y_test.reshape(-1,y_test.shape[2],y_test.shape[3],y_test.shape[4],y_test.shape[5])
        
        #This change is implemented from 20210528 onwards
        x_train = x_train.transpose(0,2,1,3,4)
        x_val = x_val.transpose(0,2,1,3,4)
        x_test = x_test.transpose(0,2,1,3,4)
        y_train = y_train.transpose(0,2,1,3,4)
        y_val = y_val.transpose(0,2,1,3,4)
        y_test = y_test.transpose(0,2,1,3,4)
        return x_train,x_val,x_test,y_train,y_val,y_test
        
    def ard_aoa_aoa_data_prep_2(self,train_start,train,val,test,thresh=1,windowed=False,window=15,dataset_interval=0,step=1):
        """Test set = view_as_windows"""
        x_data = np.zeros((len(self.crd),3,self.crd[0,0].shape[0],self.crd[0,0].shape[1]))
        x_data[:,0,:,:] = np.mean(np.abs(self.crd[:,:,:,:]),axis=1) #mean of ard
        x_data[:,1,:,:] = self.aoa[:,0,:,:]
        x_data[:,2,:,:] = self.aoa[:,1,:,:]
        #Manually set closest 3 bins to 0, for some reason they're populated by noise and ruins thresholding
        x_data[:,:,0:3,:] = 0
        
        #Threshold
        for i in range(len(x_data)):
            x_data[i,0][x_data[i,0]<np.max(x_data[i,0]*thresh)] = 0
            x_data[i,1][x_data[i,0]<np.max(x_data[i,0]*thresh)] = 0
            x_data[i,2][x_data[i,0]<np.max(x_data[i,0]*thresh)] = 0
            
            x_data[i,0] =  x_data[i,0]/np.max(x_data[i,0])
            #x_data[i,1] =  x_data[i,1]/np.max(x_data[i,1])# angle bound between -pi/2 and pi/2, don't normalise
            #x_data[i,2] =  x_data[i,2]/np.max(x_data[i,2])# angle bound between -pi/2 and pi/2, don't normalise
        

            
            
        y_data = self.images.reshape(-1, 1, self.images[0].shape[0], self.images[0].shape[1])
        
        #Train val test split
        x_train  = x_data[train_start:train]
        x_val = x_data[train:val]
        x_test = x_data[val:test]
        
        y_train = y_data[train_start:train]
        y_val = y_data[train:val]
        y_test = y_data[val:test]
        
        if windowed==True:
            
            if dataset_interval==0:
                #Organise into blocks/windows for video processing
                
                x_train = view_as_windows(x_train,(window,3,x_data[0,0].shape[0],x_data[0,0].shape[1]),step=(step,1,1,1))
                x_train = np.squeeze(x_train)
                x_train = np.transpose(x_train, (0,2,1,3,4))
                
                x_val = view_as_windows(x_val,(window,3,x_data[0,0].shape[0],x_data[0,0].shape[1]),step=(step,1,1,1))
                x_val = np.squeeze(x_val)
                x_val = np.transpose(x_val, (0,2,1,3,4))
                
                x_test = view_as_windows(x_test,(window,3,x_data[0,0].shape[0],x_data[0,0].shape[1]),step=(step,1,1,1))
                x_test = np.squeeze(x_test)
                x_test = np.transpose(x_test, (0,2,1,3,4))
                
                
                y_train = view_as_windows(y_train,(window,1,y_data[0,0].shape[0], y_data[0,0].shape[1]),step=(step,1,1,1))
                y_train = np.squeeze(y_train)
                y_train = y_train.reshape(-1,1,window,y_data[0,0].shape[0],y_data[0,0].shape[1])
                
                y_val = view_as_windows(y_val,(window,1,y_data[0,0].shape[0], y_data[0,0].shape[1]),step=(step,1,1,1))
                y_val = np.squeeze(y_val)
                y_val = y_val.reshape(-1,1,window,y_data[0,0].shape[0],y_data[0,0].shape[1])
                
                y_test = view_as_windows(y_test,(window,1,y_data[0,0].shape[0], y_data[0,0].shape[1]),step=(step,1,1,1))
                y_test = np.squeeze(y_test)
                y_test = y_test.reshape(-1,1,window,y_data[0,0].shape[0],y_data[0,0].shape[1])
            else:
                X_train = []
                X_val = []
                X_test = []
                
                Y_train = []
                Y_val= []
                Y_test = []
                
                for i in range(int(len(x_train)/dataset_interval)):
                    x_tr = view_as_windows(x_train[i*dataset_interval:(i+1)*dataset_interval],(window,3,x_data[0,0].shape[0],x_data[0,0].shape[1]),step=(step,1,1,1))
                    x_tr = np.squeeze(x_tr)
                    X_train.append(np.transpose(x_tr, (0,2,1,3,4)))
                
                for i in range(int(len(x_val)/dataset_interval)):
                    x_va = view_as_windows(x_val[i*dataset_interval:(i+1)*dataset_interval],(window,3,x_data[0,0].shape[0],x_data[0,0].shape[1]),step=(step,1,1,1))
                    x_va = np.squeeze(x_va)
                    X_val.append(np.transpose(x_va, (0,2,1,3,4)))
                
                for i in range(int(len(x_test)/dataset_interval)):
                    x_te = view_as_windows(x_test[i*dataset_interval:(i+1)*dataset_interval],(window,3,x_data[0,0].shape[0],x_data[0,0].shape[1]),step=(step,1,1,1))
                    x_te = np.squeeze(x_te)
                    X_test.append(np.transpose(x_te, (0,2,1,3,4)))
                
                for i in range(int(len(y_train)/dataset_interval)):
                    y_tr = view_as_windows(y_train[i*dataset_interval:(i+1)*dataset_interval],(window,1,y_data[0,0].shape[0], y_data[0,0].shape[1]),step=(step,1,1,1))
                    y_tr = np.squeeze(y_tr)
                    Y_train.append(y_tr.reshape(-1,1,window,y_data[0,0].shape[0],y_data[0,0].shape[1]))
                
                for i in range(int(len(y_val)/dataset_interval)):
                    y_va = view_as_windows(y_val[i*dataset_interval:(i+1)*dataset_interval],(window,1,y_data[0,0].shape[0], y_data[0,0].shape[1]),step=(step,1,1,1))
                    y_va = np.squeeze(y_va)
                    Y_val.append(y_va.reshape(-1,1,window,y_data[0,0].shape[0],y_data[0,0].shape[1]))
                    
                for i in range(int(len(y_test)/dataset_interval)):
                    y_te = view_as_windows(y_test[i*dataset_interval:(i+1)*dataset_interval],(window,1,y_data[0,0].shape[0], y_data[0,0].shape[1]),step=(step,1,1,1))
                    y_te = np.squeeze(y_te)
                    Y_test.append(y_te.reshape(-1,1,window,y_data[0,0].shape[0],y_data[0,0].shape[1]))
                x_train = np.asarray(X_train)
                x_train = x_train.reshape(-1,x_train.shape[2],x_train.shape[3],x_train.shape[4],x_train.shape[5])
                x_val = np.asarray(X_val)
                x_val = x_val.reshape(-1,x_val.shape[2],x_val.shape[3],x_val.shape[4],x_val.shape[5])
                x_test = np.asarray(X_test)
                x_test = x_test.reshape(-1,x_test.shape[2],x_test.shape[3],x_test.shape[4],x_test.shape[5])
                y_train = np.asarray(Y_train)
                y_train = y_train.reshape(-1,y_train.shape[2],y_train.shape[3],y_train.shape[4],y_train.shape[5])
                y_val = np.asarray(Y_val)
                y_val = y_val.reshape(-1,y_val.shape[2],y_val.shape[3],y_val.shape[4],y_val.shape[5])
                y_test = np.asarray(Y_test)
                y_test = y_test.reshape(-1,y_test.shape[2],y_test.shape[3],y_test.shape[4],y_test.shape[5])
        
        #This change is implemented from 20210528 onwards
        x_train = x_train.transpose(0,2,1,3,4)
        x_val = x_val.transpose(0,2,1,3,4)
        x_test = x_test.transpose(0,2,1,3,4)
        y_train = y_train.transpose(0,2,1,3,4)
        y_val = y_val.transpose(0,2,1,3,4)
        y_test = y_test.transpose(0,2,1,3,4)
        return x_train,x_val,x_test,y_train,y_val,y_test
    
    def ard_aoa_data_prep_p2go(self,train_start,train,val,test,thresh=1,windowed=False,window=15,dataset_interval=0,step=1):
        """p2go data. Test set = view_as_blocks"""
        x_data = np.zeros((len(self.crd),2,self.crd[0,0].shape[0],self.crd[0,0].shape[1]))
        x_data[:,0,:,:] = np.mean(np.abs(self.crd[:,:,:,:]),axis=1) #mean of ard
        x_data[:,1,:,:] = self.aoa[:,0,:,:]
        
        #Threshold
        for i in range(len(x_data)):
            x_data[i,0][x_data[i,0]<np.max(x_data[i,0]*thresh)] = 0
            x_data[i,1][x_data[i,0]<np.max(x_data[i,0]*thresh)] = 0
            
            x_data[i,0] =  x_data[i,0]/(max(np.max(x_data[i,0]),1e-10))
            #x_data[i,1] =  x_data[i,1]/np.max(x_data[i,1])# angle bound between -pi/2 and pi/2, don't normalise
            #x_data[i,2] =  x_data[i,2]/np.max(x_data[i,2])# angle bound between -pi/2 and pi/2, don't normalise
        

            
            
        y_data = self.images.reshape(-1, 1, self.images[0].shape[0], self.images[0].shape[1])
        
        #Train val test split
        x_train  = x_data[train_start:train]
        x_val = x_data[train:val]
        x_test = x_data[val:test]
        
        y_train = y_data[train_start:train]
        y_val = y_data[train:val]
        y_test = y_data[val:test]
        
        if windowed==True:
            
            if dataset_interval==0:
                #Organise into blocks/windows for video processing
                x_train = view_as_windows(x_train,(window,2,x_data[0,0].shape[0],x_data[0,0].shape[1]),step=(step,1,1,1))
                x_train = np.squeeze(x_train)
                x_train = np.transpose(x_train, (0,2,1,3,4))
                
                x_val = view_as_windows(x_val,(window,2,x_data[0,0].shape[0],x_data[0,0].shape[1]),step=(step,1,1,1))
                x_val = np.squeeze(x_val)
                x_val = np.transpose(x_val, (0,2,1,3,4))
                
                x_test = view_as_blocks(x_test,(window,2,x_data[0,0].shape[0],x_data[0,0].shape[1]))
                x_test = np.squeeze(x_test)
                x_test = np.transpose(x_test, (0,2,1,3,4))
                
                
                y_train = view_as_windows(y_train,(window,1,y_data[0,0].shape[0], y_data[0,0].shape[1]),step=(step,1,1,1))
                y_train = np.squeeze(y_train)
                y_train = y_train.reshape(-1,1,window,y_data[0,0].shape[0],y_data[0,0].shape[1])
                
                y_val = view_as_windows(y_val,(window,1,y_data[0,0].shape[0], y_data[0,0].shape[1]),step=(step,1,1,1))
                y_val = np.squeeze(y_val)
                y_val = y_val.reshape(-1,1,window,y_data[0,0].shape[0],y_data[0,0].shape[1])
                
                y_test = view_as_blocks(y_test,(window,1,y_data[0,0].shape[0], y_data[0,0].shape[1]))
                y_test = np.squeeze(y_test)
                y_test = y_test.reshape(-1,1,window,y_data[0,0].shape[0],y_data[0,0].shape[1])
            else:
                X_train = []
                X_val = []
                X_test = []
                
                Y_train = []
                Y_val= []
                Y_test = []
                
                for i in range(int(len(x_train)/dataset_interval)):
                    x_tr = view_as_windows(x_train[i*dataset_interval:(i+1)*dataset_interval],(window,2,x_data[0,0].shape[0],x_data[0,0].shape[1]),step=(step,1,1,1))
                    x_tr = np.squeeze(x_tr)
                    X_train.append(np.transpose(x_tr, (0,2,1,3,4)))
                
                for i in range(int(len(x_val)/dataset_interval)):
                    x_va = view_as_windows(x_val[i*dataset_interval:(i+1)*dataset_interval],(window,2,x_data[0,0].shape[0],x_data[0,0].shape[1]),step=(step,1,1,1))
                    x_va = np.squeeze(x_va)
                    X_val.append(np.transpose(x_va, (0,2,1,3,4)))
                
                for i in range(int(len(x_test)/dataset_interval)):
                    x_te = view_as_blocks(x_test[i*dataset_interval:(i+1)*dataset_interval],(window,2,x_data[0,0].shape[0],x_data[0,0].shape[1]))
                    x_te = np.squeeze(x_te)
                    X_test.append(np.transpose(x_te, (0,2,1,3,4)))
                
                for i in range(int(len(y_train)/dataset_interval)):
                    y_tr = view_as_windows(y_train[i*dataset_interval:(i+1)*dataset_interval],(window,1,y_data[0,0].shape[0], y_data[0,0].shape[1]),step=(step,1,1,1))
                    y_tr = np.squeeze(y_tr)
                    Y_train.append(y_tr.reshape(-1,1,window,y_data[0,0].shape[0],y_data[0,0].shape[1]))
                
                for i in range(int(len(y_val)/dataset_interval)):
                    y_va = view_as_windows(y_val[i*dataset_interval:(i+1)*dataset_interval],(window,1,y_data[0,0].shape[0], y_data[0,0].shape[1]),step=(step,1,1,1))
                    y_va = np.squeeze(y_va)
                    Y_val.append(y_va.reshape(-1,1,window,y_data[0,0].shape[0],y_data[0,0].shape[1]))
                    
                for i in range(int(len(y_test)/dataset_interval)):
                    y_te = view_as_blocks(y_test[i*dataset_interval:(i+1)*dataset_interval],(window,1,y_data[0,0].shape[0], y_data[0,0].shape[1]))
                    y_te = np.squeeze(y_te)
                    Y_test.append(y_te.reshape(-1,1,window,y_data[0,0].shape[0],y_data[0,0].shape[1]))
                x_train = np.asarray(X_train)
                x_train = x_train.reshape(-1,x_train.shape[2],x_train.shape[3],x_train.shape[4],x_train.shape[5])
                x_val = np.asarray(X_val)
                x_val = x_val.reshape(-1,x_val.shape[2],x_val.shape[3],x_val.shape[4],x_val.shape[5])
                x_test = np.asarray(X_test)
                x_test = x_test.reshape(-1,x_test.shape[2],x_test.shape[3],x_test.shape[4],x_test.shape[5])
                y_train = np.asarray(Y_train)
                y_train = y_train.reshape(-1,y_train.shape[2],y_train.shape[3],y_train.shape[4],y_train.shape[5])
                y_val = np.asarray(Y_val)
                y_val = y_val.reshape(-1,y_val.shape[2],y_val.shape[3],y_val.shape[4],y_val.shape[5])
                y_test = np.asarray(Y_test)
                y_test = y_test.reshape(-1,y_test.shape[2],y_test.shape[3],y_test.shape[4],y_test.shape[5])
        
        #This change is implemented from 20210528 onwards
        x_train = x_train.transpose(0,2,1,3,4)
        x_val = x_val.transpose(0,2,1,3,4)
        x_test = x_test.transpose(0,2,1,3,4)
        y_train = y_train.transpose(0,2,1,3,4)
        y_val = y_val.transpose(0,2,1,3,4)
        y_test = y_test.transpose(0,2,1,3,4)
        return x_train,x_val,x_test,y_train,y_val,y_test









#%% architerctures
def cnn_v0(input_shape):
    """ Assumes a 3D input (batch,1,range), where 1 is for channel 2."""
    kse = 5
    ksd = 5
    feats = 8
    kern_reg = None
    kern_int_e = 'glorot_uniform'
    kern_int_d = 'glorot_uniform'
    
    # 54 = 2 x 3 x 3 x 3 = 6 x 3 x 3
    # 96 = 2 x 2 x 2 x 2 x 2 x 3 = 16 x 2 x 3
    
    reset_keras()
    
    inp=Input(shape=(input_shape))
    
    ###################################################################################################################
            # Encoding
    ###################################################################################################################
    conv1 = Conv1D(128, kse, padding='same', kernel_regularizer=kern_reg, kernel_initializer=kern_int_e, data_format = 'channels_first')(inp)
    conv1 = LeakyReLU()(conv1)

    conv2 = Conv1D(128, kse, padding='same', kernel_regularizer=kern_reg, kernel_initializer=kern_int_e, data_format = 'channels_first')(
        conv1)
    conv2 = LeakyReLU()(conv2)
    
    conv3 = Conv1D(128, kse, padding='same', kernel_regularizer=kern_reg, kernel_initializer=kern_int_e, data_format = 'channels_first')(
        conv2)
    conv3 = LeakyReLU()(conv3)
    
    conv4 = Conv1D(258, kse, padding='same', kernel_regularizer=kern_reg, kernel_initializer=kern_int_e, data_format = 'channels_first')(
        conv3)
    conv4 = LeakyReLU()(conv4)    
    #batchnorm_1 = BatchNormalization()(conv4)
    
    ###################################################################################################################
        # Latent
    ###################################################################################################################
    
    reshaped = Reshape((172, 6, 16))(conv4)
    
    conv5 = Conv2D(172, ksd, kernel_initializer=kern_int_e, padding='same', data_format='channels_first')(reshaped)    
    conv5 = LeakyReLU()(conv5)
    #batchnorm_2 = BatchNormalization()(conv5)
        
    ###################################################################################################################
    #         Decoding
    ###################################################################################################################
    
    tconv1 = UpSampling2D(size=(3, 2), data_format='channels_first',interpolation='bilinear')(conv5)    
    tconv2 = Conv2D(128, ksd,  padding='same', kernel_regularizer=kern_reg, kernel_initializer=kern_int_d, data_format='channels_first')(tconv1)
    tconv2 = LeakyReLU()(tconv2)
    tconv2 = UpSampling2D(size=(3, 3), data_format='channels_first',interpolation='bilinear')(tconv2)    
    tconv3 = Conv2D(128, ksd,  padding='same', kernel_regularizer=kern_reg, kernel_initializer=kern_int_d, data_format='channels_first')(tconv2)
    tconv3 = LeakyReLU()(tconv3)
    out = Conv2D(1, 1, padding='same', activation='sigmoid', kernel_regularizer=kern_reg, kernel_initializer=kern_int_d, data_format='channels_first')(tconv3)
    
    model = tf.keras.Model(inputs=inp, outputs=out)
    
    return model

def cnn_v1(input_shape):
    """ Assumes a 3D input (batch,channel,iq), is trained on the 3 channels' IQ streams (only I is given actually)"""
    kse = 5
    ksd = 5
    feats = 1
    kern_reg = None
    kern_int_e = 'glorot_uniform'
    kern_int_d = 'glorot_uniform'
    
    # 54 = 2 x 3 x 3 x 3 = 6 x 3 x 3
    # 96 = 2 x 2 x 2 x 2 x 2 x 3 = 16 x 2 x 3
    
    reset_keras()
    
    inp=Input(shape=(input_shape))
    
    ###################################################################################################################
            # Encoding
    ###################################################################################################################
    conv1 = Conv1D(feats*128, kse, padding='same', kernel_regularizer=kern_reg, kernel_initializer=kern_int_e, data_format = 'channels_first')(inp)
    conv1 = LeakyReLU()(conv1)

    conv2 = Conv1D(feats*128, kse, padding='same', kernel_regularizer=kern_reg, kernel_initializer=kern_int_e, data_format = 'channels_first')(
        conv1)
    conv2 = LeakyReLU()(conv2)
    
    conv3 = Conv1D(feats*128, kse, padding='same', kernel_regularizer=kern_reg, kernel_initializer=kern_int_e, data_format = 'channels_first')(
        conv2)
    conv3 = LeakyReLU()(conv3)
    
    conv4 = Conv1D(feats*129, kse, padding='same', kernel_regularizer=kern_reg, kernel_initializer=kern_int_e, data_format = 'channels_first')(
        conv3)
    conv4 = LeakyReLU()(conv4)    
    #batchnorm_1 = BatchNormalization()(conv4)
    
    ###################################################################################################################
        # Latent
    ###################################################################################################################
    
    reshaped = Reshape((feats*172, 6, 16))(conv4)
    
    conv5 = Conv2D(feats*172, ksd, kernel_initializer=kern_int_e, padding='same', data_format='channels_first')(reshaped)    
    conv5 = LeakyReLU()(conv5)
    #batchnorm_2 = BatchNormalization()(conv5)
        
    ###################################################################################################################
    #         Decoding
    ###################################################################################################################
    
    tconv1 = UpSampling2D(size=(3, 2), data_format='channels_first',interpolation='bilinear')(conv5)    
    tconv2 = Conv2D(feats*128, ksd,  padding='same', kernel_regularizer=kern_reg, kernel_initializer=kern_int_d, data_format='channels_first')(tconv1)
    tconv2 = LeakyReLU()(tconv2)
    tconv2 = UpSampling2D(size=(3, 3), data_format='channels_first',interpolation='bilinear')(tconv2)    
    tconv3 = Conv2D(feats*128, ksd,  padding='same', kernel_regularizer=kern_reg, kernel_initializer=kern_int_d, data_format='channels_first')(tconv2)
    tconv3 = LeakyReLU()(tconv3)
 #   reshaped = tf.transpose(tconv3,perm=(0,2,3,1))
    out = Conv2D(1, 1, padding='same', activation='sigmoid', kernel_regularizer=kern_reg, kernel_initializer=kern_int_d, data_format='channels_first')(tconv3)
    
    model = tf.keras.Model(inputs=inp, outputs=out)
    
    return model

def cnn_v3(input_shape):
    """Previously cnn_v1 - old model"""
    kse = 8
    ksd = 8
    feats = 8
    kern_reg = None
    kern_int_e = 'glorot_uniform'
    kern_int_d = 'glorot_uniform'
    
    reset_keras()
    
    inp=Input(shape=input_shape)
    
    ###################################################################################################################
            # Encoding
    ###################################################################################################################
    conv1 = Conv2D(feats * 8, kse, activation='sigmoid', padding='same', kernel_regularizer=kern_reg, kernel_initializer=kern_int_e, data_format = 'channels_first')(inp)
    
    conv2 = Conv2D(feats * 8, kse, activation='relu', padding='same', kernel_regularizer=kern_reg, kernel_initializer=kern_int_e, data_format = 'channels_first', strides=2)(
        conv1)
    
    conv3 = Conv2D(feats * 4, kse, activation='relu', padding='same', kernel_regularizer=kern_reg, kernel_initializer=kern_int_e, data_format = 'channels_first', strides=2)(
        conv2)
    
    conv4 = Conv2D(feats * 2, kse, activation='relu', padding='same', kernel_regularizer=kern_reg, kernel_initializer=kern_int_e, data_format = 'channels_first', strides=2)(
        conv3)
    
    
    ###################################################################################################################
        # Latent
    ###################################################################################################################
    
    flattened = Flatten()(conv3)
    
    dense1 = Dense(16*6*16, activation='relu', kernel_initializer=kern_int_e)(flattened)
    
    reshaped = Reshape((16, 6, 16))(dense1)
    
    ###################################################################################################################
    #         Decoding
    ###################################################################################################################
    
    tconv1 = UpSampling2D(size=(3, 2), data_format='channels_first')(reshaped)    
    tconv2 = Conv2D(feats * 2, ksd, activation='relu', padding='same', kernel_regularizer=kern_reg, kernel_initializer=kern_int_d, data_format='channels_first')(tconv1)
    
    tconv2 = UpSampling2D(size=(3, 3), data_format='channels_first')(tconv2)    
    tconv3 = Conv2D(feats * 2, ksd, activation='relu', padding='same', kernel_regularizer=kern_reg, kernel_initializer=kern_int_d, data_format='channels_first')(tconv2)
    
    out = Conv2D(1, 1, padding='same', activation='relu', kernel_regularizer=kern_reg, kernel_initializer=kern_int_d, data_format='channels_first')(tconv3)
    
    model = tf.keras.Model(inputs=inp, outputs=out)
    

    return model

def cnn_v2(input_shape):
    """Old model"""
    kse = (9,5)
    ksd = (9,5)
    kern_reg = None
    kern_int_e = 'glorot_uniform'
    kern_int_d = 'glorot_uniform'
    
    reset_keras()
    
    inp=Input(shape=input_shape)
    
    ###################################################################################################################
            # Encoding
    ###################################################################################################################
    conv1 = Conv2D(64, kse, activation='sigmoid', padding='same', kernel_regularizer=kern_reg, kernel_initializer=kern_int_e, data_format = 'channels_first')(inp)
    
    conv2 = Conv2D(64, kse, activation='relu', padding='same', kernel_regularizer=kern_reg, kernel_initializer=kern_int_e, data_format = 'channels_first', strides=2)(
        conv1)
    
    conv3 = Conv2D(128 , kse, activation='relu', padding='same', kernel_regularizer=kern_reg, kernel_initializer=kern_int_e, data_format = 'channels_first', strides=2)(
        conv2)
       
    ###################################################################################################################
        # Latent
    ###################################################################################################################
    conv4 = Conv2D(420, kse, activation='relu', padding='same', kernel_regularizer=kern_reg, kernel_initializer=kern_int_e, data_format = 'channels_first', strides=2)(
        conv3)    
    
    reshaped = Reshape((70, 6, 16))(conv4)
    
    tconv = Conv2D(64, ksd, activation='relu', padding='same', kernel_regularizer=kern_reg, kernel_initializer=kern_int_e, data_format = 'channels_first')(reshaped)
       
    ###################################################################################################################
    #         Decoding
    ###################################################################################################################
    
    tconv1 = UpSampling2D(size=(3, 2), data_format='channels_first')(tconv)    
    tconv2 = Conv2D(32, ksd, activation='relu', padding='same', kernel_regularizer=kern_reg, kernel_initializer=kern_int_d, data_format='channels_first')(tconv1)
    
    tconv2 = UpSampling2D(size=(3, 3), data_format='channels_first')(tconv2)    
    tconv3 = Conv2D(16, ksd, activation='relu', padding='same', kernel_regularizer=kern_reg, kernel_initializer=kern_int_d, data_format='channels_first')(tconv2)
    
    out = Conv2D(1, 1, padding='same', activation='relu', kernel_regularizer=kern_reg, kernel_initializer=kern_int_d, data_format='channels_first')(tconv3)
    
    model = tf.keras.Model(inputs=inp, outputs=out)
    

    return model

def cnn3d_v0(input_shape):
    """Old model"""
    kse = (3,8,8)
    ksd = (3,8,8)
    feats = 4
    kern_reg = None
    kern_int_e = 'glorot_uniform'
    kern_int_d = 'glorot_uniform'
    reset_keras()
    
    inp=Input(shape=input_shape)
    
    ###################################################################################################################
            # Encoding
    ###################################################################################################################
    conv1 = Conv3D(feats * 8, kse, padding='same', kernel_regularizer=kern_reg, kernel_initializer=kern_int_e, data_format = 'channels_first')(inp)
    conv1 = LeakyReLU()(conv1)
    conv2 = Conv3D(feats * 8, kse, padding='same', kernel_regularizer=kern_reg, kernel_initializer=kern_int_e, data_format = 'channels_first', strides=(1,2,2))(
        conv1)
    conv2 = LeakyReLU()(conv2)
    conv3 = Conv3D(feats * 4, kse, padding='same', kernel_regularizer=kern_reg, kernel_initializer=kern_int_e, data_format = 'channels_first', strides=(3,2,2))(
        conv2)
    conv3 = LeakyReLU()(conv3)
    conv4 = Conv3D(feats * 4, kse, padding='same', kernel_regularizer=kern_reg, kernel_initializer=kern_int_e, data_format = 'channels_first', strides=(3,2,2))(
        conv3)
    conv4 = LeakyReLU()(conv4)
    conv4 = BatchNormalization(axis=1)(conv4)
    
    ###################################################################################################################
        # Latent
    ###################################################################################################################
    
    flattened = Flatten()(conv4)
    
    dense1 = Dense(16*5*6*16, kernel_initializer=kern_int_e)(flattened)
    dense1 = LeakyReLU()(dense1)
    reshaped = Reshape((16, 5, 6, 16))(dense1)
    
    ###################################################################################################################
    #         Decoding
    ###################################################################################################################
    
    tconv1 = UpSampling3D(size=(3, 3, 2), data_format='channels_first')(reshaped)    
    tconv2 = Conv3D(feats * 2, ksd, padding='same', kernel_regularizer=kern_reg, kernel_initializer=kern_int_d, data_format='channels_first')(tconv1)
    tconv2 = BatchNormalization(axis=1)(tconv2)
    tconv2 = LeakyReLU()(tconv2)
    tconv2 = UpSampling3D(size=(1, 3, 3), data_format='channels_first')(tconv2)    
    tconv3 = Conv3D(feats * 2, ksd, padding='same', kernel_regularizer=kern_reg, kernel_initializer=kern_int_d, data_format='channels_first')(tconv2)
    tconv3 = LeakyReLU()(tconv3)
    out = Conv3D(1, 1, padding='same', activation='relu', kernel_regularizer=kern_reg, kernel_initializer=kern_int_d, data_format='channels_first')(tconv3)
    
    model = tf.keras.Model(inputs=inp, outputs=out)
    
   
    return model


def cnn3d_v1(input_shape):
    """Assumes a 5D input 'batch,channel,frame,row,column', matching stacked time frames of ARD-AOA-AOA measurements"""
    kse = (3,9,5)
    ksd = (3,9,5)
    feats = 64
    kern_reg = None
    kern_int_e = 'glorot_uniform'
    kern_int_d = 'glorot_uniform'
    reset_keras()
    
    inp=Input(shape=input_shape)
    
    ###################################################################################################################
            # Encoding
    ###################################################################################################################
    conv1 = Conv3D(feats, kse, padding='same', kernel_regularizer=kern_reg, kernel_initializer=kern_int_e, data_format = 'channels_first')(inp)
    conv1 = LeakyReLU()(conv1)
    conv2 = Conv3D(feats * 2, kse, padding='same', kernel_regularizer=kern_reg, kernel_initializer=kern_int_e, data_format = 'channels_first', strides=(1,2,2))(
        conv1)
    conv2 = LeakyReLU()(conv2)
    conv3 = Conv3D(feats * 4, kse, padding='same', kernel_regularizer=kern_reg, kernel_initializer=kern_int_e, data_format = 'channels_first', strides=(3,2,1))(
        conv2)
    conv3 = LeakyReLU()(conv3)
    conv4 = BatchNormalization(axis=1)(conv3)
    
    ###################################################################################################################
        # Latent
    ###################################################################################################################
    conv5 = Conv3D(feats * 4, (1,1,3), kernel_regularizer=kern_reg, kernel_initializer=kern_int_e, data_format = 'channels_first')(
    conv4)
    conv5 = LeakyReLU()(conv5)
    reshaped = Reshape((feats * 4, 5, 6, 16))(conv5)
    
    ###################################################################################################################
    #         Decoding
    ###################################################################################################################
    
    tconv1 = UpSampling3D(size=(3, 3, 2), data_format='channels_first')(reshaped)    
    tconv2 = Conv3D(feats * 2, ksd, padding='same', kernel_regularizer=kern_reg, kernel_initializer=kern_int_d, data_format='channels_first')(tconv1)
    tconv2 = LeakyReLU()(tconv2)
    tconv2 = UpSampling3D(size=(1, 3, 3), data_format='channels_first')(tconv2)    
    tconv3 = Conv3D(feats * 1, ksd, padding='same', kernel_regularizer=kern_reg, kernel_initializer=kern_int_d, data_format='channels_first')(tconv2)
    tconv3 = LeakyReLU()(tconv3)
    tconv4 = Conv3D(int(feats/2), ksd, padding='same', kernel_regularizer=kern_reg, kernel_initializer=kern_int_d, data_format='channels_first')(tconv3)
    tconv4 = LeakyReLU()(tconv4)
    out = Conv3D(1, 1, padding='same', activation='sigmoid', kernel_regularizer=kern_reg, kernel_initializer=kern_int_d, data_format='channels_first')(tconv4)
    
    model = tf.keras.Model(inputs=inp, outputs=out)
    
   
    return model


def cnn3d_v2(input_shape):
    kse = (3,9,3)
    ksd = (3,5,5)
    feats = 64
    kern_reg = None
    kern_int_e = 'glorot_uniform'
    kern_int_d = 'glorot_uniform'
    reset_keras()
    
    inp=Input(shape=input_shape)
    
    ###################################################################################################################
            # Encoding
    ###################################################################################################################
    conv1 = Conv3D(feats, kse, padding='same', kernel_regularizer=kern_reg, kernel_initializer=kern_int_e, data_format = 'channels_first')(inp)
    conv1 = LeakyReLU()(conv1)
    conv2 = Conv3D(feats * 2, kse, padding='same', kernel_regularizer=kern_reg, kernel_initializer=kern_int_e, data_format = 'channels_first', strides=(2,2,2))(
        conv1)
    conv2 = LeakyReLU()(conv2)
    conv4 = BatchNormalization(axis=1)(conv2)
    conv3 = Conv3D(feats * 4, kse, padding='same', kernel_regularizer=kern_reg, kernel_initializer=kern_int_e, data_format = 'channels_first', strides=(2,2,1))(
        conv2)
    conv3 = LeakyReLU()(conv3)
    conv3 = BatchNormalization(axis=1)(conv3)
    conv4 = Conv3D(feats * 4, kse, padding='same', kernel_regularizer=kern_reg, kernel_initializer=kern_int_e, data_format = 'channels_first', strides=(2,2,1))(
        conv4)
    conv4 = LeakyReLU()(conv4)
    conv4 = BatchNormalization(axis=1)(conv4)
    
    ###################################################################################################################
        # Latent
    ###################################################################################################################
    conv5 = Conv3D(feats * 4, (1,1,3), kernel_regularizer=kern_reg, kernel_initializer=kern_int_e, data_format = 'channels_first')(
    conv4)
    conv5 = LeakyReLU()(conv5)
    conv5 = BatchNormalization(axis=1)(conv5)
    reshaped = Reshape((feats * 4, 15, 6, 16))(conv5)
    
    ###################################################################################################################
    #         Decoding
    ###################################################################################################################
    
    tconv1 = UpSampling3D(size=(2, 3, 2), data_format='channels_first')(reshaped)    
    tconv2 = Conv3D(feats * 2, ksd, padding='same', kernel_regularizer=kern_reg, kernel_initializer=kern_int_d, data_format='channels_first')(tconv1)
    tconv2 = LeakyReLU()(tconv2)
    tconv2 = BatchNormalization(axis=1)(tconv2)
    tconv2 = UpSampling3D(size=(2, 3, 3), data_format='channels_first')(tconv2)    
    tconv3 = Conv3D(feats * 1, ksd, padding='same', kernel_regularizer=kern_reg, kernel_initializer=kern_int_d, data_format='channels_first')(tconv2)
    tconv3 = LeakyReLU()(tconv3)
    tconv3 = BatchNormalization(axis=1)(tconv3)
    tconv4 = Conv3D(int(feats/2), ksd, padding='same', kernel_regularizer=kern_reg, kernel_initializer=kern_int_d, data_format='channels_first')(tconv3)
    tconv4 = LeakyReLU()(tconv4)
    out = Conv3D(1, 1, padding='same', activation='relu', kernel_regularizer=kern_reg, kernel_initializer=kern_int_d, data_format='channels_first')(tconv4)
    
    model = tf.keras.Model(inputs=inp, outputs=out)
    
   
    return model

def cnn3d_v3(input_shape):
    """Assumes a 5D input 'batch,frame,channel,row,column', matching stacked time frames of ARD-AOA-AOA measurements"""
    kse = (3,9,5)
    ksd = (1,9,5)
    feats = 60
    kern_reg = None
    kern_int_e = 'glorot_uniform'
    kern_int_d = 'glorot_uniform'
    reset_keras()
    
    inp=Input(shape=input_shape)
    
    ###################################################################################################################
            # Encoding
    ###################################################################################################################
    conv1 = Conv3D(feats, kse, padding='same', kernel_regularizer=kern_reg, kernel_initializer=kern_int_e, data_format = 'channels_first')(inp)
    conv1 = LeakyReLU()(conv1)
    conv2 = Conv3D(feats * 2, kse, padding='same', kernel_regularizer=kern_reg, kernel_initializer=kern_int_e, data_format = 'channels_first', strides=(1,2,2))(
        conv1)
    conv2 = LeakyReLU()(conv2)
    conv3 = Conv3D(feats * 4, kse, padding='same', kernel_regularizer=kern_reg, kernel_initializer=kern_int_e, data_format = 'channels_first', strides=(1,2,1))(
        conv2)
    conv3 = LeakyReLU()(conv3)
    conv4 = BatchNormalization(axis=1)(conv3)
    
    ###################################################################################################################
        # Latent
    ###################################################################################################################
    conv5 = Conv3D(feats * 4, (3,1,3), kernel_regularizer=kern_reg, kernel_initializer=kern_int_e, data_format = 'channels_first')(
    conv4)
    conv5 = LeakyReLU()(conv5)
    reshaped = Reshape((feats*4, 1, 6, 16))(conv5)
    
    ###################################################################################################################
    #         Decoding
    ###################################################################################################################
    
    tconv1 = UpSampling3D(size=(1, 3, 2), data_format='channels_first')(reshaped)    
    tconv2 = Conv3D(feats * 2, ksd, padding='same', kernel_regularizer=kern_reg, kernel_initializer=kern_int_d, data_format='channels_first')(tconv1)
    tconv2 = LeakyReLU()(tconv2)
    tconv2 = UpSampling3D(size=(1, 3, 3), data_format='channels_first')(tconv2)    
    tconv3 = Conv3D(feats * 1, ksd, padding='same', kernel_regularizer=kern_reg, kernel_initializer=kern_int_d, data_format='channels_first')(tconv2)
    tconv3 = LeakyReLU()(tconv3)
    tconv4 = Conv3D(int(feats/2), ksd, padding='same', kernel_regularizer=kern_reg, kernel_initializer=kern_int_d, data_format='channels_first')(tconv3)
    tconv4 = LeakyReLU()(tconv4)
    out = Conv3D(15, 1, padding='same', activation='sigmoid', kernel_regularizer=kern_reg, kernel_initializer=kern_int_d, data_format='channels_first')(tconv4)
    
    model = tf.keras.Model(inputs=inp, outputs=out)
    
   
    return model




def cnn3d_v4(input_shape):
    """Assumes a 4D input 'batch,frame,channel,iq. Channel before frame"""
    kse = (3,9)
    ksd = (1,9,5)
    feats = 60
    kern_reg = None
    kern_int_e = 'glorot_uniform'
    kern_int_d = 'glorot_uniform'
    reset_keras()
    
    inp=Input(shape=input_shape)
    
    ###################################################################################################################
            # Encoding
    ###################################################################################################################
    conv1 = Conv2D(feats, kse, padding='same', kernel_regularizer=kern_reg, kernel_initializer=kern_int_e, data_format = 'channels_first')(inp)
    conv1 = LeakyReLU()(conv1)
    conv2 = Conv2D(feats * 2, kse, padding='same', kernel_regularizer=kern_reg, kernel_initializer=kern_int_e, data_format = 'channels_first', strides=(1,2))(
        conv1)
    conv2 = LeakyReLU()(conv2)
    conv3 = Conv2D(feats * 4, kse, padding='same', kernel_regularizer=kern_reg, kernel_initializer=kern_int_e, data_format = 'channels_first', strides=(1,2))(
        conv2)
    conv3 = LeakyReLU()(conv3)
    conv4 = BatchNormalization(axis=1)(conv3)
    
    ###################################################################################################################
        # Latent
    ###################################################################################################################
    conv5 = Conv2D(feats * 4, (1,9), kernel_regularizer=kern_reg, kernel_initializer=kern_int_e, data_format = 'channels_first')(
    conv4)
    conv5 = LeakyReLU()(conv5)
    reshaped = Reshape((feats*3, 1, 6, 16))(conv5)
    
    ###################################################################################################################
    #         Decoding
    ###################################################################################################################
    
    tconv1 = UpSampling3D(size=(1, 3, 2), data_format='channels_first')(reshaped)    
    tconv2 = Conv3D(feats * 2, ksd, padding='same', kernel_regularizer=kern_reg, kernel_initializer=kern_int_d, data_format='channels_first')(tconv1)
    tconv2 = LeakyReLU()(tconv2)
    tconv2 = UpSampling3D(size=(1, 3, 3), data_format='channels_first')(tconv2)    
    tconv3 = Conv3D(feats * 1, ksd, padding='same', kernel_regularizer=kern_reg, kernel_initializer=kern_int_d, data_format='channels_first')(tconv2)
    tconv3 = LeakyReLU()(tconv3)
    tconv4 = Conv3D(int(feats/2), ksd, padding='same', kernel_regularizer=kern_reg, kernel_initializer=kern_int_d, data_format='channels_first')(tconv3)
    tconv4 = LeakyReLU()(tconv4)
    out = Conv3D(15, 1, padding='same', activation='sigmoid', kernel_regularizer=kern_reg, kernel_initializer=kern_int_d, data_format='channels_first')(tconv4)
    
    model = tf.keras.Model(inputs=inp, outputs=out)
    
   
    return model

def cnn3d_v5(input_shape):
    """Assumes a 5D input 'batch,frame,channel,row,column', matching stacked time frames of ARD-AOA measurements"""
    kse = (2,9,5)
    ksd = (1,9,5)
    feats = 60
    kern_reg = None
    kern_int_e = 'glorot_uniform'
    kern_int_d = 'glorot_uniform'
    reset_keras()
    
    inp=Input(shape=input_shape)
    
    ###################################################################################################################
            # Encoding
    ###################################################################################################################
    conv1 = Conv3D(feats, kse, padding='same', kernel_regularizer=kern_reg, kernel_initializer=kern_int_e, data_format = 'channels_first')(inp)
    conv1 = LeakyReLU()(conv1)
    conv2 = Conv3D(feats * 2, kse, padding='same', kernel_regularizer=kern_reg, kernel_initializer=kern_int_e, data_format = 'channels_first', strides=(1,2,2))(
        conv1)
    conv2 = LeakyReLU()(conv2)
    conv3 = Conv3D(feats * 4, kse, padding='same', kernel_regularizer=kern_reg, kernel_initializer=kern_int_e, data_format = 'channels_first', strides=(1,2,1))(
        conv2)
    conv3 = LeakyReLU()(conv3)
    conv4 = BatchNormalization(axis=1)(conv3)
    
    ###################################################################################################################
        # Latent
    ###################################################################################################################
    conv5 = Conv3D(feats * 4, (2,1,3), kernel_regularizer=kern_reg, kernel_initializer=kern_int_e, data_format = 'channels_first')(
    conv4)
    conv5 = LeakyReLU()(conv5)
    reshaped = Reshape((feats*4, 1, 6, 16))(conv5)
    
    ###################################################################################################################
    #         Decoding
    ###################################################################################################################
    
    tconv1 = UpSampling3D(size=(1, 3, 2), data_format='channels_first')(reshaped)    
    tconv2 = Conv3D(feats * 2, ksd, padding='same', kernel_regularizer=kern_reg, kernel_initializer=kern_int_d, data_format='channels_first')(tconv1)
    tconv2 = LeakyReLU()(tconv2)
    tconv2 = UpSampling3D(size=(1, 3, 3), data_format='channels_first')(tconv2)    
    tconv3 = Conv3D(feats * 1, ksd, padding='same', kernel_regularizer=kern_reg, kernel_initializer=kern_int_d, data_format='channels_first')(tconv2)
    tconv3 = LeakyReLU()(tconv3)
    tconv4 = Conv3D(int(feats/2), ksd, padding='same', kernel_regularizer=kern_reg, kernel_initializer=kern_int_d, data_format='channels_first')(tconv3)
    tconv4 = LeakyReLU()(tconv4)
    out = Conv3D(15, 1, padding='same', activation='sigmoid', kernel_regularizer=kern_reg, kernel_initializer=kern_int_d, data_format='channels_first')(tconv4)
    
    model = tf.keras.Model(inputs=inp, outputs=out)
    
   
    return model
