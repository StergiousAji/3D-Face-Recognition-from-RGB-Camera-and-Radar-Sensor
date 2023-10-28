#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr  8 05:56:51 2021

@author: mrawe14
"""

import math
import numpy as np
import matplotlib.pyplot as plt
from tqdm.auto import tqdm

#%%
def find_nearest(array,values):
    """
    

    Parameters
    ----------
    array : array-like
        the target array from which to search nearest elements
    values : array-like
        the list of values.

    Returns
    -------
    idxs : array
        the nearest elements of 'array' for each element of 'values'. Used to resample a data series to match the timestamps of another data series.

    """
    idxs = np.zeros(len(values))
    for i,value in enumerate(values):
        idx = np.searchsorted(array, value, side="left")
        if idx > 0 and (idx == len(array) or math.fabs(value - array[idx-1]) < math.fabs(value - array[idx])):
            idxs[i] = idx-1
        else:
            idxs[i] = idx
    return idxs

def plot_crd_image_data(crd_data,image_data,path,start=0):
    """Helper method to plot crd frames and images"""
    num_channels=crd_data.shape[1]
    for i,frame_crd in tqdm(enumerate(crd_data)):
        plt.figure(figsize = (8.53,2*4.8), dpi=100)
        for channel_id in range(num_channels):  
            plt.subplot(2, 3, (channel_id + 1))
            plt.imshow(np.abs(frame_crd[channel_id,:,:]), origin='lower', cmap=plt.get_cmap('jet'), interpolation='none', aspect='auto')
            plt.title('ch' + str(channel_id))
            plt.xlabel('velocity (a.u.)')
            plt.xticks(ticks = [0,4,8,12],labels=(-8,-4,0,4))
            plt.ylabel('range (a.u.)')
        plt.subplot(2,1,2)
        plt.imshow(image_data[i], vmin=0, vmax=7, cmap='viridis_r', interpolation='none', aspect='auto')
        cbar = plt.colorbar()
        cbar.set_label('depth (m)',rotation=270, labelpad=15)
        plt.tight_layout()
        plt.grid(False)
        # plt.savefig(r"{}/img{}.png".format(path,i+start))
        #plt.show()    
        plt.close()
