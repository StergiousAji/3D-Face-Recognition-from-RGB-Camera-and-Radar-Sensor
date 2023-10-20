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
import tensorflow as tf

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
        plt.savefig(r"{}/img{}.png".format(path,i+start))
        #plt.show()    
        plt.close()

        
def angle2pixel(ang_h,ang_v):
    """returns the Intel Realsense pixel coordinates of an registered object at ang_h and ang_v by the SOLI radar"""
    # Intel Realsense FOV 86° × 57° (horizontal vertical) https://www.intelrealsense.com/depth-camera-d435/
    # Image shape = 96 x 54 pixels
    # i.e. 43° horizontal gets mapped on column 0, -43° onto 96, etc.
    
    # Map radians to degrees
    ang_h = (180/np.pi)*ang_h.copy()
    ang_v = (180/np.pi)*ang_v.copy()
    
    ang_h = ang_h - 25
    h = ang_h*(86/96) #rescale to pixels
    h = -h #flip
    h = h + 43 #shift
   
    v = ang_v*(54/57)
    v = -v #flip
    v = v + 27
    
    h[h<0] = 0
    h[h>95] = 95 #python indexing
    
    v[v<0] = 0
    v[v>53] = 53
    
    return h,v
    
    
def plot_rp_crd_aoa_image_data(rp,crd,aoa,images,co,thresh,rang,vel,max_depth):
    """
    

    Parameters
    ----------
    rp = 3D array (dims = burst,channel,range)
        series of complex range profiles
    crd : 4D array of complex64s (dims = burst,channel,range,doppler)
        series of complex range doppler frames
    aoa : 4D array of float32s (dims = burst,horizontal/vertical,range,doppler)
        series of horizontal and vertical angle of arrival plots, calculated via phase monopulse angle estimation.
    images : 3D array (series of 2D arrays)
        3D depth frames, normalised.
    co : int
        frame number.
    thresh : float between 0 and 1
        thresholding value, as a factor of the max amplitude in the given frame
    rang : array of floats,
        the range samples of the CRD plots
    vel : array of floats,
        the velocity samples of the CRD plots
    max_depth: float,
        maximum depth in FOV of the acquisition
    Returns
    -------
    Image.

    """
    #Declare AoA
    angle_h = aoa[co,0,...] #horizontal AoA
    angle_v = aoa[co,1,...] #vertical AoA
    
    #Declare amplitudes
    amp0 = np.abs(crd[co,0])
    amp1 = np.abs(crd[co,1])
    amp2 = np.abs(crd[co,2])
    
    #Threshold using amp2
    threshold = np.max(amp2)*thresh
    angle_h_thresh = angle_h.copy()
    angle_h_thresh[amp2<threshold] = 0
    angle_v_thresh = angle_v.copy()
    angle_v_thresh[amp2<threshold] = 0
    t_ranges = rang[np.where(amp2>threshold)[0]+64]
    t_angle_v = angle_v_thresh[amp2>threshold]
    t_angle_h = angle_h_thresh[amp2>threshold]
    
    h,v = angle2pixel(t_angle_h,t_angle_v)
    
    fig = plt.figure(figsize = (12.8,7.2), dpi=100)
    gs = fig.add_gridspec(ncols=6, nrows=2, height_ratios=(2, 3),
                      hspace=0.75,wspace=1)
    plt.subplot(gs[0,0])
    plt.plot(np.abs(rp[co,0]),label='ch 0')
    plt.plot(np.abs(rp[co,1]),label='ch 1')
    plt.plot(np.abs(rp[co,2]),label='ch 2')
    plt.title('RP')
    plt.xlabel('range (m)')
    plt.ylabel('amplitude (a.u.)')
    plt.legend()
    plt.xticks(ticks = [0, 40], labels=rang[[64,64+40]])
    plt.subplot(gs[0,1])
    plt.imshow(amp0, vmin=0, origin='lower', cmap=plt.get_cmap('jet'), interpolation='none', aspect='auto')
    plt.title('ch 0 ARD')
    plt.xlabel('velocity (m/s)')
    plt.xticks(ticks = [0,8],labels=vel[[0,8]])
    plt.ylabel('range (m)')
    plt.yticks(ticks = [0, 20, 40, 60], labels=rang[[64,64+20,64+40,64+60]])
    plt.subplot(gs[0,2])
    plt.imshow(amp1, vmin=0, origin='lower', cmap=plt.get_cmap('jet'), interpolation='none', aspect='auto')
    plt.title('ch 1 ARD')
    plt.xlabel('velocity (m/s)')
    plt.xticks(ticks = [0,8],labels=vel[[0,8]])
    plt.ylabel('range (m)')
    plt.yticks(ticks = [0, 20, 40, 60], labels=rang[[64,64+20,64+40,64+60]])
    plt.subplot(gs[0,3])
    plt.imshow(amp2, vmin=0, origin='lower', cmap=plt.get_cmap('jet'), interpolation='none', aspect='auto')
    plt.title('ch 2 ARD')
    plt.xlabel('velocity (m/s)')
    plt.xticks(ticks = [0,8],labels=vel[[0,8]])
    plt.ylabel('range (m)')
    plt.yticks(ticks = [0, 20, 40, 60], labels=rang[[64,64+20,64+40,64+60]])
    plt.subplot(gs[0,4])
    plt.imshow(angle_h_thresh, vmin=-np.pi/2, vmax=np.pi/2, origin='lower', interpolation='none', aspect='auto')
    plt.title('thresholded\nhorizontal AoA (thAoA)')
    cbar = plt.colorbar()
    cbar.set_label('angle (rad)', rotation=270, labelpad=10)
    plt.xlabel('velocity (m/s)')
    plt.xticks(ticks = [0,8],labels=vel[[0,8]])
    plt.ylabel('range (m)')
    plt.yticks(ticks = [0, 20, 40, 60], labels=rang[[64,64+20,64+40,64+60]])    
    plt.subplot(gs[0,5])
    plt.imshow(angle_v_thresh, vmin=-np.pi/2, vmax=np.pi/2, origin='lower', interpolation='none', aspect='auto')
    plt.title('thresholded\nvertical AoA (tvAoA)')
    cbar = plt.colorbar()
    cbar.set_label('angle (rad)', rotation=270, labelpad=10)
    plt.xlabel('velocity (m/s)')
    plt.xticks(ticks = [0,8],labels=vel[[0,8]])
    plt.ylabel('range (m)')
    plt.yticks(ticks = [0, 20, 40, 60], labels=rang[[64,64+20,64+40,64+60]])
    
    
    ax = plt.subplot(gs[1,0:2], projection='polar')
    plt.scatter(t_angle_h,t_ranges,c=amp2[amp2>threshold],s=10,cmap='jet')
    ax.set_theta_zero_location('N')
    ax.set_rlim([0,np.max(rang)])
    ax.set_rlabel_position(90) 
    cbar = plt.colorbar(pad=0.2)
    cbar.set_label('signal strength',rotation=270,labelpad=10)
    plt.title('thAoA (deg) vs range (m)')
    ax = plt.subplot(gs[1,2:4], projection='polar')
    plt.scatter(t_angle_v,t_ranges,c=amp2[amp2>threshold],s=10,cmap='jet')
    ax.set_theta_zero_location('E')
    ax.set_rlim([0,np.max(rang)])
    ax.set_rlabel_position(90) 
    cbar = plt.colorbar(pad=0.2)
    cbar.set_label('signal strength',rotation=270,labelpad=10)
    plt.title('tvAoA (deg) vs range (m)')
    plt.subplot(gs[1,4:6])
    plt.imshow(images[co].astype('float32'),vmin=0,vmax=max_depth,cmap='viridis_r')
    plt.title('3D image')
    cbar = plt.colorbar()
    cbar.set_label('depth (m)',rotation=270,labelpad=10)
    plt.scatter(h,v,c=t_ranges,vmin=0,vmax=np.max(images),edgecolors='k',cmap='viridis_r',marker='s',s=100*(amp2[amp2>threshold]),label='soli signal')
    plt.legend(bbox_to_anchor=(1, 1.5))
    plt.show()