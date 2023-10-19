#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 25 17:33:31 2022

@author: chkaul
"""

import numpy as np
import matplotlib.pyplot as plt
import json
from mpl_toolkits.axes_grid1 import make_axes_locatable
import importlib
import sys
from skimage.util import view_as_windows
import cv2
from tqdm.auto import tqdm
from numba import jit
sys.path.insert(1,'../src')
import radar_utils as ru
import utils
importlib.reload(ru)
importlib.reload(utils)


#%% Radar metadata
samples_per_chirp = 128 #number of samples
chirps_per_burst = 16
T_c = (1/2000) #up chirp time = 1/chirp rate
B = 1e9 #bandwidth = 61GHz - 59Ghz
c = 3e8 
f_tx = 60e9 #tranmission frequency = 60GHz
wavelength = c/f_tx
spacing = 2.5e-3 #spacing between receiver antennas in m

#%% Pre-processing

def compute_rp_complex(chirp_raw, zpf = 1):
    n_rbins = chirp_raw.shape[0]

    # window
    window = np.blackman(n_rbins).astype(np.float32)
    window /= np.linalg.norm(window)

    chirp_raw = chirp_raw - np.mean(chirp_raw)  
    # range processing
    chirp_raw = chirp_raw.astype(np.complex64)
    chirp_raw *= window
    chirp_raw = chirp_raw - np.mean(chirp_raw)
    rp_complex = np.fft.fft(chirp_raw, n=n_rbins * zpf)[:int(n_rbins * zpf / 2)] / n_rbins

    return rp_complex

def get_iq_data(json_data):
    iq_data = list()
    for burst in np.asarray(json_data[0]['bursts']):
        if np.asarray(burst['chirps']).shape!=(6144,):
            print('iq failure at burst',burst['burst_id'],'size',np.asarray(burst['chirps']).shape)
        else:
            chirps_in_burst = np.asarray(burst['chirps']).reshape(16,3,-1)
            for channels_in_chirp in chirps_in_burst:
                iq_chirp = list()
                for chirp_per_channel in channels_in_chirp:
                    chirp = np.asarray(chirp_per_channel)
                    iq_chirp.append(chirp)
                iq_data.append(iq_chirp)
    return np.asarray(iq_data)

def get_rp_data(json_data):
    rp_data = list()
    for burst in np.asarray(json_data[0]['bursts']):
        if np.asarray(burst['chirps']).shape!=(6144,):
            print('rp failure at burst',burst['burst_id'],'size',np.asarray(burst['chirps']).shape)
        else:
            chirps_in_burst = np.asarray(burst['chirps']).reshape(16,3,-1)
            for channels_in_chirp in chirps_in_burst:
                rp_chirp = list()
                for chirp_per_channel in channels_in_chirp:
                    chirp = np.asarray(chirp_per_channel)
                    rp_chirp.append(compute_rp_complex(chirp))
                rp_data.append(rp_chirp)
    return np.asarray(rp_data)

def remove_clutter(range_data, clutter_alpha):
    assert range_data.ndim == 3
    clutter_map = 0
    nchirp, nchan, szr = range_data.shape
    range_clutter = range_data.copy()
    if clutter_alpha != 0:
        clutter_map = range_data[0]
        for ic in range(1, nchirp):
            clutter_map = (clutter_map * clutter_alpha + range_data[ic, ...] * (1.0 - clutter_alpha))
            range_clutter[ic, ...] -=  clutter_map
    return range_clutter

def get_chirp_timestamps(json_data):
    timestamps = []
    for burst in json_data['bursts']:
          timestamps.append(burst['timestamp_ms'])
    return np.asarray(timestamps)

def plot_abs_data(data):
  for channel_idx in range(data.shape[1]):
    fig = plt.figure(figsize=(6,2), dpi=300)
    ax = fig.add_subplot(111)
    plt.title('channel {}'.format(channel_idx))
    plt.imshow(np.abs(data[:,channel_idx,:].transpose()), origin='lower', cmap=plt.get_cmap('jet'), interpolation='none', aspect='auto')
    cbar = plt.colorbar()
    cbar.set_label('Amplitude',rotation=270,labelpad=10)
    plt.tight_layout()
    plt.xlabel('Chirp #, total time = 5s')
    plt.ylabel('Range (a.u.)')
    plt.grid(False)
    plt.show()

def get_crd_data(json_data, declutter=True,clutter_coeff=0.9, num_chirps_per_burst=16):
    rp_data = get_rp_data(json_data)
    if declutter:
        rp_clutter = remove_clutter(rp_data, clutter_coeff)
    else: 
        rp_clutter = rp_data.copy()

    window = np.blackman(num_chirps_per_burst).astype(np.float32)
    window /= np.linalg.norm(window)

    rp_transposed = np.transpose(rp_clutter, (1, 0, 2))
    result = []
    for channel_data in rp_transposed:
        channel_data = np.reshape(channel_data, (channel_data.shape[0] // num_chirps_per_burst, num_chirps_per_burst, channel_data.shape[1]))
        crp_per_channel = []
        for burst in channel_data:
            burst = np.transpose(burst)
            crp_burst = []
            for data in burst:
                data = data * window
                data = np.fft.fft(data)
                crp_burst.append(data)
            crp_burst = np.asarray(crp_burst)
            crp_per_channel.append(crp_burst)
        crp_per_channel = np.asarray(crp_per_channel)
        result.append(crp_per_channel)
    result = np.asarray(result)
    result = np.transpose(result,(1, 0, 2, 3))
    result = np.roll(result, result.shape[3]//2, axis=3)
    return result


def phase2angle(phase):
    return np.arcsin((phase*wavelength)/(2*np.pi*spacing))

def get_aoa_data(crd_data,thresh=0):
    """Returns the angle of arrival data from a 3 channel crd input, arranged such that the spacing between Rx2 and Rx1 is horizontal, and between Rx2 and Rx0 vertical"""
    aoa_data = np.abs(np.zeros_like(crd_data[:,0:2,...])).astype('float32') #bursts,horizontal-vertical,range samples,velocity samples
    for co in range(len(crd_data)):
        r0 = np.angle(crd_data[co,0])
        r1 = np.angle(crd_data[co,1])
        r2 = np.angle(crd_data[co,2])

        phase_h = r2 - r1 #horizontal phase  
        phase_h[phase_h>np.pi]=-2*np.pi+phase_h[phase_h>np.pi]
        phase_h[phase_h<-np.pi]=(2*np.pi+phase_h[phase_h<-np.pi])

        phase_v = r2 - r0 #vertical phase
        phase_v[phase_v>np.pi]=-2*np.pi+phase_v[phase_v>np.pi]
        phase_v[phase_v<-np.pi]=(2*np.pi+phase_v[phase_v<-np.pi])

        angle_h = phase2angle(phase_h) #horizontal AoA
        angle_v = phase2angle(phase_v) #vertical AoA
        if thresh!=0:
            #amp0 = np.abs(crd_data[co,0])
            #amp1 = np.abs(crd_data[co,1])
            amp2 = np.abs(crd_data[co,2])

            threshold = np.max(amp2)*thresh
            angle_v[amp2<threshold] = 0    
            angle_h[amp2<threshold] = 0

            #t_ranges = rang[np.where(amp2>threshold)[0]+64]
            #t_angle_v = angle_v[amp2>threshold]
            #t_angle_h = angle_h[amp2>threshold]
        aoa_data[co,0,...] = angle_h
        aoa_data[co,1,...] = angle_v
            
    return aoa_data
    

def plot_crd_data(crd_data):
  num_channels=crd_data.shape[1]
  for i,frame_crd in enumerate(crd_data):
    print(i)
    fig,axs=plt.subplots(nrows=1,ncols=3,gridspec_kw={'width_ratios':(1,1,1)},figsize = (8.53,4.8), dpi=100)
    for channel_id in range(num_channels):  
      #plt.subplot(1, num_channels, channel_id + 1,sharey=True)
      im = axs[channel_id].imshow(np.abs(frame_crd[channel_id,:,:]), origin='lower', cmap=plt.get_cmap('jet'), interpolation='none', aspect='auto')
      axs[channel_id].set_title('ch' + str(channel_id))
      axs[channel_id].set_xlabel('velocity (a.u.)')
      axs[channel_id].set_xticks(ticks = [0,4,8,12])
      axs[channel_id].set_xticklabels(labels=(-8,-4,0,4))
      #divider = make_axes_locatable(axs[channel_id])
      #cax = divider.append_axes('right', size='5%', pad=0.05)
      #cbar = fig.colorbar(im,cax=cax)
      #cbar.set_label('amplitude (a.u.)',rotation=270,labelpad=10)
      axs[channel_id].set_ylabel('range (a.u.)')

    plt.tight_layout()
    plt.grid(False)
    #plt.savefig(r'C:\Users\kapit\OneDrive - University of Glasgow\Single_pixel_detector\Radar_Google\data\20210319_experimenting\range_doppler\crd{}.png'.format(i))
    #plt.show()    
    plt.close()
    
#%% Range and velocity samples in units of m and m/s resp.
def range_freqsamples():
    """Returns the sample frequencies of a fourier transform (i.e. the x-axis)"""
    f_b = np.fft.fftshift(np.fft.fftfreq(samples_per_chirp,T_c/samples_per_chirp))
    """Transforms frequency into range"""
    return c*T_c*f_b/(2*B)

wavelength = c/(f_tx)
def velocity_freqsamples():
    f_d = np.fft.fftshift(np.fft.fftfreq(chirps_per_burst,(chirps_per_burst*T_c)/chirps_per_burst))
    return c*f_d/(2*f_tx)




json_soli_data = []
radar_timestamps = []

f = open('/home/chkaul/Desktop/soli/temp/guntest/GunChest_r7/radar.json')
json_soli_datas = json.load(f)
json_soli_data.append(json_soli_datas)

radar_timestamps.append(get_chirp_timestamps(json_soli_datas))

images = np.load('/home/chkaul/Desktop/soli/temp/guntest/GunChest_r7/images.npy')
image_timestampss = np.load('/home/chkaul/Desktop/soli/temp/guntest/GunChest_r7/image_timestamps.npy')


rp_data = []
crd_data = []
iq_data = []
rp_data.append(get_rp_data(json_soli_data))
iq_data.append(get_iq_data(json_soli_data))
crd_data.append(get_crd_data(json_soli_data,declutter=False))


phase = np.angle(crd_data[0][0,0])
phase[phase>np.pi]=-2*np.pi+phase[phase>np.pi]
phase[phase<-np.pi]=(2*np.pi+phase[phase<-np.pi])

plt.figure(figsize=(10,6))
plt.subplot(121)
plt.imshow(np.abs(crd_data[0][0,0]),aspect='auto',origin='lower')
plt.title('Amplitude of range doppler plot')
plt.colorbar()
plt.subplot(122)
plt.imshow(phase,aspect='auto',origin='lower')
plt.title('Phase of range doppler plot')
plt.colorbar()
plt.show()

#Plot the range profiles, just to check
plot_abs_data(rp_data[0])


#Find maximum depth in frames
maxima = []
for i in images:
    for j in i:
        maxima.append(np.max(j))
maxima = np.asarray(maxima)
maxima[np.isinf(maxima)]=0
plt.hist(maxima,bins=200)
plt.xlim([0,10])
plt.show()

# #Set upper threshold on maximum depth
max_depth = 5
images[images>max_depth]=max_depth


#Visualise image data, just to check
plt.imshow(images[3000],cmap='viridis_r',vmin=0,vmax=max_depth)
cbar = plt.colorbar()
cbar.set_label('depth (m)',rotation=270,labelpad=15)
plt.show()

radar_time = []
image_time = []
for i in range(len(radar_timestamps)):
    radar_time.append(radar_timestamps[i] - radar_timestamps[i][0]) #remove initial offset using the assumption that the streams started in sync
    image_time.append(image_timestampss[:,0] - image_timestampss[0,0])
    print('CRD frame count',len(crd_data[i]),'image frame count',len(images[i])) #I need to resample the crd_data to match the frame rate of the images
    print('Radar total time',radar_time[i][-1]-radar_time[i][0],'image total time',image_time[i][-1]-image_time[i][0])
    print('Radar start',radar_time[i][0],'image start',image_time[i][0])

import math
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

find_nearestN = find_nearest(image_time[0], radar_time[0])


image_time_resampled = []
find_nearestN = find_nearestN[5:1505].astype('int16')
images_resampled = images[find_nearestN]
crd_resampled = crd_data[0][5:1505]

crd_resampled_Awithknife = np.moveaxis(crd_resampled, 1, -1)
images_resampled_Awithknife = np.expand_dims(images_resampled, -1)



##############################################################
crd_data = []
crd_data = np.concatenate((crd_resampled_Awithknife, crd_resampled_Awithnoknife, crd_resampled_Bwithknife,crd_resampled_Bwithnoknife))
crd_data = np.abs(crd_data).astype('float64')

image_data = np.concatenate((images_resampled_Awithknife, images_resampled_Awithnoknife, images_resampled_Bwithknife,images_resampled_Bwithnoknife))
image_data = np.abs(image_data).astype('float64')


a_knife_label = np.ones((1500,1), dtype='float64')
a_noknife_label = np.zeros((1500,1), dtype='float64')
b_knife_label = np.ones((1500,1), dtype='float64')
b_noknife_label = np.zeros((1500,1), dtype='float64')

labels = np.concatenate((a_knife_label, a_noknife_label, b_knife_label, b_noknife_label))


def normalize(data):    
    data = data-np.min(data)
    data = data/np.max(data)
    return data

crd_data = normalize(crd_data)
image_data = image_data/np.max(image_data)

np.save('radar_soli.npy', crd_data)
np.save('images_soli.npy', image_data)
np.save('labels.npy', labels)


