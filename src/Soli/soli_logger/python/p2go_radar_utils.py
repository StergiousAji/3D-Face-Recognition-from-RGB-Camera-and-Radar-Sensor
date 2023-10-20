# -*- coding: utf-8 -*-
"""
Created on Tue Apr 20 16:05:44 2021

@author: Valentin Kapitany
"""
#@title Utils
# Range profile (aka preprocessed)

import numpy as np
import matplotlib.pyplot as plt
#%% Radar metadata
samples_per_chirp = 128 #number of samples
chirps_per_burst = 16
T_c = 0.3e-3 # up chirp time
B = 0.2e9 #bandwidth = 24.22GHz - 24.05GHz
c = 3e8 
f_tx = 24e9 #tranmission frequency = 24GHz
wavelength = c/f_tx
spacing = 6.25e-3 #spacing between receiver antennas in m

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

def get_iq_data(p2go_data):
    iq_data = list()
    for burst in p2go_data:
        for chirp in burst:
            iq_chirp = list()
            for samples_per_channel in chirp:
               iq_chirp.append(samples_per_channel)
            iq_data.append(iq_chirp)
    return np.asarray(iq_data)


def get_rp_data(p2go_data):
    rp_data = list()
    for burst in p2go_data:
        for chirp in burst:
            rp_chirp = list()
            for samples_per_channel in chirp:
                rp_chirp.append(compute_rp_complex(samples_per_channel))
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

def get_crd_data(p2go_data, declutter=True,clutter_coeff=0.9, num_chirps_per_burst=16):
    rp_data = get_rp_data(p2go_data)
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
    """Returns the angle of arrival data from a 2 channel crd input, arranged such that the spacing between Rx0 and Rx1 is horizontal"""
    aoa_data = np.abs(np.zeros_like(crd_data[:,0,...])).astype('float32')
    aoa_data = aoa_data.reshape(aoa_data.shape[0],1,aoa_data.shape[1],aoa_data.shape[2])
    #bursts,horizontal-vertical,range samples,velocity samples
    for co in range(len(crd_data)):
        r0 = np.angle(crd_data[co,0])
        r1 = np.angle(crd_data[co,1])

        phase_h = r1 - r0 #horizontal phase  
        phase_h[phase_h>np.pi]=-2*np.pi+phase_h[phase_h>np.pi]
        phase_h[phase_h<-np.pi]=(2*np.pi+phase_h[phase_h<-np.pi])
        phase_h[phase_h>np.pi] = np.pi
        phase_h[phase_h<-np.pi] = -np.pi

        angle_h = phase2angle(phase_h) #horizontal AoA
        if thresh!=0:
            #amp0 = np.abs(crd_data[co,0])
            #amp1 = np.abs(crd_data[co,1])
            amp2 = np.abs(crd_data[co,1])

            threshold = np.max(amp2)*thresh
            angle_h[amp2<threshold] = 0

            #t_ranges = rang[np.where(amp2>threshold)[0]+64]
            #t_angle_v = angle_v[amp2>threshold]
            #t_angle_h = angle_h[amp2>threshold]
        aoa_data[co,0,...] = angle_h
            
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
