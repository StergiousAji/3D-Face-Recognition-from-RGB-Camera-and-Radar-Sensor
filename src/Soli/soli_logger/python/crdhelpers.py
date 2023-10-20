#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 16 17:46:03 2022

@author: elle
"""

import numpy as np

def crd_phase_and_amp(crd):
    
    crd_amp = np.mean(np.abs(crd), axis=1)
    crd_ph_1 = np.angle(crd[:, 2, :, :]*np.conj(crd[:, 0, :, :]))
    crd_ph_2 = np.angle(crd[:, 2, :, :]*np.conj(crd[:, 1, :, :]))
    
    crd_amp = np.transpose(crd_amp, (1, 2, 0))
    crd_ph_1 = np.transpose(crd_ph_1, (1, 2, 0))
    crd_ph_2 = np.transpose(crd_ph_2, (1, 2, 0))
    
    return crd_amp, crd_ph_1, crd_ph_2

def process_crd(crd):
    
    crd_amp = np.mean(np.abs(crd), axis=1)
    crd_ph_1 = np.angle(crd[:, 2, :, :])
    
    crd_ph_1 = np.angle(crd[:, 2, :, :]*np.conj(crd[:, 0, :, :]))
    crd_ph_2 = np.angle(crd[:, 2, :, :]*np.conj(crd[:, 1, :, :]))
    
    real10 = crd_amp*np.cos(crd_ph_1)
    imag10 = crd_amp*np.sin(crd_ph_1)
    real12 = crd_amp*np.cos(crd_ph_2)
    imag12 = crd_amp*np.sin(crd_ph_2)
    
    real10 = np.transpose(real10, (1, 2, 0))
    imag10 = np.transpose(imag10, (1, 2, 0))
    real12 = np.transpose(real12, (1, 2, 0))
    imag12 = np.transpose(imag12, (1, 2, 0))
    
    return real10, imag10, real12, imag12

def six_to_four(real0, imag0, real1, imag1, real2, imag2):
    
    real0 = np.transpose(real0, (2, 0, 1))
    imag0 = np.transpose(imag0, (2, 0, 1))
    real1 = np.transpose(real1, (2, 0, 1))
    imag1 = np.transpose(imag1, (2, 0, 1))
    real2 = np.transpose(real2, (2, 0, 1))
    imag2 = np.transpose(imag2, (2, 0, 1))
    
    crd0 = np.expand_dims(real0+1j*imag0, axis=1)
    crd1 = np.expand_dims(real1+1j*imag1, axis=1)
    crd2 = np.expand_dims(real2+1j*imag2, axis=1)
    
    crd = np.concatenate((crd0, crd1, crd2), axis=1)
    
    real10, imag10, real12, imag12 = process_crd(crd)
    
    return real10, imag10, real12, imag12
    
def displace(crd, fd):
    if fd >= 0:
        crd_disp = np.concatenate((np.zeros((fd, np.shape(crd)[1], np.shape(crd)[2], np.shape(crd)[3])), crd), axis=0)
    else:
        crd_disp = crd[-fd:, :, :, :]
    
    return crd_disp

def threshold_phase(ard, theta, phi, thr=0.1):
    for inde in range(np.shape(ard)[2]):
        ard_i = ard[:, :, inde]
        theta_i = theta[:, :, inde]
        theta_i[ard_i < thr * np.amax(ard_i)] = 0.0
        phi_i = phi[:, :, inde]
        phi_i[ard_i < thr * np.amax(ard_i)] = 0.0
        
        if inde == 0:
            theta_new = np.expand_dims(theta_i, axis=2)
            phi_new = np.expand_dims(phi_i, axis=2)
        else:
            theta_new = np.concatenate((theta_new, np.expand_dims(theta_i, axis=2)), axis=2)
            phi_new = np.concatenate((phi_new, np.expand_dims(phi_i, axis=2)), axis=2)
            
    return theta_new, phi_new

def normalise_columns(x, SC=0):
    
    for i in range(np.shape(x)[1]):
        
        x_i = x[:, i]
        
        min_x_i = np.amin(x_i)
        max_x_i = np.amax(x_i)
        xn_i = (x_i-min_x_i)/(SC+max_x_i-min_x_i)
        
        if i == 0:
            xn = np.expand_dims(xn_i, axis=1)
        else:
            xn = np.concatenate((xn, np.expand_dims(xn_i, axis=1)), axis=1)
            
    return xn



exp_ard, exp_theta, exp_phi = crd_phase_and_amp(sample1)
exp_theta, exp_phi = threshold_phase(exp_ard, exp_theta, exp_phi)
hf = np.concatenate((np.expand_dims(exp_ard, axis=3), np.expand_dims(exp_theta, axis=3), np.expand_dims(exp_phi, axis=3)), axis=3)
high_fidelity = np.expand_dims(hf, axis=0)

