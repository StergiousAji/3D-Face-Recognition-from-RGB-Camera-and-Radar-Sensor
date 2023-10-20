#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 16 19:09:09 2022

@author: elle
"""

import sys

import zmq

from soli_logging import SoliLogParser, get_crd_data_bin

import numpy as np



# def compute_rp_complex(chirp_raw, n_rbins, window, zpf=1):
#     # range processing
#     chirp_raw = chirp_raw - np.mean(chirp_raw)  
#     chirp_raw *= window
#     chirp_raw = chirp_raw - np.mean(chirp_raw)
#     rp_complex = np.fft.fft(chirp_raw, n=n_rbins * zpf)[:int(n_rbins * zpf / 2)] / n_rbins
#     return rp_complex

# def get_rp_data_bin(chirps):
#     rp_data = []

#     # n_rbins and window were previously calculated on each call to compute_rp_complex
#     # but they only depend on the chirp length which will be the same for every
#     # chirp in a burst, so just calculate them once here to make things a bit faster
#     n_rbins = 128
#     window = np.blackman(n_rbins).astype(np.float32)
#     window /= np.linalg.norm(window)

#     for chirp in chirps:
#         rp_chirp = []
#         for channel in range(3):
#             rp_chirp.append(compute_rp_complex(chirp[channel, :], n_rbins, window))
#         rp_data.append(rp_chirp)
#     return np.asarray(rp_data)







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

def get_rp_data(chirps):
    rp_data = list()
    for channels_in_chirp in chirps:
        rp_chirp = list()
        for chirp_per_channel in channels_in_chirp:
            chirp = np.asarray(chirp_per_channel)
            rp_chirp.append(compute_rp_complex(chirp))
        rp_data.append(rp_chirp)
    return np.asarray(rp_data)


def get_crd_data(chirps, num_chirps_per_burst=16):
    rp_data = get_rp_data(chirps)

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









import tensorflow as tf

# Seed value (can actually be different for each attribution step)
seed_value= 0

# 1. Set `PYTHONHASHSEED` environment variable at a fixed value
import os
os.environ['PYTHONHASHSEED']=str(seed_value)

# 2. Set `python` built-in pseudo-random generator at a fixed value
import random
random.seed(seed_value)

# 3. Set `numpy` pseudo-random generator at a fixed value
np.random.seed(seed_value)

# 4. Set `tensorflow` pseudo-random generator at a fixed value
tf.random.set_seed(seed_value)

# 5. Configure a new global `tensorflow` session
from keras import backend as K
session_conf = tf.compat.v1.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)
sess = tf.compat.v1.Session(graph=tf.compat.v1.get_default_graph(), config=session_conf)
K.set_session(sess)


model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(filters=128, kernel_size=3, activation='relu', input_shape=(60, 16, 3)),
    #tf.keras.layers.Conv2D(filters=128, kernel_size=3, activation='relu'),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

STEPS_PER_EPOCH = 5400//32
lr_schedule = tf.keras.optimizers.schedules.InverseTimeDecay(
  0.001,
  decay_steps=STEPS_PER_EPOCH*1000,
  decay_rate=1,
  staircase=False)

def get_optimizer():
  return tf.keras.optimizers.Adam(lr_schedule)

optimizer = get_optimizer()

model.compile(optimizer=optimizer,
              loss=tf.keras.losses.BinaryCrossentropy(),
              metrics=['accuracy'])

callbacks = [
    tf.keras.callbacks.ModelCheckpoint(
        # Path where to save the model
        # The two parameters below mean that we will overwrite
        # the current checkpoint if and only if
        # the `val_loss` score has improved.
        # The saved model name will include the current epoch.
        filepath="mymodel_{epoch}",
        save_best_only=True,  # Only save a model if `val_loss` has improved.
        monitor="val_accuracy",
        verbose=1,
    )
]


model = tf.keras.models.load_model('/home/elle/soli_logger/python/mymodel_rt_slab_25')




if __name__ == '__main__':
    #if len(sys.argv) != 2:
    #    print('testzmq.py <remote IP>')
    #    sys.exit(0)

    ctx = zmq.Context()
    sock = ctx.socket(zmq.SUB)
    sock.connect('tcp://{}:5556'.format('127.0.0.1'))
    sock.setsockopt(zmq.SUBSCRIBE, 'soli'.encode('utf-8'))
    
    buffer = None
    data = []
    buflen = 1
    
    parser = SoliLogParser()
    try:
        while True:
            topic, msg = sock.recv_multipart()
            print(topic.decode('utf-8'), len(msg))
            params, burst = parser.parse_burst(msg, clear_params=True)
            #print('> Received burst ID: {}'.format(burst.burst_id))
            if burst.burst_id % 1 == 0:
                print('> Received burst ID: {}'.format(burst.burst_id))
                chirp = burst.chirps
            latest_crd = get_crd_data(chirp)
            #range_profile = get_rp_data_bin([burst])
            if buffer is None:
                buffer = latest_crd
                #buffer = range_profile
            elif buffer.shape[0] < buflen:
                buffer = np.concatenate([buffer, latest_crd], axis=0)
                #buffer = np.concatenate([buffer, range_profile], axis=0)
            else:
                buffer = np.delete(buffer, 0, axis=0)
                buffer = np.concatenate([buffer, latest_crd], axis=0)
                #buffer = np.delete(buffer, 0, axis=0)
                #buffer = np.concatenate([buffer, range_profile], axis=0)
                
                buffertest = np.abs(buffer)
                buffertest = np.moveaxis(buffertest, 1, -1) 

                buffertest = buffertest.astype('float64') # [:,4:,:,:]
                #buffertest = np.squeeze(buffertest, axis=0)
                #buffertest= buffertest[:,4:,:,:]
                #data.append(buffertest)
                
                #buffertest = buffertest.reshape(buffertest.shape[0], buffertest.shape[1]*buffertest.shape[2]*buffertest.shape[3])
                
                #buffertest = normalize(buffertest)
                
                #buffer = np.squeeze(buffer, axis =0)
                #buffer_resampled = np.moveaxis(buffer, 1, -1)
                #buffer_resampled = buffer_resampled[:,16:-16,:,:]
                
                #min_data = np.min(buffer_resampled)
                #max_data = np.max(buffer_resampled)
                
                #buffer_resampled = normalize(buffer_resampled, min_data, max_data).astype('float32')
                
                #pred_class = model.predict_classes(buffer_resampled)
                
                pred = model.predict(buffertest)
                out = np.where(pred > 0.5, 1,0)
                print(pred)
                #print(pred > 0.5)
                data.append(out)
                #m = tf.keras.metrics.Accuracy()

                
                #buffer_resampled = np.expand_dims(buffer_resampled, axis=0)
                #data.append(buffertest)
                
                #Q = deque(maxlen=(16))
            
                #with tf.device('/cpu:0'):
                    #pred = model.predict(buffer_resampled)
                    #print(pred>0.5)
                    #Q.append(pred)
                    
                    #result = np.array(Q).mean(axis=0)
                    #print(np.argmax(result))
                    #print(result >0.2)
                    
    except KeyboardInterrupt:
        pass
    
data = np.squeeze(np.asarray(data), axis=1)    
lab = np.ones((654,1))    
accuracy = (data == lab).sum() / 654