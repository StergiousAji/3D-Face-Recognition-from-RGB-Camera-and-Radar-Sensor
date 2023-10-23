import sys

import zmq
import numpy as np

import matplotlib
matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt
import matplotlib.animation as ani

from collections import deque

from scipy.stats import mode



# import tensorflow as tf
# from tensorflow.keras.layers import TimeDistributed, Conv2D, MaxPooling2D, Dropout, LSTM, Dense, Flatten

# # Seed value (can actually be different for each attribution step)
# seed_value= 0

# # 1. Set `PYTHONHASHSEED` environment variable at a fixed value
# import os
# os.environ['PYTHONHASHSEED']=str(seed_value)

# # 2. Set `python` built-in pseudo-random generator at a fixed value
# import random
# random.seed(seed_value)

# # 3. Set `numpy` pseudo-random generator at a fixed value
# import numpy as np
# np.random.seed(seed_value)

# # 4. Set `tensorflow` pseudo-random generator at a fixed value
# tf.random.set_seed(seed_value)

# # 5. Configure a new global `tensorflow` session
# from keras import backend as K
# session_conf = tf.compat.v1.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)
# sess = tf.compat.v1.Session(graph=tf.compat.v1.get_default_graph(), config=session_conf)
# K.set_session(sess)


# model = tf.keras.Sequential([
#     tf.keras.layers.Conv2D(filters=128, kernel_size=3, activation='relu', input_shape=(60, 16, 3)),
#     #tf.keras.layers.Conv2D(filters=128, kernel_size=3, activation='relu'),
#     tf.keras.layers.Flatten(),
#     tf.keras.layers.Dense(64, activation='relu'),
#     tf.keras.layers.Dense(1, activation='sigmoid')
# ])

# STEPS_PER_EPOCH = 5400//32
# lr_schedule = tf.keras.optimizers.schedules.InverseTimeDecay(
#   0.001,
#   decay_steps=STEPS_PER_EPOCH*1000,
#   decay_rate=1,
#   staircase=False)

# def get_optimizer():
#   return tf.keras.optimizers.Adam(lr_schedule)

# optimizer = get_optimizer()

# model.compile(optimizer=optimizer,
#               loss=tf.keras.losses.BinaryCrossentropy(),
#               metrics=['accuracy'])

# callbacks = [
#     tf.keras.callbacks.ModelCheckpoint(
#         # Path where to save the model
#         # The two parameters below mean that we will overwrite
#         # the current checkpoint if and only if
#         # the `val_loss` score has improved.
#         # The saved model name will include the current epoch.
#         filepath="mymodel_{epoch}",
#         save_best_only=True,  # Only save a model if `val_loss` has improved.
#         monitor="val_accuracy",
#         verbose=1,
#     )
# ]


# model = tf.keras.models.load_model('/home/elle/soli_logger/python/mymodel_22')


from soli_logging import SoliLogParser, get_crd_data_bin, plot_crd_data, get_rp_data_bin, get_crd_data

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
            latest_crd = get_crd_data_bin(params, [burst], remove_clutter_=False)
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

                buffertest = buffertest[:,4:,:,:].astype('float64')
                
                #buffertest = buffertest.reshape(buffertest.shape[0], buffertest.shape[1]*buffertest.shape[2]*buffertest.shape[3])
                
                #buffertest = normalize(buffertest)
                
                #buffer = np.squeeze(buffer, axis =0)
                #buffer_resampled = np.moveaxis(buffer, 1, -1)
                #buffer_resampled = buffer_resampled[:,16:-16,:,:]
                
                #min_data = np.min(buffer_resampled)
                #max_data = np.max(buffer_resampled)
                
                #buffer_resampled = normalize(buffer_resampled, min_data, max_data).astype('float32')
                
                #pred_class = model.predict_classes(buffer_resampled)
                
                #pred = model.predict(buffertest)
                #print(pred)
                #print(pred > 0.5)
                #m = tf.keras.metrics.Accuracy()

                
                #buffer_resampled = np.expand_dims(buffer_resampled, axis=0)
                data.append(buffertest)
                
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

