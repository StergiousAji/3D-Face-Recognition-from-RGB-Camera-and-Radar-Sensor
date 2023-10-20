#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 22 17:00:40 2022

@author: elle
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 22 11:38:35 2022

@author: elle
"""


import numpy as np

knife1 = np.load('/home/elle/soli_logger/python/kevslab1.npy')
knife2 = np.load('/home/elle/soli_logger/python/kevslab2.npy')
noknife1 = np.load('/home/elle/soli_logger/python/kevnoslab1.npy')
noknife2 = np.load('/home/elle/soli_logger/python/kevnoslab2.npy')

knife1 = np.squeeze(knife1, axis=1)
knife2 = np.squeeze(knife2, axis=1)
noknife1 = np.squeeze(noknife1, axis=1)
noknife2 = np.squeeze(noknife2, axis=1)

train = np.concatenate((knife1, knife2, noknife1, noknife2))
#test = np.concatenate((knife2, noknife2))

#train = train[:,4:,:,:]
#test = test[:,4:,:,:]

#train = np.concatenate((gun1, nogun1))
#test = np.concatenate((gun2, nogun2))

#train = np.squeeze(train, axis = 1)
#test = np.squeeze(test, axis=1)

#train_label = np.concatenate((gun1lab, nogun1lab))
#test_label = np.concatenate((gun2lab, nogun2lab))


knife1lab = np.ones((len(knife1),1), dtype='float64')
knife2lab = np.ones((len(knife2),1), dtype='float64')
noknife1lab = np.zeros((len(noknife1),1), dtype='float64')
noknife2lab = np.zeros((len(noknife2),1), dtype='float64')
train_label = np.concatenate((knife1lab, knife2lab, noknife1lab, noknife2lab))
#test_label = np.concatenate((knife2lab, noknife2lab))




import tensorflow as tf
from tensorflow.keras.layers import TimeDistributed, Conv2D, MaxPooling2D, Dropout, LSTM, Dense, Flatten

# Seed value (can actually be different for each attribution step)
seed_value= 0

# 1. Set `PYTHONHASHSEED` environment variable at a fixed value
import os
os.environ['PYTHONHASHSEED']=str(seed_value)

# 2. Set `python` built-in pseudo-random generator at a fixed value
import random
random.seed(seed_value)

# 3. Set `numpy` pseudo-random generator at a fixed value
import numpy as np
np.random.seed(seed_value)

# 4. Set `tensorflow` pseudo-random generator at a fixed value
tf.random.set_seed(seed_value)

# 5. Configure a new global `tensorflow` session
from keras import backend as K
session_conf = tf.compat.v1.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)
sess = tf.compat.v1.Session(graph=tf.compat.v1.get_default_graph(), config=session_conf)
K.set_session(sess)


model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(filters=128, kernel_size=3, activation='relu', input_shape=(64, 16, 3)),
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
        filepath="mymodel_rt_slab_{epoch}",
        save_best_only=True,  # Only save a model if `val_loss` has improved.
        monitor="val_accuracy",
        verbose=1,
    )
]



model.fit(train, train_label, batch_size=32, epochs=25, callbacks=callbacks)

model.save('mymodel_rt_slab_25')
