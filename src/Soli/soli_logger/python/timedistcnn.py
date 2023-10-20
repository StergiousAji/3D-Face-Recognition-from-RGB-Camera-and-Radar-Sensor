#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 10 15:55:59 2022

@author: chkaul
"""

import numpy as np
knife1 = np.load('/home/elle/soli_logger/python/chaitanyaknife1.npy')
#knife1= np.resize(knife1, (len(knife1)*32,3,64,16))
knife1 = np.abs(np.squeeze(knife1, axis=1))
knife1 = np.moveaxis(knife1, 1, -1)

knife2 = np.load('/home/elle/soli_logger/python/chaitanyaknife2.npy')
#knife2= np.resize(knife1, (len(knife2)*32,3,64,16))
knife2 = np.abs(np.squeeze(knife2, axis=1))
knife2 = np.moveaxis(knife2, 1, -1)

noknife1 = np.load('/home/elle/soli_logger/python/chaitanyanoknife1.npy')
#noknife1= np.resize(knife1, (len(noknife1)*32,3,64,16))
noknife1 = np.abs(np.squeeze(noknife1, axis=1))
noknife1 = np.moveaxis(noknife1, 1, -1)

noknife2 = np.load('/home/elle/soli_logger/python/chaitanyanoknife2.npy')
#noknife2= np.resize(knife1, (len(noknife2)*32,3,64,16))
noknife2 = np.abs(np.squeeze(noknife2, axis=1))
noknife2 = np.moveaxis(noknife2, 1, -1)


# knifehigher1 = np.load('/home/elle/soli_logger/python/dataknifetrainhigher.npy')
# knifehigher1 = np.abs(np.squeeze(knifehigher1, axis=1))
# knifehigher1 = np.moveaxis(knifehigher1, 1, -1)

# knifehigher2 = np.load('/home/elle/soli_logger/python/dataknifetesthigher.npy')
# knifehigher2 = np.abs(np.squeeze(knifehigher2, axis=1))
# knifehigher2 = np.moveaxis(knifehigher2, 1, -1)


# train = np.concatenate((knife1, knifehigher1, noknife2)).astype('float32')
# test = np.concatenate((knife2, knifehigher2, noknife1)).astype('float32')

train = np.concatenate((knife1, noknife1)).astype('float32')
test = np.concatenate((knife2, noknife2)).astype('float32')

train = train[:,16:-16,4:-4,:]
test = test[:,16:-16,4:-4,:]

train_knife = np.ones((len(knife1),1), dtype='float32')
#trainknifehigher = np.ones((len(knifehigher1),1), dtype='float32')
train_noknife = np.zeros((len(noknife1),1), dtype = 'float32')

test_knife = np.ones((len(knife2),1), dtype='float32')
#testknifehigher = np.ones((len(knifehigher2),1), dtype='float32')
test_noknife = np.zeros((len(noknife2),1), dtype = 'float32')

train = train.reshape(train.shape[0],train.shape[1]*train.shape[2]*train.shape[3])
test = test.reshape(test.shape[0], test.shape[1]*test.shape[2]*test.shape[3])

# train_labels = np.concatenate((train_knife, trainknifehigher, train_noknife))
# test_labels = np.concatenate((test_knife, testknifehigher, test_noknife))

train_labels = np.concatenate((train_knife, train_noknife))
test_labels = np.concatenate((test_knife, test_noknife))

def normalize(data):    
    data = data-np.min(data)
    data = data/np.max(data)
    return data


train = normalize(train)
test = normalize(test)

realtrain = np.concatenate((train,test))
realtrainlab = np.concatenate((train_labels,test_labels))

import tensorflow as tf
from tensorflow.keras.layers import TimeDistributed, Conv2D, MaxPooling2D, Dropout, LSTM, Dense, Flatten

model = tf.keras.Sequential()
#model.add(Conv2D(128, (3, 3), activation='relu', input_shape=(32, 16, 3)))
#model.add(Conv2D(128, (3, 3), activation='relu'))
#model.add(MaxPooling2D(2,2))
model.add(Dense(256, activation='relu', input_shape=(32*8*3,)))
#model.add(Flatten())
#model.add(Dropout(0.2))
model.add(Dense(128, activation='relu'))
#model.add(Dense(64, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

model.summary()


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

#model.fit(train, train_labels, batch_size=16, validation_data=(test,test_labels), epochs=50, callbacks=callbacks)

model.fit(realtrain, realtrainlab, batch_size=16, epochs=50)

model.save('mymodel_50')

# model = tf.keras.models.load_model('/home/elle/soli_logger/python/mymodel_26')

# pred = model.predict(test)


# model = tf.keras.models.load_model('/home/chkaul/Desktop/soli/src/mymodel_3')
# res = model.predict(test) > 0.5
# m = tf.keras.metrics.Accuracy()
# m.update_state(res, test_labels)
# m.result().numpy()


# pred_class = model.predict_classes(test)


# model.evaluate(test, test_labels)