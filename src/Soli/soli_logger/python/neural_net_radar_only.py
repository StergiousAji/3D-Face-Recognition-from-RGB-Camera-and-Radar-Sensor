#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 26 18:32:54 2022

@author: chkaul
"""


import tensorflow as tf

# Helper libraries
import numpy as np
import matplotlib.pyplot as plt

model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(filters=128, kernel_size=3, activation='relu', input_shape=(64, 16, 3)),
    #tf.keras.layers.Conv2D(filters=128, kernel_size=3, activation='relu'),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

STEPS_PER_EPOCH = 5400//32
lr_schedule = tf.keras.optimizers.schedules.InverseTimeDecay(
  0.01,
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

model.fit(crd_data, labels , validation_data=(crd_resampled_gun2, gun_label), batch_size=32, epochs=50, callbacks=callbacks)


model.fit(crd_data[0:3000], labels[0:3000] , validation_data=(crd_data[3000:6000], labels[3000:6000]), batch_size=32, epochs=50)


model = tf.keras.models.load_model('/home/chkaul/Desktop/soli/src/mymodel_3')
import time
start = time.clock() 
with tf.device('/cpu:0'):
    pred = model.evaluate(crd_resampled_gun2, gun_label)
end = time.clock()

print("Time per image: {} ".format((end-start)/1500)) 

train_radar1 = crd_data[0:3000-300]
train_radar2 = crd_data[3000+300:]
train_radar = np.concatenate((train_radar1,train_radar2))
val_radar = crd_data[3000-300:3000+300]

train_classif1 = labels[0:3000-300]
train_classif2 = labels[3000+300:]
train_classif = np.concatenate((train_classif1,train_classif2))
val_classif = labels[3000-300:3000+300]

model.fit(train_radar, train_classif, validation_data=(val_radar, val_classif), batch_size=32, epochs=50)



train_radar1 = crd_data[0:3000-300]
train_radar2 = crd_data[3000+300:]
train_radar = np.concatenate((train_radar1,train_radar2))
val_radar = crd_data[3000-300:3000+300]

train_classif1 = labels[0:1500-150]
train_classif2 = labels[1500+150:]
train_classif = np.concatenate((train_classif1,train_classif2))
val_classif = labels[1500-150:1500+150]

model.fit(train_radar, train_classif, validation_data=(val_radar, val_classif), batch_size=32, epochs=50)






train_radar1 = crd_data_knifenoknife[0:3000-300]
train_radar2 = crd_data_knifenoknife[3000+300:]
train_radar = np.concatenate((train_radar1,train_radar2))
val_radar = crd_data_knifenoknife[3000-300:3000+300]

train_classif1 = labels[0:3000-300]
train_classif2 = labels[3000+300:]
train_classif = np.concatenate((train_classif1,train_classif2))
val_classif = labels[3000-300:3000+300]

model.fit(train_radar, train_classif, validation_data=(val_radar, val_classif), batch_size=32, epochs=50)


