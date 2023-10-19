
import matplotlib.pyplot as plt
plt.ion()
import numpy as np
import time
from datetime import datetime
from skimage.util import view_as_blocks
import cv2

import pyrealsense2 as rs
#import argparse
import time

import os
from concurrent.futures import ProcessPoolExecutor
from invoke import run

import socket 

#%% References on Intel Realsense python wrapper

# https://intelrealsense.github.io/librealsense/python_docs/_generated/pyrealsense2.html
# 
# https://intelrealsense.github.io/librealsense/python_docs/_generated/pyrealsense2.frame.html#pyrealsense2.frame.data
# 
# https://intelrealsense.github.io/librealsense/python_docs/_generated/pyrealsense2.composite_frame.html

# https://github.com/IntelRealSense/librealsense/issues/1887#issuecomment-397529590

#%% Define functions to configure the IRS D435 and to grab frames

def config():
    print("Executing image Task on Process {}".format(os.getpid()))
    config = rs.config()
    #config.enable_stream(rs.stream.depth, 480, 270, rs.format.any, 60) #Params: stream, resolution_x, resolution_y, module, frame rate
    config.enable_stream(rs.stream.color, 640, 480, rs.format.any, 30) #Params: stream, resolution_x, resolution_y, module, frame rate
    
    #Create a pipeline, which is an object with the try_wait_for_frames() method
    global pipeline 
    pipeline = rs.pipeline()
    profile = pipeline.start(config)
    
    
    print("Configured")
    
def grab_frames():    
    #Declare variables to hold frame and system timestamps
    time_1 = 0
    time_2 = 0
    
    #Set the number of frames to capture. Ideally, this should match the length of time for which the 
    #SOLI will run. E.g. if the SOLI acquires frames over 20s, and the IRS is set to 15 FPS, we want 300 frames
    frame_number = 5*60
    
    #Instantiate empty array for depth frames
    depth_frames_list = np.zeros((frame_number,54,96))
    timestamps = np.zeros((frame_number,2))
    print("Grabbing frames...")
    try:   
       for i in range(frame_number):        
            #_,frames = pipeline.try_wait_for_frames()
            #if _ == True:

                #depth_frames_list[i] = depth_scale*np.asanyarray(frames[0].data).astype('float16') #the depth frames are in units of m       
                #timestamps[i,0] = frames.timestamp
                #timestamps[i,1] = time.time()
                #print('Frame',i,'device time difference',frames.timestamp-time_1,'system time difference',(time.time()-time_2)*1e3)
                #time_1 = frames.timestamp  
                #time_2 = time.time()
                
            frames = pipeline.wait_for_frames()
            image = np.asanyarray(frames[0].data)#the depth frames are in units of m      
            print(image.shape,image.dtype) 
            cv2.imshow('image',image)
            cv2.waitKey(1)
	    
    finally:
        pipeline.stop()
        #np.save('../data/images.npy',depth_frames_list)
        #np.save('../data/image_timestamps.npy',timestamps)
config()  
grab_frames()
