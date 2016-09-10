#!/usr/bin/python
import sys
import os 
sys.path.append('/usr/local/lib/python2.7/site-packages')
import cv2
import faceDetector
import numpy as np
import glob

# Camera 0 is the integrated web cam on my netbook
#camera_port = 0
#Number of frames to throw away while the camera adjusts to light levels
ramp_frames = 1
# Now we can initialize the camera capture object with the cv2.VideoCapture class.
# All it needs is the index to a camera port.
savePath = "./data_facetrimmer/output"

# Captures a single image from the camera and returns it in PIL format
#def get_image():
# read is the easiest way to get a full image out of a VideoCapture object.

# return im
# Ramp the camera - these frames will be discarded and are only used to allow v4l2
# to adjust light levels, if necessary
#camera = cv2.VideoCapture(camera_port)
tmp = glob.glob('./data_facetrimmer/input/*.JPG')
n = 0
for index in tmp:
    #print index
    n+=1
    camera_capture = cv2.imread(index,1)
    f = os.path.basename(index)
    #for i in xrange(ramp_frames):
    # cap = get_image()
    
    # cap = cv2.imread(tmp[ramp_frames],1)
    #print("Taking image...")
    
    # Take the actual image we want to keep
    
    # print camera_capture
    
    flag = faceDetector.faceDetector(camera_capture, savePath, f)
print n

# A nice feature of the imwrite method is that it will automatically choose the
# correct format based on the file extension you provide. Convenient!
# You'll want to release the camera, otherwise you won't be able to create a new
# capture object until your script exits
