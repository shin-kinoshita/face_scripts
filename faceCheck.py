#!/usr/bin/env python
#! -*- coding: utf-8 -*-

import sys
import numpy as np
import tensorflow as tf
import cv2
import threading
import Queue

NUM_CLASSES = 6
IMAGE_SIZE = 28
IMAGE_PIXELS = IMAGE_SIZE*IMAGE_SIZE*3

def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                          strides=[1, 2, 2, 1], padding='SAME')

x = tf.placeholder("float", shape=[None, IMAGE_PIXELS])
y_ = tf.placeholder("float", shape=[None, NUM_CLASSES])
W_conv1 = weight_variable([5, 5, 3, 32])
b_conv1 = bias_variable([32])
x_image = tf.reshape(x, [-1, IMAGE_SIZE, IMAGE_SIZE, 3])
h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
h_pool1 = max_pool_2x2(h_conv1)

W_conv2 = weight_variable([5, 5, 32, 64])
b_conv2 = bias_variable([64])
h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2)

W_fc1 = weight_variable([IMAGE_SIZE * IMAGE_SIZE * 64 / 16, 1024])
b_fc1 = bias_variable([1024])
h_pool2_flat = tf.reshape(h_pool2, [-1, IMAGE_SIZE * IMAGE_SIZE * 64 / 16])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

keep_prob = tf.placeholder("float")
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

W_fc2 = weight_variable([1024, NUM_CLASSES])
b_fc2 = bias_variable([NUM_CLASSES])

y_conv = tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)

correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

def faceDetector(queue, image):
    f = False
    cascade_path = "/usr/local/opt/opencv/share/OpenCV/haarcascades/haarcascade_frontalface_alt.xml"
    if(image is None):
    	print "no image....."
    	quit()
    image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    cascade = cv2.CascadeClassifier(cascade_path)
    rect = cascade.detectMultiScale(image_gray, scaleFactor=1.2, minNeighbors=2, minSize=(10, 10))
    if len(rect)>0:
        if len(rect[0])>0:
            x = rect[0][0]
            y = rect[0][1]
            width = rect[0][2]
            height = rect[0][3]
            dst = image[y:y+height, x:x+width]
            queue.put((rect, dst))
    else:
        queue.put(([], 0))

if __name__ == '__main__':
    queue = Queue.Queue()
    model = "models/face_model.ckpt"
    # Camera 0 is the integrated web cam on my netbook
    cap = cv2.VideoCapture(0)
    n = 0
    result = 0
    name = ""
    sess = tf.InteractiveSession()
    saver = tf.train.Saver()
    sess.run(tf.initialize_all_variables())
    saver.restore(sess, model)
    faceAndRect = []
    x_face, y_face, width, height = 0, 0, 0, 0
    while(True):
        # Take the actual image we want to keep
        ret, frame = cap.read()
        t = threading.Thread(target = faceDetector, args = (queue, frame, ))
        t.start()
        t.join()
        rect, dst = queue.get()
        print rect
        if len(rect) > 0:
            if len(rect[0]) > 0:
                face = cv2.resize(dst, (IMAGE_SIZE, IMAGE_SIZE))
                face = face.flatten().astype(np.float32)/255.0
                face = np.asarray(face)
                with sess.as_default():
                    result = np.argmax(y_conv.eval(feed_dict={
                        x:[face],
                        keep_prob: 1.0 })[0])
                x_face = rect[0][0]
                y_face = rect[0][1]
                width = rect[0][2]
                height = rect[0][3]

                if result == 0:
                    name = "anitha"
                elif result == 1:
                    name = "gouri"
                elif result == 2:
                    name = "kento"
                elif result == 3:
                    name = "sangram"
                elif result == 4:
                    name = "vishal"
                elif result == 5:
                    name = "yogesh"

        print name
        frame = cv2.resize(frame, (1280,720))
        frame = cv2.putText(frame, name ,(x_face, y_face),cv2.FONT_HERSHEY_SIMPLEX, 3 ,(255,255,0))
        cv2.rectangle(frame, (x_face, y_face), (x_face+width, y_face+width), (0, 255, 0), 3)
        #cv2.rectangle(im, (x, y),(x+width, y+height), (255, 255, 255) , thickness=2)
        cv2.imshow('camera capture', frame)
        k = cv2.waitKey(1)
        if k == 27:
            break

    # キャプチャを解放する
    cap.release()
    cv2.destroyAllWindows()
