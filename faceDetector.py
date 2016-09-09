# -*- coding:utf-8 -*-
import cv2
import sys
import os
import shutil

def faceDetector(image, savePath, filename):
    
    f = False
    cascade_path = "/usr/local/share/OpenCV/haarcascades/haarcascade_frontalface_alt2.xml"
    filename = filename.split(".")
    #ファイル読み込み
    #image = cv2.imread(image_path)
    if(image is None):
    	print "no image....."
    	quit()
    #グレースケール変換
    image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    #カスケード分類器の特徴量を取得する
    cascade = cv2.CascadeClassifier(cascade_path)
    print cascade
    #物体認識（顔認識）の実行
    facerect = cascade.detectMultiScale(image_gray, scaleFactor=1.2, minNeighbors=2, minSize=(10, 10))
    
    #ディレクトリの作成
    if len(facerect) > 0:
    	if os.path.isdir(savePath):
    		pass
        else:
    	    os.mkdir(savePath)
    for rect in facerect:
    	#顔だけ切り出して保存
    	x = rect[0]
    	y = rect[1]
    	width = rect[2]
    	height = rect[3]
        if width > 180 and height > 180:
            dst = image[y:y+height, x:x+width]
            new_image_path = savePath + '/'+ filename[0] + "." + filename[1]
            cv2.imwrite(new_image_path, dst)
            f = True
            print 'ok'
    return f
