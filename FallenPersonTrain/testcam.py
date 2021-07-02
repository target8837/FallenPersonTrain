#!/usr/bin/env python
# -*- coding: utf-8 -*-

from keras.models import Sequential
from keras.layers import Dense
from keras.models import model_from_yaml
import numpy
import os
import tensorflow as tf
import pandas as pd
import h5py
import glob
import cv2
import csv
import functools

import imutils
from imutils.video import VideoStream
from imutils.video import FPS
import time

# fix random seed for reproducibility

BODY_PARTS = { "Nose": 0, "Neck": 1, "RShoulder": 2, "RElbow": 3, "RWrist": 4,
                "LShoulder": 5, "LElbow": 6, "LWrist": 7, "MidHip": 8, "RHip":9, "RKnee": 10,
                "RAnkle": 11, "LHip": 12, "LKnee": 13, "LAnkle": 14, "REye":15, "LEye":16, "REar":17,"LEar":18,"LBigToe":19, "LSmallToe":20, "LHeel":21,
		"RBigToe":22, "RSmallToe":23, "RHeel":24, "Background": 25 }

POSE_PAIRS = [ ["Nose", "Neck"], ["Neck", "RShoulder"], ["RShoulder", "RElbow"],
                ["RElbow", "RWrist"], ["Neck", "LShoulder"], ["LShoulder", "LElbow"],
                ["LElbow", "LWrist"], ["Neck", "MidHip"], ["MidHip", "RHip"], ["RHip", "RKnee"],
                ["RKnee", "RAnkle"], ["MidHip", "LHip"], ["LHip", "LKnee"], ["LKnee", "LAnkle"], 
		["Nose", "REye"],["REye","REar"], ["Nose", "LEye"],["REye","LEar"],
		["RAnkle","RHeel"], ["RHeel", "RBigToe"], ["RHeel", "RSmallToe"],
		["LAnkle","LHeel"], ["LHeel", "LBigToe"], ["LHeel", "LSmallToe"],
		 ]

seed = 7
numpy.random.seed(seed)
 
# load YAML and create model
yaml_file = open('model.yaml', 'r')
loaded_model_yaml = yaml_file.read()
yaml_file.close()
loaded_model = model_from_yaml(loaded_model_yaml)
# load weights into new model
loaded_model.load_weights("model.h5")

tf.__version__
print("Loaded model from disk")
tf.__version__

protoFile = "/home/kwan/openpose/models/pose/body_25/pose_deploy.prototxt"
weightsFile = "/home/kwan/openpose/models/pose/body_25/pose_iter_584000.caffemodel"
 
net = cv2.dnn.readNetFromCaffe(protoFile, weightsFile)
vs = cv2.VideoCapture('/home/kwan/move.MOV')
time.sleep(2.0)

fps = FPS().start()
(w, h) = (None, None)

tf.__version__
while True :
    # 이미지 읽어오기
    xmin = 10000
    xmax = 0
    ymin = 10000
    ymax = 0
    #cap = cv2.VideoCapture(2)
    ret, frame = vs.read()
    frame = imutils.resize(frame,width=600)
    if w is None or h is None:
        (h, w) = frame.shape[:2]
    inpBlob = cv2.dnn.blobFromImage(cv2.resize(frame, (300,300)), 1.0/255, (300,300), (0, 0, 0), swapRB=False, crop=False)
    # network에 넣어주기
    net.setInput(inpBlob)
    # 결과 받아오기
    output = net.forward()

    # output.shape[0] = 이미지 ID, [1] = 출력 맵의 높이, [2] = 너비
    H = output.shape[2]
    W = output.shape[3]
    # 키포인트 검출시 이미지에 그려줌
    points = []
    train_points = []
    for i in range(0,25):
        # 해당 신체부위 신뢰도 얻음.
        probMap = output[0, i, :, :]
 
        # global 최대값 찾기
        minVal, prob, minLoc, point = cv2.minMaxLoc(probMap)

        # 원래 이미지에 맞게 점 위치 변경
        x = (w * point[0]) / W
        y = (h * point[1]) / H

        # 키포인트 검출한 결과가 0.1보다 크면(검출한곳이 위 BODY_PARTS랑 맞는 부위면) points에 추가, 검출했는데 부위가 없으면 None으로    
        if prob > 0.3 :    
            if xmin > x :
                xmin = x
            if xmax <= x :
                xmax = x
            if ymin > y :
                ymin = y
            if ymax <= y :
                ymax = y
    xmax = xmax + 10
    xmin = xmin - 10
    ymax = ymax + 10
    ymin = ymin - 10
    xlen = xmax-xmin
    ylen = ymax-ymin
    print("Rectangle Done")
    for i in range(0,25):
        # 해당 신체부위 신뢰도 얻음.
        probMap = output[0, i, :, :]
 
        # global 최대값 찾기
        minVal, prob, minLoc, point = cv2.minMaxLoc(probMap)

        # 원래 이미지에 맞게 점 위치 변경
        x = (w * point[0]) / W
        y = (h * point[1]) / H

        # 키포인트 검출한 결과가 0.1보다 크면(검출한곳이 위 BODY_PARTS랑 맞는 부위면) points에 추가, 검출했는데 부위가 없으면 None으로    
        if prob > 0.3 :    
            #cv2.circle(image, (int(x), int(y)), 3, (0, 255, 255), thickness=-1, lineType=cv2.FILLED)       # circle(그릴곳, 원의 중심, 반지름, 색)
            #cv2.putText(image, "{}".format(i), (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, lineType=cv2.LINE_AA)
            points.append((int(x), int(y)))
            if i < 25 :
                train_points.append((x-xmin)/xlen)
                train_points.append((y-ymin)/ylen)
        else :
            points.append(None)
            if i < 25 :
                train_points.append(0)
                train_points.append(0)
    #train_points.append(0)
    print(len(train_points))
    sum_point = sum(train_points)
    numpy_points = numpy.asarray(train_points)

    print(numpy_points)

    imageC = frame
    
    loaded_model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
    test_X = numpy_points[0:50]
    if (test_X.ndim == 1):
        test_X = numpy.array([test_X])
    pred = loaded_model.predict(test_X)
    
    if sum_point == 0 :
        predict = "None Detected"
    else :
        if pred*100 > 50:
            imageC = cv2.rectangle(imageC,(int(xmin), int(ymin)),(int(xmax), int(ymax)),(0,0,255),1)
            predict = "FALL  " + str(int(pred*100)) + "%"
        else :
            imageC = cv2.rectangle(imageC,(int(xmin), int(ymin)),(int(xmax), int(ymax)),(0,0,255),1)
            predict = "STAND  " + str(100-int(pred*100)) + "%"

    detects=[]
    xpoint = xmin + 30
    ypoint = ymax - 30
    cv2.putText(imageC,predict,(30,30),cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255),1,cv2.LINE_AA)
    
    # 각 POSE_PAIRS별로 선 그어줌 (머리 - 목, 목 - 왼쪽어깨, ...)
    for pair in POSE_PAIRS:
        partA = pair[0]             # Head
        partA = BODY_PARTS[partA]   # 0
        partB = pair[1]             # Neck
        partB = BODY_PARTS[partB]   # 1
        #print(partA," 와 ", partB, " 연결\n")
        if points[partA] and points[partB]:
            cv2.line(imageC, points[partA], points[partB], (0, 255, 0), 2)
    cv2.imshow("Output-Keypoints",imageC)
    cv2.waitKey(1)
#vs.release()
cv2.destroyAllWindows()
#"/home/kwan/openpose/build/python"