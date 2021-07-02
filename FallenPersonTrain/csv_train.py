#!/usr/bin/env python
# -*- coding: utf-8 -*-

# fashion_pose.py : MPII를 사용한 신체부위 검출
import cv2
import csv
import glob

# MPII에서 각 파트 번호, 선으로 연결될 POSE_PAIRS
'''
BODY_PARTS = { "Head": 0, "Neck": 1, "RShoulder": 2, "RElbow": 3, "RWrist": 4,
                "LShoulder": 5, "LElbow": 6, "LWrist": 7, "RHip": 8, "RKnee": 9,
                "RAnkle": 10, "LHip": 11, "LKnee": 12, "LAnkle": 13, "Chest": 14,
                "Background": 15 }
'''

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

INPUT_CSV = ["survived", "Nose_x","Nose_y", "Neck_x", "Neck_y","RShoulder_x","RShoulder_y","RElbow_x","RElbow_y","RWrist_x","RWrist_y","LShoulder_x", "LShoulder_y",
            "LElbow_x","LElbow_y" ,"LWrist_x", "LWrist_y","MidHip_x", "MidHip_y","RHip_x","RHip_y" ,"RKnee_x", "RKnee_y",
            "RAnkle_x", "RAnkle_y","LHip_x", "LHip_y","LKnee_x","LKnee_y","LAnkle_x","LAnkle_y","REye_x", "REye_y",
            "LEye_x","LEye_y","REar_x", "REar_y","LEar_x", "LEar_y","LBigToe_x", "LBigToe_y","LSmallToe_x","LSmallToe_y" ,"LHeel_x", "LHeel_y","RBigToe_x", "RBigToe_y",
            "RSmallToe_x","RSmallToe_y" ,"RHeel_x","RHeel_y"]
    
# 각 파일 path
protoFile = "/home/kwan/openpose/models/pose/body_25/pose_deploy.prototxt"
weightsFile = "/home/kwan/openpose/models/pose/body_25/pose_iter_584000.caffemodel"
 
# 위의 path에 있는 network 불러오기
net = cv2.dnn.readNetFromCaffe(protoFile, weightsFile)


# 여기부터 시작
pathfall = glob.glob("/home/kwan/PoseTrain/pic/Capture/train/fall/*")
pathstand = glob.glob("/home/kwan/PoseTrain/pic/Capture/train/stand/*")
#pathfall = glob.glob("/home/kwan/PoseTrain/pic/train/fall/*")
#pathstand = glob.glob("/home/kwan/PoseTrain/pic/train/stand/*")

fallcnt = 0
standcnt = 0

trains = []
for img in pathfall :
    # 이미지 읽어오기
    xmin = 10000
    xmax = 0
    ymin = 10000
    ymax = 0
    image = cv2.imread(img)
    # frame.shape = 불러온 이미지에서 height, width, color 받아옴
    imageHeight, imageWidth, _ = image.shape
    # network에 넣기위해 전처리
    inpBlob = cv2.dnn.blobFromImage(image, 1.0 / 255, (imageWidth, imageHeight), (0, 0, 0), swapRB=False, crop=False)
    # network에 넣어주기
    net.setInput(inpBlob)
    # 결과 받아오기
    output = net.forward()

    # output.shape[0] = 이미지 ID, [1] = 출력 맵의 높이, [2] = 너비
    H = output.shape[2]
    W = output.shape[3]
    fallcnt = fallcnt + 1
    print("이미지 ID : ", len(output[0]), ", H : ", output.shape[2], ", W : ",output.shape[3]) # 이미지 ID
    print("진행 게이지 : ", fallcnt, "/",len(pathfall)) # 이미지 ID

    # 키포인트 검출시 이미지에 그려줌
    points = []
    train_points = []

    for i in range(0,25):
        # 해당 신체부위 신뢰도 얻음.
        probMap = output[0, i, :, :]
 
        # global 최대값 찾기
        minVal, prob, minLoc, point = cv2.minMaxLoc(probMap)

        # 원래 이미지에 맞게 점 위치 변경
        x = (imageWidth * point[0]) / W
        y = (imageHeight * point[1]) / H

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
    print("Lectangle Done")
    for i in range(0,25):
        # 해당 신체부위 신뢰도 얻음.
        probMap = output[0, i, :, :]
 
        # global 최대값 찾기
        minVal, prob, minLoc, point = cv2.minMaxLoc(probMap)

        # 원래 이미지에 맞게 점 위치 변경
        x = (imageWidth * point[0]) / W
        y = (imageHeight * point[1]) / H

        # 키포인트 검출한 결과가 0.1보다 크면(검출한곳이 위 BODY_PARTS랑 맞는 부위면) points에 추가, 검출했는데 부위가 없으면 None으로    
        if prob > 0.3 :    
            cv2.circle(image, (int(x), int(y)), 3, (0, 255, 255), thickness=-1, lineType=cv2.FILLED)       # circle(그릴곳, 원의 중심, 반지름, 색)
            cv2.putText(image, "{}".format(i), (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, lineType=cv2.LINE_AA)
            points.append((int(x), int(y)))
            '''
            if i < 25 :
                if x != xmin :
                    train_points.append((x-xmin)/xlen)
                else : 
                    train_points.append(0)
                if y != ymin : 
                    train_points.append((y-ymin)/ylen)
                else : 
                    train_points.append(0)
                    '''
            if i < 25 :
                train_points.append((x-xmin)/xlen)
                train_points.append((y-ymin)/ylen)
        else :
            points.append(None)
            if i < 25 :
                train_points.append(0)
                train_points.append(0)
    train_points.append(1)
    trains.append(train_points)

for img in pathstand :
    # 이미지 읽어오기
    xmin = 10000
    xmax = 0
    ymin = 10000
    ymax = 0
    image = cv2.imread(img)
    # frame.shape = 불러온 이미지에서 height, width, color 받아옴
    imageHeight, imageWidth, _ = image.shape
    # network에 넣기위해 전처리
    inpBlob = cv2.dnn.blobFromImage(image, 1.0 / 255, (imageWidth, imageHeight), (0, 0, 0), swapRB=False, crop=False)
    # network에 넣어주기
    net.setInput(inpBlob)
    # 결과 받아오기
    output = net.forward()

    # output.shape[0] = 이미지 ID, [1] = 출력 맵의 높이, [2] = 너비
    H = output.shape[2]
    W = output.shape[3]

    standcnt = standcnt + 1
    print("이미지 ID : ", len(output[0]), ", H : ", output.shape[2], ", W : ",output.shape[3]) # 이미지 ID
    print("진행 게이지 : ", standcnt, "/",len(pathstand)) # 이미지 ID

    # 키포인트 검출시 이미지에 그려줌
    points = []
    train_points = []

    for i in range(0,25):
        # 해당 신체부위 신뢰도 얻음.
        probMap = output[0, i, :, :]
 
        # global 최대값 찾기
        minVal, prob, minLoc, point = cv2.minMaxLoc(probMap)

        # 원래 이미지에 맞게 점 위치 변경
        x = (imageWidth * point[0]) / W
        y = (imageHeight * point[1]) / H

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

    for i in range(0,25):
        # 해당 신체부위 신뢰도 얻음.
        probMap = output[0, i, :, :]
        # global 최대값 찾기
        minVal, prob, minLoc, point = cv2.minMaxLoc(probMap)

        # 원래 이미지에 맞게 점 위치 변경
        x = (imageWidth * point[0]) / W
        y = (imageHeight * point[1]) / H

        # 키포인트 검출한 결과가 0.1보다 크면(검출한곳이 위 BODY_PARTS랑 맞는 부위면) points에 추가, 검출했는데 부위가 없으면 None으로    
        if prob > 0.4 :    
            cv2.circle(image, (int(x), int(y)), 3, (0, 255, 255), thickness=-1, lineType=cv2.FILLED)       # circle(그릴곳, 원의 중심, 반지름, 색)
            cv2.putText(image, "{}".format(i), (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, lineType=cv2.LINE_AA)
            points.append((int(x), int(y)))
            '''
            if i < 25 :
                if x != xmin :
                    train_points.append((x-xmin)/xlen)
                else : 
                    train_points.append(0)
                if y != ymin : 
                    train_points.append((y-ymin)/ylen)
                else : 
                    train_points.append(0)
                    '''
            if i < 25 :
                train_points.append((x-xmin)/xlen)
                train_points.append((y-ymin)/ylen)
        else :
            points.append(None)
            if i < 25 :
                train_points.append(0)
                train_points.append(0)
    train_points.append(0)
    trains.append(train_points)

print(len(trains))
with open('tr_json.csv', 'w', newline='') as csvfile:
    wr = csv.writer(csvfile)
    #wr.writerow(INPUT_CSV)
    for i in trains:
        wr.writerow(i)

cv2.destroyAllWindows()
