# -*- coding: utf-8 -*-
"""
Created on Wed Nov 4 
@author: jongwon Kim 
         Deep.I Inc.
"""

import cv2

capture = cv2.VideoCapture(0)
# 영상 검출기
def videoDetector(cam,cascade,age_net,gender_net,MODEL_MEAN_VALUES,age_list,gender_list):

    while True:

        # 캡처 이미지 불러오기
        ret,img = cam.read()
        # 영상 압축
        try:
            img = cv2.resize(img,dsize=None,fx=1.0,fy=1.0)
        except: break

        img = cv2.flip(img, 1)
        
        # 그레이 스케일 변환
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) 
        # cascade 얼굴 탐지 알고리즘 
        results = cascade.detectMultiScale(gray,            # 입력 이미지
                                           scaleFactor= 1.1,# 이미지 피라미드 스케일 factor
                                           minNeighbors=5,  # 인접 객체 최소 거리 픽셀
                                           minSize=(20,20)  # 탐지 객체 최소 크기
                                           )

        for box in results:
            x, y, w, h = box
            face = img[int(y):int(y+h),int(x):int(x+h)].copy()
            blob = cv2.dnn.blobFromImage(face, 1, (227, 227), MODEL_MEAN_VALUES, swapRB=False)

            # gender detection
            gender_net.setInput(blob)
            gender_pred = gender_net.forward()
            gender = gender_pred.argmax()
            # Predict age
            age_net.setInput(blob)
            age_pred = age_net.forward()
            age = age_pred.argmax()
            
            info = str(gender_list[gender]) +' '+ age_list[age]

            cv2.rectangle(img, (x,y), (x+w, y+h), (255,255,255), thickness=2)
            cv2.putText(img,info,(x,y-15),0, 0.5, (0, 255, 0), 1)
            

         # 영상 출력
        
        cv2.imshow('facenet',img)

        if cv2.waitKey(1) > 0: 

            break

# 사진 검출기


# 얼굴 탐지 모델 가중치
cascade_filename = 'haarcascade_frontalface_alt.xml'
# 모델 불러오기
cascade = cv2.CascadeClassifier(cascade_filename)

MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)

age_net = cv2.dnn.readNetFromCaffe(
	'deploy_age.prototxt',
	'age_net.caffemodel')

gender_net = cv2.dnn.readNetFromCaffe(
	'deploy_gender.prototxt',
	'gender_net.caffemodel')

age_list = ['(0 ~ 2)','(4 ~ 6)','(8 ~ 12)','(15 ~ 20)',
            '(25 ~ 32)','(38 ~ 43)','(48 ~ 53)','(60 ~ 100)']
gender_list = ['Male', 'Female']


# 영상 탐지기
while cv2.waitKey(1) != ord('q'):  #아무 키나 누르면 끝난다.
    #웹캠으로부터 영상 가져옴
    hasFrame, frame = capture.read()  
    
    #영상이 커서 느리면 사이즈를 줄이자
    frame=cv2.resize(frame,dsize=(320,240),interpolation=cv2.INTER_AREA)
    
    #웹캠으로부터 영상을 가져올 수 없으면 웹캠 중지
    if not hasFrame:
        cv2.waitKey()
        break


    # 결과 받아오기
    videoDetector(capture,cascade,age_net,gender_net,MODEL_MEAN_VALUES,age_list,gender_list )

capture.release()  #카메라 장치에서 받아온 메모리 해제
cv2.destroyAllWindows() #모든 윈도우 창 닫음