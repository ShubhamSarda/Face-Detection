import cv2
import numpy as np
from PIL import Image
import os

rec = cv2.face.LBPHFaceRecognizer_create()
rec.read("training-data/trainingdata.yml")

detect = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
cam = cv2.VideoCapture(0)
font = cv2.FONT_HERSHEY_SIMPLEX
#font = cv2.InitFont(cv2.cv.CV_FONT_HERSHEY_SIMPLEX, 1, 1, 0, 1, 1)

while True:
    check, img = cam.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = detect.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5)
    for (x,y,w,h) in faces:
        cv2.rectangle(img,(x,y),(x+w,y+h),(0,0,255),2)
        Id, conf = rec.predict(gray[y:y+h,x:x+w])
        print(conf)
        print(Id)
        if(conf<50):
            if(Id == 1):
                Id = "Shubham"
            elif(Id==2):
                Id="Harshit"
            else:
                Id="You Are Not In Our System"
        else:
            Id="Low Confidence Level, Try Again!"
        cv2.putText(img,str(Id + " " + str(conf)),(x,y+h), font, 4,(255,255,255),2,cv2.LINE_AA)
	
	#Resize Frame.
    iw=int(img.shape[1]/2)
    ih=int(img.shape[0]/2)
    rimg = cv2.resize(img,(iw,ih))
	
    cv2.imshow("Window Screen", rimg)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cam.release()
cv2.destroyAllWindows()

#Important Doubts And answers.
#https://stackoverflow.com/questions/20518632/importerror-numpy-core-multiarray-failed-to-import
#http://www.lfd.uci.edu/~gohlke/pythonlibs/#opencv
#https://stackoverflow.com/questions/28568070/filename-whl-is-not-supported-wheel-on-this-platform
#http://answers.opencv.org/question/171886/using-trained-file-for-predictions/
#https://github.com/opencv/opencv_contrib/issues/1267
