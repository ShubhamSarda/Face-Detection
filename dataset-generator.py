import cv2
import numpy as np

faceDetect = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

cam = cv2.VideoCapture(0) 
#Video Souce Link Or In-build Webcam Number. 
#Use "cam = cv2.VideoCapture(0)" For Webcam.

id = input("Enter User ID - ")
name = input("Enter Name - ")
sample = int(0)

while True:
    check, img = cam.read()
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    faces = faceDetect.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5)
    for (x,y,w,h) in faces:
        sample = sample+1
        cv2.imwrite("dataset/User." + str(id) + "." + str(sample) + "." + str(name) + ".jpg", gray[y:y+h,x:x+w])
        cv2.rectangle(img,(x,y),(x+w,y+h),(0,0,255),2)
        cv2.waitKey(1)
    if sample == 50:
        break
	
	#Re-Sizing Frame.
    iw=int(img.shape[1]/3)
    ih=int(img.shape[0]/3)
    rimg = cv2.resize(img,(iw,ih)) 
	
	#Display Result Using Resized Frame.
    cv2.imshow("Window Screen", rimg) 

    if cv2.waitKey(1) == ord('q'):
        break

cam.release()
cv2.destroyAllWindows()
