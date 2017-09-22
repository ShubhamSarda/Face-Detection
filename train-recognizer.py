import os
import cv2
import numpy as np
from PIL import Image

rec =  cv2.face.LBPHFaceRecognizer_create()

path = "dataset-generator"

def getImages(path):
    imagePaths = [os.path.join(path,f) for f in os.listdir(path)]
    faces = []
    IDs = []
    for i in imagePaths: #i = ImagePath
        faceImg = Image.open(i).convert('L')
        
        faceNp = np.array(faceImg,'uint8')
        ID = int(os.path.split(i)[-1].split('.')[1])
        faces.append(faceNp)
        print(ID)
        IDs.append(ID)
        cv2.imshow("Training Mode",faceNp)
        cv2.waitKey(10)
    return IDs, faces

Ids, faces = getImages(path)
rec.train(faces,np.array(Ids))
rec.write("training-data/trainingdata.yml")
cv2.destroyAllWindows()

#Useful Link - 
#http://docs.opencv.org/2.4/modules/contrib/doc/facerec/facerec_tutorial.html
#https://stackoverflow.com/questions/13576161/convert-opencv-image-into-pil-image-in-python-for-use-with-zbar-library