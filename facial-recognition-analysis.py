import cv2
import numpy as np
import matplotlib.pyplot as plt
from deepface import DeepFace

img = cv2.imread('testfaces\\asian_sad_face.png')
plt.imshow(img)
plt.show()




'''
video = cv2.VideoCapture(0)

while True:
    ret, frame = video.read() # ret is a boolean that checks if the camera is on or not, frame is a variable the camera feed is being saved in
    cv2.imshow('frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break




video.release()
cv2.destroyAllWindows()
'''
