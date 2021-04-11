import cv2
import numpy as np
from deepface import DeepFace


video = cv2.VideoCapture(1)

while True:
    ret, frame = video.read() # ret is a boolean that checks if the camera is on or not, frame is a variable the camera feed is being saved in
    cv2.imshow('frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break




video.release()
cv2.destroyAllWindows()



