import cv2
import numpy as np
import matplotlib.pyplot as plt
from deepface import DeepFace


video = cv2.VideoCapture(0)
if not video.isOpened():
    print("Unable to open webcam...")

while True:
    ret, frame = video.read() # ret is a boolean that checks if the camera is on or not, frame is a variable the camera feed is being saved in
 

    predictions = DeepFace.analyze(frame)

    start_point = (predictions['region']['x'], predictions['region']['y'])
    end_point = (predictions['region']['x'] + predictions['region']['w'], predictions['region']['y'] + predictions['region']['h'])
    box_colour = (0,255,0)
    text_colour = (255,0,0)
    thickness = 8
    font = cv2.FONT_HERSHEY_DUPLEX

    gender_start_point = (int((predictions['region']['x']/3)),int(((predictions['region']['y']*0.75)*0.25)))
    age_start_point = (int((predictions['region']['x']/3)),int(((predictions['region']['y']*0.75)*0.5)))
    race_start_point = (int((predictions['region']['x']/3)),int(((predictions['region']['y']*0.75)*0.75)))
    emotion_start_point = (int((predictions['region']['x']/3)),int(((predictions['region']['y']*0.75)/1)))

    frame = cv2.rectangle(frame, start_point, end_point, box_colour, thickness)
    cv2.putText(frame, "Emotion: " + predictions['dominant_emotion'], emotion_start_point, font, 0.5, text_colour, 1)
    cv2.putText(frame, "Race: " + predictions['dominant_race'], race_start_point, font, 0.5, text_colour, 1)
    cv2.putText(frame, "Gender: " + predictions['gender'], gender_start_point, font, 0.5, text_colour, 1)
    cv2.putText(frame, "Age: " + str(predictions['age']), age_start_point, font, 0.5, text_colour, 1)

    cv2.imshow("Facial Recognition Analysis",frame)


    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video.release()
cv2.destroyAllWindows()
