#Importing Libraries
import time
import cv2
from cv2 import VideoCapture        #for image processing functions
import numpy as np                  #for array related functions
import dlib                         #for face landmark detection
import math                         #for basic mathematical opera
from math import hypot              #for finding the euclidean distance
import playsound                    #for playing the alarm sound
from threading import Thread        #for creating and starting threads to implement multithreading
#selecting the font style for the alert message
font= cv2.FONT_HERSHEY_SIMPLEX
#function for the sound to be played for alarm
ALARM_ON = False
def sound_alarm_sleeping(path):
	# play an alarm sound
	playsound.playsound("D:/b.tech/6th sem/open lab/Smart accident prevention/major-project-main/1649424327956.wav")
#initializing the camera for capturing the instances
cap = VideoCapture(0)
#initializing the face detector
detector = dlib.get_frontal_face_detector()
#intializing the landmark detector
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
#calculating the midpoint for finding the points on the upper and lower lash line
def midpoint(p1 ,p2):
    return int((p1.x + p2.x)/2), int((p1.y + p2.y)/2)
#reading video into variable 'frame' read using videostream module
while True:
    _, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#detecting the co-ordinates for making a rectangle around the face in the video 
    faces = detector(gray)
    #detected face in faces array
    for face in faces:
        x, y = face.left(), face.top()
        x1, y1 = face.right(), face.bottom()
        cv2.rectangle(frame, (x, y), (x1, y1), (0, 255, 0), 2)

        #marks the landmark points on the eyes
        landmarks1 = predictor(gray, face)
        left_point1 = (landmarks1.part(36).x, landmarks1.part(36).y)
        right_point1 = (landmarks1.part(39).x, landmarks1.part(39).y)
        center_top1 = midpoint(landmarks1.part(37), landmarks1.part(38))
        center_bottom1 = midpoint(landmarks1.part(41), landmarks1.part(40))

        #hor_line1 = cv2.line(frame, left_point1, right_point1, (0, 255, 0), 2)
        #ver_line1 = cv2.line(frame, center_top1, center_bottom1, (0, 255, 0), 2)

        #marks the landmark points on the eyes
        landmarks2 = predictor(gray, face)
        left_point2 = (landmarks2.part(42).x, landmarks2.part(42).y)
        right_point2 = (landmarks2.part(45).x, landmarks2.part(45).y)
        center_top2 = midpoint(landmarks2.part(43), landmarks2.part(44))
        center_bottom2 = midpoint(landmarks2.part(47), landmarks2.part(46))

        #hor_line2 = cv2.line(frame, left_point2, right_point2, (0, 255, 0), 2)
        #ver_line2 = cv2.line(frame, center_top2, center_bottom2, (0, 255, 0), 2)

        #calculating the euclidean distance b/w point A and point B using linear algebra module of numpy
        hor_line_length = hypot((left_point1[0] - right_point1[0]), (left_point1[1] - right_point1[1]))
        ver_line_length = hypot((center_top1[0] - center_bottom1[0]), (center_top1[1] - center_bottom1[1]))
        
        ratio= hor_line_length//ver_line_length
        #checking the driver's state (sleeping/Drowsy/Active)
        if ratio==5.0:
            cv2.putText(frame, "SLEEPING!!!", (275,30), font, 1, (0, 0, 255),2)
            ALARM_ON = True
            t = Thread(target=sound_alarm_sleeping, args=(["alarm"],))
            t.deamon = True
            t.start()
            time.sleep(2)
        elif ratio ==4.0:
            cv2.putText(frame, "DROWSY!!!", (275,30), font, 1, (0, 0, 255),2)
        elif ratio <=3.0:
            cv2.putText(frame, "ACTIVE!!!", (275,30), font, 1, (0, 0, 255),2)        
        #highlighting the left and right right eyes by the grren line    
        left_eye_region = np.array([(landmarks1.part(36).x, landmarks1.part(36).y),
                                    (landmarks1.part(37).x, landmarks1.part(37).y),
                                    (landmarks1.part(38).x, landmarks1.part(38).y),
                                    (landmarks1.part(39).x, landmarks1.part(39).y),
                                    (landmarks1.part(40).x, landmarks1.part(40).y),
                                    (landmarks1.part(41).x, landmarks1.part(41).y)], np.int32)

        cv2.polylines(frame, [left_eye_region], True, (0, 255, 0), 2)
        
        right_eye_region = np.array([(landmarks2.part(42).x, landmarks2.part(42).y),
                                     (landmarks2.part(43).x, landmarks2.part(43).y),
                                     (landmarks2.part(44).x, landmarks2.part(44).y),
                                     (landmarks2.part(45).x, landmarks2.part(45).y),
                                     (landmarks2.part(46).x, landmarks2.part(46).y),
                                     (landmarks2.part(47).x, landmarks2.part(47).y)], np.int32)
        
        cv2.polylines(frame, [right_eye_region], True, (0, 255, 0), 2)
            
        min1_x = np.min(left_eye_region[:, 0])
        max1_x = np.max(left_eye_region[:, 0])
        min1_y = np.min(left_eye_region[:, 1])
        max1_y = np.max(left_eye_region[:, 1])
        
        eye1 = frame[min1_y: max1_y, min1_x: max1_x]
        
        min2_x = np.min(right_eye_region[:, 0])
        max2_x = np.max(right_eye_region[:, 0])
        min2_y = np.min(right_eye_region[:, 1])
        max2_y = np.max(right_eye_region[:, 1])
        
        eye2 = frame[min2_y: max2_y, min2_x: max2_x]
        
        #eye1 = cv2.resize(eye1, None, fx=5, fy=5)
        
        #eye2 = cv2.resize(eye2, None, fx=5, fy=5)
        
        #cv2.imshow("Eye1", eye1)
        
        #cv2.imshow("Eye2", eye2)

        print(ratio)
        #printing the EAR on the video frame along with the driver's state
        cv2.putText(frame, "EAR: " + str(ratio), (10,30), font, 1, (0, 0, 255), 2)
                       
    cv2.imshow("Frame", frame)

    key = cv2.waitKey(1)
    if key == 27:
        break

cap.release()
cv2.destroyAllWindows()