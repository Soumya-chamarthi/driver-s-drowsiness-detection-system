# import the necessary packages
from scipy.spatial import distance as dist   #for calculating distance
from imutils.video import FileVideoStream    #
from imutils.video import VideoStream        #
from threading import Thread                 #for creating and starting threads to implement multithreading
import numpy as np                           #for array related functions
from imutils import face_utils               #for face detection
import playsound                             #for playing the alarm sound
import imutils                               #for image processing
import time                                  #for working with time
import dlib                                  #for face landmark detection
import cv2                                   #for image processing functons
#function for the sound to be played for alarm
def sound_alarm(path):
	# play an alarm sound
	playsound.playsound("C:/Users/Soumya/OneDrive/Desktop/Alarm.wav")
#function for the lip distance 
def lip_distance(shape):
    top_lip = shape[50:53]
    top_lip = np.concatenate((top_lip, shape[61:64]))

    low_lip = shape[56:59]
    low_lip = np.concatenate((low_lip, shape[65:68]))
    #calculating the mean of top lip and low lip
    top_mean = np.mean(top_lip, axis=0)
    low_mean = np.mean(low_lip, axis=0)
    #calculating the difference b/w the top mean and low mean
    distance = abs(top_mean[1] - low_mean[1])
    return distance
#function for detecting the face and starting the video stream
def main():

    YAWN_THRESH = 35
    ALARM_ON = False
    #detecting the face
    detector = dlib.get_frontal_face_detector()
    #marking the landmark point arounf the lips
    predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')

    #strting the video streaming
    vs= VideoStream().start()
    time.sleep(1.0)
#reading video into variable 'frame' read using videostream module
    while True:
        frame = vs.read()
        frame = imutils.resize(frame, width=450)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # detect faces in the grayscale frame
        rects = detector(gray, 0)

        # loop over the face detections
        for rect in rects:
            shape = predictor(gray, rect)
            shape = face_utils.shape_to_np(shape)
            #calling the lip_distance function to calculate the distance of the shape
            distance = lip_distance(shape)



            lip = shape[48:60]
            #for drawing the lines on the detected lip area to highlight it
            cv2.drawContours(frame, [lip], -1, (0, 255, 0), 1)
            #check
            if (distance > YAWN_THRESH):
                cv2.putText(frame, "Yawn Alert", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

                if not ALARM_ON:
                    ALARM_ON = True
                    t = Thread(target=sound_alarm,
                               args=(["alarm"],))
                    t.deamon = True
                    t.start()

            else:
                ALARM_ON = False

            cv2.putText(frame, "YAWN: {:.2f}".format(distance), (300, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        # show the frame
        cv2.imshow("Frame", frame)
        key = cv2.waitKey(1) & 0xFF

        # if the `q` key was pressed, break from the loop
        if key == ord("q"):
            break

    # do a bit of cleanup
    cv2.destroyAllWindows()
    vs.stop()

if __name__ == '__main__':
    main()
