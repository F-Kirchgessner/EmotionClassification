from imutils.video import VideoStream
from imutils import face_utils
import datetime
import argparse
import imutils
import time
import dlib
import cv2

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('data/shape_predictor_68_face_landmarks.dat')

# initialize the video stream and allow the cammera sensor to warmup
vs = VideoStream(usePiCamera=0).start()
time.sleep(2.0)

while True:
    frame = vs.read()
    rects = detector(frame, 0)
    for rect in rects:
        cv2.rectangle(frame, (rect.left(), rect.top()), (rect.right(), rect.bottom()), (1, 1, 255))

    cv2.imshow("Frame", frame)
    # ESC to exit
    key = cv2.waitKey(20)
    if key == 27:
        break
