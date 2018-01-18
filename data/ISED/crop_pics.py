"""
put this file into data/ISED/.
run this file to take ISED face pics from Bilder folder, crop the faces out of them and put them into pics folder.
"""
import sys
import dlib
import cv2
import numpy as np
import os

predictor_path = "../shape_predictor_68_face_landmarks.dat"

detector = dlib.get_frontal_face_detector()
sp = dlib.shape_predictor(predictor_path)

#record errors
errors = []

#check for pics directory 
if os.path.exists('pics/') != True:
    os.mkdir('pics/')

filenames = list(os.listdir('Bilder/'))
for filename in filenames:
    face_input_path = 'Bilder/' + filename

    # 0=load greyscale
    img = cv2.imread(face_input_path, 0)
    dets = detector(img, 0)

    faces = dlib.full_object_detections()
    for detection in dets:
        faces.append(sp(img, detection))

    # fake RBG because dlib doesnt like grey people
	# check for empty list	
    if faces:
       	images = dlib.get_face_chips(cv2.cvtColor(img, cv2.COLOR_GRAY2BGR), faces, size=256)
    else:
        errors.append(filename)
    for i, image in enumerate(images):
        file_index = int(filename.split('.')[0])
        if i > 0:
            face_output_path = 'pics/%04d_%d.jpg' % (file_index, i)
        else:
            face_output_path = 'pics/%04d.jpg' % (file_index)

        # 7=convert RBG back to Gray
        cv2.imwrite(face_output_path, cv2.cvtColor(image, 7))
        print(face_input_path, '->', face_output_path)


cv2.destroyAllWindows()
print(errors)
print('Done.')
