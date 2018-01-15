import cv2
import sys
import dlib
import numpy as np
import cnn

def faceAlignment(img, predictor_path = "shape_predictor_68_face_landmarks.dat"):
	# Load all the models we need: a detector to find the faces, a shape predictor
	# to find face landmarks so we can precisely localize the face
	detector = dlib.get_frontal_face_detector()
	sp = dlib.shape_predictor(predictor_path)

	# Ask the detector to find the bounding boxes of each face. The 1 in the
	# second argument indicates that we should upsample the image 1 time. This
	# will make everything bigger and allow us to detect more faces.
	dets = detector(img, 0)

	num_faces = len(dets)
	if num_faces == 0:
		return [], []

	# Find the 5 face landmarks we need to do the alignment.
	faces = dlib.full_object_detections()
	frames = []
	for d in dets:
		faces.append(sp(img, d))
		frames.append([(d.left(), d.top()), (d.right(), d.bottom())])

	# Get the aligned face images
	# Optionally: 
	# images = dlib.get_face_chips(img, faces, size=160, padding=0.25)
	imagesAligned = []
	images = dlib.get_face_chips(img, faces, size=320)
	for i, image in enumerate(images):
		#cv_bgr_img = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
		image = cv2.resize(image, (200, 200), interpolation = cv2.INTER_LINEAR)
		imagesAligned.append(image)
		
	#cv2.destroyAllWindows()
	
	return imagesAligned, frames
	# It is also possible to get a single chip
	#image = dlib.get_face_chip(img, faces[0])
	#cv_bgr_img = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
	#cv2.imshow('image',cv_bgr_img)
	#cv2.waitKey(0)

 
# Captures a single image from the camera and returns it in PIL format
def getImage(camera):
	# read is the easiest way to get a full image out of a VideoCapture object.
	retval, im = camera.read()
	return im
	
	
def takeSingleImage(cameraPort, adjustmentFrames = 30):
	# Initialize camera
	camera = cv2.VideoCapture(cameraPort)

	# Discard frames while camera adjusts to light condition
	for i in range(adjustmentFrames):
		temp = getImage(camera)
		
	print("Taking image...")
	# Take the actual image we want to keep
	image = getImage(camera)

	del(camera)
	return image
	

def runSingleImage(cameraPort):
	image = takeSingleImage(cameraPort)

	imagesAligned, frames = faceAlignment(image)

	cnn.runCNN(imagesAligned[0])

	cv2.imshow('Emotion Classification',imagesAligned[0])
	cv2.waitKey(0)
	
	
def runRealtimeStream(cameraPort):
	cv2.namedWindow("Emotion Classification")
	vc = cv2.VideoCapture(cameraPort)

	if vc.isOpened(): # try to get the first frame
		rval, image = vc.read()
	else:
		rval = False

	while rval:
		rval, image = vc.read()
		
		imagesAligned, frames = faceAlignment(image)
		
		if imagesAligned != []:
			results = cnn.runCNN(imagesAligned[0])
			
			# Add rectangle and text
			cv2.rectangle(image, frames[0][0], frames[0][1], (1, 1, 255))
			text = "%s" %(results[0][0])
			cv2.putText(image, text, (frames[0][1][0], frames[0][0][1] + 25), 0, 1, (1, 1, 255))
			
		cv2.imshow("Emotion Classification", image)
		
		# ESC to exit
		key = cv2.waitKey(20)
		if key == 27:
			break
			
	cv2.destroyWindow("Emotion Classification")


# Which camera
cameraPort = 0

#runSingleImage(cameraPort)
runRealtimeStream(cameraPort)



