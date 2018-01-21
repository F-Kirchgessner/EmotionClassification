from imutils.video import VideoStream
import cv2
import dlib
import numpy as np

import torch
from torch.autograd import Variable


def faceAlignment(img, detector, shapePredictor):
    # Find bounding boxes
    dets = detector(img, 0)

    num_faces = len(dets)
    if num_faces == 0:
        return [], []

    # Find face landmarks we need to do the alignment.
    faces = dlib.full_object_detections()
    frames = []
    for d in dets:
        faces.append(shapePredictor(img, d))
        frames.append([(d.left(), d.top()), (d.right(), d.bottom())])

    # Get the aligned face images
    # Optionally:
    # images = dlib.get_face_chips(img, faces, size=160, padding=0.25)
    imagesAligned = []
    images = dlib.get_face_chips(img, faces, size=320)
    for i, image in enumerate(images):
        #cv_bgr_img = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        image = cv2.resize(image, (256, 256), interpolation = cv2.INTER_LINEAR)
        imagesAligned.append(image)
        
    #cv2.destroyAllWindows()
    
    return imagesAligned, frames

 
# Captures a single image from the camera and returns it in PIL format
def getImage(camera):
    # read is the easiest way to get a full image out of a VideoCapture object.
    retval, im = camera.read()
    return im
    
    
def takeSingleImage(cameraPort, adjustmentFrames=30):
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
    

def runSingleImage(cameraPort, modelPath, predictorPath):
    # Preload
    model = torch.load(modelPath)
    model.eval()
    detector = dlib.get_frontal_face_detector()
    shapePredictor = dlib.shape_predictor(predictorPath)

    image = takeSingleImage(cameraPort)

    imagesAligned, frames = faceAlignment(image, detector, shapePredictor)

    runCNN(imagesAligned[0], model)

    cv2.imshow('Emotion Classification',imagesAligned[0])
    cv2.waitKey(0)
    
    
def runRealtimeStream(cameraPort, modelPath, predictorPath):
    # Preload
    model = None
    try:
        model = torch.load(modelPath)
    except:
        model = torch.load(modelPath, map_location=lambda storage, loc: storage)

    model.eval()
    detector = dlib.get_frontal_face_detector()
    shapePredictor = dlib.shape_predictor(predictorPath)

    # Display window
    cv2.namedWindow("Emotion Classification")
    vc = VideoStream(cameraPort, usePiCamera=0).start()

    while True:
        image = vc.read()
        
        imagesAligned, frames = faceAlignment(image, detector, shapePredictor)
        
        if imagesAligned != []:
            results = runCNN(imagesAligned[0], model)
            
            # Add rectangle and text
            cv2.rectangle(image, frames[0][0], frames[0][1], (1, 1, 255))
            text = createDisplayText(results)
            cv2.putText(image, text[0], (frames[0][1][0], frames[0][0][1] + 25), 0, 0.7, (1, 1, 255))
            cv2.putText(image, text[1], (frames[0][1][0], frames[0][0][1] + 45), 0, 0.7, (1, 1, 255))
            cv2.putText(image, text[2], (frames[0][1][0], frames[0][0][1] + 65), 0, 0.7, (1, 1, 255))
            
        cv2.imshow("Emotion Classification", image)
        
        # ESC to exit
        key = cv2.waitKey(20)
        if key == 27:
            break
            
    cv2.destroyWindow("Emotion Classification")


def createDisplayText(results):
    emotions = {0: 'neutral', 1: 'anger', 2: 'contempt', 3: 'disgust', 4: 'fear', 5: 'happy', 6: 'sadness', 7: 'surprise'}
    totalScore = sum([res[1] for res in results])
    text = []

    text.append("%s: %0.1f%%" %(emotions[results[0][0]], (results[0][1] / totalScore) * 100))
    text.append("%s: %0.1f%%" %(emotions[results[1][0]], (results[1][1] / totalScore) * 100))
    text.append("%s: %0.1f%%" %(emotions[results[2][0]], (results[2][1] / totalScore) * 100))
    
    return text


def runCNN(img, model):
    img = img[np.newaxis,:,:,:]
    img = Variable(torch.Tensor(img))
    img = img.permute(0,3,1,2)
    
    out = model.forward(img).data.numpy()

    #Get results
    emotions = np.argsort(-out)[0]
    percentages = -np.sort(-out)[0]
    results = []

    for i in range(len(emotions)):
        results.append((emotions[i], percentages[i] - percentages[-1]))
    
    return results


if __name__ == "__main__":
    # WSettings
    cameraPort = 1
    modelPath = "models/Basic_300.model"
    predictorPath = "data/shape_predictor_68_face_landmarks.dat"

    #runSingleImage(cameraPort, modelPath, predictorPath, emotions)
    runRealtimeStream(cameraPort, modelPath, predictorPath)



