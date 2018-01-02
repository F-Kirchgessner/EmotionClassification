# EmotionClassification
 
## Introduction
State-of-the-art face detection algorithms have become widely available, which raises the desire to be able to recognize and classify emotions as well. In this project, we seek to further this field of interest. Our goal is to build a pipeline that receives video sequences as input and extracts the faces at regular time-steps to classify the emotions of those faces. Finally, overlaying the video with boundary boxes and emotion classification of the faces on a nearly frame-by-frame basis.

### Related Works
* We plan on using an implementation from the open source project OpenFace for recognizing faces in images. As this project is based on Google's FaceNet paper, Google's paper is seen as state of the art for face recognition. 
* The winner of the Emotion Recognition in the Wild Challenge 2016 is currently one of the best models for video based emotion recognition and can be considered as state of the art for this topic.

## Dataset
We are working with already existing datasets, that provide the labels themselves. Those datasets take a picture as an input and give the classified emotion as output. The VGG-Face Dataset is used to pretrain our network. For the video sequences, we consider using the EmotiW2016 dataset.

## Methodology
Using transfer learning, we are going to build our emotion classification network architecture on top of pre-trained models, e.g. VGG-Face. Initially, our goal is classifying a dataset of fully frontal faces in ideal lighting conditions. If successful, we will continue to train on faces in non perfect environments.
The next step involves the segmentation of faces in full body images with the framework OpenFace. By extracting individual faces we can pass images similar to our training data to the classification network. Finally, the input images will be single frames of an input video.
We intend to use the Google Cloud Platform to train our network on the GPU.


## Outcome
We seek to extract faces from the input video frames and classify their emotions with our convolutional neural network. Configuring the network architecture on top of a pretrained model and training the network will be the core of this project. In the end, we want to output the input video with boundary boxes around the faces with frame-by-frame classified emotions.

Additionally, we could try to improve our emotion classification accuracy by leveraging the temporal correlation of the input videos using popular recurrent methods on top of our existing network. On top of that, we might try to implement real-time inference of our network.
