# object-detection
Vehicle detection using deep learning with tensorflow and Python


This tutorial explains how to train your own convolutional neural network object detection classifier for multiple objects, starting from scratch. At the end of this, one can identify and detect specific objects in pictures, videos, or in a webcam feed.

There are several applications available for implementing TensorFlow’s Object Detection API to train a classifier for a single object. To set up TensorFlow to train a model on Windows, there are several steps that need to be used.

The program is written for Windows 10, and it will also work for Windows 7 and 8. I used TensorFlow-CPU v1.5 but the program will work for future version.

TensorFlow-GPU allows PC to use the video card to provide extra processing power while training.

Steps

1. Install Anaconda

Visit https://www.anaconda.com/distribution/#download-section
Downlaod and install Anaconda 

Visit https://www.tensorflow.org/
TensorFlow's website describes for installation details.

2. Set up TensorFlow Directory and Anaconda Virtual Environment

The TensorFlow Object Detection API requires using the specific directory structure provided in its GitHub repository. It also requires several Python packages, PATH and PYTHONPATH variables.

2a. Download TensorFlow Object Detection API repository from GitHub
Create a folder directly in C: and name it “tensorflow12”. This working directory will contain the full TensorFlow object detection framework, as well training images, training data, trained classifier, configuration files, and everything else needed for the object detection classifier.
Download the full TensorFlow object detection repository located at https://github.com/tensorflow/models by clicking the “Clone or Download” button and downloading the zip file. Open the downloaded zip file and extract the “models-master” folder directly into the C:\tensorflow1 directory you just created. Rename “models-master” to just “models”.

2b. Download the Faster-RCNN-Inception-V2-COCO model from TensorFlow's model zoo
TensorFlow provides several object detection models (pre-trained classifiers with specific neural network architectures) in its model zoo such as the SSD-MobileNet model, Faster-RCNN model.

This progarm use the Faster-RCNN-Inception-V2 model. Download the model here. Open the downloaded faster_rcnn_inception_v2_coco_2018_01_28.tar.gz file with a file archiver such as WinZip or 7-Zip and extract the faster_rcnn_inception_v2_coco_2018_01_28 folder to the C:\tensorflow1\models\research\object_detection folder.

2c. Download this program repository from GitHub


2d. Set up new Anaconda virtual environment

From the Start menu in Windows, search for the Anaconda Prompt utility, right click on it, and click “Run as Administrator”. If Windows asks you if you would like to allow it to make changes to your computer, click Yes.

In the command terminal that pops up, create a new virtual environment called “tensorflow1” by issuing the following command:




