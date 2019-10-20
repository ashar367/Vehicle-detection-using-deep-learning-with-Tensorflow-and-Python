# object-detection
Vehicle detection using deep learning with tensorflow and Python


This tutorial explains how to train your own convolutional neural network object detection classifier for multiple objects, starting from scratch. At the end of this, one can identify and detect specific objects in pictures, videos, or in a webcam feed.

There are several applications available for implementing TensorFlow’s Object Detection API to train a classifier for a single object. To set up TensorFlow to train a model on Windows, there are several steps that need to be used.

The program is written for Windows 10, and it will also work for Windows 7 and 8. I used TensorFlow-CPU v1.5 but the program will work for future version.

TensorFlow-GPU allows PC to use the video card to provide extra processing power while training.

Steps

# 1. Install Anaconda

Visit https://www.anaconda.com/distribution/#download-section
Downlaod and install Anaconda 

Visit https://www.tensorflow.org/
TensorFlow's website describes for installation details.

# 2. Set up TensorFlow Directory and Anaconda Virtual Environment

The TensorFlow Object Detection API requires using the specific directory structure provided in its GitHub repository. It also requires several Python packages, PATH and PYTHONPATH variables.

### 2a. Download TensorFlow Object Detection API repository from GitHub
Create a folder directly in C: and name it “tensorflow12”. This working directory will contain the full TensorFlow object detection framework, as well training images, training data, trained classifier, configuration files, and everything else needed for the object detection classifier.
Download the full TensorFlow object detection repository located at https://github.com/tensorflow/models by clicking the “Clone or Download” button and downloading the zip file. Open the downloaded zip file and extract the “models-master” folder directly into the C:\tensorflow1 directory you just created. Rename “models-master” to just “models”.

### 2b. Download the Faster-RCNN-Inception-V2-COCO model from TensorFlow's model zoo
TensorFlow provides several object detection models (pre-trained classifiers with specific neural network architectures) in its model zoo such as the SSD-MobileNet model, Faster-RCNN model.

This progarm use the Faster-RCNN-Inception-V2 model. Download the model here. Open the downloaded faster_rcnn_inception_v2_coco_2018_01_28.tar.gz file with a file archiver such as WinZip or 7-Zip and extract the faster_rcnn_inception_v2_coco_2018_01_28 folder to the C:\tensorflow1\models\research\object_detection folder.

### 2c. Download this program repository from GitHub


### 2d. Set up new Anaconda virtual environment

Go to the start menu in Windows, search for the Anaconda Prompt utility, right click on it, and click “Run as Administrator”. The  Windows will asks if you would like to allow it to make changes to your computer, click Yes.

In the command terminal that pops up, create a new virtual environment called “tensorflow12” by issuing the following command:

C:\> conda create -n tensorflow12 pip python=3.5

Activate the environment and update pip:
C:\> activate tensorflow12

(tensorflow1) C:\>python -m pip install --upgrade pip

Install tensorflow-cpu in this environment by issuing:
(tensorflow1) C:\> pip install --ignore-installed --upgrade tensorflow

Install the other necessary packages using commands:
(tensorflow12) C:\> conda install -c anaconda protobuf

(tensorflow12) C:\> pip install pillow

(tensorflow12) C:\> pip install lxml

(tensorflow12) C:\> pip install Cython

(tensorflow12) C:\> pip install contextlib2

(tensorflow12) C:\> pip install jupyter

(tensorflow12) C:\> pip install matplotlib

(tensorflow12) C:\> pip install pandas

(tensorflow12) C:\> pip install opencv-python

( The ‘pandas’ and ‘opencv-python’ packages are not needed by TensorFlow. They are used in the Python scripts to generate TFRecords and to work with images, videos, and webcam feeds.)

### 2e. Configure PYTHONPATH environment variable

A PYTHONPATH variable must be created that points to the \models, \models\research, and \models\research\slim directories. Use the following commands (from any directory):
(tensorflow12) C:\> set PYTHONPATH=C:\tensorflow12\models;C:\tensorflow12\models\research;C:\tensorflow12\models\research\slim

### 2f. Compile Protobufs and run setup.py

Next, compile the Protobuf files, which are used by TensorFlow to configure model and training parameters.
In the Anaconda Command Prompt, change directories to the \models\research directory:

(tensorflow12) C:\> cd C:\tensorflow12\models\research

Copy and paste the following command into the command line and press Enter:
protoc --python_out=. .\object_detection\protos\anchor_generator.proto .\object_detection\protos\argmax_matcher.proto .\object_detection\protos\bipartite_matcher.proto .\object_detection\protos\box_coder.proto .\object_detection\protos\box_predictor.proto .\object_detection\protos\eval.proto .\object_detection\protos\faster_rcnn.proto .\object_detection\protos\faster_rcnn_box_coder.proto .\object_detection\protos\grid_anchor_generator.proto .\object_detection\protos\hyperparams.proto .\object_detection\protos\image_resizer.proto .\object_detection\protos\input_reader.proto .\object_detection\protos\losses.proto .\object_detection\protos\matcher.proto .\object_detection\protos\mean_stddev_box_coder.proto .\object_detection\protos\model.proto .\object_detection\protos\optimizer.proto .\object_detection\protos\pipeline.proto .\object_detection\protos\post_processing.proto .\object_detection\protos\preprocessor.proto .\object_detection\protos\region_similarity_calculator.proto .\object_detection\protos\square_box_coder.proto .\object_detection\protos\ssd.proto .\object_detection\protos\ssd_anchor_generator.proto .\object_detection\protos\string_int_label_map.proto .\object_detection\protos\train.proto .\object_detection\protos\keypoint_box_coder.proto .\object_detection\protos\multiscale_anchor_generator.proto .\object_detection\protos\graph_rewriter.proto .\object_detection\protos\calibration.proto .\object_detection\protos\flexible_grid_anchor_generator.proto

Finally, run the following commands from the C:\tensorflow12\models\research directory:
(tensorflow12) C:\tensorflow12\models\research> python setup.py build
(tensorflow12) C:\tensorflow12\models\research> python setup.py install

### 2g. Test TensorFlow setup to verify it works
From the \object_detection directory, issue this command:
(tensorflow12) C:\tensorflow12\models\research\object_detection> jupyter notebook object_detection_tutorial.ipynb


# 3. Gather and Label Pictures

We need to provide the images it will use to train a new detection classifier.

### 3a. Gather Pictures

Make sure the images aren’t too large. They should be less than 200KB each, and their resolution shouldn’t be more than 720x1280. The larger the images are, the longer it will take to train the classifier. 

Move 20% of them to the \object_detection\images\test directory, and 80% of them to the \object_detection\images\train directory.

### 3b. Label Pictures

LabelImg is a tool for labeling images.

LabelImg GitHub link

LabelImg download link

Download and install LabelImg, point it to your \images\train directory, and then draw a box around each object in each image. Repeat the process for all the images in the \images\test directory.

LabelImg saves a .xml file containing the label data for each image. These .xml files will be used to generate TFRecords, which are one of the inputs to the TensorFlow trainer. Once you have labeled and saved each image, there will be one .xml file for each image in the \test and \train directories.

# 4. Generate Training Data

With the images labeled, now generate the TFRecords that serve as input data to the TensorFlow training model. 

First, the image .xml data will be used to create .csv files containing all the data for the train and test images. From the \object_detection folder, issue the following command in the Anaconda command prompt:

(tensorflow12) C:\tensorflow12\models\research\object_detection> python xml_to_csv.py

This creates a train_labels.csv and test_labels.csv file in the \object_detection\images folder.

Next, open the generate_tfrecord.py file in a text editor. Replace the label map starting at line 31 with your own label map, where each object is assigned an ID number. This same number assignment will be used when configuring the labelmap.pbtxt file in Step 5b.

For example, say you are training a classifier to detect basketballs, shirts, and shoes. You will replace the following code in generate_tfrecord.py:

### TO-DO replace this with label map
def class_text_to_int(row_label):

    if row_label == 'nine':
        return 1
        
    elif row_label == 'ten':
        return 2
        
    elif row_label == 'jack':
        return 3
        
    elif row_label == 'queen':
        return 4
        
    elif row_label == 'king':
        return 5
        
    elif row_label == 'ace':
        return 6
        
    else:
        None
        
With this:


### TO-DO replace this with label map
def class_text_to_int(row_label):

    if row_label == 'basketball':
        return 1
        
    elif row_label == 'shirt':
        return 2
        
    elif row_label == 'shoe':
        return 3
        
    else:
        None
        
        
Then, generate the TFRecord files by issuing these commands from the \object_detection folder:

python generate_tfrecord.py --csv_input=images\train_labels.csv --image_dir=images\train --output_path=train.record

python generate_tfrecord.py --csv_input=images\test_labels.csv --image_dir=images\test --output_path=test.record


These generate a train.record and a test.record file in \object_detection. These will be used to train the new object detection classifier.

# 5. Create Label Map and Configure Training

### 5a. Label map
Use a text editor to create a new file and save it as labelmap.pbtxt in the 

C:\tensorflow1\models\research\object_detection\training folder.

item {
  id: 1
  name: ''
}

item {
  id: 2
  name: ''
}


item {
  id: 3
  name: ''
}


item {
  id: 4
  name: ''
}


item {
  id: 5
  name: ''
}


item {
  id: 6
  name: ''
}

The label map ID numbers should be the same as what is defined in the generate_tfrecord.py file. For the basketball, shirt, and shoe detector example mentioned in Step 4, the labelmap.pbtxt file will look like:

tem {
  id: 1
  name: 'basketball'
}


item {
  id: 2
  name: 'shirt'
}


item {
  id: 3
  name: 'shoe'
}

# 5b. Configure training

Navigate to 
C:\tensorflow1\models\research\object_detection\samples\configs 
and copy the faster_rcnn_inception_v2_pets.config file into the 
\object_detection\training directory. Then, open the file with a text editor.
There are several changes to make to the .config file, mainly changing the number of classes and examples, and adding the file paths to the training data.

Make the following changes to the faster_rcnn_inception_v2_pets.config file.

* Line 9. Change num_classes to the number of different objects you want the classifier to detect. For the above basketball, shirt, and shoe detector, it would be num_classes : 3 .

* Line 106. Change fine_tune_checkpoint to:

fine_tune_checkpoint : "C:/tensorflow1/models/research/object_detection/faster_rcnn_inception_v2_coco_2018_01_28/model.ckpt"
Lines 123 and 125. In the train_input_reader section, change input_path and label_map_path to:

input_path : "C:/tensorflow1/models/research/object_detection/train.record"
label_map_path: "C:/tensorflow1/models/research/object_detection/training/labelmap.pbtxt"
Line 130. Change num_examples to the number of images you have in the \images\test directory.

Lines 135 and 137. In the eval_input_reader section, change input_path and label_map_path to:

input_path : "C:/tensorflow1/models/research/object_detection/test.record"
label_map_path: "C:/tensorflow1/models/research/object_detection/training/labelmap.pbtxt"
Save the file after the changes have been made. That’s it! The training job is all configured and ready to go!
