# Object-detection
*Vehicle detection using deep learning with tensorflow and Python*

This programs explains how to train your own convolutional neural network (CNN) in object detection for multiple objects, starting from scratch. Using the tutorial one can identify and detect specific objects in pictures, videos, or in a webcam feed.

The steps used to train model on window (10, 7, 8) in the Tensorflow envionment are decribed below. I used TensorFlow-v1.5 but the program will work for future version.


# Steps

# 1. Install Anaconda

Visit https://www.anaconda.com/distribution/#download-section
Download and install Anaconda 

Visit 
https://www.tensorflow.org/
https://www.tensorflow.org/install#install-tensorflow-2
TensorFlow's website describes installation details.

# 2. Set up TensorFlow Directory and Anaconda Virtual Environment

The TensorFlow Object Detection API and Anaconda Virtual Environment requires configuration. There are several sub steps as explained below.

### 2a. Download TensorFlow Object Detection API repository from GitHub
*Create a folder in C: and name it “tensorflow12" *

This working directory will contain the full TensorFlow object detection framework, as well training images, training data, trained classifier, configuration files, and everything else needed for the object detection classifier.

Download the full TensorFlow object detection repository located at https://github.com/tensorflow/models by clicking the “Clone or Download” button and downloading the zip file. Open the downloaded zip file and extract the “models-master” folder directly into the C:\tensorflow12 directory. Rename “models-master” to just “models”.

### 2b. Download the Faster-RCNN-Inception-V2-COCO model from TensorFlow's model zoo
TensorFlow provides several object detection models (pre-trained classifiers with specific neural network architectures) in its model zoo such as the SSD-MobileNet model, Faster-RCNN model.

This progarm use the Faster-RCNN-Inception-V2 model. Download the model. 
Open the downloaded 
faster_rcnn_inception_v2_coco_2018_01_28.tar.gz file with a file archiver such as WinZip or 7-Zip and extract the faster_rcnn_inception_v2_coco_2018_01_28 folder to the C:\tensorflow12\models\research\object_detection folder.

### 2c. Download this program repository from GitHub

Download the full repository located on this page (scroll to the top and click Clone or Download) and extract all the contents directly into the C:\tensorflow12\models\research\object_detection directory.

At this point, here is what your \object_detection folder should look like:

This repository contains the images, annotation data, .csv files, and TFRecords needed to train a "vehicle" detector.

To train own object detector (vehicle), delete the following files (do not delete the folders):

All files in \object_detection\images\train and \object_detection\images\test
The “test_labels.csv” and “train_labels.csv” files in \object_detection\images
All files in \object_detection\training
All files in \object_detection\inference_graph

### 2d. Set up new Anaconda virtual environment

Go to the start menu in Windows, search for the Anaconda Prompt utility, right click on it, and click “Run as Administrator”. The  Windows will asks if you would like to allow it to make changes to your computer, click Yes.

In the command terminal that pops up, create a new virtual environment called “tensorflow12” by issuing the following command:

C:\> conda create -n tensorflow12 pip python=3.5

Activate the environment and update pip:
```
C:\> activate tensorflow12
(tensorflow12) C:\>python -m pip install --upgrade pip
```

Install tensorflow in this environment by issuing:
```
(tensorflow12) C:\> pip install --ignore-installed --upgrade tensorflow
```

Install the other necessary packages using commands:
```
(tensorflow12) C:\> conda install -c anaconda protobuf

(tensorflow12) C:\> pip install pillow

(tensorflow12) C:\> pip install lxml

(tensorflow12) C:\> pip install Cython

(tensorflow12) C:\> pip install contextlib2

(tensorflow12) C:\> pip install jupyter

(tensorflow12) C:\> pip install matplotlib

(tensorflow12) C:\> pip install pandas

(tensorflow12) C:\> pip install opencv-python
```

( The ‘pandas’ and ‘opencv-python’ packages are not needed by TensorFlow. They are used in the Python scripts to generate TFRecords and to work with images, videos, and webcam feeds.)

### 2e. Configure PYTHONPATH environment variable

A PYTHONPATH variable must be created that points to the \models, \models\research, and \models\research\slim directories. Use the following commands (from any directory):
```
(tensorflow12) C:\> set PYTHONPATH=C:\tensorflow12\models;C:\tensorflow12\models\research;C:\tensorflow12\models\research\slim
```

### 2f. Compile Protobufs and run setup.py

Next, compile the Protobuf files, which are used by TensorFlow to configure model and training parameters.
In the Anaconda Command Prompt, change directories to the \models\research directory:
```
(tensorflow12) C:\> cd C:\tensorflow12\models\research
```

Copy and paste the following command into the command line and press Enter:

```
protoc --python_out=. .\object_detection\protos\anchor_generator.proto .\object_detection\protos\argmax_matcher.proto .\object_detection\protos\bipartite_matcher.proto .\object_detection\protos\box_coder.proto .\object_detection\protos\box_predictor.proto .\object_detection\protos\eval.proto .\object_detection\protos\faster_rcnn.proto .\object_detection\protos\faster_rcnn_box_coder.proto .\object_detection\protos\grid_anchor_generator.proto .\object_detection\protos\hyperparams.proto .\object_detection\protos\image_resizer.proto .\object_detection\protos\input_reader.proto .\object_detection\protos\losses.proto .\object_detection\protos\matcher.proto .\object_detection\protos\mean_stddev_box_coder.proto .\object_detection\protos\model.proto .\object_detection\protos\optimizer.proto .\object_detection\protos\pipeline.proto .\object_detection\protos\post_processing.proto .\object_detection\protos\preprocessor.proto .\object_detection\protos\region_similarity_calculator.proto .\object_detection\protos\square_box_coder.proto .\object_detection\protos\ssd.proto .\object_detection\protos\ssd_anchor_generator.proto .\object_detection\protos\string_int_label_map.proto .\object_detection\protos\train.proto .\object_detection\protos\keypoint_box_coder.proto .\object_detection\protos\multiscale_anchor_generator.proto .\object_detection\protos\graph_rewriter.proto .\object_detection\protos\calibration.proto .\object_detection\protos\flexible_grid_anchor_generator.proto
```

Finally, run the following commands from the 
C:\tensorflow12\models\research directory:
```
(tensorflow12) C:\tensorflow12\models\research> python setup.py build
(tensorflow12) C:\tensorflow12\models\research> python setup.py install
```

### 2g. Test TensorFlow setup to verify it works
From the \object_detection directory, issue this command:
```
(tensorflow12) C:\tensorflow12\models\research\object_detection> jupyter notebook object_detection_tutorial.ipynb
```

![Image of objects](https://github.com/tensorflow/models/raw/master/research/object_detection/g3doc/img/kites_detections_output.jpg)


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

```
(tensorflow12) C:\tensorflow12\models\research\object_detection> python xml_to_csv.py
```

This creates a train_labels.csv and test_labels.csv file in the \object_detection\images folder.

Next, open the generate_tfrecord.py file in a text editor. Replace the label map starting at line 31 with your own label map, where each object is assigned an ID number. This same number assignment will be used when configuring the labelmap.pbtxt file in Step 5b.

For example, say you are training a classifier to detect basketballs, shirts, and shoes. You will replace the following code in generate_tfrecord.py:

### TO-DO replace this with label map
def class_text_to_int(row_label):

    if row_label == 'car':
        return 1
        
    elif row_label == 'truck':
        return 2     
    
    else:
        None
        
With this:

### TO-DO replace this with label map
def class_text_to_int(row_label):

    if row_label == 'car':
        return 1
        
    elif row_label == 'truck':
        return 2   
      
    else:
        None
        
        
Then, generate the TFRecord files by issuing these commands from the \object_detection folder:

```
python generate_tfrecord.py --csv_input=images\train_labels.csv --image_dir=images\train --output_path=train.record

python generate_tfrecord.py --csv_input=images\test_labels.csv --image_dir=images\test --output_path=test.record
```

These generate a train.record and a test.record file in \object_detection. These will be used to train the new object detection classifier.

# 5. Create Label Map and Configure Training

### 5a. Label map
Use a text editor to create a new file and save it as labelmap.pbtxt in the 

C:\tensorflow1\models\research\object_detection\training folder.

item {
  id: 1
  name: 'car'
}

item {
  id: 2
  name: 'truck'
}

The label map ID numbers should be the same as what is defined in the generate_tfrecord.py file. For the basketball, shirt, and shoe detector example mentioned in Step 4, the labelmap.pbtxt file will look like:

### 5b. Configure training

Navigate to 
C:\tensorflow12\models\research\object_detection\samples\configs and copy the faster_rcnn_inception_v2_pets.config file into the 
\object_detection\training directory. 

Then, open the file with a text editor for adding changes.

Make the following changes to the faster_rcnn_inception_v2_pets.config file.

* Line 9. Change num_classes to the number of different objects you want the classifier to detect. For the above car and truck detector, it would be num_classes : 2 .

* Line 106. Change fine_tune_checkpoint to:

fine_tune_checkpoint : "C:/tensorflow12/models/research/object_detection/faster_rcnn_inception_v2_coco_2018_01_28/model.ckpt"

*Lines 123 and 125. In the train_input_reader section, change input_path and label_map_path to:

input_path : "C:/tensorflow12/models/research/object_detection/train.record"
label_map_path: "C:/tensorflow12/models/research/object_detection/training/labelmap.pbtxt"

* Line 130. Change num_examples to the number of images you have in the \images\test directory.

* Lines 135 and 137. In the eval_input_reader section, change input_path and label_map_path to:

input_path : "C:/tensorflow12/models/research/object_detection/test.record"
label_map_path: "C:/tensorflow12/models/research/object_detection/training/labelmap.pbtxt"

Save the file after the changes have been made. The training job is all configured.

# 6. Run the Training

From the \object_detection directory, issue the following command to begin training:
```
python train.py --logtostderr --train_dir=training/ --pipeline_config_path=training/faster_rcnn_inception_v2_pets.config
```

If everything has been set up correctly, TensorFlow will initialize the training. The initialization can take up to 30 seconds before the actual training begins. When training begins, it will look like this:


Each step of training reports the loss. It will start high and get lower and lower as training progresses. It will take about 40,000 steps, or about 2 hours (depending on how powerful your CPU and GPU are). Note: The loss numbers will be different if a different model is used. MobileNet-SSD starts with a loss of about 20, and should be trained until the loss is consistently under 2.

You can view the progress of the training job by using TensorBoard. Open a new instance of Anaconda Prompt, activate the tensorflow12 virtual environment, change to the

C:\tensorflow12\models\research\object_detection directory, and issue the following command:
```
(tensorflow12) C:\tensorflow12\models\research\object_detection>tensorboard --logdir=training
```

This will create a webpage on the local machine, which can be viewed through a web browser. The TensorBoard page provides information and graphs that show how the training is progressing. One important graph is the Loss graph, which shows the overall loss of the classifier over time.

# 7. Export Inference Graph

From the \object_detection folder, issue the following command, where “XXXX” in “model.ckpt-XXXX” should be replaced with the highest-numbered .ckpt file in the training folder:

```
python export_inference_graph.py --input_type image_tensor --pipeline_config_path training/faster_rcnn_inception_v2_pets.config --trained_checkpoint_prefix training/model.ckpt-XXXX --output_directory inference_graph
```

This creates a frozen_inference_graph.pb file in the \object_detection\inference_graph folder. The .pb file contains the object detection classifier.

# 8. Use Newly Trained Object Detection (Vehicle) Classifier

To run any of the scripts, type “idle” in the Anaconda Command Prompt (with the “tensorflow12” virtual environment activated) and press ENTER. This will open IDLE, and from there, you can open any of the scripts and run them.

If everything is working properly, the object detector will initialize for about 10 seconds and then display a window showing the objects it’s detected in the image!

![Image of objects](https://github.com/ashar367/Vehicle-detection-using-deep-learning-with-Tensorflow-and-Python/blob/master/car.png)

If you encounter errors, check out on Stack Exchange or in TensorFlow’s Issues on GitHub.
