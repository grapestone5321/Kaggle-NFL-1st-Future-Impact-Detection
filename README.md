# Kaggle-NFL-1st-Future-Impact-Detection

https://www.kaggle.com/c/nfl-impact-detection


## End Date (Final Submission Deadline): 
January 4, 2021 11:59 PM UTC

-------

## Task
In this task, you will segment helmet collisions in videos of football plays using bounding boxes. 

In this competition, you’ll develop a computer vision model that automatically detects helmet impacts that occur on the field. 

Kick off with a dataset of more than one thousand definitive head impacts from thousands of game images, labeled video from the sidelines and end zones, and player tracking data. 

This information is sourced from the NFL’s Next Gen Stats (NGS) system, which documents the position, speed, acceleration, and orientation for every player on the field during NFL games.

-------

## Evaluation
This competition is evaluated using a micro F1 score at an Intersection over Union (IoU) threshold of 0.35.

## Submission File

Due to the custom metric, this competition relies on an evaluation pipeline which is slightly different than a typical code competition. 

Your notebook must import and submit via the custom nflimpact python module available in Kaggle notebooks. 

To submit, simply add these three lines at the end of your code:

     import nflimpact
     env = nflimpact.make_env()
     env.predict(df) # df is a pandas dataframe of your entire submission file

### The dataframe should be in the following format: 

Each row in your submission represents a single predicted bounding box for the given frame. 

Note that it is not required to include labels of which players had an impact, only a bounding box where it occurred.

     gameKey,playID,view,video,frame,left,width,top,height
     57590,3607,Endzone,57590_003607_Endzone.mp4,1,1,1,1,1
     57590,3607,Sideline,57590_003607_Sideline.mp4,1,1,1,1,1
     57595,1252,Endzone,57595_001252_Endzone.mp4,1,1,1,1,1
     57595,1252,Sideline,57595_001252_Sideline.mp4,1,1,1,1,1
     etc.

-------

## Data Overview
### There are three different types of data provided for this problem:

### Image Data: 
Almost 10,000 images and associated helmet labels for the purpose of building a helmet detection computer vision system.

The labeled image dataset consists of 9947 labeled images and a .csv file named image_labels.csv that contains the labeled bounding boxes for all images. This dataset is provided to support the development of helmet detection algorithms.

### Video Data: 
120 videos (60 plays) from both a sideline and endzone point of view (one each per play) with associated helmet and helmet impact labels for the purpose of building a helmet impact detection computer vision system.

The labeled video dataset provides video for 60 plays observed from both the sideline and endzone perspective (120 videos total). The video_labels.csv file contains labeled bounding boxes for every helmet that is visible in every frame of every video.

### Tracking Data: 
Tracking data for all players that participate in the provided 60 plays.

The player track file in .csv format includes player position, direction, and orientation data for each player during the entire course of the play collected using the Next Gen Stats (NGS) system. This data is indexed by gameKey, playID, and player, with the time variable providing a temporal index within an individual play.

-------



# EfficientDet

## Blog (Google AI Blog)


### EfficientDet: Towards Scalable and Efficient Object Detection
https://ai.googleblog.com/2020/04/efficientdet-towards-scalable-and.html

In “EfficientDet: Scalable and Efficient Object Detection”, accepted at CVPR 2020, we introduce a new family of scalable and efficient object detectors. 

Building upon our previous work on scaling neural networks (EfficientNet), and incorporating a novel bi-directional feature network (BiFPN) and new scaling rules, EfficientDet achieves state-of-the-art accuracy while being up to 9x smaller and using significantly less computation compared to prior state-of-the-art detectors. 

The following figure shows the overall network architecture of our models.

## Paper

### EfficientDet: Scalable and Efficient Object Detection
https://arxiv.org/pdf/1911.09070.pdf

### EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks
https://arxiv.org/pdf/1905.11946.pdf


## GitHub
### EfficientDet
https://github.com/google/automl/tree/master/efficientdet

-------

# Detects 2 class objects by tito
Object Detection part is based on EfficientDet notebook for global wheat detection competition by shonenkov,  which is using github repos efficientdet-pytorch by @rwightman.

- class1: helmet without impact

- class2: helmet with impact
### global wheat detection competition by shonenkov

- [Training] EfficientDet

https://www.kaggle.com/shonenkov/training-efficientdet

- [Inference] EfficientDet

https://www.kaggle.com/shonenkov/inference-efficientdet/data

### github repos efficientdet-pytorch by @rwightman
- rwightman/pytorch-image-models

https://github.com/rwightman/pytorch-image-models


-------


### Training part: 2Class Object Detection Training
https://www.kaggle.com/its7171/2class-object-detection-training

### Inference part: 2Class Object Detection Inference
https://www.kaggle.com/its7171/2class-object-detection-inference


### Dataset by tito
### nfl-lib: 
https://www.kaggle.com/its7171/nfl-lib

- pkgs.tgz

- timm-0.1.26-py3-none-any.whl


### nfl-models:
https://www.kaggle.com/its7171/nfl-models

- best-checkpoint-002epoch.bin

- best-checkpoint-018epoch.bin

- efficientdet_d5-ef44aea8.pth

-------
## 2Class Object Detection strict filter: by Art
### both zones 2Class Object Detection strict filter
https://www.kaggle.com/artkulak/both-zones-2class-object-detection-strict-filter

- A notebook with filter to remove some of the False Positives by leaving only predictions which are present in both "Endzone" and "Sideline" views. 

- Here is one more filtering idea which is similar but achieves a bit higher public LB score. 

- Don't forget to properly validate your solutions, before adding those postprocessing extra steps to your pipelines.

### This notebook is based on
https://www.kaggle.com/its7171/2class-object-detection-inference

### 2Class Object Detection Inference with filtering: Here is another version with filtering: 
https://www.kaggle.com/artkulak/2class-object-detection-inference-with-filtering



-----

## Progress

### Current Best LB Score: 0.2260

-------








