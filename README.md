# Kaggle-NFL-1st-Future-Impact-Detection


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

## Progress

### Current BestLB Score: 0.1329

-------

## Detects 2 class objects by tito
### Object Detection part is based on EfficientDet notebook for global wheat detection competition by shonenkov, which is using github repos efficientdet-pytorch by @rwightman.

- class1: helmet without impact

- class2: helmet with impact

### Object Detection part
: based on EfficientDet notebook for global wheat detection competition by shonenkov, which is using github repos efficientdet-pytorch by @rwightman.

### [Training] EfficientDet
https://www.kaggle.com/shonenkov/training-efficientdet

------

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







