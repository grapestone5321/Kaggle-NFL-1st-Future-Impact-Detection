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

## Paper

### EfficientDet: Scalable and Efficient Object Detection
https://arxiv.org/pdf/1911.09070.pdf

### EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks
https://arxiv.org/pdf/1905.11946.pdf


## GitHub
### EfficientDet
https://github.com/google/automl/tree/master/efficientdet


### About EfficientDet Models

EfficientDets are a family of object detection models, which achieve state-of-the-art 55.1mAP on COCO test-dev, yet being 4x - 9x smaller and using 13x - 42x fewer FLOPs than previous detectors. 

Our models also run 2x - 4x faster on GPU, and 5x - 11x faster on CPU than other detectors.

-------

## Paper-2

### Encoder-Decoder with Atrous Separable Convolution for Semantic Image Segmentation
https://arxiv.org/pdf/1802.02611.pdf

### Rethinking Atrous Convolution for Semantic Image Segmentation
https://arxiv.org/pdf/1706.05587.pdf

### DeepLab: Semantic Image Segmentation with Deep Convolutional Nets, Atrous Convolution, and Fully Connected CRFs
https://arxiv.org/pdf/1606.00915.pdf

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

### 2Class Object Detection Inference
https://www.kaggle.com/its7171/2class-object-detection-inference

### 2Class Object Detection Inference with filtering: Here is another version with filtering: 
https://www.kaggle.com/artkulak/2class-object-detection-inference-with-filtering

-------

# Process: 
## both zones 2Class Object Detection strict filter
https://www.kaggle.com/artkulak/both-zones-2class-object-detection-strict-filter

## def seed_everything(seed):

## SET CONSTANTS

     DETECTION_THRESHOLD = 0.4  ##default
     DETECTOR_FILTERING_THRESHOLD = 0.3  ##default

## mk_images

     def mk_images(video_name, video_labels, video_dir, out_dir, only_with_impact=True):

     if IS_PRIVATE:

     def get_valid_transforms():

     class DatasetRetriever(Dataset):

## load_net

     def load_net(checkpoint_path):
         config = get_efficientdet_config('tf_efficientdet_d5')
         net = EfficientDet(config, pretrained_backbone=False)
         config.num_classes = 2
         config.image_size=512
         net.class_net = HeadNet(config, num_outputs=config.num_classes, norm_kwargs=dict(eps=.001, momentum=.01))
         checkpoint = torch.load(checkpoint_path)
         net.load_state_dict(checkpoint['model_state_dict'])
         net = DetBenchEval(net, config)
         net.eval();
         return net.cuda()
     if IS_PRIVATE:
         net = load_net('../input/nfl-models//best-checkpoint-002epoch.bin')

## dataset

     dataset = DatasetRetriever(

     def collate_fn(batch):

     data_loader = DataLoader

## make_predictions

     def make_predictions(images, score_threshold=0.5):
         images = torch.stack(images).cuda().float()
         box_list = []
         score_list = []
         with torch.no_grad():
             det = net(images, torch.tensor([1]*images.shape[0]).float().cuda())
             for i in range(images.shape[0]):
                 boxes = det[i].detach().cpu().numpy()[:,:4]    
                 scores = det[i].detach().cpu().numpy()[:,4]   
                 label = det[i].detach().cpu().numpy()[:,5]
                 # useing only label = 2
                 indexes = np.where((scores > score_threshold) & (label == 2))[0]
                 boxes[:, 2] = boxes[:, 2] + boxes[:, 0]
                 boxes[:, 3] = boxes[:, 3] + boxes[:, 1]
                 box_list.append(boxes[indexes])
                 score_list.append(scores[indexes])
         return box_list, score_list
     import matplotlib.pyplot as plt

## check prediction

     cnt = 0
     for images, image_ids in data_loader:
         box_list, score_list = make_predictions(images, score_threshold=DETECTION_THRESHOLD)
         for i in range(len(images)):
             sample = images[i].permute(1,2,0).cpu().numpy()
             boxes = box_list[i].astype(np.int32).clip(min=0, max=511)
             scores = score_list[i]
             if len(scores) >= 1:
                 fig, ax = plt.subplots(1, 1, figsize=(16, 8))
                 sample = cv2.resize(sample , (int(1280), int(720)))
                 for box,score in zip(boxes,scores):
                     box[0] = box[0] * 1280 / 512
                     box[1] = box[1] * 720 / 512
                     box[2] = box[2] * 1280 / 512
                     box[3] = box[3] * 720 / 512
                     cv2.rectangle(sample, (box[0], box[1]), (box[2], box[3]), (1, 0, 0), 3)
                 ax.set_axis_off()
                 ax.imshow(sample);
                 cnt += 1
         if cnt >= 10:
             break

## Results

     result_image_ids = []
     results_boxes = []
     results_scores = []

     box_df = pd.DataFrame(np.concatenate(results_boxes), columns=['left', 'top', 'width', 'height'])
     test_df = pd.DataFrame({'scores':np.concatenate(results_scores), 'image_name':result_image_ids})
     test_df = pd.concat([test_df, box_df], axis=1)

     rest_df = test_df[test_df.scores > DETECTOR_FILTERING_THRESHOLD]

     test_df.shape
     
     test_df


## FILTER

     dropIDX = []
     for keys in test_df.groupby(['gameKey', 'playID']).size().to_dict().keys():
         tmp_df = test_df.query('gameKey == @keys[0] and playID == @keys[1]')
    
         for index, row in tmp_df.iterrows():
            
             currentFrame = row['frame']

             bboxCount1 = tmp_df.query('view == "Sideline" and abs(frame - @currentFrame) <= 0').shape[0]
             bboxCount2 = tmp_df.query('view == "Endzone" and abs(frame - @currentFrame) <= 0').shape[0]
             if bboxCount1 != bboxCount2:
                 dropIDX.append(index)

## FILTER: 2Class Object Detection Inference with filtering

     dropIDX = []
     for keys in test_df.groupby(['gameKey', 'playID']).size().to_dict().keys():
         tmp_df = test_df.query('gameKey == @keys[0] and playID == @keys[1]')
    
         for index, row in tmp_df.iterrows():
             if row['view'] == 'Endzone':
                 check_df = tmp_df.query('view == "Sideline"')
                 if check_df['frame'].apply(lambda x: np.abs(x - row['frame']) <= 4).sum() == 0:
                     dropIDX.append(index)
        
             if row['view'] == 'Sideline':
                 check_df = tmp_df.query('view == "Endzone"')
                 if check_df['frame'].apply(lambda x: np.abs(x - row['frame']) <= 4).sum() == 0:
                     dropIDX.append(index)

## dropIDX

     dropIDX = []
      for keys in test_df.groupby(['gameKey', 'playID']).size().to_dict().keys():

     test_df = test_df.drop(index = dropIDX).reset_index(drop = True)
 
      !mv * /tmp/

## submission

     import nflimpact
     env = nflimpact.make_env()

     if IS_PRIVATE:
          env.predict(test_df) # df is a pandas dataframe of your entire submission file
     else:
          sub = pd.read_csv('../input/nfl-impact-detection/sample_submission.csv')
          env.predict(sub)


-------

## Progress

### Current Best LB Score: 0.2393

-------

## SET CONSTANTS
### both zones 2Class Object Detection strict filter:

-------

## (DETECTION_THRESHOLD, DETECTOR_FILTERING_THRESHOLD) = (0.3885, 0.3) : Standard ##ver12

### DETECTION_THRESHOLD: default=0.4

:DETECTOR_FILTERING_THRESHOLD = 0.3
     
     DETECTION_THRESHOLD = 0.5:    test_df.shape = (52, 6)   #LB= 0.0439  ##ver4
     DETECTION_THRESHOLD = 0.4:    test_df.shape = (336, 6)  #LB= 0.2260  ##ver1
     DETECTION_THRESHOLD = 0.3895: test_df.shape = (391, 6)  #LB= 0.2393  ##ver30  --- best
     DETECTION_THRESHOLD = 0.389:  test_df.shape = (393, 6)  #LB= 0.2393  ##ver9   --- best
     DETECTION_THRESHOLD = 0.3885: test_df.shape = (396, 6)  #LB= 0.2393  ##ver12  --- best
     DETECTION_THRESHOLD = 0.388:  test_df.shape = (397, 6)  #LB= 0.2393  ##ver10  --- best
     DETECTION_THRESHOLD = 0.3875: test_df.shape = (399, 6)  #LB= 0.2393  ##ver31  --- best
     DETECTION_THRESHOLD = 0.387:  test_df.shape = (404, 6)  #LB= 0.2352  ##ver11
     DETECTION_THRESHOLD = 0.38:   test_df.shape = (444, 6)  #LB= 0.2170  ##ver8
     DETECTION_THRESHOLD = 0.35:   test_df.shape = (638, 6)  #LB= 0.1939  ##ver6
     DETECTION_THRESHOLD = 0.3:    test_df.shape = (1104, 6) #LB= 0.1608  ##ver5


### DETECTOR_FILTERING_THRESHOLD: default=0.3

:DETECTION_THRESHOLD = 0.4

     DETECTOR_FILTERING_THRESHOLD = 0.4: test_df.shape = (336, 6)    #LB= 0.2260   ##ver2
     DETECTOR_FILTERING_THRESHOLD = 0.3: test_df.shape = (336, 6)    #LB= 0.2260   ##ver1
     DETECTOR_FILTERING_THRESHOLD = 0.2: test_df.shape = (336, 6)    #LB= 0.2260   ##ver3

:DETECTION_THRESHOLD = 0.3885

     DETECTOR_FILTERING_THRESHOLD = 0.9: test_df.shape = (0, 6)      #LB= 0.0000   ##ver13
     DETECTOR_FILTERING_THRESHOLD = 0.7: test_df.shape = (0, 6)      #LB= 0.0000   ##ver15
     DETECTOR_FILTERING_THRESHOLD = 0.5: test_df.shape = (52, 6)     #LB= 0.0439   ##ver16
     DETECTOR_FILTERING_THRESHOLD = 0.4: test_df.shape = (336, 6)    #LB= 0.2260   ##ver39
     DETECTOR_FILTERING_THRESHOLD = 0.3: test_df.shape = (396, 6)    #LB= 0.2393   ##ver12  --- best
     DETECTOR_FILTERING_THRESHOLD = 0.1: test_df.shape = (396, 6)    #LB= 0.2393   ##ver17  --- best
     DETECTOR_FILTERING_THRESHOLD = 0.01:test_df.shape = (396, 6)    #LB= 0.2393   ##ver18  --- best

:DETECTION_THRESHOLD = 0.38

     DETECTOR_FILTERING_THRESHOLD = 0.5: test_df.shape = (52, 6)    #LB= 0.0439  ##ver35
     DETECTOR_FILTERING_THRESHOLD = 0.3: test_df.shape = (444, 6)   #LB= 0.2170  ##ver8
     DETECTOR_FILTERING_THRESHOLD = 0.2: test_df.shape = (444, 6)   #LB= 0.2170  ##ver33
     DETECTOR_FILTERING_THRESHOLD = 0.01:test_df.shape = (444, 6)   #LB= 0.2170  ##ver34

DETECTION_THRESHOLD = 0.3:    

     DETECTOR_FILTERING_THRESHOLD = 0.5:   test_df.shape = (52, 6)   #LB= 0.0439  ##ver36
     DETECTOR_FILTERING_THRESHOLD = 0.4:   test_df.shape = (336, 6)  #LB= 0.2260  ##ver38
     DETECTOR_FILTERING_THRESHOLD = 0.39:  test_df.shape = (389, 6)  #LB= 0.2352  ##ver42
     DETECTOR_FILTERING_THRESHOLD = 0.389: test_df.shape = (393, 6)  #LB= 0.2393  ##ver44     
     DETECTOR_FILTERING_THRESHOLD = 0.388: test_df.shape = (397, 6)  #LB= 0.2393  ##ver45
     DETECTOR_FILTERING_THRESHOLD = 0.3875:test_df.shape = (399, 6)  #LB= 0.2393  ##ver47
     DETECTOR_FILTERING_THRESHOLD = 0.387: test_df.shape = (404, 6)  #LB= 0.2352  ##ver46     
     DETECTOR_FILTERING_THRESHOLD = 0.385: test_df.shape = (416, 6)  #LB= 0.2352  ##ver43
     DETECTOR_FILTERING_THRESHOLD = 0.38:  test_df.shape = (444, 6)  #LB= 0.2170  ##ver41
     DETECTOR_FILTERING_THRESHOLD = 0.35:  test_df.shape = (638, 6)  #LB= 0.1939  ##ver40
     DETECTOR_FILTERING_THRESHOLD = 0.3:   test_df.shape = (1104, 6) #LB= 0.1608  ##ver5
     DETECTOR_FILTERING_THRESHOLD = 0.01:  test_df.shape = (1104, 6) #LB= 0.1608  ##ver37
     
-------

## score_threshold:  default=0.5
### Note:box_list, score_list = make_predictions(images, score_threshold=DETECTION_THRESHOLD)

     def make_predictions(images, score_threshold=0.5):   #LB= 0.2393   ##ver12
     def make_predictions(images, score_threshold=0.4):   #LB= 0.2393   ##ver19


## batch_size: default=16
:data_loader = DataLoader(
   
     batch_size =   8:   #LB= 0.2393    ##ver25
     batch_size =  16:   #LB= 0.2393    ##ver12
     batch_size =  32:   #LB= 0.2393    ##ver21
     batch_size =  64:   #LB= 0.2393    ##ver22
     batch_size = 128:   #LB= 0.2393    ##ver23
     batch_size = 256:   #LB= error     ##ver24
    
## num_workers: default=4
:data_loader = DataLoader(
   
     num_workers =  2:   #LB= 0.2393    ##ver26
     num_workers =  4:   #LB= 0.2393    ##ver12
     num_workers =  8:   #LB= 0.2393    ##ver27
     
## shuffle: default=False

    shuffle = False:    #LB= 0.2393    ##ver12    
    shuffle = True:     #LB= 0.2393    ##ver28 
     
## drop_last=False,: default=False

    drop_last=False:    #LB= 0.2393    ##ver12    
    drop_last=True:     #LB= 0.2393    ##ver29     
          
-------

## FILTER:

             #LB= 0.2393: test_df.shape = (396, 6)    ##ver12
             bboxCount1 = tmp_df.query('view == "Sideline" and abs(frame - @currentFrame) <= 0').shape[0]
             bboxCount2 = tmp_df.query('view == "Endzone" and abs(frame - @currentFrame) <= 0').shape[0]
             
             #LB= 0.1913: test_df.shape = (396, 6)    ##ver48
             bboxCount1 = tmp_df.query('view == "Sideline" and abs(frame - @currentFrame) <= 1').shape[0]
             bboxCount2 = tmp_df.query('view == "Endzone" and abs(frame - @currentFrame) <= 1').shape[0]
             
             #LB= 0.1295: test_df.shape = (396, 6)    ##ver49
             bboxCount1 = tmp_df.query('view == "Sideline" and abs(frame - @currentFrame) < 0').shape[0]
             bboxCount2 = tmp_df.query('view == "Endzone" and abs(frame - @currentFrame) < 0').shape[0]
             
             #LB= 0.2393: test_df.shape = (396, 6)    ##ver50
             bboxCount1 = tmp_df.query('view == "Sideline" and abs(frame - @currentFrame) < 0.1').shape[0]
             bboxCount2 = tmp_df.query('view == "Endzone" and abs(frame - @currentFrame) < 0.1').shape[0]

## def load_net(checkpoint_path):

### momentum: default=0.1
     #LB= 0.2393: test_df.shape = (396, 6)    ##ver12
     net.class_net = HeadNet(config, num_outputs=config.num_classes, norm_kwargs=dict(eps=.001, momentum=.01))
     
     #LB= 0.2393: test_df.shape = (396, 6)    ##ver51
     net.class_net = HeadNet(config, num_outputs=config.num_classes, norm_kwargs=dict(eps=.001, momentum=.05))

### eps: =.001

     #LB= 0.1904: test_df.shape =  (245, 6)   ##ver52
     net.class_net = HeadNet(config, num_outputs=config.num_classes, norm_kwargs=dict(eps=.01, momentum=.01))

     #LB=       : test_df.shape =             ##ver53
     net.class_net = HeadNet(config, num_outputs=config.num_classes, norm_kwargs=dict(eps=.0001, momentum=.01))
-------
