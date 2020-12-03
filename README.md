# Kaggle-NFL-1st-Future-Impact-Detection


## End Date (Final Submission Deadline): 
January 4, 2021 11:59 PM UTC

## Task
In this task, you will segment helmet collisions in videos of football plays using bounding boxes. 

In this competition, you’ll develop a computer vision model that automatically detects helmet impacts that occur on the field. 

Kick off with a dataset of more than one thousand definitive head impacts from thousands of game images, labeled video from the sidelines and end zones, and player tracking data. 

This information is sourced from the NFL’s Next Gen Stats (NGS) system, which documents the position, speed, acceleration, and orientation for every player on the field during NFL games.

## Evaluation
This competition is evaluated using a micro F1 score at an Intersection over Union (IoU) threshold of 0.35.

## Submission File

Due to the custom metric, this competition relies on an evaluation pipeline which is slightly different than a typical code competition. 

Your notebook must import and submit via the custom nflimpact python module available in Kaggle notebooks. 

To submit, simply add these three lines at the end of your code:

     import nflimpact
     env = nflimpact.make_env()
     env.predict(df) # df is a pandas dataframe of your entire submission file


## Progress

### Current BestLB Score: 0.1329



