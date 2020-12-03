# Kaggle-NFL-1st-Future-Impact-Detection


## End Date (Final Submission Deadline): 
January 4, 2021 11:59 PM UTC

## task
In this task, you will segment helmet collisions in videos of football plays using bounding boxes. 

This competition is evaluated using a micro F1 score at an Intersection over Union (IoU) threshold of 0.35.

## Submission File

Due to the custom metric, this competition relies on an evaluation pipeline which is slightly different than a typical code competition. 

Your notebook must import and submit via the custom nflimpact python module available in Kaggle notebooks. 

To submit, simply add these three lines at the end of your code:

     import nflimpact
     env = nflimpact.make_env()
     env.predict(df) # df is a pandas dataframe of your entire submission file


