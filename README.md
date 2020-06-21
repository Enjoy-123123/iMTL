# Interactive Multi-Task Learning Framework

This is the code for the Interactive Multi-Task Learning (iMTL) Framework for Next POI Recommendation with Uncertain Check-ins.  iMTL exploits the interplay between activity and location preference through the temporal-aware activity encoder and spatial-aware location preference encoder.


# Pre-requisits
## Running environment
 - python 3.6.5
 - numpy 1.16.2
 - pandas 1.0.1
 - progressbar 2.5
 - tensorflow 1.13.1
 - tensorflow-estimator 1.13.0
## Datasets
### Raw dataset:
 - Foursquare: [Dingqi Yang, Daqing Zhang, and Bingqing Qu. Participatory cultural mapping based on collective behavior data in location-based social networks.](https://sites.google.com/site/yangdingqi/home/foursquare-dataset) 
 - Yelp: [www.yelp.com/dataset/challenge](https://www.yelp.com/dataset/challenge)

# Modules of iMTL
### Data_Preparation.py
This module is in charge of preprocessing the raw data to formulate iMLT framework inputs. The processed data would be the five sequences recording information about time, POI, category, type and distance. 
### iMTL_Model.py
This module is the main body of the training and testing model. The model consists of three tasks:
 - category prediction
 - type  prediction 
 - POI prediction
### Helper_Functions.py
This module is a library providing useful functions for the previous two modules.
