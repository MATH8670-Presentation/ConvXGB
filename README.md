# ConvXGB

Our attempted implementation of the ConvXGB machine learning model proposed in https://doi.org/10.1016/j.net.2020.04.008. Although we were unable to do this,
we were able to compare CNN, XGBoost, and an XGBoost model using the feature layer output of a pretrained CNN model. 

## File Description

### CNN_Model.ipynb

CNN Model creation, training and experiment

### XGBoost.ipynb

XGBoost model creation, training and experiment

### XGBoost_plus_pretrainedCNN.ipynb

CNN+XGBoost model experiment

### experiment_results.ipynb

Contains plot of model accuracies

### Retreive_Data.ipynb

Contains code to grab and clean data

### Others

data_dct, the CNN_models folder and DATA folder are supporting files or source data. 

## Data Source

The sensorless drive diagnosis dataset can be found at https://archive.ics.uci.edu/ml/machine-learning-databases/00325/Sensorless_drive_diagnosis.txt.