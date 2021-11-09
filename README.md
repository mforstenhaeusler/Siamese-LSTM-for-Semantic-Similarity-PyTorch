# Semantic-Similarity-Using-a-LSTM-NN

This repositpory entails an implementation of a Deep Learning Pipline that can be used evaulates the semantic similarity of two sentenences using pytorch. The model of choice is a Siamese LSTM Neural Network. 

It consists of 2 modules: \
    - a data module \
    - a model module

The data module handles the data preparation and data loading. 
The model module handles the model configuration and the training and evaluation algorithms.

The dataset was downloaded from https://www.kaggle.com/quora/question-pairs-dataset. In order to simplifiy the setup, the dataset can be modified to only used 50k examples and a question length of 30 to 50 characters. The modified dataset is included in the repository. 