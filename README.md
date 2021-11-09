## Siamese-LSTM-for-Semantic-Similarity

This repositpory entails an implementation of a Deep Learning Pipeline that can be used to evaulate the semantic similarity of two sentenences using PyTorch. The model of choice is a Siamese LSTM Neural Network. 

It consists of 2 modules: \
    - a data module that handles the data preparation and data loading  \
    - a model module that handles the model configuration, the training, evaluation and prediction algorithms

The dataset used for this task was downloaded from https://www.kaggle.com/quora/question-pairs-dataset. In order to simplifiy the setup, the dataset can be modified to only used 50k examples and a question length of 30 to 50 characters. The modified dataset is included in the repository. 
