import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns

import gensim
import torch


def create_pretrained_weights(google_embbeding_path, embedding_dim, language):
    """ Load pretrained weight and create pretrained weights for the embedding layer of the moel from pre-trained embbedings """
    n_words_vocab = len(language.word2index)
    # adopted from https://mccormickml.com/2016/04/12/googles-pretrained-word2vec-model-in-python/
    # link to embeddings https://drive.google.com/file/d/0B7XkCwpI5KDYNlNUTTlSS21pQmM/edit?resourcekey=0-wjGZdNAUop6WykTtMip30g
    # Load pre-trained embeddings from word2vec
    word2vec_model = gensim.models.KeyedVectors.load_word2vec_format(google_embbeding_path, binary=True)
    
    # Convert word2vec embeddings into FloatTensor
    word2vec_weights = torch.FloatTensor(word2vec_model.vectors)

    # Create a random weight tensor of the shape (n_words_vocab + 1, embedding_dim) and place each word's embedding from word2vec at the index assigned to that word
    # Two key points:
    # 1. Weights tensor has been initialized randomly so that the words which are part of our dataset vocabulary but are not present in word2vec are given a random embedding
    # 2. Embedding at 0 index is all zeros. This is the embedding for the padding that we will do for batch processing
    weights = torch.randn(n_words_vocab + 1, embedding_dim)
    weights[0] = torch.zeros(embedding_dim)
    for word, lang_word_index in language.word2index.items():
        if word in word2vec_model:
            weights[lang_word_index] = torch.FloatTensor(word2vec_model.word_vec(word))
    
    return weights


def save_model(model, path):
    """ Saves a pytorch model locally """
    return torch.save(model.state_dict(), path)

def load_model(model, path):
    """ Loads a model locally """
    m = model(n_classes=10)
    # load the state dict and pass it to the load_state_dict function
    return m.load_state_dict(torch.load("./model.pt"))


def plotConfusionMatrix(y, y_pred, classes, title=None):
    """ Plots a confusion matrix """
    cm = confusion_matrix(y, y_pred)
    ax = sns.heatmap(cm, xticklabels=classes, yticklabels=classes, annot=True, fmt='0.2g', cmap=plt.cm.Blues)
    bottom, top = ax.get_ylim()
    ax.set_ylim(bottom + 0.5, top - 0.5)
    if title:
        ax.set_title(title)
    ax.set_xlabel('Predictions')
    ax.set_ylabel('Test Set')
    plt.show()

    return cm