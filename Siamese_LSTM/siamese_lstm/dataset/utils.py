import re
import nltk 
from nltk.stem import WordNetLemmatizer, SnowballStemmer
from nltk.corpus import stopwords


def text_to_wordlist(text, remove_stopwords, stem_words):
    """ 
    This function was adapoted from 
    https://www.kaggle.com/currie32/quora-question-pairs/the-importance-of-cleaning-text
    
    Description:
        - Clean the text, with the option to remove stopwords and to stem words.
        - Convert words to lower case and split them 
    
    Params:
    -------
    text : str
           question string 
    remove_stopwords : bool
                       if True --> removes stopwords, if False --> does not remove stopwords 
    stem_words : bool
                 if True --> stem stopwords, if False --> normal
    
    return:
    -------
    text : str
           cleaned questing string 
    """
    text = text.lower().split()

    # Optionally, remove stop words
    if remove_stopwords:
        stops = set(stopwords.words("english"))
        text = [w for w in text if not w in stops]
    
    text = " ".join(text)

    # Clean the text
    text = re.sub(r"[^A-Za-z0-9^,!.\/'+-=]", " ", text)
    text = re.sub(r"what's", "what is ", text)
    text = re.sub(r"\'s", " ", text)
    text = re.sub(r"\'ve", " have ", text)
    text = re.sub(r"can't", "cannot ", text)
    text = re.sub(r"n't", " not ", text)
    text = re.sub(r"i'm", "i am ", text)
    text = re.sub(r"\'re", " are ", text)
    text = re.sub(r"\'d", " would ", text)
    text = re.sub(r"\'ll", " will ", text)
    text = re.sub(r",", " ", text)
    text = re.sub(r"\.", " ", text)
    text = re.sub(r"!", " ! ", text)
    text = re.sub(r"\/", " ", text)
    text = re.sub(r"\^", " ^ ", text)
    text = re.sub(r"\+", " + ", text)
    text = re.sub(r"\-", " - ", text)
    text = re.sub(r"\=", " = ", text)
    text = re.sub(r"'", " ", text)
    text = re.sub(r"(\d+)(k)", r"\g<1>000", text)
    text = re.sub(r":", " : ", text)
    text = re.sub(r" e g ", " eg ", text)
    text = re.sub(r" b g ", " bg ", text)
    text = re.sub(r" u s ", " american ", text)
    text = re.sub(r"\0s", "0", text)
    text = re.sub(r" 9 11 ", "911", text)
    text = re.sub(r"e - mail", "email", text)
    text = re.sub(r"j k", "jk", text)
    text = re.sub(r"\s{2,}", " ", text)
    
    # Optionally, shorten words to their stems
    if stem_words:
        text = text.split()
        stemmer = SnowballStemmer('english')
        stemmed_words = [stemmer.stem(word) for word in text]
        text = " ".join(stemmed_words)
    
    # Return a list of words
    text = text.strip()
    return text


def convert_data_to_tuples(df, remove_stopwords, stem_words):
    questions_pair = []
    labels = []
    for _, row in df.iterrows():

        q1 = text_to_wordlist(str(row['question1']), remove_stopwords, stem_words)
        q2 = text_to_wordlist(str(row['question2']), remove_stopwords, stem_words)
        label = int(row['is_duplicate'])
        if q1 and q2:
            questions_pair.append((q1, q2))
            labels.append(label)

    print ('Question Pairs: ', len(questions_pair))
    return questions_pair, labels


