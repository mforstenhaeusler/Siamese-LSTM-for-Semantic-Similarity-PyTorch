
class Language:
    def __init__(self):
        """ 
        Language class keeps track of the datasets vocabulary and creates 
        a words to index dictionary that will be required in the pytroch dataset
        """
        self.word2index = {}  # sets index accodringly to unique ness - most common lower index e.g.1 
        self.word2count = {}  # counts each unique word 
        self.index2word = {}  # reverse of word3index
        self.n_words = 0

    def addSentence(self, sentence):
        for word in sentence.split(' '):
            self.addWord(word)

    def addWord(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.n_words + 1
            self.word2count[word] = 1
            self.index2word[self.n_words + 1] = word
            self.n_words += 1
        else:
            self.word2count[word] += 1