from torch.utils.data import Dataset

class QuoraDataset(Dataset):
    def __init__(self, questions_list, word2index, labels):
        """
        Params:
        -------
        questions_list : list
                         list with tuples of all the questions pairs 
        
        word2index : dict
                     vocbulary of the dataset
        labels : list 
                 list of the corrsponding labels to the question pairs 
        
        """
        self.questions_list = questions_list
        self.labels = labels
        self.word2index = word2index
        
    def __len__(self):
        return len(self.questions_list)
    
    def __getitem__(self, index):
        questions_pair = self.questions_list[index]
        q1 = questions_pair[0]
        q1_indices = []
        for word in q1.split():
            q1_indices.append(self.word2index[word])
            
        q2 = questions_pair[1]
        q2_indices = []
        for word in q2.split():
            q2_indices.append(self.word2index[word])
            
        # q1_indices and q2_indices are lists of indices against words used in the sentence 
        return {
            'q1': q1,
            'q2': q2,
            'q1_token': q1_indices, 
            'q2_token': q2_indices, 
            'labels': self.labels[index], 
        }


def collate(batch):
    q1_text_list = []
    q2_text_list = []
    q1_list = []
    q2_list = []
    labels = []
    for item in batch:
        q1_text_list.append(item['q1'])
        q2_text_list.append(item['q2'])
        q1_list.append(item['q1_token'])
        q2_list.append(item['q2_token'])
        labels.append(item['labels'])
          
        
    q1_lengths = [len(q) for q in q1_list]
    q2_lengths = [len(q) for q in q2_list]
    
    return {
        'q1_text': q1_text_list,
        'q2_text': q2_text_list, 
        'q1_token': q1_list, 
        'q2_token': q2_list,
        'q1_lengths': q1_lengths, 
        'q2_lengths': q2_lengths,
        'labels': labels
    }