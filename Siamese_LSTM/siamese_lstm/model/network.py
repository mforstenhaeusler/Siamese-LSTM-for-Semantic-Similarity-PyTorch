import torch
import torch.nn as nn
from torch.nn.modules import dropout
from torch.nn.utils.rnn import pad_sequence, pad_packed_sequence, pack_padded_sequence
import numpy as np

from .utils import create_pretrained_weights

class EmbeddingLSTMNet(nn.Module):
    def __init__(
            self,
            embedding_dim,
            hidden_cells,
            num_layers, 
            embedding_rquires_grad,
            pretrained_weights,
            dropout,
            simple,
            ):
        super(EmbeddingLSTMNet, self).__init__()
        """ 
        LSTM Network and embeddings from pretrained weights
        
            - 1 lstm is enough since weights are shared
        embedding_dim : int
                        embedding dimnesion
        hidden_cells : int 
                       number of hidden cells in LSTM
        num_layers :  int
                      number of layers
        embedding_requires_grad : bool
        pretrained_weights : torch.tensor
                             pre-trained weights tensor 
        dropout : float
                  indicates the dropout percentage
        simple : bool
                 selects the simplest model, only LSTM layer
        """
        self.dropout = nn.Dropout(dropout)
        self.lstm = nn.LSTM(
            input_size=embedding_dim, 
            hidden_size=hidden_cells, 
            num_layers=num_layers, 
            batch_first=True
        )
        self.fc1 = nn.Linear(hidden_cells, hidden_cells)
        self.fc = nn.Linear(hidden_cells, hidden_cells)
        self.relu = nn.ReLU()
        # initialize embeddings 
        self.embedding = nn.Embedding.from_pretrained(pretrained_weights)
        self.embedding.weight.requires_grad = embedding_rquires_grad

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.simple = simple

    def forward(self, question, lengths):
        """ 
        Params:
        -------
        question : (batch dim, sequence)
                   i.e. [ [i1, i2, i3],
                          [j1, j2, j4, j5] ]
        lenghts : list
                  list all the lengths of each question  
        
        Return:
        -------
        result : torch.tensor
                 output tesnor of of forward pass 
        """
        # Reverse the sequence lengths indices in decreasing order (pytorch requirement for pad and pack)
        sorted_indices = np.flipud(np.argsort(lengths))
        lengths = np.flipud(np.sort(lengths))
        lengths = lengths.copy()
        
        # Reorder questions in the decreasing order of their lengths
        ordered_questions = [torch.LongTensor(question[i]).to(self.device) for i in sorted_indices]
        # Pad sequences with 0s to the max length sequence in the batch
        ordered_questions = pad_sequence(ordered_questions, batch_first=True)
        # Retrieve Embeddings
        embeddings = self.embedding(ordered_questions).to(self.device)
        
        
        # Model forward 
        embeddings = self.dropout(embeddings)
        # Pack the padded sequences and pass it through LSTM
        packed = pack_padded_sequence(embeddings, lengths, batch_first=True)
        out, (hn, cn) = self.lstm(packed)
        # Unpack the padded sequence and pass it through the linear layers 
        unpacked, unpacked_len = pad_packed_sequence(out, batch_first=True, total_length=int(lengths[0]))
        if self.simple == False:
            out = self.fc1(unpacked)
            out = self.relu(out)
            out = self.fc(out)
        else:
            out = unpacked
        
        # Reorder the output to the original order in which the questions were passed
        result = torch.FloatTensor(out.size())
        for i, encoded_matrix in enumerate(out):
            result[sorted_indices[i]] = encoded_matrix
        return result


class SiameseNetwork(nn.Module):
    def __init__(self, embedding_lstm_net):
        super(SiameseNetwork, self).__init__()
        """
        Siamese LSTM Network 

        Params:
        -------
        embedding_lstm_net : nn.Module
                             embedded LSTM Network 
        """
        self.embedding = embedding_lstm_net
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
    def forward(self, q1, q2, q1_lengths, q2_lengths):
        """ Forward pass 
        Params:
        -------
        q1 : pad sequence tensor 
             question 1  
        q2 : pad sequence tensor 
             question 2  
        q1_lengths : torch.tensor
                      original lengths of each question 1
        q2_lengths : torch.tensor
                      original lengths of each question 1
        Returns:
        --------
        similarity_score : torch.tensor
        """
        output_q1 = self.embedding(q1, q1_lengths)
        output_q2 = self.embedding(q2, q2_lengths)
        similarity_score = torch.zeros(output_q1.size()[0]).to(self.device)
        # Calculate Similarity Score between both questions in a single pair
        for index in range(output_q1.size()[0]):
            # Sequence lenghts are being used to index and retrieve the activations before the zero padding since they were not part of original question
            q1 = output_q1[index, q1_lengths[index] - 1, :]
            q2 = output_q2[index, q2_lengths[index] - 1, :]
            similarity_score[index] = self.manhattan_distance(q1, q2)
        
        return similarity_score
    
    def manhattan_distance(self, q1, q2):
        """ Computes the Mannhatten distance between the two question tokens """
        return torch.exp(-torch.sum(torch.abs(q1 - q2), dim=0)).to(self.device)
