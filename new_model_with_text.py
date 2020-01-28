import torch.nn as nn
import torch
import numpy as np
import json
import pandas as pd

''' This file contains the LSTM model with text embeddings. '''

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class LSTM(nn.Module):
    """ Recurrent Neural Network (RNN) model"""

    def __init__(self, input_dimensions=[2, 13, 4], embedding_dimensions=[1, 7, 2] , hidden_nodes=64, n_layers=1,
                 n_classes=13, input_classes=['speaker', 'dialogue_act', 'level', 'utterance_length']):
        """ Initialises of the RNN

            Args:
                hidden_nodes    = the number of nodes in each hidden layer
                n_layers        = the number of hidden layers
                n_classes       = the number of output classes
        """

        super(LSTM, self).__init__()

        # Embed the data classes 'speaker', 'dialogue_act' and 'level'.
        self.speaker_embedding = nn.Embedding(input_dimensions[0], embedding_dimensions[0])
        self.DA_embedding = nn.Embedding(input_dimensions[1], embedding_dimensions[1])
        self.level_embedding = nn.Embedding(input_dimensions[2], embedding_dimensions[2])

        # Embed the words in the utterance text.
        weights_matrix = torch.tensor(np.load('weights_matrix.npy'))
        num_embeddings, embedding_dim = weights_matrix.shape
        self.word_embedding = nn.Embedding(num_embeddings, embedding_dim)

        # The word embedder has pretrained weights that should not be trained during runtime.
        self.word_embedding.load_state_dict({'weight': weights_matrix})
        self.word_embedding.weight.requires_grad = False

        # Stores the embeddings' dimensions.
        embedding_dimensions.append(embedding_dim)
        self.embedding_dimensions = embedding_dimensions

        # Stored the correct mapping from words to the corresponding word vector.
        # load the
        with open('word_vector_mapping.json', 'r') as f:
            self.word_vector_mapping = json.load(f)

        # The input is the embedding of the chosen classes.
        self.input_classes = input_classes
        self.input_dimension = 0
        self.input_dimension += embedding_dimensions[0]
        self.input_dimension += embedding_dimensions[1]
        self.input_dimension += embedding_dimensions[2]
        self.input_dimension += embedding_dimensions[3]
        self.input_dimension += 1

        # Model.
        self.lstm = nn.LSTM(self.input_dimension, hidden_nodes, n_layers)
        self.decoder = nn.Linear(hidden_nodes, n_classes)

        # Some parameters.
        self.hidden_nodes = hidden_nodes
        self.n_layers = n_layers
        self.n_classes = n_classes

        self.data = pd.read_csv('data/DA_labeled_belc_2019.csv')

    def forward(self, data, hidden_state):
        """ Takes the input, performs forward propagation and returns the current output and the hidden state

            Args:
                input           = the input data
                hidden_state    = the hidden state of the RNN storing memory
        """
        dims = data.shape
        data = data.long()

        # Constructs the right input according to which classes are wanted.
        input = torch.empty(dims[0], dims[1], 0).to(device)
        input = torch.cat((input, self.speaker_embedding(data[:, :, 0]).float()), dim=2)
        input = torch.cat((input, self.DA_embedding(data[:, :, 1]).float()), dim=2)
        input = torch.cat((input, self.level_embedding(data[:, :, 2]).float()), dim=2)
        input = torch.cat((input, data[:, :, 3].unsqueeze(2).float()), dim=2)

        # Compute the utterance text embedding for the batch.
        utterances = torch.empty(0, dims[1], self.embedding_dimensions[3]).to(device)
        for i in range(dims[0]):
            sub_sequence_indices = data[i, :, 4]
            sub_sequence_texts = self.data.loc[sub_sequence_indices.cpu().numpy(), ['text']]
            sub_sequence_texts = [str(t[0]).split() for t in sub_sequence_texts.values.tolist()]
            batch_utterances = torch.empty(0, self.embedding_dimensions[3]).to(device)
            for utterance_text in sub_sequence_texts:
                utterance_vector = torch.zeros(1, self.embedding_dimensions[3]).to(device)
                for word in utterance_text:
                    index = self.word_vector_mapping[word]
                    utterance_vector += self.word_embedding(torch.tensor([index]).to(device)).float()

                # Stores the i'th subsequence's utterance text of all the inputs in the batch.
                batch_utterances = torch.cat((batch_utterances, utterance_vector), dim=0)

            # Stores the subsequences of the batch together.
            utterances = torch.cat((utterances, batch_utterances.unsqueeze(0)), dim=0)

        # Concatenates the utterance text embeddings to the rest of the data.
        input = torch.cat((input, utterances), dim=2)

        output, hidden = self.lstm(input, hidden_state)
        decoded_output = self.decoder(output)
        return decoded_output, hidden
