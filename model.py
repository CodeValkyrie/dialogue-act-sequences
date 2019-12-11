import torch.nn as nn
import torch


class LSTM(nn.Module):
    """ Recurrent Neural Network (RNN) model"""

    def __init__(self, input_dimensions=[2, 13, 4], embedding_dimensions=[4, 20, 10] , hidden_nodes=64, n_layers=1,
                 n_classes=13, input_classes=['speaker', 'dialogue_act', 'level', 'utterance_length']):
        """ Initialises of the RNN

            Args:
                hidden_nodes    = the number of nodes in each hidden layer
                n_layers        = the number of hidden layers
                n_classes       = the number of output classes
        """

        super(LSTM, self).__init__()

        # Embed the data classes 'speaker', 'dialogue_act' and 'level'
        self.speaker_embedding = nn.Embedding(input_dimensions[0], embedding_dimensions[0])
        self.DA_embedding = nn.Embedding(input_dimensions[1], embedding_dimensions[1])
        self.level_embedding = nn.Embedding(input_dimensions[2], embedding_dimensions[2])

        # The input is the embedding of the chosen classes.
        self.input_classes = input_classes
        self.input_dimension = 0
        if 'speaker' in self.input_classes:
            self.input_dimension += embedding_dimensions[0]
        if 'dialogue_act' in self.input_classes:
            self.input_dimension += embedding_dimensions[1]
        if 'level' in self.input_classes:
            self.input_dimension += embedding_dimensions[2]
        if 'utterance_length' in self.input_classes:
            self.input_dimension += 1

        self.embedding_dimensions = embedding_dimensions

        # Model.
        self.lstm = nn.LSTM(self.input_dimension, hidden_nodes, n_layers)
        self.decoder = nn.Linear(hidden_nodes, n_classes)

        # Some parameters.
        self.hidden_nodes = hidden_nodes
        self.n_layers = n_layers
        self.n_classes = n_classes

    def forward(self, data, hidden_state):
        """ Takes the input, performs forward propagation and returns the current output and the hidden state

            Args:
                input           = the input data
                hidden_state    = the hidden state of the RNN storing memory
        """
        dims = data.shape
        data = data.long()

        # Constructs the right input according to which classes are wanted.
        input = torch.empty(dims[0], dims[1], 0)
        if 'speaker' in self.input_classes:
            input = torch.cat((input, self.speaker_embedding(data[:,:,0]).float()), dim=2)
        if 'dialogue_act' in self.input_classes:
            input = torch.cat((input, self.DA_embedding(data[:,:,1]).float()), dim=2)
        if 'level' in self.input_classes:
            input = torch.cat((input, self.level_embedding(data[:,:,2]).float()), dim=2)
        if 'utterance_length' in self.input_classes:
            input = torch.cat((input, data[:, :, 3].unsqueeze(2).float()), dim=2)

        output, hidden = self.lstm(input, hidden_state)
        decoded_output = self.decoder(output)
        return decoded_output, hidden
