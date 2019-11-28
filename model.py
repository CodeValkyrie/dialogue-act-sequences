import torch.nn as nn

class RNN(nn.Module):
    """ Recurrent Neural Network (RNN) model"""

    def __init__(self, n_classes, hidden_nodes, n_layers):
        """ Initialises of the RNN

            Args:
                hidden_nodes    = the number of nodes in each hidden layer
                n_layers        = the number of hidden layers
                n_classes       = the number of output classes
        """

        super(RNN, self).__init__()
        self.rnn = nn.RNN(n_classes, hidden_nodes, n_layers).double()
        self.decoder = nn.Linear(hidden_nodes, n_classes).double()

        self.hidden_nodes = hidden_nodes
        self.n_layers = n_layers
        self.n_classes = n_classes

    def forward(self, input, hidden_state):
        """ Takes the input, performs forward propagation and returns the current output and the hidden state

            Args:
                input           = the input data
                hidden_state    = the hidden state of the RNN storing memory
        """
        output, hidden = self.rnn(input, hidden_state)
        decoded_output = self.decoder(output)
        return decoded_output, hidden