# coding: utf-8
import torch
import torch.nn as nn
import torch.optim as optim
import model

# Global Variables initialisation
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def main():
    """ Runs the RNN algorithm. """

    model = model.RNNModel(n_input, hidden_nodes, n_layers, n_classes).to(device)
    return 0

###################################################################################
#                            HELPER FUNCTIONS                                     #
###################################################################################

def train(model, data, learning_rate, batch_size, epochs):
    """ Trains a given RNN model on a given preprocessed data set with a specified learning rate,
        batch size and number of epochs.

    Args
        model           = the RNN model that is to be trained
        data            = the data set that the RNN model is going to train on
        learning_rate   = learning rate
        batch_size      = the number of data points used in training at the same time
        epochs          = the number of times the RNN trains on the same data points
    """
    criterion = nn.CrossEntropyLoss()

    # ADJUST FOR OTHER OPTIMISERS
    optimiser = optim.Adam(model.parameters())

    model.train()
    total_loss = 0
    for batch, labels in data:
        optimiser.zero_grad()

def evaluate(model, data, x):
    """ Returns the prediction evaluation scores precision, recall and F1 of the RNN model
        on a data sequences of length x after 10-fold cross-validation

        Args:
            model                      = RNN model to be evaluated
            data                       = data on which the RNN is evaluated
            x                          = the length of the sequences in the evaluation data

        Returns:
            (Precision, recall, F1)    = a tuple containing the scores of the precision, recall and F1 measures
    """

    return 0



def predict(model, input):
    """ Returns the predicted output of the RNN model given the input.

        Args:
            model   = a trained RNN model
            input   = a data point or several independent data points

        Returns:
            output  = predicted next data point or data points given the input
    """
    return 0

def predict_sequence(model, input, x):
    """ Returns the predicted output of the RNN model given the input sequence of length x.

            Args:
                model   = a trained RNN model
                input   = a data point sequence
                x       = the length of the input sequence

            Returns:
                output  = predicted next data point given the data point sequence
    """

#####################################################################################

main()