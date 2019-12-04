# coding: utf-8
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
from model import LSTM
from data import DataSet

# Global Variables initialisation
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main():
    """ Runs the RNN algorithm. """

    # Makes a Dataset object from the dataset.
    dataset = DataSet()

    # Defines hyperparameters for model initialisation.
    n_classes = dataset.get_number_of_classes()
    n_layers = 1
    hidden_nodes = 64

    lstm = LSTM(n_classes, hidden_nodes, n_layers).to(device)

    # Defines hyperparameters for training initialisation.
    learning_rate = 5e-3
    batch_size = 16
    epochs = 10
    train(lstm, dataset, learning_rate, batch_size, epochs)
    return 0

###################################################################################
#                            HELPER FUNCTIONS                                     #
###################################################################################


def train(model, data, learning_rate, batch_size, epochs):
    """ Trains a given RNN model on a given preprocessed data set with a specified learning rate,
        batch size and number of epochs.

    Args
        model           = the RNN model that is to be trained
        data            = the preprocessed data set that the RNN model is going to train on
        learning_rate   = learning rate
        batch_size      = the number of data points used in training at the same time
        epochs          = the number of times the RNN trains on the same data points
    """
    criterion = nn.CrossEntropyLoss()

    # ADJUST FOR OTHER OPTIMISERS
    optimiser = optim.RMSprop(model.parameters(), lr=learning_rate)

    model.train()
    counter = 0
    total_loss = 0
    for epoch in range(epochs):
        for dialogue in data:
            batches_labels = data.get_batch_labels(dialogue, batch_size)

            for batch, labels in batches_labels:
                optimiser.zero_grad()
                batch = batch.to(device)
                labels = labels.to(device)
                output, hidden = model(batch, None)

                # The output needs to be transposed to (batch, number_of_classes, sequence_length) for the criterion.
                # The output can stay 3D but the labels must be 2D, so the following takes the argmax of the labels
                loss = criterion(output.permute(1, 2, 0), torch.argmax(labels, dim=2).permute(1, 0))

                loss.backward()

                # Sets the maximum gradient norm to counteract the exploding gradient problem.
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)

                optimiser.step()
                total_loss += loss.item()
                counter += 1
        print(total_loss / counter)


def evaluate(model, data):
    """ Returns the prediction evaluation scores precision, recall and F1 of the RNN model
        on a data sequences of length x after 10-fold cross-validation

        Args:
            model                      = RNN model to be evaluated
            data                       = data on which the RNN is evaluated
            labels                     = the labels of the data

        Returns:
            (Precision, recall, F1)    = a tuple containing the scores of the precision, recall and F1 measures
    """
    i = 0
    accuracy_total = 0
    model.eval()
    for dialogue in data:
        batches_labels = data.get_batch_labels(dialogue, batch_size=16)
        for batch, labels in batches_labels:
            labels = torch.argmax(labels, dim=2).numpy().reshape(-1)
            prediction = torch.argmax(predict(model, batch), dim=2).detach().numpy().reshape(-1)
            accuracy_total += accuracy_score(labels, prediction)
            i += 1
    return accuracy_total / i


def predict(model, input):
    """ Returns the predicted output of the RNN model given the input.

        Args:
            model   = a trained RNN model
            input   = a data point/sequence or several independent data points/sequences

        Returns:
            output  = predicted next data point or data points given the input
    """
    return torch.softmax(model(input, None)[0], dim=2)

def generate_sequence(model, input, x):
    """ Returns a generated sequence of length x given an input.

            Args:
                model   = a trained RNN model
                input   = a data point
                x       = the length of the generated sequence

            Returns:
                output  = generated data sequence of length x
    """
    hidden_state = None
    sequence = [input]
    for i in range(x):
        output, hidden_state = model(sequence[-1], hidden_state)
        sequence.append(output)
    return sequence

#####################################################################################

main()