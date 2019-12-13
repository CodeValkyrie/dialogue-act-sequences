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
    n_classes = 13
    n_layers = 1
    hidden_nodes = 64

    input_dimensions = [2, 13, 4]
    embedding_dimensions = [4, 20, 10]

    lstm = LSTM(input_dimensions, embedding_dimensions, hidden_nodes, n_layers, n_classes).to(device)

    # Defines hyperparameters for training initialisation.
    learning_rate = 5e-3
    batch_size = 32
    epochs = 20
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
                output, hidden = model(batch[:, :, :4], None)

                # The output needs to be transposed to (batch, number_of_classes, sequence_length) for the criterion.
                # The output can stay 3D but the labels must be 2D, so the following takes the argmax of the labels
                loss = criterion(output.permute(1, 2, 0), labels.permute(1, 0).long())

                loss.backward()

                # Sets the maximum gradient norm to counteract the exploding gradient problem.
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)

                optimiser.step()
                total_loss += loss.item()
                counter += 1
        print(total_loss / counter)
        break


def evaluate(model, data, save_labels_predictions=False):
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
    labels_predictions = None
    for dialogue in data:
        batches_labels = data.get_batch_labels(dialogue, batch_size=16)
        for batch, labels in batches_labels:

            if labels_predictions is None:
                labels_predictions = np.empty((batch.shape[0], 0, 3))

            # If the predictions and labels must be stored, stores the labels and predictions with their input's index.
            if save_labels_predictions:
                labels_to_store = np.expand_dims(labels, axis=2)
                index_to_store = np.expand_dims(batch[:, :, 4], axis=2)
                predictions = np.expand_dims(torch.argmax(predict(model, batch[:, :, :4]), dim=2).detach().numpy(), axis=2)
                labels_predictions_batch = np.concatenate((index_to_store, labels_to_store, predictions), axis=2)
                labels_predictions = np.concatenate((labels_predictions, labels_predictions_batch), axis=1)

            # Computes the accuracy score.
            labels = labels.numpy().reshape(-1)
            predictions = predictions.reshape(-1)
            accuracy_total += accuracy_score(labels, predictions)
            i += 1
    print('accuracy', accuracy_total / i)
    if save_labels_predictions:
        return labels_predictions, accuracy_total / i
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

# main()