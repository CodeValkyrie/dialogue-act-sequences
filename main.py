# coding: utf-8
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import precision_score, recall_score, f1_score
from model import LSTM
from data import Dataset, Preprocessing
import numpy as np

# Global Variables initialisation
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def main():
    """ Runs the RNN algorithm. """

    # Preprocesses the data.
    preprocessed = Preprocessing('data/DA_labeled_belc_2019.csv')
    preprocessed.save_dialogue_IDs()
    preprocessed.save_class_representation()
    preprocessed.save_dialogues_as_matrices(sequence_length=7)

    # Makes a Dataset object from the dataset.
    dataset = Dataset()

    dataset.make_k_fold_cross_validation_split([1,2,3], 10)
    # print(dataset.cross_validation_test_IDs)
    # print(len(dataset.cross_validation_test_IDs))
    # print(dataset.cross_validation_train_IDs)
    # print(len(dataset.cross_validation_train_IDs))

    # Defines hyperparameters for model initialisation.
    n_classes = dataset.get_number_of_classes()
    n_layers = 1
    hidden_nodes = 64

    rnn = LSTM(n_classes, hidden_nodes, n_layers).to(device)

    # Defines hyperparameters for training initialisation.
    learning_rate = 5e-3
    batch_size = 16
    epochs = 10
    train(rnn, dataset, learning_rate, batch_size, epochs)
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
            dialogue = data.get_dialogue(index, k, train_test):
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

def evaluate(model, data, labels):
    """ Returns the prediction evaluation scores precision, recall and F1 of the RNN model
        on a data sequences of length x after 10-fold cross-validation

        Args:
            model                      = RNN model to be evaluated
            data                       = data on which the RNN is evaluated
            labels                     = the labels of the data

        Returns:
            (Precision, recall, F1)    = a tuple containing the scores of the precision, recall and F1 measures
    """
    prediction = predict(model, data)
    precision = precision_score(labels, prediction)
    recall = recall_score(labels, prediction)
    f1 = f1_score(labels, prediction)
    return precision, recall, f1



def predict(model, input):
    """ Returns the predicted output of the RNN model given the input.

        Args:
            model   = a trained RNN model
            input   = a data point/sequence or several independent data points/sequences

        Returns:
            output  = predicted next data point or data points given the input
    """
    return torch.softmax(model(input, None)[0])

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