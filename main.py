# coding: utf-8
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
from new_model_with_text import LSTM
from data import DataSet
from nltk.util import ngrams
from nltk.lm import MLE
from nltk.lm.preprocessing import padded_everygram_pipeline

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


def train(model, data, learning_rate, batch_size, epochs, weighted='unweighted'):
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

    # Initialises the CrossEntropyLoss with weights for each class according to the distribution.
    if weighted == 'weighted':
        distribution = pd.read_csv('analyses/dialogue_act_distribution.csv', index_col=[0], header=None).sort_index()
        distribution = distribution.max() / distribution
        criterion = nn.CrossEntropyLoss(weight=torch.tensor(distribution.to_numpy().flatten()).float().to(device))

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
                loss = criterion(output.permute(1, 2, 0), labels.permute(1, 0).long())

                loss.backward()

                # Sets the maximum gradient norm to counteract the exploding gradient problem.
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)

                optimiser.step()
                total_loss += loss.item()
                counter += 1
        print('loss', total_loss / counter)


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
            batch = batch.to(device)
            labels = labels.to(device)
            if labels_predictions is None:
                labels_predictions = np.empty((0, 3))

            # If the predictions and labels must be stored, stores the labels and predictions with their input's index.
            predictions = torch.argmax(predict(model, batch), dim=2)
            if save_labels_predictions:

                # Only stores the last labels and predictions in a sequence because these are the most fine-tuned.
                labels_to_store = np.expand_dims(labels[-1].detach().cpu().numpy(), axis=1)
                index_to_store = np.expand_dims(batch[-1, :, 4].detach().cpu().numpy(), axis=1)
                predictions_to_store = np.expand_dims(predictions[-1, :].detach().cpu().numpy(), axis=1)
                labels_predictions_batch = np.concatenate((index_to_store, labels_to_store, predictions_to_store), axis=1)
                labels_predictions = np.concatenate((labels_predictions, labels_predictions_batch), axis=0)

            # Computes the accuracy score.
            labels = labels.detach().cpu().numpy().reshape(-1)
            predictions = predictions.detach().cpu().numpy().reshape(-1)
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

def train_n_gram(data, ids, n):
    """ Returns an n-gram model.

        Args:
            data    - the original data frame containing all the data
            ids     - the IDs of the dialogues the n-gram model will train on
            n       - the length of the n-gram model

        Returns:
            NLTK NgramCounter containing the counts of all the n-grams in the training set.
    """
    # Extracts all the dialogue act classes.
    unique_dialogue_acts = sorted(list(set(data['dialogue_act'])))

    # Makes n-grams of the all the dialogues with the given ids.
    training_dialogues = []
    for ID in ids:
        training_dialogue = list(data[data['dialogue_id'] == ID]['dialogue_act'])
        training_dialogues.append(training_dialogue)

    # Get the every n-gram up to n from the training dialogues.
    n_grams = []
    for i in range(n):
        n_grams = n_grams + [list(ngrams(dialogue, n - i)) for dialogue in training_dialogues]

    # Trains the n-gram model on the dialogue n-grams and the unique dialogue acts.
    lm = MLE(n)
    lm.fit(n_grams, unique_dialogue_acts)
    return lm

def evaluate_n_gram(model, data, ids, n):
    """ Returns a DataFrame containing the predictions and labels of the n-gram model indexed by the input indices.

            Args:
                model   - NLTK NgramCounter containing the counts of all the n-grams during training
                data    - the original data frame containing all the data
                ids     - the IDs of the dialogues the n-gram model will test on
                n       - the length of the n-gram model

            Returns:
                DataFrame containing the predictions and labels of the n-gram model indexed by the input indices.
        """

    # Makes predictions for all the dialogue acts in the dialogues with the given ids.
    test_labels_predictions = pd.DataFrame()
    for ID in ids:
        test_dialogue = pd.DataFrame(data[data['dialogue_id'] == ID]['dialogue_act'])

        # The labels of the inputs are the inputs with an offset of 1 down.
        test_dialogue['labels_' + str(n) + '_gram'] = test_dialogue['dialogue_act'].shift(-1)

        # Gets the list of inputs of which the predictions must be computed.
        test_dialogue_inputs = list(test_dialogue['dialogue_act'].to_numpy())

        # The first and second input are needed for the trigram model, so the first inputs prediction is not made.
        test_dialogue_predictions = [np.nan]

        # Makes predictions for each of the dialogue input turns according to the model used.
        for i in range(1, len(test_dialogue_inputs) - 1):
            sequence = []

            # Compute the right context for the n-gram model.
            if n == 2:
                sequence = [test_dialogue_inputs[i]]
            elif n == 3:
                sequence = [test_dialogue_inputs[i-1], test_dialogue_inputs[i]]

            # Predict the next DA given the context
            test_dialogue_predictions.append(model.generate(text_seed=sequence))

        # The last input of a dialogue does not have a label so the last prediction is NaN
        test_dialogue_predictions.append(np.nan)

        # Stores the labels and predictions of the dialogues, together with their input indices, into a DataFrame.
        test_dialogue['predictions_' + str(n) + '_gram'] = test_dialogue_predictions
        test_labels_predictions = pd.concat([test_labels_predictions, test_dialogue[['labels_' + str(n) + '_gram', 'predictions_' + str(n) + '_gram']]])
    return test_labels_predictions

#####################################################################################

# main()
