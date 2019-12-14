import torch
import pandas as pd
import numpy as np
from crossvalidation import CrossValidation
from data import DataSet, Preprocessing
from model import LSTM

""" This is a script that performs cross-validation on an lstm model given the different settings below and keeps track
    of the predictions being made.

    The variables that need to be specified:
        sequence_lengths    = a list containing different sequence lengths
        levels              = a list containing the ability levels in the data to be considered
        k                   = scalar specifying the fold of the cross-validation
        number_of_layers    = scalar specifying the number of layers in the lstm model
        hidden_nodes        = scalar specifying the number of hidden nodes in the LSTM model's layers
        input_classes       = a list containing the classes that must be used as input
        learning_rate       = scalar specifying the learning rate of the LSTM's training 
        batch_size          = scalar specifying the batch size of the LSTM's training
        epochs              = scalar specifying the number of epochs during the LSTM's training

    The script outputs a matrix containing the mean accuracy of the cross-validation per sequence length        
"""

# Analysis parameters.
# sequence_lengths = [20, 15, 10, 7, 5, 3, 2]
sequence_lengths = [3]
levels = [1, 2, 3, 4]
k = 10
weighted = 'unweighted'

# Model hyper parameters.
number_of_layers = 1
hidden_nodes = 64
input_classes = ['dialogue_act', 'speaker', 'level', 'utterance_length']

# Training hyper parameters.
learning_rate = 5e-3
batch_size = 16
epochs = 20

output = np.empty((1, 3))
for sequence_length in sequence_lengths:
    print("Cross-validation for sequence length {}".format(sequence_length))

    # Preprocesses the data for the sequence length.
    preprocessed = Preprocessing('data/DA_labeled_belc_2019.csv')
    preprocessed.save_dialogues_as_matrices(sequence_length=sequence_length, store_index=True)
    data_frame = preprocessed.data
    data = DataSet()

    # Initialise cross validator.
    cross_validation = CrossValidation(data, k)
    cross_validation.make_k_fold_cross_validation_split(levels)

    # Performs cross validation on different subsets of classes as input parameters.
    for subsection in range(len(input_classes)):
        classes = input_classes[:subsection + 1]
        input = '_'.join([c[0] for c in classes])
        print("Cross-validation for input {}".format(classes))

        # Performs cross-validation.
        labels_predictions, scores = cross_validation.validate(learning_rate, batch_size, epochs,
                                                               classes, save_labels_predictions=True)

        # Stores the labels and predictions in a DataFrame.
        input_frame = pd.DataFrame(labels_predictions)
        columns = input_frame.columns
        input_frame = input_frame.set_index(columns[0]).rename_axis(None)
        input_frame = input_frame.rename(columns={columns[1]: 'labels_' + input, columns[2]: 'predictions_' + input})
        input_frame = input_frame.astype(str)

        # Replaces all the numerical values of the labels and predictions with their name.
        DAs = preprocessed.DAs
        for i in range(len(DAs)):
            input_frame = input_frame.replace({str(i) + '.0': DAs[i]})

        # Adds the labels and predictions for this input as columns to the original data in one DataFrame.
        data_frame = data_frame.merge(input_frame, how='left', left_index=True, right_index=True)

    # Saves the DataFrame containing all the labels and predictions for the different input settings.
    data_frame.to_csv('analyses/' + weighted + '_model_sequence_length_' + str(sequence_length) + '_predictions.csv')