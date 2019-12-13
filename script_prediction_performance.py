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

# Model hyper parameters.
number_of_layers = 1
hidden_nodes = 64
input_classes = ['speaker', 'dialogue_act', 'level', 'utterance_length']

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
    data = DataSet()
    n_classes = data.get_number_of_classes()

    # This is the shape of the predictions and labels after the folds (44595, 3)

    # (15219, 7)
    print(preprocessed.data.shape)

    # 45539
    print(preprocessed.data.shape[0] * 3 - preprocessed.number_of_dialogues)

    # Performs cross-validation.
    cross_validation = CrossValidation(data, k)
    cross_validation.make_k_fold_cross_validation_split(levels)
    scores = cross_validation.validate(n_classes, hidden_nodes, number_of_layers, learning_rate, batch_size, epochs,
                                       input_classes, save_labels_predictions=True)
    entry = np.array([sequence_length, np.mean(scores), np.std(scores)]).reshape(-1, 3)
    output = np.concatenate((output, entry), axis=0)

# Makes a pandas DataFrame to output to a .csv file.
data = pd.DataFrame(output, columns=['sequence length', 'accuracy', 'SD'])
data.to_csv('analyses/accuracy_per_sequence_length.csv', index=None, header=True)