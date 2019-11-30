import torch
import pandas as pd
import numpy as np
from crossvalidation import CrossValidation
from data import DataSet, Preprocessing
from model import LSTM


""" This is a scrpt that performs cross-validation on an lstm model given the different settings below.

    The variables that need to be specified:
        sequence_lengths    = a list containing different sequence lengths
        levels              = a list containing the ability levels in the data to be considered
        k                   = scalar specifying the fold of the cross-validation
        number_of_layers    = scalar specifying the number of layers in the lstm model
        hidden_nodes        = scalar specifying the number of hidden nodes in the LSTM model's layers
        learning_rate       = scalar specifying the learning rate of the LSTM's training 
        batch_size          = scalar specifying the batch size of the LSTM's training
        epochs              = scalar specifying the number of epochs during the LSTM's training
        
    The script outputs a matrix containing the mean accuracy of the cross-validation per sequence length        
"""


# Analysis parameters.
sequence_lengths = [20, 15, 10, 7, 5, 3, 2]
levels = [1, 2, 3, 4]
k = 10

# Model hyper parameters.
number_of_layers = 1
hidden_nodes = 64

# Training hyper parameters.
learning_rate = 5e-3
batch_size = 16
epochs = 20

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

output = np.empty(1, 2)
for sequence_length in sequence_lengths:

    # Preprocesses the data for the sequence length.
    preprocessed = Preprocessing('data/DA_labeled_belc_2019.csv')
    preprocessed.save_dialogues_as_matrices(sequence_length=sequence_length)
    data = DataSet()
    n_classes = data.get_number_of_classes()

    # Initialises LSTM model.
    lstm = LSTM(n_classes, hidden_nodes, number_of_layers).to(device)

    # Performs cross-validation.
    cross_validation = CrossValidation(data, k)
    cross_validation.make_k_fold_cross_validation_split(levels)
    scores = cross_validation.validate(lstm, learning_rate, batch_size, epochs)
    output = np.concatenate((output, np.mean(scores)), axis=0)

# Makes a pandas DataFrame to output to a .csv file.
data = pd.DataFrame(output, columns=['sequence length','accuracy'])
data.to_csv('analyses/accuracy_per_sequence_length.csv', index=None, header=True)
