import pandas as pd
import numpy as np
from crossvalidation import CrossValidation
from data import DataSet, Preprocessing


""" This is a script that performs cross-validation on an lstm model given the different settings below and keeps track
    of the predictions being made.

    The variables that need to be specified:
        sequence_lengths    = a list containing different sequence lengths
        levels              = a list containing the ability levels in the data to be considered
        k                   = scalar specifying the fold of the cross-validation
        weighted            = a string specifying whether the model should be weighted or not {'weighted', 'unweighted'}
        number_of_layers    = scalar specifying the number of layers in the lstm model
        hidden_nodes        = scalar specifying the number of hidden nodes in the LSTM model's layers
        learning_rate       = scalar specifying the learning rate of the LSTM's training 
        batch_size          = scalar specifying the batch size of the LSTM's training
        epochs              = scalar specifying the number of epochs during the LSTM's training

    The script outputs a matrix containing the mean accuracy of the cross-validation per sequence length        
"""

# Analysis parameters.
# sequence_lengths = [20, 15, 10, 7, 5, 3, 2]
sequence_lengths = [20, 15, 10, 7, 5, 3, 2]
levels = [1, 2, 3, 4]
k = 10
weighted = 'weighted'

# Model hyper parameters.
number_of_layers = 1
hidden_nodes = 50

# Training hyper parameters.
learning_rate = 0.001
batch_size = 16
epochs = 20

data_frame = pd.read_csv('data/DA_labeled_belc_2019.csv')

output = np.empty((1, 3))
for sequence_length in sequence_lengths:
    print("Cross-validation for sequence length {}".format(sequence_length))

    # Preprocesses the data for the sequence length.
    preprocessed = Preprocessing('data/DA_labeled_belc_2019.csv')
    preprocessed.save_dialogues_as_matrices(sequence_length=sequence_length, store_index=True)
    data = DataSet()

    # Initialise cross validator.
    cross_validation = CrossValidation(data, k)
    cross_validation.make_k_fold_cross_validation_split(levels)

    # Performs cross-validation.
    labels_predictions, scores = cross_validation.validate(learning_rate, batch_size, epochs,
                                                           hidden_nodes=hidden_nodes, save_labels_predictions=True,
                                                           weighted=weighted)

    # Stores the labels and predictions in a DataFrame.
    input_frame = pd.DataFrame(labels_predictions)
    columns = input_frame.columns
    input_frame = input_frame.set_index(columns[0]).rename_axis(None)
    input_frame = input_frame.rename(columns={columns[1]: 'labels_seq_len_' + str(sequence_length),
                                     columns[2]: 'predictions_seq_len_' + str(sequence_length)})
    input_frame = input_frame.astype(str)

    # Replaces all the numerical values of the labels and predictions with their name.
    DAs = preprocessed.DAs
    for i in range(len(DAs)):
        input_frame = input_frame.replace({str(i) + '.0': DAs[i]})

    # Adds the labels and predictions for this input as columns to the original data in one DataFrame.
    data_frame = data_frame.merge(input_frame, how='left', left_index=True, right_index=True)

# Saves the DataFrame containing all the labels and predictions for the different input settings.
data_frame.to_csv('analyses/' + weighted + '_model_predictions.csv')
