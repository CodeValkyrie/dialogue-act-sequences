import pandas as pd
import numpy as np
from crossvalidation_new_model import CrossValidation
from data import DataSet, Preprocessing

""" This is a script that performs cross-validation on an LSTM model with sentence embeddings given the different 
    settings below.

    The variables that need to be specified:
        hidden_dimensions       = a list containing numbers of hidden nodes to do the search over.    

    The script outputs a matrix containing the mean accuracy of the cross-validation per hyper parameter setting.     
"""

# All the hyper parameters this search will loop over.
hidden_dimensions = [30, 35, 40, 45, 50]

# Hyper parameters that are fixed.
levels = [1, 2, 3, 4]
weighted = 'weighted'
sequence_length = 3
k = 10
number_of_layers = 1
learning_rate = 0.001
batch_size = 16
epochs = 20

# Need to store convergence in table to determine number of epochs.

# Preprocesses the data for the sequence length.
preprocessed = Preprocessing('data/train_belc_das_2020.csv')
preprocessed.save_dialogues_as_matrices(sequence_length=3, store_index=True)

# Loops over all the settings, computes the accuracy and outputs it into a data frame.
output = np.empty((1, 3)).astype(str)

for hidden_dimension in hidden_dimensions:
    print("Cross-validation for hidden dimension {}".format(hidden_dimension))

    data = DataSet()

    # Performs cross-validation.
    cross_validation = CrossValidation(data, k)
    cross_validation.make_k_fold_cross_validation_split(levels)
    scores = cross_validation.validate(learning_rate, batch_size, epochs, hidden_nodes=hidden_dimension, weighted=weighted)

    # Store the mean accuracy and standard deviation over the cross validation per setting in a Numpy array.
    setting_name = '_'.join([weighted, 'lr', str(learning_rate), 'hidden', str(hidden_dimension)])
    entry = np.array([setting_name, str(np.mean(scores)), str(np.std(scores))]).reshape(-1, 3)
    output = np.concatenate((output, entry), axis=0)
    print(output)

# Makes a pandas DataFrame to output to a .csv file.
data = pd.DataFrame(output, columns=['sequence length', 'accuracy', 'SD'])
data.to_csv('analyses/accuracy_per_hyper_parameter_setting.csv', index=None, header=True)
