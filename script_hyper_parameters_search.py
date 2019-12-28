import pandas as pd
import numpy as np
from crossvalidation import CrossValidation
from data import DataSet, Preprocessing

""" This is a scrpt that performs cross-validation on an lstm model given the different settings below.

    The variables that need to be specified:
    

    The script outputs a matrix containing the mean accuracy of the cross-validation per hyper parameter setting.     
"""

# All the hyper parameters this search will loop over.
learning_rates = [0.1, 0.01, 0.001, 0.0001, 0.00001]
hidden_dimensions = [12, 16, 20]
embedding_dimensions = [[1, 7, 2, 1], [2, 13, 4, 1]]
weighted = ['weighted', 'unweighted']

# Hyper parameters that are fixed.
batch_size = 16
epochs = 20
sequence_length = 3
levels = [1, 2, 3, 4]
k = 10
number_of_layers = 1
input_classes = ['dialogue_act', 'speaker', 'level', 'utterance_length']

# Training hyper parameters.
batch_size = 16
epochs = 20

# Need to store convergence in table to determine number of epochs.

# Preprocesses the data for the sequence length.
preprocessed = Preprocessing('data/DA_labeled_belc_2019.csv')
preprocessed.save_dialogues_as_matrices(sequence_length=3)

# Loops over all the settings, computes the accuracy and outputs it into a data frame.
output = np.empty((1, 3)).astype(str)
for model in weighted:
    print("Cross-validation for {} model".format(model))
    for lr in learning_rates:
        print("Cross-validation for learning rate {}".format(lr))
        for hidden_dimension in hidden_dimensions:
            print("Cross-validation for hidden dimension {}".format(hidden_dimension))
            for embedding_dimension in embedding_dimensions:
                print("Cross-validation for embedding dimension {}".format(embedding_dimension))

                data = DataSet()

                # Performs cross-validation.
                cross_validation = CrossValidation(data, k)
                cross_validation.make_k_fold_cross_validation_split(levels)
                scores = cross_validation.validate(lr, batch_size, epochs, input_classes,
                                                   embedding_dimensions=embedding_dimension,
                                                   hidden_nodes=hidden_dimension, weighted=model)

                # Store the mean accuracy and standard deviation over the cross validation per setting in a Numpy array.
                setting_name = '_'.join(model, 'lr', str(lr), 'hidden', str(hidden_dimension), 'emb', str(embedding_dimension[0]))
                entry = np.array([setting_name, str(np.mean(scores)), str(np.std(scores))]).reshape(-1, 3)
                output = np.concatenate((output, entry), axis=0)
                print(output)

# Makes a pandas DataFrame to output to a .csv file.
data = pd.DataFrame(output, columns=['sequence length', 'accuracy', 'SD'])
data.to_csv('analyses/accuracy_per_hyper_parameter_setting.csv', index=None, header=True)
