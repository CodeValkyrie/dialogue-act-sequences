import pandas as pd
import numpy as np
from crossvalidation import CrossValidation
from data import DataSet, Preprocessing


""" This is a script that performs cross-validation on a n-gram language models given the different settings 
    below and keeps track of the predictions being made.

    The variables that need to be specified:
        levels  = a list containing the ability levels in the data to be considered
        k       = scalar specifying the fold of the cross-validation
        n       = a list containing the different lengths of n-grams over which to measure prediction performance

    The script outputs a .csv containing the predictions and correct labels per n-gram model.     
"""

levels = [1, 2, 3, 4]
k = 10

# Prediction performance for bigram and trigram models.
n = [2, 3]

output = np.empty((1, 3))

# Preprocesses the data for the sequence length.
preprocessed = Preprocessing('data/DA_labeled_belc_2019.csv')
data_frame = preprocessed.data
data = DataSet()

# Initialise cross validator.
cross_validation = CrossValidation(data, k)
cross_validation.make_k_fold_cross_validation_split(levels)

# Performs cross validation on different n-gram language models.
for n_gram in n:
    print("Cross-validation for input {}-gram".format(n_gram))

    # Performs cross-validation.
    labels_predictions, scores = cross_validation.validate_n_gram(data_frame, n_gram)

    # Adds the labels and predictions for this input as columns to the original data in one DataFrame.
    data_frame = data_frame.merge(labels_predictions, how='left', left_index=True, right_index=True)

# Saves the DataFrame containing all the labels and predictions for the different n_gram models.
data_frame.to_csv('analyses/n_gram_models_predictions.csv')
