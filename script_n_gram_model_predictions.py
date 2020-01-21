import pandas as pd
import numpy as np
from crossvalidation import CrossValidation
from data import DataSet, Preprocessing, Statistics


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
    labels_predictions= cross_validation.validate_n_gram(data_frame, n_gram)

    # Adds the labels and predictions for this input as columns to the original data in one DataFrame.
    data_frame = data_frame.merge(labels_predictions, how='left', left_index=True, right_index=True)

# Saves the DataFrame containing all the labels and predictions for the different n_gram models.
data_frame.to_csv('analyses/n_gram_models_predictions.csv')

########################################################################################################################
#                               COMPUTING THE PRECISION, RECALL AND F1-SCORES                                          #
########################################################################################################################

# Reads in the data containing the predictions of a model under the given settings.
data = Preprocessing('analyses/n_gram_models_predictions.csv')
statistics = Statistics(data)

# Gets the precision, recall and f1-score for every dialogue act for different model input settings.
for n_gram in n:
    accuracy_dict = dict()
    accuracy_frame = data.data[['labels_' + str(n_gram) + '_gram', 'predictions_' + str(n_gram) + '_gram']].dropna()
    accuracy = accuracy_frame[accuracy_frame['labels_' + str(n_gram) + '_gram'] ==
                              accuracy_frame['predictions_' + str(n_gram) + '_gram']].shape[0] / accuracy_frame.shape[0]
    print("The accuracy of the " + str(n_gram) + "-gram model is: ", accuracy)

    columns = ['labels_' + str(n_gram) + '_gram', 'predictions_' + str(n_gram) + '_gram']
    confusion_matrix = (statistics.get_normalised_confusion_matrix(data.data, columns) * 100).round(2)
    confusion_matrix.to_csv('analyses/' + str(n_gram) + 'gram_error_analysis.csv')
    for dialogue_act in data.DAs:
        precision, recall, f1 = statistics.precision_recall_f1(data.data, columns, dialogue_act)

        if 'all_levels' not in accuracy_dict.keys():
            accuracy_dict['all_levels'] = dict()
            accuracy_dict['all_levels']['p'] = dict()
            accuracy_dict['all_levels']['r'] = dict()
            accuracy_dict['all_levels']['f1'] = dict()
        accuracy_dict['all_levels']['p'][dialogue_act] = precision
        accuracy_dict['all_levels']['r'][dialogue_act] = recall
        accuracy_dict['all_levels']['f1'][dialogue_act] = f1

        for level in data.levels:
            level_data = data.data[data.data['level'] == level]
            precision, recall, f1 = statistics.precision_recall_f1(level_data, columns, dialogue_act)

            if 'level_' + str(level) not in accuracy_dict.keys():
                accuracy_dict['level_' + str(level)] = dict()
                accuracy_dict['level_' + str(level)]['p'] = dict()
                accuracy_dict['level_' + str(level)]['r'] = dict()
                accuracy_dict['level_' + str(level)]['f1'] = dict()
            accuracy_dict['level_' + str(level)]['p'][dialogue_act] = precision
            accuracy_dict['level_' + str(level)]['r'][dialogue_act] = recall
            accuracy_dict['level_' + str(level)]['f1'][dialogue_act] = f1

    # Code below adapted from https://stackoverflow.com/questions/13575090/construct-pandas-dataframe-from-items-in-nested-dictionary
    # Prepares the separate parts of the dictionary to be stored in one DataFrame.
    level_accuracies = []
    frames = []
    for level_accuracy, accuracy_scores in accuracy_dict.items():
        level_accuracies.append(level_accuracy)
        frames.append(pd.DataFrame.from_dict(accuracy_scores, orient='index'))

    # Stores all the accuracy scores for all the levels over all the dialogue acts into a DataFrame.
    accuracies = pd.concat(frames, keys=level_accuracies).T.round(4)

    # Saves the accuracy DataFrame to a .csv file.
    accuracy_file = 'analyses/model_' + str(n_gram) + 'gram_baseline_accuracy.csv'
    accuracies.to_csv(accuracy_file)

