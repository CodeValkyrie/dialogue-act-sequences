import torch
import pandas as pd
import numpy as np
from crossvalidation import CrossValidation
from data import Preprocessing, Statistics

weighted = 'unweighted'
sequence_lengths = [3]
input_settings = ['_d', '_d_s', '_d_s_l', '_d_s_l_u']

for sequence_length in sequence_lengths:

    # Reads in the data containing the predictions of a model under the given settings.
    filename = 'analyses/' + str(weighted) + '_model_sequence_length_' + str(sequence_length) + '_predictions.csv'
    data = Preprocessing(filename)
    statistics = Statistics(data)

    # Gets the precision, recall and f1-score for every dialogue act for different model input settings.
    for input_setting in input_settings:
        accuracy_dict = dict()
        for dialogue_act in data.DAs:
            columns = ['labels' + input_setting, 'predictions' + input_setting]
            precision, recall, f1 = statistics.precision_recall_f1(data.data, columns, dialogue_act)

            if 'all_levels' in accuracy_dict.keys():
                accuracy_dict['all_levels']['p'][dialogue_act] = precision
                accuracy_dict['all_levels']['r'][dialogue_act] = recall
                accuracy_dict['all_levels']['f1'][dialogue_act] = f1
            else:
                accuracy_dict['all_levels'] = dict()
                accuracy_dict['all_levels']['p'] = dict()
                accuracy_dict['all_levels']['r'] = dict()
                accuracy_dict['all_levels']['f1'] = dict()

            for level in data.levels:
                level_data = data.data[data.data['level'] == level]
                precision, recall, f1 = statistics.precision_recall_f1(level_data, columns, dialogue_act)

                if 'level_' + str(level) in accuracy_dict.keys():
                    accuracy_dict['level_' + str(level)]['p'][dialogue_act] = precision
                    accuracy_dict['level_' + str(level)]['r'][dialogue_act] = recall
                    accuracy_dict['level_' + str(level)]['f1'][dialogue_act] = f1
                else:
                    accuracy_dict['level_' + str(level)] = dict()
                    accuracy_dict['level_' + str(level)]['p'] = dict()
                    accuracy_dict['level_' + str(level)]['r'] = dict()
                    accuracy_dict['level_' + str(level)]['f1'] = dict()

        # Code below adapted from https://stackoverflow.com/questions/13575090/construct-pandas-dataframe-from-items-in-nested-dictionary
        user_ids = []
        frames = []

        for user_id, d in accuracy_dict.items():
            user_ids.append(user_id)
            frames.append(pd.DataFrame.from_dict(d, orient='index'))

        # Stores all the accuracy scores for all the levels over all the dialogue acts into a DataFrame.
        accuracies = pd.concat(frames, keys=user_ids).T.round(4)

        # Saves the accuracy DataFrame to a .csv file.
        accuracy_file = 'analyses/' + weighted + '_model_sequence_length_' + str(sequence_length) + input_setting + '_accuracy.csv'
        accuracies.to_csv(accuracy_file)