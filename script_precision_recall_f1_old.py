import pandas as pd
from data import Preprocessing, Statistics


""" This is a script that stores the prediction performance accuracy scores per dialogue act given different input 
    settings and sequence lengths. The data is read in from .csv files containing the predictions of different models 
    and input settings.

    The variables that need to be specified:
        weighted           = chosen from {"weighted", "unweighted"} to specify which model's predictions are to be used.
        sequence_lengths   = the sequence length with which the model made the predictions.
        input_settings     = the input settings used: a subset from 
                             {'dialogue act', 'speaker', 'level', 'utterance length'}. The list must consist of 
                             abbreviations in the format '_<first letter>', for example ['_d', '_d_u'], which uses first 
                             only dialogue acts and then dialogue acts and utterance lengths.

    The script outputs csv files containing the accuracy scores of the dialogue acts (rows) per level and accuracy 
    metric (columns).       
"""

weighted = 'weighted'
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
        # Prepares the separate parts of the dictionary to be stored in one DataFrame.
        level_accuracies = []
        frames = []
        for level_accuracy, accuracy_scores in accuracy_dict.items():
            level_accuracies.append(level_accuracy)
            frames.append(pd.DataFrame.from_dict(accuracy_scores, orient='index'))

        # Stores all the accuracy scores for all the levels over all the dialogue acts into a DataFrame.
        accuracies = pd.concat(frames, keys=level_accuracies).T.round(4)

        # Saves the accuracy DataFrame to a .csv file.
        accuracy_file = 'analyses/' + weighted + '_model_sequence_length_' + str(sequence_length) + input_setting + \
                        '_accuracy.csv'
        accuracies.to_csv(accuracy_file)
