import pandas as pd
from data import Preprocessing, Statistics


""" This is a script that stores the prediction performance accuracy scores for different baselines. 
    The data is read in from a .csv file containing the predictions of different models 
    and input settings. Only the labels will be used as these stay the same for all the model settings and baselines.

    The variables that need to be specified:
        filename    = a .csv file containing the labels belonging to the inputs.
        baselines   = a list containing the baselines used, choose from {'majority_class', 'random', 'weighted_random'}

    The script outputs a .csv file containing the accuracy scores of the dialogue acts (rows) and the accuracy 
    metric per baseline (columns).       
"""

filename = 'analyses/n_gram_models_predictions.csv'
baselines = ['majority_class', 'random', 'weighted_random']

# Reads in the data containing the predictions of a model under the given settings.
data = Preprocessing(filename)
for baseline in baselines:
    data.add_baseline(baseline)
statistics = Statistics(data)

data.data.to_csv('analyses/simple_baselines_predictions.csv')

# Gets the precision, recall and f1-score for every dialogue act for different baselines.
for baseline in baselines:
    accuracy_dict = dict()
    columns = ['labels_2_gram', baseline]
    confusion_matrix = (statistics.get_normalised_confusion_matrix(data.data, columns) * 100).round(2)
    confusion_matrix.to_csv('analyses/' + baseline + '_error_analysis.csv')
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
    print(accuracies)

    # Saves the accuracy DataFrame to a .csv file.
    accuracy_file = 'analyses/model_' + baseline + '_baseline_accuracy.csv'
    accuracies.to_csv(accuracy_file)
