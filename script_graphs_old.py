import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from data import Preprocessing, Statistics


""" This is a script that plots the f1-scores per dialogue act given different input settings and sequence lengths. 
    The data is read in from .csv files containing the accuracy scores of different models and input settings.

    The variables that need to be specified:
        weighted           = chosen from {"weighted", "unweighted"} to specify which model's predictions are to be used.
        sequence_lengths   = the sequence length with which the model made the predictions.
        input_settings     = a list containing the input settings used: a subset from 
                             {'dialogue act', 'speaker', 'level', 'utterance length'}. The list must consist of 
                             abbreviations in the format '_<first letter>', for example ['_d', '_d_u'], which uses first 
                             only dialogue acts and then dialogue acts and utterance lengths.
        colors             = a list containing the colors of the bars in the graphs that correspond to the 
                             input settings. 

    The script outputs png plots containing the f1-scores of the dialogue acts per input setting.       
"""

# Sets the font size of the plot labels.
plt.rcParams['xtick.labelsize'] = 6

weighted = 'unweighted'
sequence_lengths = [3]
input_settings = ['_d', '_d_s', '_d_s_l', '_d_s_l_u']
baselines = ['majority_class', 'random', 'weighted_random']
colors = ['b', 'r', 'y', 'g']

# Initialises variables to be defined later.
names = []
x = []

preprocessed = Preprocessing('data/DA_labeled_belc_2019.csv')
statistics = Statistics(preprocessed)
da_sorted_by_occurance = list(statistics.get_da_distribution().index)

for sequence_length in sequence_lengths:

    # Initialises the plot format.
    fig, ax = plt.subplots()

    # Gets the precision, recall and f1-score for every dialogue act for different model input settings.
    for input_setting in input_settings:

        # Loads in the data for the plots.
        filename = 'analyses/' + weighted + '_model_sequence_length_' + str(sequence_length) + input_setting + \
                   '_accuracy.csv'
        accuracies = pd.read_csv(filename, index_col=[0], header=[0, 1])

        # Sort the accuracies by the overall occurance ratio of the classes.
        accuracies = accuracies.reindex(index=da_sorted_by_occurance)

        # Gets the correct labels and coordinates for the x-axis.
        names = list(accuracies.index)
        names = [s[:5] for s in names]
        x = np.arange(len(names))

        # Gets the f1 score of all the dialogue acts over all the levels.
        values = list(accuracies['all_levels']['f1'].to_numpy())

        # Gets the index of the input setting and the offset on the x axis and width of each bar.
        i = input_settings.index(input_setting)
        j = i
        if j > 1:
            j -= 4
        width = 0.8

        # Plots the f1-scores for each dialogue act per input setting
        setting = ax.bar(x - j * width / 4, values, width=0.2, color=colors[i], align='center', label=input_setting[1:])

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel('F1-score')
    ax.set_xlabel('Dialogue Act')
    ax.set_title('Prediction performance of DAs given different settings')
    ax.set_xticks(x)
    ax.set_xticklabels(names)
    ax.legend()

    # Saves the plot to a file.
    figure_file = 'analyses/' + weighted + '_model_sequence_length_' + str(sequence_length) + \
                  '_histogram_per_setting.png'
    plt.savefig(figure_file)

# Initialises the plot format.
    fig, ax = plt.subplots()
for baseline in baselines:

    # Loads in the data for the plots.
    filename = 'analyses/model_' + baseline + '_baseline_accuracy.csv'
    accuracies = pd.read_csv(filename, index_col=[0], header=[0, 1])

    # Sort the accuracies by the overall occurance ratio of the classes.
    accuracies = accuracies.reindex(index=da_sorted_by_occurance)

    # Gets the correct labels and coordinates for the x-axis.
    names = list(accuracies.index)
    names = [s[:5] for s in names]
    x = np.arange(len(names))

    # Gets the f1 score of all the dialogue acts over all the levels.
    values = list(accuracies['all_levels']['f1'].to_numpy())

    # Gets the index of the input setting and the offset on the x axis and width of each bar.
    i = baselines.index(baseline)
    j = i
    if j > 1:
        j -= 3
    width = 0.8

    # Plots the f1-scores for each dialogue act per input setting
    setting = ax.bar(x - j * width / 3, values, width=0.2, color=colors[i], align='center', label=baseline)

    # Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_ylabel('F1-score')
ax.set_xlabel('Dialogue Act')
ax.set_title('Prediction performance of DAs for different baselines')
ax.set_xticks(x)
ax.set_xticklabels(names)
ax.legend()

# Saves the plot to a file.
figure_file = 'analyses/model_baselines_histogram.png'
plt.savefig(figure_file)
