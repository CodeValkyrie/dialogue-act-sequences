import pandas as pd
import numpy as np
from data import Preprocessing, Statistics
import matplotlib.pyplot as plt

plt.rcParams['xtick.labelsize'] = 6

weighted = 'unweighted'
sequence_lengths = [3]
input_settings = ['_d', '_d_s', '_d_s_l', '_d_s_l_u']
colors = ['b', 'r', 'y', 'g']

for sequence_length in sequence_lengths:

    # Initialises the plot format.
    fig, ax = plt.subplots()

    # Gets the precision, recall and f1-score for every dialogue act for different model input settings.
    for input_setting in input_settings:

        # Loads in the data for the plots.
        filename = 'analyses/' + weighted + '_model_sequence_length_' + str(sequence_length) + input_setting + '_accuracy.csv'
        accuracies = pd.read_csv(filename, index_col=[0], header=[0, 1])

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
    figure_file = 'analyses/' + weighted + '_model_sequence_length_' + str(sequence_length) + '_histogram_per_setting.png'
    plt.savefig(figure_file)