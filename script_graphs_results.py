import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats


""" This is a script that plots the f1-scores per dialogue act given different input settings and sequence lengths. 
    The data is read in from .csv files containing the accuracy scores of different models and input settings.
    The script also plots the confusion matrices of the predictions for the models that are in .csv files.

    The variables that need to be specified:
        weighted           = chosen from {"weighted", "unweighted"} to specify which model's predictions are to be used.
        sequence_lengths   = a list containing the sequence lengths with which the model made the predictions.
        baselines          = a list containing all the baselines of which the f1-scores should be included

    The script outputs .png plots containing the f1-scores of the dialogue acts per input setting.       
"""

weighted = 'weighted'
sequence_lengths = [2, 3, 5, 7, 10, 15, 20]
baselines = ['majority_class', 'random', 'weighted_random', '2gram', '3gram']
old_models = ['weighted']
levels = [1, 2, 3, 4]
level_names = ['level_1.0', 'level_2.0', 'level_3.0', 'level_4.0']
level_mapping = {'all_levels': 'All Levels',
                 'level_1.0': 'Level 1',
                 'level_2.0': 'Level 2',
                 'level_3.0': 'Level 3',
                 'level_4.0': 'Level 4'}

########################################################################################################################
#                   CONSTRUCT DATAFRAME THAT CONCATENATES ALL THE NEEDED F1-SCORE COLUMNS                              #
########################################################################################################################

# Initialises the DataFrame that will be used for the Seaborn graphs.
data_frame = pd.DataFrame(columns=["Dialogue Act", "F1-Score", "Model"])

# Gets the f1-scores over all levels for every baseline model.
for baseline in baselines:
    accuracies = pd.read_csv('analyses/model_' + baseline + '_baseline_accuracy.csv', index_col=[0], header=[0, 1])
    accuracies = accuracies['all_levels']['f1']
    accuracies = accuracies.reset_index()
    accuracies.columns = ["Dialogue Act", "F1-Score"]
    accuracies["Model"] = baseline
    data_frame = pd.concat([data_frame, accuracies], ignore_index=True)

# Gets the f1-scores over all levels for every sequence length the model was run on.
for sequence_length in sequence_lengths:
    filename = 'analyses/weighted_model_with_txt_sequence_length_' + str(sequence_length) + '_accuracy.csv'
    accuracies = pd.read_csv(filename, index_col=[0], header=[0, 1])
    accuracies = accuracies['all_levels']['f1']
    accuracies = accuracies.reset_index()
    accuracies.columns = ["Dialogue Act", "F1-Score"]
    accuracies["Model"] = "Sequence Length " + str(sequence_length)
    data_frame = pd.concat([data_frame, accuracies], ignore_index=True)

for model in old_models:
    filename = 'analyses/old_' + weighted + '_model_sequence_length_3_accuracy.csv'
    accuracies = pd.read_csv(filename, index_col=[0], header=[0, 1])
    accuracies = accuracies['all_levels']['f1']
    accuracies = accuracies.reset_index()
    accuracies.columns = ["Dialogue Act", "F1-Score"]
    accuracies["Model"] = "Old " + model + " Model"
    data_frame = pd.concat([data_frame, accuracies], ignore_index=True)

########################################################################################################################
#                   MAKE PLOT WITH DA ON X-AXIS AND F1-SCORE ON Y-AXIS PER MODEL IN ONE GRAPH                          #
########################################################################################################################

# Sets colour palette for the graph.
colours = ['red', 'orange', 'yellow', 'green', 'lime green']
baseline_colours = sns.xkcd_palette(colours)
model_colours = sns.color_palette('Blues', len(sequence_lengths) + 1)
old_model_colours = sns.xkcd_palette(['light purple', 'purple'])
colour_palette = baseline_colours + model_colours[1:] + old_model_colours
sns.set_palette(colour_palette)

# Plots the F1-Scores per Dialogue Act for Different Models.
distribution_order = pd.read_csv('analyses/dialogue_act_distribution.csv', index_col=[0], header=None)
graph = sns.catplot(x="Dialogue Act", y="F1-Score", hue="Model", data=data_frame, kind="bar", height=8, aspect=1.5,
                    order=distribution_order.index, legend=False)
graph.set_xticklabels(rotation=45, horizontalalignment='right')
plt.title('F1-Scores per Dialogue Act for Different Models')
plt.legend(loc='upper right')
plt.ylim(0, 0.6)
plt.tight_layout(2)
plt.savefig('analyses/f1_per_dialogue_act_histogram.png')
plt.clf()

########################################################################################################################
#           MAKE PLOT WITH MODELS ON X-AXIS AND AVERAGE PER CLASS F1-SCORE WITH ERROR ON Y-AXIS                        #
########################################################################################################################

# Sets colour palette for the graph.
colours = ['red', 'orange', 'yellow', 'green', 'lime green']
baseline_colours = sns.xkcd_palette(colours)
model_colours = sns.color_palette('Blues', len(sequence_lengths) + 1)
old_model_colours = sns.xkcd_palette(['light purple', 'purple'])
colour_palette = baseline_colours + model_colours[1:] + old_model_colours
sns.set_palette(colour_palette)

# Plots the Average F1-Scores over the Dialogue Acts per Model with error bars.
# sns.set(style="whitegrid")
plt.figure(figsize=(10, 5))
graph = sns.barplot(x="Model", y="F1-Score", data=data_frame, capsize=0.1)
_, labels = plt.xticks()
graph.set_xticklabels(labels, rotation=45, horizontalalignment='right', fontsize='x-small')
plt.title('Average per Class F1-Scores over the Dialogue Acts per Model')
plt.ylim(0, 0.35)
plt.tight_layout()
plt.savefig('analyses/average_f1_per_model_histogram.png')
plt.clf()


########################################################################################################################
#                                             MAKE PLOT PER LEVEL                                                      #
########################################################################################################################
# Initialises the DataFrame that will be used for the Seaborn graphs.
data_frame = pd.DataFrame(columns=["Dialogue Act", "F1-Score", "Model", "Level"])

# Gets the f1-scores over all levels for every baseline model.
for baseline in baselines:
    for level in level_names:
        accuracies = pd.read_csv('analyses/model_' + baseline + '_baseline_accuracy.csv', index_col=[0], header=[0, 1])
        accuracies = accuracies[level]['f1']
        accuracies = accuracies.reset_index()
        accuracies.columns = ["Dialogue Act", "F1-Score"]
        accuracies["Model"] = baseline
        accuracies["Level"] = level_mapping[level]
        data_frame = pd.concat([data_frame, accuracies], ignore_index=True)

# Gets the f1-scores over all levels for every sequence length the model was run on.
for sequence_length in sequence_lengths:
    filename = 'analyses/weighted_model_with_txt_sequence_length_' + str(sequence_length) + '_accuracy.csv'
    for level in level_names:
        accuracies = pd.read_csv(filename, index_col=[0], header=[0, 1])
        accuracies = accuracies[level]['f1']
        accuracies = accuracies.reset_index()
        accuracies.columns = ["Dialogue Act", "F1-Score"]
        accuracies["Model"] = "Sequence Length " + str(sequence_length)
        accuracies["Level"] = level_mapping[level]
        data_frame = pd.concat([data_frame, accuracies], ignore_index=True)

for model in old_models:
    filename = 'analyses/old_' + weighted + '_model_sequence_length_3_accuracy.csv'
    for level in level_names:
        accuracies = pd.read_csv(filename, index_col=[0], header=[0, 1])
        accuracies = accuracies[level]['f1']
        accuracies = accuracies.reset_index()
        accuracies.columns = ["Dialogue Act", "F1-Score"]
        accuracies["Model"] = "Old " + model + " Model"
        accuracies["Level"] = level_mapping[level]
        data_frame = pd.concat([data_frame, accuracies], ignore_index=True)

sns.set_palette(sns.color_palette('Blues', 6)[1:])

# Plots the Average F1-Scores over the Dialogue Acts per Model with error bars for each level.
# sns.set(style="whitegrid")
plt.figure(figsize=(10, 5))
graph = sns.barplot(x="Model", y="F1-Score", hue='Level', data=data_frame,
                    hue_order=['Level 1', 'Level 2', 'Level 3', 'Level 4'], errwidth=1, capsize=0.1)
_, labels = plt.xticks()
graph.set_xticklabels(labels, rotation=45, horizontalalignment='right', fontsize='x-small')
plt.title('Average per Class F1-Scores over the Dialogue Acts per Level for Different Models')
plt.ylim(0, 0.35)
plt.tight_layout()
plt.savefig('analyses/average_f1_per_level_per_model_histogram.png')
plt.clf()

for level in levels:
    for sequence_length in [2, 10, 20]:
        data = data_frame[data_frame["Model"] == "Sequence Length " + str(sequence_length)][['Dialogue Act', 'F1-Score',
                                                                                             'Level']]
        graph = sns.barplot(x="Dialogue Act", y="F1-Score", hue='Level', data=data, order=distribution_order.index)
        _, labels = plt.xticks()
        graph.set_xticklabels(labels, rotation=45, horizontalalignment='right', fontsize='x-small')
        plt.title('F1-Scores per Dialogue Act for Different Levels of Embedded Text Model with Sequence Length'+ str(sequence_length))
        plt.ylim(0, 1.0)
        plt.tight_layout()
        plt.savefig('analyses/f1_per_da_per_level_model_seq_len_' + str(sequence_length) + 'histogram.png')
        plt.clf()


########################################################################################################################
#                   CONSTRUCT DATAFRAME THAT CONCATENATES ALL THE NEEDED PRECISION COLUMNS                             #
########################################################################################################################

# Initialises the DataFrame that will be used for the Seaborn graphs.
data_frame = pd.DataFrame(columns=["Dialogue Act", "Precision", "Model"])

# Gets the f1-scores over all levels for every baseline model.
for baseline in baselines:
    accuracies = pd.read_csv('analyses/model_' + baseline + '_baseline_accuracy.csv', index_col=[0], header=[0, 1])
    accuracies = accuracies['all_levels']['p']
    accuracies = accuracies.reset_index()
    accuracies.columns = ["Dialogue Act", "Precision"]
    accuracies["Model"] = baseline
    data_frame = pd.concat([data_frame, accuracies], ignore_index=True)

# Gets the f1-scores over all levels for every sequence length the model was run on.
for sequence_length in sequence_lengths:
    filename = 'analyses/weighted_model_with_txt_sequence_length_' + str(sequence_length) + '_accuracy.csv'
    accuracies = pd.read_csv(filename, index_col=[0], header=[0, 1])
    accuracies = accuracies['all_levels']['p']
    accuracies = accuracies.reset_index()
    accuracies.columns = ["Dialogue Act", "Precision"]
    accuracies["Model"] = "Sequence Length " + str(sequence_length)
    data_frame = pd.concat([data_frame, accuracies], ignore_index=True)

for model in old_models:
    filename = 'analyses/' + model + '_model_sequence_length_3_d_s_l_u_accuracy.csv'
    accuracies = pd.read_csv(filename, index_col=[0], header=[0, 1])
    accuracies = accuracies['all_levels']['p']
    accuracies = accuracies.reset_index()
    accuracies.columns = ["Dialogue Act", "Precision"]
    accuracies["Model"] = "Old " + model + " Model"
    data_frame = pd.concat([data_frame, accuracies], ignore_index=True)
########################################################################################################################
#                   MAKE PLOT WITH DA ON X-AXIS AND PRECISION ON Y-AXIS PER MODEL IN ONE GRAPH                         #
########################################################################################################################

# Sets colour palette for the graph.
colours = ['red', 'orange', 'yellow', 'green', 'lime green']
baseline_colours = sns.xkcd_palette(colours)
model_colours = sns.color_palette('Blues', len(sequence_lengths) + 1)
old_model_colours = sns.xkcd_palette(['light purple', 'purple'])
colour_palette = baseline_colours + model_colours[1:] + old_model_colours
sns.set_palette(colour_palette)

# Plots the F1-Scores per Dialogue Act for Different Models.
distribution_order = pd.read_csv('analyses/dialogue_act_distribution.csv', index_col=[0], header=None)
graph = sns.catplot(x="Dialogue Act", y="Precision", hue="Model", data=data_frame, kind="bar", height=8, aspect=1.5,
                    order=distribution_order.index, legend=False)
graph.set_xticklabels(rotation=45, horizontalalignment='right', fontsize='x-small')
plt.title('Precision per Dialogue Act for Different Models')
plt.legend(loc='upper right')
plt.ylim(0, 1.0)
plt.tight_layout(2)
plt.savefig('analyses/precision_per_dialogue_act_histogram.png')
plt.clf()

########################################################################################################################
#           MAKE PLOT WITH MODELS ON X-AXIS AND AVERAGE PER CLASS PRECISION WITH ERROR ON Y-AXIS                       #
########################################################################################################################

# Sets colour palette for the graph.
colours = ['red', 'orange', 'yellow', 'green', 'lime green']
baseline_colours = sns.xkcd_palette(colours)
model_colours = sns.color_palette('Blues', len(sequence_lengths) + 1)
old_model_colours = sns.xkcd_palette(['light purple', 'purple'])
colour_palette = baseline_colours + model_colours[1:] + old_model_colours
sns.set_palette(colour_palette)

# Plots the Average F1-Scores over the Dialogue Acts per Model with error bars.
# sns.set(style="whitegrid")
plt.figure(figsize=(10, 5))
graph = sns.barplot(x="Model", y="Precision", data=data_frame, capsize=0.1)
_, labels = plt.xticks()
graph.set_xticklabels(labels, rotation=45, horizontalalignment='right', fontsize='x-small')
plt.title('Average per Class Precision over the Dialogue Acts per Model')
plt.ylim(0, 0.35)
plt.tight_layout()
plt.savefig('analyses/average_precision_per_model_histogram.png')
plt.clf()


########################################################################################################################
#                   MAKE PLOT PER LEVEL                          #
########################################################################################################################

########################################################################################################################
#                   CONSTRUCT DATAFRAME THAT CONCATENATES ALL THE NEEDED RECALL COLUMNS                                #
########################################################################################################################

# Initialises the DataFrame that will be used for the Seaborn graphs.
data_frame = pd.DataFrame(columns=["Dialogue Act", "Recall", "Model"])

# Gets the f1-scores over all levels for every baseline model.
for baseline in baselines:
    accuracies = pd.read_csv('analyses/model_' + baseline + '_baseline_accuracy.csv', index_col=[0], header=[0, 1])
    accuracies = accuracies['all_levels']['r']
    accuracies = accuracies.reset_index()
    accuracies.columns = ["Dialogue Act", "Recall"]
    accuracies["Model"] = baseline
    data_frame = pd.concat([data_frame, accuracies], ignore_index=True)

# Gets the f1-scores over all levels for every sequence length the model was run on.
for sequence_length in sequence_lengths:
    filename = 'analyses/weighted_model_with_txt_sequence_length_' + str(sequence_length) + '_accuracy.csv'
    accuracies = pd.read_csv(filename, index_col=[0], header=[0, 1])
    accuracies = accuracies['all_levels']['r']
    accuracies = accuracies.reset_index()
    accuracies.columns = ["Dialogue Act", "Recall"]
    accuracies["Model"] = "Sequence Length " + str(sequence_length)
    data_frame = pd.concat([data_frame, accuracies], ignore_index=True)

for model in old_models:
    filename = 'analyses/' + model + '_model_sequence_length_3_d_s_l_u_accuracy.csv'
    accuracies = pd.read_csv(filename, index_col=[0], header=[0, 1])
    accuracies = accuracies['all_levels']['r']
    accuracies = accuracies.reset_index()
    accuracies.columns = ["Dialogue Act", "Recall"]
    accuracies["Model"] = "Old " + model + " Model"
    data_frame = pd.concat([data_frame, accuracies], ignore_index=True)
########################################################################################################################
#                   MAKE PLOT WITH DA ON X-AXIS AND RECALL ON Y-AXIS PER MODEL IN ONE GRAPH                            #
########################################################################################################################

# Sets colour palette for the graph.
colours = ['red', 'orange', 'yellow', 'green', 'lime green']
baseline_colours = sns.xkcd_palette(colours)
model_colours = sns.color_palette('Blues', len(sequence_lengths) + 1)
old_model_colours = sns.xkcd_palette(['light purple', 'purple'])
colour_palette = baseline_colours + model_colours[1:] + old_model_colours
sns.set_palette(colour_palette)

# Plots the F1-Scores per Dialogue Act for Different Models.
distribution_order = pd.read_csv('analyses/dialogue_act_distribution.csv', index_col=[0], header=None)
graph = sns.catplot(x="Dialogue Act", y="Recall", hue="Model", data=data_frame, kind="bar", height=8, aspect=1.5,
                    order=distribution_order.index, legend=False)
graph.set_xticklabels(rotation=45, horizontalalignment='right', fontsize='x-small')
plt.title('Recall per Dialogue Act for Different Models')
plt.legend(loc='upper right')
plt.ylim(0, 1.0)
plt.tight_layout(2)
plt.savefig('analyses/recall_per_dialogue_act_histogram.png')
plt.clf()

########################################################################################################################
#           MAKE PLOT WITH MODELS ON X-AXIS AND AVERAGE PER CLASS RECALL WITH ERROR ON Y-AXIS                          #
########################################################################################################################

# Sets colour palette for the graph.
colours = ['red', 'orange', 'yellow', 'green', 'lime green']
baseline_colours = sns.xkcd_palette(colours)
model_colours = sns.color_palette('Blues', len(sequence_lengths) + 1)
old_model_colours = sns.xkcd_palette(['light purple', 'purple'])
colour_palette = baseline_colours + model_colours[1:] + old_model_colours
sns.set_palette(colour_palette)

# Plots the Average F1-Scores over the Dialogue Acts per Model with error bars.
# sns.set(style="whitegrid")
plt.figure(figsize=(10, 5))
graph = sns.barplot(x="Model", y="Recall", data=data_frame, capsize=0.1)
_, labels = plt.xticks()
graph.set_xticklabels(labels, rotation=45, horizontalalignment='right', fontsize='x-small')
plt.title('Average per Class Recall over the Dialogue Acts per Model')
plt.ylim(0, 0.35)
plt.tight_layout()
plt.savefig('analyses/average_recall_per_model_histogram.png')
plt.clf()


########################################################################################################################
#                                            MAKE PLOT PER LEVEL                                                       #
########################################################################################################################


########################################################################################################################
#                                        MAKE HEATMAP PLOTS FOR ERROR ANALYSIS                                         #
########################################################################################################################

# Makes heatmaps for the new model for different sequence lengths.
average_data = None
for sequence_length in sequence_lengths:
    data = pd.read_csv('analyses/weighted_model_with_txt_sequence_length_' + str(sequence_length) +
                       '_error_analysis.csv', index_col=[0], header=[0]).astype(float)
    graph = sns.heatmap(data, vmin=0, vmax=100, square=True, cmap='Blues')
    plt.ylabel('Labels')
    plt.xlabel('Predictions')
    plt.title('Embedded Text Model Error Analysis for Sequence Length ' + str(sequence_length))
    plt.tight_layout()
    plt.savefig('analyses/weighted_model_with_txt_sequence_length_' + str(sequence_length) + '_error_analysis.png')
    plt.clf()
    if average_data is None:
        average_data = data
    else:
        average_data = average_data + data
average_data = average_data.divide(len(sequence_lengths))
graph = sns.heatmap(average_data, vmin=0, vmax=100, square=True, cmap='Blues')
plt.ylabel('Labels')
plt.xlabel('Predictions')
plt.title('Embedded Text Model Average per Sequence Length Error Analysis')
plt.tight_layout()
plt.savefig('analyses/weighted_model_with_txt_average_error_analysis.png')
plt.clf()


# Makes heatmaps for the old model for different settings.
for setting in old_models:
    data = pd.read_csv('analyses/old_' + setting + '_model_sequence_length_3_error_analysis.csv', index_col=[0],
                       header=[0]).astype(float)
    graph = sns.heatmap(data, vmin=0, vmax=100, square=True, cmap='Blues')
    plt.ylabel('Labels')
    plt.xlabel('Predictions')
    if setting == 'weighted':
        plt.title('Error Analysis for the Old Weighted Model')
    elif setting == 'unweighted':
        plt.title('Error Analysis for the Old Unweighted Model')
    plt.tight_layout()
    plt.savefig('analyses/old_' + setting + '_model_sequence_length_3_error_analysis.png')
    plt.clf()

baseline_mapping = {'majority_class': 'Majority Class',
                    'random': 'Random Class',
                    'weighted_random': 'Weighted Random Class',
                    '2gram': 'Bigram',
                    '3gram': 'Trigram'}

for baseline in baselines:
    data = pd.read_csv('analyses/' + baseline + '_error_analysis.csv', index_col=[0], header=[0]).astype(float)
    graph = sns.heatmap(data, vmin=0, vmax=100, square=True, cmap='Blues')
    plt.ylabel('Labels')
    plt.xlabel('Predictions')
    plt.title('Error Analysis for ' + baseline_mapping[baseline] + ' Model')
    plt.tight_layout()
    plt.savefig('analyses/' + baseline + '_error_analysis.png')
    plt.clf()

# Plots the heatmap of the average confusion matrix per level of all the sequence lenghts.
for level in levels:
    level_data = None
    for sequence_length in sequence_lengths:
        data = pd.read_csv('analyses/weighted_model_with_txt_sequence_length_' + str(sequence_length) + '_level_' +
                           str(level) + '_error_analysis.csv', index_col=[0], header=[0]).astype(float)
        if level_data is None:
            level_data = data
        else:
            level_data = level_data + data
    level_data = level_data.divide(len(sequence_lengths))
    level_data = level_data[level_data.index]
    graph = sns.heatmap(level_data, vmin=0, vmax=100, square=True, cmap='Blues')
    plt.ylabel('Labels')
    plt.xlabel('Predictions')
    plt.title('Embedded Text Model Average per Sequence Length Error Analysis at Level ' + str(level))
    plt.tight_layout()
    plt.savefig('analyses/weighted_model_with_txt_level_' + str(level) + '_average_error_analysis.png')
    plt.clf()

    for setting in old_models:
        data = pd.read_csv('analyses/old_' + setting + '_model_sequence_length_3_level_' + str(level) +
                           '_error_analysis.csv', index_col=[0], header=[0]).astype(float)
        graph = sns.heatmap(data, vmin=0, vmax=100, square=True, cmap='Blues')
        plt.ylabel('Labels')
        plt.xlabel('Predictions')
        plt.title('Error Analysis for the Old ' + setting + ' Model Level' + str(level))
        plt.tight_layout()
        plt.savefig('analyses/old_' + setting + '_model_sequence_length_3_level_' + str(level) + '_error_analysis.png')
        plt.clf()
