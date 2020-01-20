import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


""" This is a script that plots the f1-scores per dialogue act given different input settings and sequence lengths. 
    The data is read in from .csv files containing the accuracy scores of different models and input settings.

    The variables that need to be specified:
        weighted           = chosen from {"weighted", "unweighted"} to specify which model's predictions are to be used.
        sequence_lengths   = a list containing the sequence lengths with which the model made the predictions.
        baselines          = a list containing all the baselines of which the f1-scores should be included

    The script outputs png plots containing the f1-scores of the dialogue acts per input setting.       
"""

weighted = 'weighted'
sequence_lengths = [2, 3, 5, 7, 10, 15, 20]
baselines = ['majority_class', 'random', 'weighted_random', '2gram', '3gram']
old_models = ['weighted', 'unweighted']

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
    print('PRINT' + baseline, data_frame)

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
plt.ylim(0, 1.0)
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
plt.figure(figsize=(7, 7))
graph = sns.barplot(x="Model", y="F1-Score", data=data_frame)
_, labels = plt.xticks()
graph.set_xticklabels(labels, rotation=45, horizontalalignment='right', fontsize='x-small')
plt.title('Average F1-Scores over the Dialogue Acts per Model')
plt.ylim(0, 1.0)
plt.tight_layout(2)
plt.savefig('analyses/average_f1_per_model_histogram.png')
plt.clf()


########################################################################################################################
#                   MAKE PLOT PER LEVEL                          #
########################################################################################################################



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
plt.figure(figsize=(7, 7))
graph = sns.barplot(x="Model", y="Precision", data=data_frame)
_, labels = plt.xticks()
graph.set_xticklabels(labels, rotation=45, horizontalalignment='right', fontsize='x-small')
plt.title('Average precision over the Dialogue Acts per Model')
plt.ylim(0, 1.0)
plt.tight_layout(2)
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
plt.figure(figsize=(7, 7))
graph = sns.barplot(x="Model", y="Recall", data=data_frame)
_, labels = plt.xticks()
graph.set_xticklabels(labels, rotation=45, horizontalalignment='right', fontsize='x-small')
plt.title('Average recall over the Dialogue Acts per Model')
plt.ylim(0, 1.0)
plt.tight_layout(2)
plt.savefig('analyses/average_recall_per_model_histogram.png')
plt.clf()


########################################################################################################################
#                   MAKE PLOT PER LEVEL                          #
########################################################################################################################
