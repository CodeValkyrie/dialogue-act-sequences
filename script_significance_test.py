import pandas as pd
import numpy as np
from scipy import stats

''' This script performs significance tests on the f1 scores between the models. '''

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
baseline_mapping = {'majority_class': 'Majority Class',
                    'random': 'Random Class',
                    'weighted_random': 'Weighted Random Class',
                    '2gram': "Bigram",
                    '3gram': "Trigram"}

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
#                                       DOES SIGNIFICANCE TESTS ON THE DATA                                            #
########################################################################################################################
def cohen_d(x,y):
    nx = len(x)
    ny = len(y)
    dof = nx + ny - 2
    return (np.mean(x) - np.mean(y)) / np.sqrt(((nx-1)*np.std(x, ddof=1) ** 2 + (ny-1)*np.std(y, ddof=1) ** 2) / dof)

data_stats = pd.DataFrame(columns=['t-stat', 'p-value', 'mean 1', 'mean 2', 'cohen-d'])
for one_sequence_length in sequence_lengths:
    for two_sequence_length in sequence_lengths:
        one = data_frame[data_frame["Model"] == "Sequence Length " + str(one_sequence_length)][["F1-Score"]].to_numpy()
        two = data_frame[data_frame["Model"] == "Sequence Length " + str(two_sequence_length)][["F1-Score"]].to_numpy()
        t_stat, p_value = stats.ttest_ind(one, two)
        mean1 = np.mean(one)
        mean2 = np.mean(two)
        c_d = cohen_d(one, two)
        df = pd.DataFrame({'t-stat': t_stat, 'p-value': p_value, 'mean 1': mean1, 'mean 2': mean2, 'cohen-d': c_d},
                          index=["Sequence Lengths " + str(one_sequence_length) + " and " + str(two_sequence_length)])
        data_stats = pd.concat([data_stats, df])

for baseline in baselines:
    for sequence_length in [2, 10, 15, 20]:
        one = data_frame[data_frame["Model"] == "Sequence Length " + str(sequence_length)][["F1-Score"]].to_numpy()
        two = data_frame[data_frame["Model"] == baseline][["F1-Score"]].to_numpy()
        t_stat, p_value = stats.ttest_ind(one, two)
        mean1 = np.mean(one)
        mean2 = np.mean(two)
        c_d = cohen_d(one, two)
        df = pd.DataFrame({'t-stat': t_stat, 'p-value': p_value, 'mean 1': mean1, 'mean 2': mean2, 'cohen-d': c_d},
                          index=["Sequence Length " + str(sequence_length) + " and " + baseline_mapping[baseline] +
                                 "Baseline"])
        data_stats = pd.concat([data_stats, df])

    for base in baselines:
        one = data_frame[data_frame["Model"] == base][["F1-Score"]].to_numpy()
        two = data_frame[data_frame["Model"] == baseline][["F1-Score"]].to_numpy()
        t_stat, p_value = stats.ttest_ind(one, two)
        mean1 = np.mean(one)
        mean2 = np.mean(two)
        c_d = cohen_d(one, two)
        df = pd.DataFrame({'t-stat': t_stat, 'p-value': p_value, 'mean 1': mean1, 'mean 2': mean2, 'cohen-d': c_d},
                          index=[baseline_mapping[baseline] + " and " + baseline_mapping[base] +
                                 "Baselines"])
        data_stats = pd.concat([data_stats, df])

for sequence_length in sequence_lengths:
    one = data_frame[data_frame["Model"] == "Old weighted Model"][["F1-Score"]].to_numpy()
    two = data_frame[data_frame["Model"] == "Sequence Length " + str(sequence_length)][["F1-Score"]].to_numpy()
    t_stat, p_value = stats.ttest_ind(one, two)
    mean1 = np.mean(one)
    mean2 = np.mean(two)
    c_d = cohen_d(one, two)
    df = pd.DataFrame({'t-stat': t_stat, 'p-value': p_value, 'mean 1': mean1,
                       'mean 2': mean2, 'cohen-d': c_d},
                      index=["Old Weighted Model and Sequence Length " + str(sequence_length)])
    data_stats = pd.concat([data_stats, df])

for baseline in baselines:
    one = data_frame[data_frame["Model"] == "Old weighted Model"][["F1-Score"]].to_numpy()
    two = data_frame[data_frame["Model"] == baseline][["F1-Score"]].to_numpy()
    t_stat, p_value = stats.ttest_ind(one, two)
    mean1 = np.mean(one)
    mean2 = np.mean(two)
    c_d = cohen_d(one, two)
    df = pd.DataFrame({'t-stat': t_stat, 'p-value': p_value, 'mean 1': mean1,
                       'mean 2': mean2, 'cohen-d': c_d},
                      index=["Old Weighted Model and " + baseline_mapping[baseline] + " Baseline"])
    data_stats = pd.concat([data_stats, df])

one = pd.read_csv('analyses/old_weighted_model_sequence_length_3_accuracy.csv', index_col=[0], header=[0, 1])
one = one['all_levels']['f1'].to_numpy()
two = pd.read_csv('analyses/old_unweighted_model_sequence_length_3_accuracy.csv', index_col=[0], header=[0, 1])
two = two['all_levels']['f1'].to_numpy()
t_stat, p_value = stats.ttest_ind(one, two)
mean1 = np.mean(one)
mean2 = np.mean(two)
c_d = cohen_d(one, two)
df = pd.DataFrame({'t-stat': t_stat, 'p-value': p_value, 'mean 1': mean1, 'mean 2': mean2, 'cohen-d': c_d},
                  index=["Old Weighted Model and Old Unweighted Model"])
data_stats = pd.concat([data_stats, df])

data_stats = data_stats[data_stats["p-value"] != 1.0]
data_stats = data_stats.drop_duplicates(subset=['p-value'])
data_stats = data_stats.round(4)
data_stats.to_csv("analyses/significance_tests_average_f1_models.csv")

# t-statistic, p-value, mean1 mean2, cohens D for each pair of values to report whether the effect sizes are significantly different.

########################################################################################################################
#                                      SIGNIFICANCE FOR PLOT PER LEVEL                                                 #
########################################################################################################################

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

data_stats = pd.DataFrame(columns=['t-stat', 'p-value', 'mean 1', 'mean 2', 'cohen-d'])
for baseline in ['2gram', '3gram']:
    for level1 in levels:
        for level2 in levels:
            one = data_frame.loc[(data_frame["Model"] == baseline) & (data_frame["Level"] == "Level " +
                                                                      str(level1))][["F1-Score"]].to_numpy()
            two = data_frame.loc[(data_frame["Model"] == baseline) & (data_frame["Level"] == "Level " +
                                                                      str(level2))][["F1-Score"]].to_numpy()
            t_stat, p_value = stats.ttest_ind(one, two)
            mean1 = np.mean(one)
            mean2 = np.mean(two)
            c_d = cohen_d(one, two)
            df = pd.DataFrame({'t-stat': t_stat, 'p-value': p_value, 'mean 1': mean1, 'mean 2': mean2, 'cohen-d': c_d},
                              index=[baseline_mapping[baseline] + " Level " + str(level1) + " and " + str(level1)])
            data_stats = pd.concat([data_stats, df])

for sequence_length in sequence_lengths:
    for level1 in levels:
        for level2 in levels:
            one = data_frame.loc[(data_frame["Model"] == "Sequence Length " + str(sequence_length)) &
                                 (data_frame["Level"] == "Level " + str(level1))][["F1-Score"]].to_numpy()
            two = data_frame.loc[(data_frame["Model"] == "Sequence Length " + str(sequence_length)) &
                                 (data_frame["Level"] == "Level " + str(level2))][["F1-Score"]].to_numpy()
            t_stat, p_value = stats.ttest_ind(one, two)
            mean1 = np.mean(one)
            mean2 = np.mean(two)
            c_d = cohen_d(one, two)
            df = pd.DataFrame({'t-stat': t_stat, 'p-value': p_value, 'mean 1': mean1, 'mean 2': mean2, 'cohen-d': c_d},
                              index=["Sequence Length " + str(sequence_length) + " Level " + str(level1) + " and " +
                                     str(level2)])
            data_stats = pd.concat([data_stats, df])

data_stats = data_stats[data_stats["p-value"] != 1.0]
data_stats = data_stats.drop_duplicates(subset=['p-value'])
data_stats = data_stats.round(4)
data_stats.to_csv("analyses/significance_tests_average_f1_per_level_models.csv")
