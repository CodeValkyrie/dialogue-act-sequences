import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from data import Statistics, Preprocessing

# Preprocesses the data.
preprocessed = Preprocessing('data/DA_labeled_belc_2019.csv')
statistics = Statistics(preprocessed)


#######################################################################################################################
#                                 PLOTTING DIALOGUE ACT DISTRIBUTIONAL DATA                                           #
#######################################################################################################################
statistics.get_da_distribution()
statistics.get_da_distributions(['participant', 'interviewer'], [1, 2, 3, 4])
statistics.get_da_counts(preprocessed.data, 'dialogue_act', [1, 2, 3, 4])
statistics.get_average_utterance_length(['participant', 'interviewer'], [1, 2, 3, 4])

# Get the dialogue act distribution.
distribution_order = pd.read_csv('analyses/dialogue_act_distribution.csv', index_col=[0], header=None)

# Plot the dialogue act distribution.
sns.set_palette(sns.color_palette('Blues_r', 13))
graph = distribution_order.plot.bar()
plt.legend().remove()
_, labels = plt.xticks()
graph.set_xticklabels(labels, rotation=45, horizontalalignment='right', fontsize='x-small')
plt.title('Dialogue Act Distribution')
plt.xlabel("Dialogue Act")
plt.ylabel("Distribution")
plt.tight_layout(2)
plt.savefig('analyses/dialogue_act_distribution_histogram.png')

# Get the dialogue act counts per level.
dialogue_counts = pd.read_csv('analyses/dialogue_act_column_dialogue_act_counts.csv', index_col=[0],
                              names=['Level 1', 'Level 2', 'Level 3', 'Level 4'], header=0)
dialogue_counts['All Levels'] = dialogue_counts.sum(axis=1, skipna=True)
dialogue_counts = dialogue_counts.reindex(distribution_order.index)

# Plot the dialogue act counts per level.
sns.set_palette(sns.color_palette('Blues', 6)[1:])
graph = dialogue_counts.plot.bar()
plt.title('Dialogue Act Counts per Level')
_, labels = plt.xticks()
graph.set_xticklabels(labels, rotation=45, horizontalalignment='right', fontsize='x-small')
plt.xlabel("Dialogue Act")
plt.ylabel("Counts")
plt.tight_layout(2)
plt.savefig('analyses/dialogue_act_count_per_level_histogram.png')

# Gets the dialogue act distribution per speaker for each level
tutor_distribution = pd.read_csv('analyses/interviewer_dialogue_act_distributions.csv', index_col=[0],
                                 names=['1', '2', '3', '4'], header=0)
tutor_distribution['5'] = tutor_distribution.mean(axis=1, skipna=True)
student_distribution = pd.read_csv('analyses/participant_dialogue_act_distributions.csv', index_col=[0],
                                   names=['1', '2', '3', '4'], header=0)
student_distribution['5'] = student_distribution.mean(axis=1, skipna=True)

# Plots the dialogue act distribution per speaker for each level separately.
for level in range(1, 6):
    level_distribution = tutor_distribution[[str(level)]]
    level_distribution = level_distribution.merge(student_distribution[[str(level)]], how='left', left_index=True,
                                                  right_index=True)
    level_distribution.columns = ['Tutor', 'Student']
    level_distribution = level_distribution.reindex(distribution_order.index)
    # print(level_distribution)

    sns.set_palette([sns.color_palette('Blues', 7)[3], sns.color_palette('Blues', 7)[6]])
    graph = level_distribution.plot.bar()
    if level == 5:
        level = 'Total'
    plt.title('Dialogue Act Distribution per Speaker for Level ' + str(level))
    _, labels = plt.xticks()
    graph.set_xticklabels(labels, rotation=45, horizontalalignment='right', fontsize='x-small')
    plt.xlabel("Dialogue Act")
    plt.ylabel("Distribution")
    plt.ylim(0, 1.0)
    plt.tight_layout(2)
    plt.savefig('analyses/dialogue_act_distribution_per_speaker_level_' + str(level) + '_histogram.png')

# Plots the average utterance length per speaker per level
utterance_lengths = pd.read_csv('analyses/average_utterance_length.csv', index_col=[0], names=['Student', 'Tutor'],
                                header=0)[['Tutor', 'Student']]
total = utterance_lengths.mean(axis=0)
utterance_lengths = utterance_lengths.append(total, ignore_index=True)
utterance_lengths.index =['Level 1', 'Level 2', 'Level 3', 'Level 4', 'Level Total']
# plt.clf()

# Plot the average utterance lengths per speaker per level.
graph = utterance_lengths.plot.bar()
plt.title('Average Utterance Length per Speaker per Level')
_, labels = plt.xticks()
graph.set_xticklabels(labels, rotation=45, horizontalalignment='right', fontsize='x-small')
plt.xlabel("Level")
plt.ylabel("Utterance Length")
plt.tight_layout(2)
plt.savefig('analyses/average_utterance_length_per_speaker_per_level_histogram.png')






