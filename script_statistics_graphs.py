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
statistics.get_speaker_ratios([1, 2, 3, 4])

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
# dialogue_counts['All Levels'] = dialogue_counts.sum(axis=1, skipna=True)
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

for speaker in ['interviewer', 'participant']:
    speaker_distribution = pd.read_csv('analyses/' + speaker + '_dialogue_act_distributions.csv', index_col=[0],
                                     names=['Level 1', 'Level 2', 'Level 3', 'Level 4'], header=0)
    speaker_distribution['Level Total'] = speaker_distribution.mean(axis=1, skipna=True)
    speaker_distribution = speaker_distribution.reindex(distribution_order.index)
    speaker_distribution = speaker_distribution.dropna()
    graph = speaker_distribution.plot.bar()
    if speaker == 'interviewer':
        plt.title('Dialogue Act Distribution per Level for Tutor')
    elif speaker == 'participant':
        plt.title('Dialogue Act Distribution per Level for Student')
    _, labels = plt.xticks()
    graph.set_xticklabels(labels, rotation=45, horizontalalignment='right', fontsize='x-small')
    plt.xlabel("Dialogue Act")
    plt.ylabel("Distribution")
    plt.ylim(0, 1.0)
    plt.tight_layout(2)
    plt.savefig('analyses/' + speaker + '_dialogue_act_distribution_per_level_histogram.png')

# Plots the average utterance length per speaker per level
sns.set_palette([sns.color_palette('Blues', 7)[3], sns.color_palette('Blues', 7)[6]])
utterance_lengths = pd.read_csv('analyses/average_utterance_length.csv', index_col=[0], names=['Student', 'Tutor'],
                                header=0)[['Tutor', 'Student']]
total = utterance_lengths.mean(axis=0)
utterance_lengths = utterance_lengths.append(total, ignore_index=True)
utterance_lengths.index =['Level 1', 'Level 2', 'Level 3', 'Level 4', 'Level Total']
graph = utterance_lengths.plot.bar()
plt.title('Average Utterance Length per Speaker per Level')
_, labels = plt.xticks()
graph.set_xticklabels(labels, rotation=45, horizontalalignment='right', fontsize='x-small')
plt.xlabel("Level")
plt.ylabel("Utterance Length")
plt.tight_layout(2)
plt.savefig('analyses/average_utterance_length_per_speaker_per_level_histogram.png')

speaker_ratios = pd.read_csv('analyses/speaker_turn_ratios.csv', index_col=[0], header=[0])
speaker_ratios['Level Total'] = speaker_ratios.mean(axis=1, skipna=True)
graph = speaker_ratios.T.plot.bar()
plt.title('Speaker Dialogue Ratios per Level')
_, labels = plt.xticks()
graph.set_xticklabels(labels, rotation=45, horizontalalignment='right', fontsize='x-small')
plt.xlabel("Levels")
plt.ylabel("Ratio")
plt.ylim(0, 1.0)
plt.tight_layout(2)
plt.savefig('analyses/speaker_ratios_per_level_histogram.png')



#######################################################################################################################
#                                         PLOTTING DIALOGUE ACT BIGRAM DATA                                           #
#######################################################################################################################

sns.set_palette(sns.color_palette('Blues_r', 6))
statistics.get_bigram_distribution()
for level in [1, 2, 3, 4]:
    # sns.set_palette(sns.color_palette('Blues_r', 6)[level])
    bigram_distribution = pd.read_csv('analyses/level_' + str(level) + '_dialogue_bigram_distribution.csv',
                                      index_col=[0, 1], names=['Distribution'], header=0)
    bigram_distribution = bigram_distribution.sort_values(by=['Distribution'], ascending=False)
    bigram_distribution.to_csv('analyses/sorted_level_' + str(level) + '_dialogue_bigram_distribution.csv')

    # Plots the bigram distribution per level.
    graph = bigram_distribution.plot.line()
    plt.ylim(0, 0.1)
    _, labels = plt.xticks()
    graph.set_xticklabels(labels, visible=False)
    plt.xlabel("Bigrams")
    plt.ylabel("Distribution")
    plt.title('Bigram Distribution Level ' + str(level))
    plt.tight_layout(2)
    plt.savefig('analyses/bigram_distribution_level_' + str(level) + '_line_plot.png')

    # Plots the top 40 most occuring bigrams per level.
    graph = bigram_distribution[:40].sort_values(by='Distribution', ascending=True).plot.barh()
    _, labels = plt.yticks()
    graph.set_yticklabels(labels, fontsize='x-small')
    plt.xlabel("Occurance")
    plt.ylabel("Bigrams")
    plt.title('Top 40 Most Occuring Bigrams at Level ' + str(level))
    plt.xlim(0, 0.1)
    plt.tight_layout()
    plt.savefig('analyses/most_occuring_bigrams_level_' + str(level) + '_histogram.png')






