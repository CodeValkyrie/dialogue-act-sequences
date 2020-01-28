from data import Preprocessing, Statistics
import pandas as pd


preprocessed = Preprocessing('data/DA_labeled_belc_2019.csv')
statistics = Statistics(preprocessed)
# statistics.get_average_utterance_length(['participant', 'interviewer'], [1, 2, 3, 4])
# statistics.get_da_distribution()
# statistics.get_da_distributions(['participant', 'interviewer'], [1, 2, 3, 4])
# statistics.get_bigram_distribution()
# statistics.get_most_common_bigrams(10, [1, 2, 3, 4])
# data = pd.read_csv('analyses/unweighted_model_sequence_length_3_predictions.csv', index_col=[0])
# statistics.get_da_counts(data, 'dialogue_act', [1, 2, 3, 4])
# statistics.get_n_dialogues_average_length([1, 2, 3, 4])

