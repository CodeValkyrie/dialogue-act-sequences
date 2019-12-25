from data import Preprocessing, Statistics


preprocessed = Preprocessing('data/DA_labeled_belc_2019.csv')
statistics = Statistics(preprocessed)
# statistics.get_average_utterance_length(['participant', 'interviewer'], [1, 2, 3, 4])
# statistics.get_da_distribution()
statistics.get_da_distributions(['participant', 'interviewer'], [1, 2, 3, 4])
# statistics.get_bigram_distribution()
# statistics.get_most_common_bigrams(10, [1, 2, 3, 4])
