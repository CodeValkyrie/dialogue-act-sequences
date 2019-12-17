from data import Preprocessing, Statistics

weighted = 'unweighted'
sequence_lengths = [3]
input_settings = ['_d', '_d_s', '_d_s_l', '_d_s_l_u']

for sequence_length in sequence_lengths:

    # Reads in the data containing the predictions of a model under the given settings.
    filename = 'analyses/' + str(weighted) + '_model_sequence_length_' + str(sequence_length) + '_predictions.csv'
    data = Preprocessing(filename)
    statistics = Statistics(data)

    # Gets the precision, recall and f1-score for every dialogue act for different model input settings.
    for input_setting in input_settings:
        columns = ['labels' + input_setting, 'predictions' + input_setting]
        matrix = (statistics.get_normalised_confusion_matrix(data.data, columns) * 100).round(2)
        error_file = 'analyses/' + weighted + '_model_sequence_length_' + str(sequence_length) + input_setting + '_error_analysis.csv'
        matrix.to_csv(error_file)