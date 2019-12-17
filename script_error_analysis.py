from data import Preprocessing, Statistics


""" This is a script that stores the normalised confusion matrix of the dialogue act prediction.

    The variables that need to be specified:
        weighted           = chosen from {"weighted", "unweighted"} to specify which model's predictions are to be used.
        sequence_lengths   = the sequence length with which the model made the predictions.
        input_settings     = the input settings used: a subset from 
                             {'dialogue act', 'speaker', 'level', 'utterance length'}. The list must consist of 
                             abbreviations in the format '_<first letter>', for example ['_d', '_d_u'], which uses first 
                             only dialogue acts and then dialogue acts and utterance lengths.

    The script outputs csv files containing the normalised distribution of predictions (columns) for each label (rows).
"""

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
        error_file = 'analyses/' + weighted + '_model_sequence_length_' + str(sequence_length) + input_setting + \
                     '_error_analysis.csv'
        matrix.to_csv(error_file)
