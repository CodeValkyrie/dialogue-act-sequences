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

settings = ['unweighted', 'weighted']
sequence_lengths = [2, 3, 5, 7, 10, 15, 20]
levels = [1, 2, 3, 4]

# Reads in the data containing the predictions of the model with sentence embeddings for different sequence lengths
filename = 'analyses/new_model_predictions.csv'
data = Preprocessing(filename)
statistics = Statistics(data)

# Computes the confusion matrix for different sequence lengths.
for sequence_length in sequence_lengths:
    columns = ['labels_seq_len_' + str(sequence_length), 'predictions_seq_len_' + str(sequence_length)]
    matrix = (statistics.get_normalised_confusion_matrix(data.data, columns) * 100).round(2)
    error_file = 'analyses/weighted_model_with_txt_sequence_length_' + str(sequence_length) + '_error_analysis.csv'
    matrix.to_csv(error_file)

    # Computes the confusion matrix for different levels with the sequence length.
    for level in levels:
        level_data = data.data[data.data['level'] == level]
        matrix = (statistics.get_normalised_confusion_matrix(level_data, columns) * 100).round(2)
        error_file = 'analyses/weighted_model_with_txt_sequence_length_' + str(sequence_length) + '_level_' + \
                     str(level) + '_error_analysis.csv'
        matrix.to_csv(error_file)

# Reads in the data containing the predictions of the old weighted and unweighted models.
filename = 'analyses/old_model_sequence_length_3_test_set_predictions.csv'
data = Preprocessing(filename)
statistics = Statistics(data)

# Computes the confusion matrix for the old weighted and unweighted models.
for weighted in settings:
    columns = ['labels_' + weighted, 'predictions_' + weighted]
    matrix = (statistics.get_normalised_confusion_matrix(data.data, columns) * 100).round(2)
    error_file = 'analyses/old_' + weighted + '_model_sequence_length_3_error_analysis.csv'
    matrix.to_csv(error_file)
    for level in levels:
        level_data = data.data[data.data['level'] == level]
        matrix = (statistics.get_normalised_confusion_matrix(level_data, columns) * 100).round(2)
        error_file = 'analyses/old_' + weighted + '_model_sequence_length_3_level_' + \
                     str(level) + '_error_analysis.csv'
        matrix.to_csv(error_file)

