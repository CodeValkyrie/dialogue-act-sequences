import pandas as pd
import numpy as np
from model import LSTM
from data import Preprocessing, DataSet
from main import train, evaluate

sequence_lengths = [2, 3, 5, 7, 10, 15, 20]
levels = [1, 2, 3, 4]
k = 10
weighted = 'weighted'

# Model hyper parameters.
number_of_layers = 1
hidden_nodes = 50
input_classes = ['dialogue_act', 'speaker', 'level', 'utterance_length']
embedding_dimensions = [1, 7, 2]

# Training hyper parameters.
learning_rate = 0.001
batch_size = 16
epochs = 20

# Makes predictions for the weighted and unweighted model and stores them.
for sequence_length in sequence_lengths:
    print("Prediction performance for sequence length " + str(sequence_length))

    # Preprocesses the training data for the sequence length.
    preprocessed_train = Preprocessing('data/DA_labeled_belc_2019.csv')
    preprocessed_train.save_dialogues_as_matrices(sequence_length=sequence_length, store_index=True)
    preprocessed_train.save_dialogue_ids()
    preprocessed_train.save_class_representation()
    train_data = DataSet()

    # Preprocesses the test data for the sequence length.
    preprocessed_test = Preprocessing('data/DA_labeled_belc_2019.csv')
    preprocessed_test.save_dialogues_as_matrices(sequence_length=sequence_length, store_index=True)
    preprocessed_test.save_dialogue_ids()
    preprocessed_test.save_class_representation()
    data_frame = preprocessed_test.data
    test_data = DataSet()

    output = np.empty((1, 3))

    # Initialises model.
    model = LSTM(input_dimensions=[2, 13, 4], embedding_dimensions=embedding_dimensions,
                 hidden_nodes=hidden_nodes, n_layers=1, n_classes=13, input_classes=input_classes)

    # Trains the model on the training set.
    train(model, train_data, learning_rate, batch_size, epochs, weighted)

    # Tests the model on the test set and stores the labels and predictions.
    labels_predictions, accuracy = evaluate(model, test_data, save_labels_predictions=True)
    labels_predictions = pd.DataFrame(labels_predictions.reshape(-1, 3))

    # Stores the labels and predictions in a DataFrame.
    input_frame = pd.DataFrame(labels_predictions)
    columns = input_frame.columns
    input_frame = input_frame.set_index(columns[0]).rename_axis(None)
    input_frame = input_frame.rename(columns={columns[1]: 'labels_seq_len_' + str(sequence_length),
                                              columns[2]: 'predictions_seq_len_' + str(sequence_length)})
    input_frame = input_frame.astype(str)

    # Replaces all the numerical values of the labels and predictions with their name.
    DAs = preprocessed_train.DAs
    for i in range(len(DAs)):
        input_frame = input_frame.replace({str(i) + '.0': DAs[i]})

    # Adds the labels and predictions for test set as columns to the original test data in one DataFrame.
    data_frame = data_frame.merge(input_frame, how='left', left_index=True, right_index=True)

# Saves the DataFrame containing all the labels and predictions for the different input settings.
data_frame.to_csv('analyses/new_model_predictions.csv')