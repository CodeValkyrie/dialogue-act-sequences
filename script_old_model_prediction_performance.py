import pandas as pd
import numpy as np
import torch
from old_model_without_text import LSTM
from data import Preprocessing, DataSet
from train import train, evaluate

''' This script trains the weighted LSTM model without sentence embeddings on the training set and stores the prediction 
    made on the test set. 
'''

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


sequence_length = 3
levels = [1, 2, 3, 4]
k = 10
models = ['weighted', 'unweighted']

# Model hyper parameters.
number_of_layers = 1
hidden_nodes = 16
input_classes = ['dialogue_act', 'speaker', 'level', 'utterance_length']
embedding_dimensions = None

# Training hyper parameters.
learning_rate = 0.001
batch_size = 16
epochs = 20

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

# Makes predictions for the weighted and unweighted model and stores them.
for weighted in models:
    print("Prediction performance for " + weighted + " model")

    if weighted == 'weighted':
        embedding_dimensions = [1, 7, 2]
    elif weighted == 'unweighted':
        embedding_dimensions = [2, 13, 4, 1]
    output = np.empty((1, 3))

    # Initialises model.
    model = LSTM(input_dimensions=[2, 13, 4], embedding_dimensions=embedding_dimensions,
                 hidden_nodes=hidden_nodes, n_layers=1, n_classes=13, input_classes=input_classes).to(device)

    # Trains the model on the training set.
    train(model, train_data, learning_rate, batch_size, epochs, weighted)

    # Tests the model on the test set and stores the labels and predictions.
    labels_predictions, accuracy = evaluate(model, test_data, save_labels_predictions=True)
    labels_predictions = pd.DataFrame(labels_predictions.reshape(-1, 3))

    # Stores the labels and predictions in a DataFrame.
    input_frame = pd.DataFrame(labels_predictions)
    columns = input_frame.columns
    input_frame = input_frame.set_index(columns[0]).rename_axis(None)
    input_frame = input_frame.rename(columns={columns[1]: 'labels_' + weighted, columns[2]: 'predictions_' + weighted})
    input_frame = input_frame.astype(str)

    # Replaces all the numerical values of the labels and predictions with their name.
    DAs = preprocessed_train.DAs
    for i in range(len(DAs)):
        input_frame = input_frame.replace({str(i) + '.0': DAs[i]})

    # Adds the labels and predictions for test set as columns to the original test data in one DataFrame.
    data_frame = data_frame.merge(input_frame, how='left', left_index=True, right_index=True)

# Saves the DataFrame containing all the labels and predictions for the different input settings.
data_frame.to_csv('analyses/old_model_sequence_length_3_test_set_predictions.csv')
