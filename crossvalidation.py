import numpy as np
import pandas as pd
import torch
from main import train, evaluate
from model import LSTM


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class CrossValidation:

    def __init__(self, data, k):
        """ Initialises the CrossValidation object with a data set and a value for k."""

        self.data = data
        self.k = k

        # Creating the k-fold train and test sets.
        self.train_ids = [[]] * self.k
        self.test_ids = [[]] * self.k

    def make_k_fold_cross_validation_split(self, levels):
        """ Returns train and test splits for the given levels.

            Args:
                levels = a list containing integers denoting the ability levels, chosen from {1, 2, 3, 4}

            Output:
                It adjusts the object variables self.train_IDs and self.test_IDs, so that they hold k lists of IDs
                for the training and testing respectively.
        """

        # Extracts the IDs belonging to the different levels that need to be split.
        ids = self.data.dialogue_ids
        ids_per_level = dict()
        for level in levels:
            level_ids = []
            for i in ids:
                if i[0] == str(level):
                    level_ids.append(i)
                elif i[0] not in str(levels):
                    ids.remove(i)
            ids_per_level[level] = level_ids

        # Resets the k-fold train and test sets.
        self.train_ids = [[]] * self.k
        self.test_ids = [[]] * self.k

        # Selects k sets of train and test data IDs with equal distribution over the levels.
        for level, ids in ids_per_level.items():

            # The test IDs are as evenly as possible distributed over the k folds.
            test_samples = []
            min_size = int(len(ids) / self.k)
            remainder = len(ids) % self.k
            already_allotted = 0
            for i in range(self.k):
                bucket_size = min_size + (1 if i < remainder else 0)
                test_samples.append(ids[already_allotted:already_allotted + bucket_size])
                already_allotted = already_allotted + bucket_size

            # The training IDs are the compliment of the IDs with the test IDs.
            train_samples = [list(set(ids) - set(test_ids)) for test_ids in test_samples]

            # Adds the train and test sets of each level to the overall train and test sets.
            self.train_ids = [x + y for x, y in zip(self.train_ids, train_samples)]
            self.test_ids = [x + y for x, y in zip(self.test_ids, test_samples)]

    def validate(self, n_classes, hidden_nodes, number_of_layers, lr, batch_size, epochs, input_classes, save_labels_predictions=False):
        """ Performs k-fold cross-validation on the objects data set with a given model on given levels of the data
            set with given training parameters.

            Args:
                n_classes       = the number of classes is the dataset
                hidden_nodes    = the number of hidden nodes in the hidden layers of the LSTM model
                number_of_layers= the number of layers of the LSTM model
                lr              = learning rate, default is 5e-3
                batch_size      = the batch size
                epochs          = the number of epochs

            Returns:
                - A tuple containing the average (precision, recall, F1-measure).

        """
        scores = np.empty(self.k)
        labels_predictions = pd.DataFrame()
        for i in range(self.k):

            # Initialises model.
            model = LSTM(input_dimensions=[2, 13, 4], embedding_dimensions=[4, 20, 10] , hidden_nodes=64, n_layers=1, n_classes=13, input_classes=input_classes).to(device)

            # Trains the model on the training set belonging to the iteration of the k-fold.
            self.data.set_dialogue_ids(self.train_ids[i])
            train(model, self.data, lr, batch_size, epochs)

            # Tests the model on the test set belonging to the iteration of the k-fold.
            self.data.set_dialogue_ids(self.test_ids[i])
            if save_labels_predictions:
                labels_predictions, scores[i] = evaluate(model, self.data, save_labels_predictions)
                print(pd.DataFrame(labels_predictions))
            else:
                scores[i] = evaluate(model, self.data, save_labels_predictions)
        return scores
