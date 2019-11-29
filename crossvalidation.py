import torch
from data import DataSet
from main import train, evaluate
from model import LSTM


class CrossValidation:

    def __init__(self, data, k):
        """ Initialises the CrossValidation object with a data set and a value for k."""

        self.data = data
        self.k = k

        # Creating the k-fold train and test sets.
        self.train_IDs = [[]] * self.k
        self.test_IDs = [[]] * self.k

    def make_k_fold_cross_validation_split(self, levels):
        """ Returns train and test splits for the given levels.

            Args:
                levels = a list containing integers denoting the ability levels, chosen from {1, 2, 3, 4}

            Output:
                It adjusts the object variables self.train_IDs and self.test_IDs, so that they hold k lists of IDs
                for the training and testing respectively.
        """

        # Extracts the IDs belonging to the different levels that need to be split.
        ids = self.data.dialogue_IDs
        ids_per_level = dict()
        for level in levels:
            level_ids = []
            for ID in ids:
                if ID[0] == str(level):
                    level_ids.append(ID)
                elif ID[0] not in str(levels):
                    ids.remove(ID)
            ids_per_level[level] = level_ids

        # Resets the k-fold train and test sets.
        self.train_IDs = [[]] * self.k
        self.test_IDs = [[]] * self.k

        # Selects k sets of train and test data IDs with equal distribution over the levels.
        for level, IDs in ids_per_level.items():

            # The test IDs are as evenly as possible distributed over the k folds.
            test_samples = []
            min_size = int(len(IDs) / self.k)
            remainder = len(IDs) % self.k
            already_allotted = 0
            for i in range(self.k):
                if i < remainder:
                    test_samples.append(IDs[already_allotted:already_allotted + min_size + 1])
                    already_allotted = already_allotted + min_size + 1
                else:
                    test_samples.append(IDs[already_allotted:already_allotted + min_size])
                    already_allotted = already_allotted + min_size

            # The training IDs are the compliment of the IDs with the test IDs.
            train_samples = [list(set(IDs) - set(test_IDs)) for test_IDs in test_samples]

            # Adds the train and test sets of each level to the overall train and test sets.
            self.train_IDs = [x + y for x, y in zip(self.train_IDs, train_samples)]
            self.test_IDs = [x + y for x, y in zip(self.test_IDs, test_samples)]

    def validate(self, model, lr=5e-3, batch_size=16, epochs=5):
        """ Performs k-fold cross-validation on the objects data set with a given model on given levels of the data
            set with given training parameters.

            Args:
                model       = a neural network model that needs to be evaluated
                lr          = learning rate, default is 5e-3
                batch_size  = the batch size
                epochs      = the number of epochs

            Returns:
                - A tuple containing the average (precision, recall, F1-measure).

        """
        for i in range(self.k):

            # Trains the model on the training set belonging to the iteration of the k-fold.
            self.data.set_dialogue_ids(self.train_IDs[i])
            train(model, self.data, lr, batch_size, epochs)

            # Tests the model on the test set belonging to the iteration of the k-fold.
            self.data.set_dialogue_ids(self.test_IDs[i])
            evaluate(model, self.data)


# Global Variables initialisation
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

dataset = DataSet()
 # Defines hyperparameters for model initialisation.
n_classes = dataset.get_number_of_classes()
n_layers = 1
hidden_nodes = 64

lstm = LSTM(n_classes, hidden_nodes, n_layers).to(device)

crossval = CrossValidation(DataSet(), 10)
crossval.make_k_fold_cross_validation_split([1,4])
#crossval.validate(lstm)