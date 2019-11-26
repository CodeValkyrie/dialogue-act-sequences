import pandas as pd
import numpy as np

class Dataset():

    def __init__(self, dialogue_IDs):
        'Initialization'

        self.dialogue_IDs = dialogue_IDs

    def __len__(self):
        'Denotes the total number of samples'
        return len(self.dialogue_IDs)

    def __getitem__(self, index):
        """ Generates the matrix representation of the dialogue with the given index"""
        ID = self.dialogue_IDs[index]

        return torch.load('-'.join(['data/dialogue', ID, "level" + level, '.pt']))

# Extracting statistics from data
# Reading in all dialogues and saving them in matrix format to .pt files onder the dialogue ID name
# Making batches and labels of the dialogues



class Preprocessing():

    def __init__(self, filename, sequence_length=7):
        """ Reads in the data file containing all the dialogues and stores the matrix representation
            (sequence_length, number_of_utterances, number_of_classes) of each dialogue in a separate .pt file.
            It also stores the dialogue IDs in a file named dialogue_ids.txt and a dictionary containing the class
            vector representations of (speaker, DA) tuples in a file named class_vectors.py.

            Args:
                filename        = the name of the .csv file containing the data to be preprocessed
                sequence_length = the length of the training sequences after which the hidden state is reset.
                                  Default is 7 to counteract the vanishing gradient problem.

            Saves:
                - Data files in the format 'dialogue<ID>_level<levelint>.pt'
                - File containing the unique dialogue IDs: dialogue_ids.txt
                - File containing the class vector representation of the unique (speaker, DA) tuples: class_vectors.py
        """
        self.data = pd.read_csv(filename)

        # Dialogue information
        self.dialogue_IDs = sorted(list(set(self.data['dialogue_id'])))
        self.number_of_dialogues = len(self.dialogue_IDs)

        # Dialogue Act information
        self.DAs = sorted(list(set(self.data['dialogue_act'])))
        self.number_of_DAs = len(self.DAs)

        # Extracts the unique (speaker, DA) tuples from the dataset
        speaker_DA_tuples = self.data[['speaker','dialogue_act']].drop_duplicates().values
        speaker_DA_tuples = [tuple(pair) for pair in speaker_DA_tuples]

        self.number_of_classes = len(speaker_DA_tuples)
        self.class_dict = dict()

        # Constructs a class representation vector for each unique (speaker, DA) tuple and stores them in a dictionary
        class_vectors = np.identity(self.number_of_classes)
        for i in range(self.number_of_classes):
            self.class_dict[speaker_DA_tuples[i]] = class_vectors[i]

        # Converts every dialogue into a matrix representation and saves it to its own file
        for ID in self.dialogue_IDs:

            # Extracting the turn tuples of the dialogue corresponding to the dialogue ID
            dialogue_data = self.data[self.data['dialogue_id'] == ID]
            turns = dialogue_data[['speaker', 'dialogue_act']].values
            turns = [tuple(pair) for pair in turns]
            dialogue_length = len(turns)

            # Converting the turn tuple sequence to a numerical 2D matrix representation
            dialogue_matrix = np.array([]).reshape(-1, self.number_of_classes)
            for turn in turns:
                class_vector = self.class_dict[turn].reshape(-1, self.number_of_classes)
                dialogue_matrix = np.concatenate((dialogue_matrix, class_vector), axis=0)

            # Converting the dialogue matrix to a 3D matrix containing all the possible sequences of sequence_length
            dialogue_representation = np.array([]).reshape(-1, dialogue_length, self.number_of_classes)
            for i in range(dialogue_length - (sequence_length - 1)):
                sequence = dialogue_matrix[i:i+sequence_length, ].reshape(-1, dialogue_length, self.number_of_classes)
                dialogue_representation = np.concatenate((dialogue_representation, sequence), axis=0)

            # Converting the 3D dialogue sequences matrix to a tensor and saving it in a file
            #dialogue_tensor =
            save_name = 'dialogue' + ID + '-level' + str(dialogue_data['level'].iloc[0]) + '.pt'
            #torch.save(dialogue_tensor, save_name)

def dialogue_to_matrix(dialogue, sequence_length):
    """ Returns the dialogue data in matrix format of shape N x sequence_length

        Args:
            dialogue        = the dialogue data
            sequence_length = the length of the sequences the model is to train on

        Returns:
            (X, L)          = a matrix X with the preprocessed data points on the rows and the sequences on the columns;
                              a column vector L containing the labels of X on the indices corresponding to the data
                              points in X
    """
    return 0


""" !!!!!! STATISTICS EXTRACTION FUNCTIONS !!!!!!"""

Preprocessing('data/DA_labeled_belc_2019.csv')