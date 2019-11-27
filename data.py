import pandas as pd
import numpy as np
import torch

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

    def __init__(self, filename):
        """ Reads in the data file containing all the dialogues and stores important information about the data.
            It also stores the dialogue IDs in a file named dialogue_ids.txt and a dictionary containing the class
            vector representations of (speaker, DA) tuples in a file named class_vectors.py.

            Args:
                filename        = the name of the .csv file containing the data to be preprocessed

            Output:
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

        # Constructs a class representation vector for each unique (speaker, DA) tuple and stores them in a dictionary
        self.class_dict = dict()
        class_vectors = np.identity(self.number_of_classes)
        for i in range(self.number_of_classes):
            self.class_dict[speaker_DA_tuples[i]] = class_vectors[i]

    def save_dialogues_as_matrices(self, sequence_length=7):
        """ Reads in the data file containing all the dialogues and stores the matrix representation
            (sequence_length, number_of_utterances, number_of_classes) of each dialogue in a separate .pt file.

            Args:
                sequence_length = the length of the training sequences after which the hidden state is reset.
                                  Default is 7 to counteract the vanishing gradient problem.

            Output:
                - Data files in the format 'dialogue<ID>_level<levelint>.pt'
        """
        for ID in self.dialogue_IDs:

            # Extracts the turn tuples of the dialogue corresponding to the dialogue ID.
            dialogue_data = self.data[self.data['dialogue_id'] == ID]
            turns = dialogue_data[['speaker', 'dialogue_act']].values
            turns = [tuple(pair) for pair in turns]
            dialogue_length = len(turns)

            # Converts the turn tuple sequence to a numerical 2D matrix representation.
            dialogue_matrix = np.array([]).reshape(-1, self.number_of_classes)
            for turn in turns:
                class_vector = self.class_dict[turn].reshape(-1, self.number_of_classes)
                dialogue_matrix = np.concatenate((dialogue_matrix, class_vector), axis=0)

            # Converts the dialogue matrix to a 3D matrix containing all the possible sequences of sequence_length.
            dialogue_representation = np.array([]).reshape(sequence_length, -1, self.number_of_classes)
            for i in range(dialogue_length - (sequence_length - 1)):
                sequence = dialogue_matrix[i:i+sequence_length, ].reshape(sequence_length, -1, self.number_of_classes)
                """
                Checking if the sequences make sense
                for i in range(7):
                    for turn, clas in self.class_dict.items():
                        if np.array_equal(clas, sequence[i,0]):
                            print(turn)
                            print(sequence[i,0])
                """
                dialogue_representation = np.concatenate((dialogue_representation, sequence), axis=1)
            # Converts the 3D dialogue sequences matrix to a tensor and saving it in a file.
            dialogue_tensor = torch.from_numpy(dialogue_representation)
            save_name = 'dialogue' + ID + '-level-' + str(int(dialogue_data['level'].iloc[0])) + '.pt'
            #!!!!torch.save(dialogue_tensor, save_name)!!!!

    def save_dialogue_IDs(self):
        """ Returns and stores the unique dialogue IDs of the data set in a file named dialogue_ids.txt. """
        return self.dialogue_IDs

    def save_class_representation(self):
        """ Returns and stores a dictionary containing the class vector representations of (speaker, DA) tuples
            in a file named class_vectors.py. """
        return self.class_dict

    def get_DAs(self):
        """ Returns a list containing the unique Dialogue Acts of the data set. """
        return self.DAs

    def average_sentence_length(self, speakers, levels):
        """ Returns the average sentence length of the given speaker(s) on the given level(s) of language ability.

            Args:
                speakers = a list containing one or both of: 'participant', 'interviewer'
                levels   = a list containing a subset of {1, 2, 3, 4}

            Returns:
                 A dictionary containing the average count for every input speaker and level.
        """
        if speakers == [] or levels == []:
            return None

        for speaker in speakers:
            speaker_data = self.data[self.data['speaker'] == speaker]
            level_average = dict()
            for level in levels:
                level_data = speaker_data[speaker_data['level'] == level]
                utterance_texts = level(level_data['text'].values)
                #level_average[level] =
        return 0

    def average_word_length(self, speaker, level):
        """ Returns the average word length of the given speaker(s) on the given level(s) of language ability.

            Args:
                speaker = a list containing one or both of: 'participant', 'interviewer'
                level   = a list containing a subset of {1, 2, 3, 4}

            Returns:
                 A matrix containing the average count on index (speaker, level), for every input speaker and level.
        """
        return 0


""" !!!!!! STATISTICS EXTRACTION FUNCTIONS !!!!!!"""

preprocessed = Preprocessing('data/DA_labeled_belc_2019.csv')
preprocessed.average_sentence_length(['participant'], [1.0])