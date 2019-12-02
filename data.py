import pandas as pd
import numpy as np
import torch
import json


class DataSet:

    def __init__(self):
        """ Initialization consisting of reading in all the dialogue IDs and the mapping from vector representations
            of the classes to the textual representation. """

        self.dialogue_ids = self.__load_dialogue_ids("data/dialogue_ids.txt")
        self.class_dict = self.__load_class_representation("data/class_vectors.txt")

    def __len__(self):
        """ Denotes the total number of dialogues. """
        return len(self.dialogue_ids)

    def __getitem__(self, index):
        """ Generates the matrix representation of the dialogue with the given index. """
        id = self.dialogue_ids[index]
        return torch.load('data/dialogue-' + id + '.pt')

    def __load_dialogue_ids(self, filename):
        """ Loads and returns the list of unique dialogue IDs of the data set from filename. """
        with open(filename, "r") as file:
            return file.readlines()[0].split()

    def __load_class_representation(self, filename):
        """ Loads and returns the dictionary containing the class vector representations of (speaker, DA) tuples
            from filename. """

        # Reads in the reverse dictionary from the given file.
        with open('data/class_vectors.txt') as file:
            return json.load(file)

    def get_batch_labels(self, dialogue, batch_size=16):
        """ Slices a given dialogue in batches and the corresponding labels and returns a list
            containing (batch, labels) for each batch.

            Args:
                dialogue    = a PyTorch tensor of shape (sequence_length, number_of_turns, number_of_classes)
                batch_size  = the chosen batch size to split the dialogue into

            Returns:
                - A list of (batch, labels) tuples
                batch   = a PyTorch tensor of shape (sequence_length, batch_size, number_of_classes)
                labels  = a PyTorch tensor of shape (sequence_length, batch_size, number_of_classes)
        """
        dialogue = dialogue.numpy()
        dialogue_length = dialogue.shape[1]
        batch_tuples = []

        # Following adapted from https://stackoverflow.com/questions/48702808/numpy-slicing-with-batch-size.
        for i in range(0, dialogue_length - 1, batch_size):
            batch = dialogue[:, i:min(i + batch_size, dialogue_length - 1), :]

            # The labels of the sequences are just the next labels in the sequence.
            labels = dialogue[:, (i + 1):min((i + 1) + batch_size, dialogue_length), :]
            batch_tuples.append((torch.from_numpy(batch), torch.from_numpy(labels)))
        return batch_tuples

    def get_number_of_classes(self):
        """ Returns the number of classes in the data set. """
        return len(self.class_dict.keys())

    def get_class_decoder(self):
        """ Returns the dictionary containing the class labels and their verctor representations. """
        return self.class_dict

    def set_dialogue_ids(self, dialogue_ids):
        """ Sets the list of dialogue ids to dialogue_ids. """
        self.dialogue_ids = dialogue_ids


class Preprocessing:
    """ Class defining the variables and functions belonging to a preprocessed data object. """

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

        # Dialogue information.
        self.dialogue_ids = sorted(list(set(self.data['dialogue_id'])))
        self.number_of_dialogues = len(self.dialogue_ids)

        # Dialogue Act information.
        self.DAs = sorted(list(set(self.data['dialogue_act'])))
        self.number_of_DAs = len(self.DAs)

        # Extracts the unique (speaker, DA) dialogue turn tuples from the data set.
        speaker_DA_tuples = self.data[['speaker','dialogue_act']].drop_duplicates().values
        speaker_DA_tuples = [tuple(pair) for pair in speaker_DA_tuples]

        # Constructs a dictionary consisting of the unique turn tuples and their corresponding vector representation.
        self.number_of_classes = len(speaker_DA_tuples)
        self.class_dict = dict()
        class_vectors = np.identity(self.number_of_classes)
        for i in range(self.number_of_classes):
            self.class_dict[speaker_DA_tuples[i]] = class_vectors[i]

    def save_dialogues_as_matrices(self, sequence_length=7):
        """ Stores the matrix representation (sequence_length, number_of_utterances, number_of_classes) of each
            dialogue in the data set into a separate .pt file.

            Args:
                sequence_length = the length of the training sequences after which the hidden state is reset.
                                  Default is 7 to counteract the vanishing gradient problem.

            Output:
                - Data files in the format 'dialogue<ID>_level<levelint>.pt'
        """
        for ID in self.dialogue_ids:

            # Extracts the turn tuples of the dialogue corresponding to the dialogue ID.
            dialogue_data = self.data[self.data['dialogue_id'] == ID]
            turns = dialogue_data[['speaker', 'dialogue_act']].values
            turns = [tuple(pair) for pair in turns]
            dialogue_length = len(turns)

            # Converts the turn tuple sequence to a numerical 2D matrix representation (samples, classes).
            dialogue_matrix = np.array([]).reshape(-1, self.number_of_classes)
            for turn in turns:
                class_vector = self.class_dict[turn].reshape(-1, self.number_of_classes)
                dialogue_matrix = np.concatenate((dialogue_matrix, class_vector), axis=0)

            # Converts the dialogue matrix to a 3D matrix containing all the possible sequences of sequence_length.
            dialogue_representation = np.array([]).reshape(sequence_length, -1, self.number_of_classes)
            for i in range(dialogue_length - (sequence_length - 1)):
                sequence = dialogue_matrix[i:i+sequence_length, ].reshape(sequence_length, -1, self.number_of_classes)
                dialogue_representation = np.concatenate((dialogue_representation, sequence), axis=1)

            # Converts the 3D dialogue sequences matrix to a tensor and saves it in a file.
            dialogue_tensor = torch.from_numpy(dialogue_representation)
            torch.save(dialogue_tensor, 'data/dialogue-' +  ID + '.pt')

    def save_dialogue_IDs(self):
        """ Returns and stores the list of unique dialogue IDs of the data set in a file named dialogue_ids.txt. """
        with open("data/dialogue_ids.txt", "w") as file:
            for ID in self.dialogue_ids:
                file.write(ID + " ")
        return self.dialogue_ids

    def save_class_representation(self):
        """ Returns and stores the dictionary containing the class vector representations of (speaker, DA) tuples
            in a file named class_vectors.txt. """
        class_dict = {}
        for key, value in self.class_dict.items():
            class_dict['-'.join(key)] = list(value)
        with open('data/class_vectors.txt', 'w') as file:
            json.dump(class_dict, file)
        return class_dict

    def get_DAs(self):
        """ Returns a list containing the unique Dialogue Acts of the data set. """
        return self.DAs


class Statistics:

    def __init__(self, data):
        """ Initialises the Statistics object for a preprocessed data set.

            Input:
                data    = an object of the Preprocessing class
        """
        self.data = data

    def get_average_utterance_length(self, speakers, levels):
        """ Returns and saves the average utterance length of the given speaker(s) on the given level(s) of
            language ability.

            Args:
                speakers = a list containing one or both of: 'participant', 'interviewer'
                levels   = a list containing a subset of {1, 2, 3, 4}

            Output:
                 - Returns a dictionary containing the average count for every input speaker and level.
                 - Saves adictionary containing the average counts per speaker per level to a .csv file.
        """
        if speakers == [] or levels == []:
            return None

        average_utterance_length = dict()
        for speaker in speakers:
            average_utterance_length[speaker] = dict()
            speaker_data = self.data.data[self.data.data['speaker'] == speaker]
            for level in levels:
                level_data = speaker_data[speaker_data['level'] == level]
                utterance_texts = level_data['text'].values

                # Computing the utterance lengths without the punctuation
                utterance_lengths = [len(str(utterance).split()) - 1 for utterance in utterance_texts]
                l = 'level ' + str(level)
                average_utterance_length[speaker][l] = float(round(sum(utterance_lengths) / len(utterance_lengths), 2))
        table = pd.DataFrame(average_utterance_length)
        table.to_csv('analyses/average_utterance_length.csv')
        return table

    def get_most_common_bigrams(self, n, speakers, levels):
        """ Returns and saves the n most common bigrams per level per speaker and their normalised occurance.

            Args:
                n        = the number of most common bigrams the function must return. If the number is higher than the
                           total number of bigrams, the total number of bigrams is returned
                speakers = a list containing one or both of: 'participant', 'interviewer'
                levels   = a list containing a subset of {1, 2, 3, 4}

            Output:
                 - ????
                 - ????
        """
        for speaker in speakers:
            speaker_data = self.data.data[self.data.data['speaker'] == speaker]

            # Initialises a dictionary to count the bigrams per level.
            level_bigrams = dict()
            for level in levels:
                level_bigrams[str(level)] = dict()

            total_bigrams_per_level = np.zeros(4)
            # Counts the bigrams per level
            for dialogue_id in self.data.dialogue_ids:
                level = dialogue_id[0]
                dialogue_data = speaker_data[speaker_data['dialogue_id'] == dialogue_id]
                dialogue_DAs = dialogue_data['dialogue_act'].values
                total_bigrams_per_level[int(level) - 1] += len(dialogue_DAs)
                for i in range(len(dialogue_DAs)-1):
                    bigram = (dialogue_DAs[i], dialogue_DAs[i + 1])
                    if bigram in level_bigrams[level].keys():
                        level_bigrams[level][bigram] += 1
                    else:
                        level_bigrams[level][bigram] = 1
            return 0

    def get_da_distributions(self, speakers, levels):
        """ Saves the distribution of DAs per level per speaker.

            Args:
                speakers = a list containing one or both of: 'participant', 'interviewer'
                levels   = a list containing a subset of {1, 2, 3, 4}

            Output:
                 -  Saves a DataFrame containing the dialogue act distributions per level per speaker to
                    <speaker>_dialogue-act_distributions.csv.
        """
        for speaker in speakers:
            speaker_data = self.data.data[self.data.data['speaker'] == speaker]
            distributions = pd.DataFrame(index=sorted(list(set(speaker_data['dialogue_act']))))
            for level in levels:
                level_data = speaker_data[speaker_data['level'] == level]
                distribution = level_data['dialogue_act'].value_counts()
                nd = (distribution / distribution.sum(axis=0, skipna=True)).round(3)
                normalised_distribution = pd.DataFrame(nd.values, index=nd.index, columns=['level ' + str(level)])
                distributions = distributions.merge(normalised_distribution, how='left', left_index=True, right_index=True)
                distributions.to_csv('analyses/' + speaker + '_dialogue_act_distributions.csv')

    def get_da_distribution(self):
        """ Returns the DA distribution and saves it to a csv file.

            Output:
                - Returns a DataFrame containing the percentages of occurance of each dialogue act
                - Saves the distribution to a file named 'dialogue_act_distribution.csv'
        """
        distribution = self.data.data['dialogue_act'].value_counts()
        normalised_distribution = (distribution / distribution.sum(axis=0, skipna=True)).round(3)
        normalised_distribution.to_csv('analyses/dialogue_act_distribution.csv', index=True, header=False)
        return normalised_distribution
