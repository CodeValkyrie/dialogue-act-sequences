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
        matrix_id = self.dialogue_ids[index]
        return torch.load('data/dialogue-' + matrix_id + '.pt')

    def __load_dialogue_ids(self, filename):
        """ Loads and returns the list of unique dialogue IDs of the data set from filename. """
        with open(filename, "r") as file:
            return file.readlines()[0].split()

    def __load_class_representation(self, filename):
        """ Loads and returns the dictionary containing the class vector representations of (speaker, DA) tuples
            from filename. """

        # Reads in the reverse dictionary from the given file.
        with open(filename) as file:
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
            labels = dialogue[:, (i + 1):min((i + 1) + batch_size, dialogue_length), 1]
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

        # Speaker information.
        self.speakers = sorted(list(set(self.data['speaker'])))

        # Level information.
        self.levels = sorted(list(set(self.data['level'])))

        # Extracts the unique (speaker, DA) dialogue turn tuples from the data set.
        speaker_da_tuples = self.data[['speaker', 'dialogue_act']].drop_duplicates().values
        speaker_da_tuples = [tuple(pair) for pair in speaker_da_tuples]

        # Constructs a dictionary consisting of the unique turn tuples and their corresponding vector representation.
        self.number_of_classes = len(speaker_da_tuples)
        self.class_dict = dict()
        class_vectors = np.identity(self.number_of_classes)
        for i in range(self.number_of_classes):
            self.class_dict[speaker_da_tuples[i]] = class_vectors[i]

    def save_dialogues_as_matrices(self, sequence_length=3,
                                   classes=['speaker', 'dialogue_act', 'level', 'utterance_length'], store_index=False):
        """ Stores the matrix representation (sequence_length, number_of_utterances, number_of_classes) of each
            dialogue in the data set into a separate .pt file.

            Args:
                sequence_length = the length of the training sequences after which the hidden state is reset.
                                  Default is 7 to counteract the vanishing gradient problem.

            Output:
                - Data files in the format 'dialogue<ID>_level<levelint>.pt'
        """
        number_of_classes = len(classes)
        if store_index:
            number_of_classes += 1

        # Edits the DataFrame for every dialogue and saves the dialogue in a csv file.
        for ID in self.dialogue_ids:

            # Extracts the turn tuples of the dialogue corresponding to the dialogue ID.
            dialogue_data = self.data[self.data['dialogue_id'] == ID]

            # Replace the speaker values with their speaker set index.
            for speaker in self.speakers:
                dialogue_speakers = dialogue_data['speaker'].replace({speaker: str(self.speakers.index(speaker))})
                dialogue_data = dialogue_data.assign(speaker=dialogue_speakers)

            # Replace the level values with their level set index.
            for level in self.levels:
                dialogue_levels = dialogue_data['level'].replace({level: self.levels.index(level)})
                dialogue_data = dialogue_data.assign(level=dialogue_levels)

            # Replace the dialogue act values with their dialogue act set index.
            for DA in self.DAs:
                dialogue_das = dialogue_data['dialogue_act'].replace({DA: str(self.DAs.index(DA))})
                dialogue_data = dialogue_data.assign(dialogue_act=dialogue_das)

            # Computes the utterance lengths without the punctuation and adds it to the DataFrame.
            utterance_texts = dialogue_data['text'].values
            utterance_lengths = [len(str(utterance).split()) - 1 for utterance in utterance_texts]
            dialogue_data = dialogue_data.assign(utterance_length=utterance_lengths)

            # Makes a Numpy array out of the DataFrame containing the values for the four classes.
            index = dialogue_data[classes].index.to_numpy().reshape(-1, 1)
            dialogue_matrix = dialogue_data[classes].to_numpy().astype(int)

            # Stores the index of the inputs in the original data frame.
            if store_index:
                dialogue_matrix = np.concatenate((dialogue_matrix, index), axis=1)

            # Computes the dialogue length.
            dialogue_length = dialogue_matrix.shape[0]

            # Makes a 3D Numpy array of sequences.
            dialogue_representation = np.array([]).reshape(sequence_length, -1, number_of_classes)
            for i in range(dialogue_length - (sequence_length - 1)):
                sequence = dialogue_matrix[i:i + sequence_length, ].reshape(sequence_length, -1, number_of_classes)
                dialogue_representation = np.concatenate((dialogue_representation, sequence), axis=1)

            # Converts the 3D dialogue sequences matrix to a tensor and saves it in a file.
            dialogue_tensor = torch.from_numpy(dialogue_representation)
            torch.save(dialogue_tensor, 'data/dialogue-' + ID + '.pt')

    def save_dialogues_as_matrices_old(self, sequence_length=7):
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
            torch.save(dialogue_tensor, 'data/dialogue-' + ID + '.pt')

    def save_dialogue_ids(self):
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

    def get_dialogue_acts(self):
        """ Returns a list containing the unique Dialogue Acts of the data set. """
        return self.DAs

    def add_baseline(self, baseline):
        label_indices = list(self.data[self.data['labels_2_gram'].notnull()]['labels_2_gram'].index)
        number_labels = len(label_indices)
        if baseline == 'majority_class':
            majority_class = pd.DataFrame(['statement'] * number_labels, index=label_indices, columns=[baseline])
            self.data = self.data.merge(majority_class, how='left', left_index=True, right_index=True)
        elif baseline == 'random':
            random_class = np.random.randint(13, size=number_labels)
            random_class = [self.DAs[random] for random in random_class]
            random_class = pd.DataFrame(random_class, index=label_indices, columns=[baseline])
            self.data = self.data.merge(random_class, how='left', left_index=True, right_index=True)
        elif baseline == 'weighted_random':
            distribution = self.data['dialogue_act'].value_counts()
            normalised_distribution = (distribution / distribution.sum(axis=0, skipna=True))
            dialogue_acts = list(normalised_distribution.index)
            da_distribution = normalised_distribution.to_numpy().flatten()
            weighted_random = np.random.choice(dialogue_acts, number_labels, p=da_distribution)
            weighted_random = pd.DataFrame(weighted_random, index=label_indices, columns=[baseline])
            self.data = self.data.merge(weighted_random, how='left', left_index=True, right_index=True)

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
                level_name = 'level ' + str(level)
                average_utterance_length[speaker][level_name] = float(round(sum(utterance_lengths) /
                                                                            len(utterance_lengths), 2))
        table = pd.DataFrame(average_utterance_length)
        table.to_csv('analyses/average_utterance_length.csv')
        return table

    def get_da_distributions(self, speakers, levels):
        """ Saves the distribution of DAs per level per speaker.

            Args:
                speakers = a list containing one or both of: 'participant', 'interviewer'
                levels   = a list containing a subset of {1, 2, 3, 4}

            Output:
                 -  Saves a DataFrame containing the dialogue act distributions per level per speaker to
                    <speaker>_dialogue_act_distributions.csv.
        """
        for speaker in speakers:
            speaker_data = self.data.data[self.data.data['speaker'] == speaker]

            # All the levels' average distributions over the dialogues are stored into the columns of a DataFrame.
            distributions = pd.DataFrame(index=sorted(list(set(speaker_data['dialogue_act']))))
            for level in levels:
                level_data = speaker_data[speaker_data['level'] == level]
                dialogue_ids = sorted(list(set(level_data['dialogue_id'])))
                number_of_dialogues = len(dialogue_ids)

                # The average distribution over every dialogue in the given level is stored in a DataFrame.
                level_distributions = pd.DataFrame(index=sorted(list(set(speaker_data['dialogue_act']))))
                for ID in dialogue_ids:
                    dialogue_data = level_data[level_data['dialogue_id'] == ID]

                    # The dialogue acts in the dialogue are counted and then the counts are normalised.
                    distribution = dialogue_data['dialogue_act'].value_counts()
                    nd = (distribution / distribution.sum(axis=0, skipna=True)).round(3)
                    normalised_distribution = pd.DataFrame(nd.values, index=nd.index, columns=[ID])

                    # The normalised counts for the dialogue are added to the level DataFrame.
                    level_distributions = level_distributions.merge(normalised_distribution, how='left',
                                                                    left_index=True, right_index=True)

                # The average distribution is taken over all the dialogues in the level.
                level_distributions = (level_distributions.sum(axis=1, skipna=True) / number_of_dialogues).round(3)
                level_distributions = pd.DataFrame(level_distributions.values, index=level_distributions.index,
                                                   columns=['level' + str(level)])

                # The average distribution of the level is added to a DataFrame containing the other levels as well.
                distributions = distributions.merge(level_distributions, how='left', left_index=True, right_index=True)

            # Saves the dialogue act distributions per level to a .csv file.
            distributions.to_csv('analyses/' + speaker + '_dialogue_act_distributions.csv', index=True, header=True)

    def get_da_distribution(self):
        """ Returns the DA distribution and saves it to a csv file.

            Output:
                - Returns a DataFrame containing the percentages of occurance of each dialogue act
                - Saves the distribution to a file named 'dialogue_act_distribution.csv'
        """
        distribution = self.data.data['dialogue_act'].value_counts()
        normalised_distribution = (distribution / distribution.sum(axis=0, skipna=True)).round(3)
        normalised_distribution = normalised_distribution.sort_values(ascending=False)
        normalised_distribution.to_csv('analyses/dialogue_act_distribution.csv', index=True, header=False)
        return normalised_distribution

    def get_bigram_distribution(self):
        """ Saves the distributions of ((speaker, DA)|(speaker, DA)) bigrams per level to a csv file.

            Output:
                 -  A csv file of the name 'analyses/level_'<level>_dialogue_bigram_distribution.csv' for every level
                    in the data set. The file contains the bigram distribution of two turns.
        """

        # Changes the annotation of the speakers to T (for 'tutor') and S (for 'student').
        data = self.data.data
        data = data.replace({'participant': 'S', 'interviewer': 'T'})

        # Changes the annotation of the dialogue acts.
        data = data.replace({'statement': 'ST',
                             'general_other_question': 'GOQ',
                             'spanish': 'SP',
                             'wh_question': 'WQ',
                             'signal_non_understanding': 'SNU',
                             'yes_answers': 'YA',
                             'backchannel_acknowledge': 'BA',
                             'yes_no_question': 'YNQ',
                             'backchannel_question': 'BQ',
                             'response_acknowledgement': 'RA',
                             'no_answers': 'NA',
                             'repeat_phrase': 'RP',
                             'declarative_yes_no_question': 'DQ'})

        # Saves a DataFrame containing the bigram distributions to a csv file for every level.
        for level in range(1, 5):
            level_data = data[data['level'] == level]
            dialogue_ids = sorted(list(set(level_data['dialogue_id'])))
            number_of_dialogues = len(dialogue_ids)
            dialogue_dict = dict()

            # Takes the average distribution of the bigrams over all the dialogues.
            for ID in dialogue_ids:
                dialogue_data = level_data[level_data['dialogue_id'] == ID][['speaker', 'dialogue_act']]
                dialogue_length = len(dialogue_data.index) - 1
                start = dialogue_data.index[0]
                end = start + dialogue_length
                dialogue_dict[ID] = dict()

                # Extracts and counts the bigrams per dialogue.
                for i in range(start, end):
                    speakers = dialogue_data.at[i, 'speaker'] + dialogue_data.at[i + 1, 'speaker']
                    dialogue_acts = dialogue_data.at[i, 'dialogue_act'] + '-' + dialogue_data.at[i + 1, 'dialogue_act']
                    bigram = (speakers, dialogue_acts)
                    if bigram in dialogue_dict[ID].keys():
                        dialogue_dict[ID][bigram] += 1
                    else:
                        dialogue_dict[ID][bigram] = 1

                dialogue_dict[ID] = {bigram: count / dialogue_length for bigram, count in dialogue_dict[ID].items()}

            # The average distribution over all the dialogues is stored in a new DataFrame for each level.
            level_dialogue = pd.DataFrame(dialogue_dict)
            level_dialogue = (level_dialogue.sum(axis=1, skipna=True) / number_of_dialogues).sort_index()

            # Convert values to percentages.
            level_dialogue = level_dialogue.round(4)

            # The average distributions are saved to a csv file.
            level_dialogue.to_csv('analyses/level_' + str(level) + '_dialogue_bigram_distribution.csv', header=['%'])

    def get_most_common_bigrams(self, n, levels):

        # Saves the n most common occuring bigrams to a csv file for every level in levels
        for level in levels:

            # Gets the bigram distributions for the level.
            data = pd.read_csv('analyses/level_' + str(level) + '_dialogue_bigram_distribution.csv', header=0)

            # Gives the columns practical names.
            data.columns = ['speakers', 'dialogue_acts', 'distribution']

            # The top ten bigrams are calculated for every speaker bigram pair for the level.
            speakers = sorted(list(set(data['speakers'])))
            for speaker_bigram in speakers:
                bigram_data = data[data['speakers'] == speaker_bigram]
                bigram_data = bigram_data.sort_values(by='distribution', axis=0, ascending=False)
                top_n_bigrams = bigram_data[0:n][['dialogue_acts', 'distribution']].set_index('dialogue_acts')
                top_n_bigrams.columns = [speaker_bigram]

                # The top n bigrams and their distributions are saved to a csv file.
                filename = '_'.join(['analyses/level', str(level), speaker_bigram, 'top', str(n), 'bigrams.csv'])
                top_n_bigrams.to_csv(filename)

    def get_da_counts(self, data, column, levels):

        # All the levels' average distributions over the dialogues are stored into the columns of a DataFrame.
        distributions = pd.DataFrame(index=sorted(list(set(data['dialogue_act']))))
        for level in levels:
            level_data = data[data['level'] == level]
            dialogue_ids = sorted(list(set(level_data['dialogue_id'])))

            # The average distribution over every dialogue in the given level is stored in a DataFrame.
            level_distributions = pd.DataFrame(index=sorted(list(set(data['dialogue_act']))))
            for ID in dialogue_ids:
                dialogue_data = level_data[level_data['dialogue_id'] == ID]

                # The dialogue acts in the dialogue are counted and then the counts are added to the data frame.
                distribution = dialogue_data['dialogue_act'].value_counts()
                distribution = pd.DataFrame(distribution.values, index=distribution.index, columns=[ID])
                level_distributions = level_distributions.merge(distribution, how='left', left_index=True,
                                                                right_index=True)

            # The average distribution is taken over all the dialogues in the level.
            level_distributions = level_distributions.sum(axis=1, skipna=True)
            level_distributions = pd.DataFrame(level_distributions.values, index=level_distributions.index,
                                               columns=['level' + str(level)])

            # The average distribution of the level is added to a DataFrame containing the other levels as well.
            distributions = distributions.merge(level_distributions, how='left', left_index=True, right_index=True)

        # Saves the dialogue act distributions per level to a .csv file.
        distributions.to_csv('analyses/' + column + '_column_dialogue_act_counts.csv', index=True, header=True)

    def precision_recall_f1(self, data, columns, dialogue_act):
        """ Returns the precision, recall and f1-score for a dialogue act given a label and a prediction column in a
            DataFrame.

            Args:
                data            = a DataFrame consisting of columns containing predictions and labels
                column_names    = the columns in the DataFrame over which the precision and recall must be calculated
                dialogue_act    = the dialogue act for which the prediction, recall and must be calculated.
       """

        # True positive:    label == DA && prediction == DA
        # False positive:   label != DA && prediction == DA
        # False negative:   label == DA && prediction != DA
        # True negative:    label != DA && prediction != DA

        data_columns = data[columns]
        labels = columns[0]
        predictions = columns[1]

        # Computes the precision = true positives / (true positives + false positives).
        # All the DataFrame rows with the dialogue act in the prediction column (true positives + false positives)
        predictions_of_da = data_columns[data_columns[predictions] == dialogue_act]

        # The true positives are the ones of which the label and the prediction are both the dialogue act.
        true_positives = len(predictions_of_da[predictions_of_da[labels] == dialogue_act])

        # The number of the predictions of the dialogue act is the true positives and false positives combined.
        true_positives_and_false_positives = len(predictions_of_da)

        # Computes the recall.
        # All the DataFrame rows with the dialogue act in the label column (true positives + false negatives)
        true_positives_and_false_negatives = len(data_columns[data_columns[labels] == dialogue_act])

        # If the number of labels as the dialogue act is 0, the recall is set to NaN.
        recall = np.nan
        if true_positives_and_false_negatives != 0:
            recall = true_positives / true_positives_and_false_negatives

        # If the number of predictions as the dialogue act is 0, the precision is set to NaN.
        precision = np.nan
        if true_positives_and_false_positives != 0:
            precision = true_positives / true_positives_and_false_positives
        elif recall == 0:
            precision = 0

        # If either the precision or the recall is NaN, the f1-score is set to NaN as well.
        f1 = np.nan
        if recall == 0:
            f1 = 0
        elif precision != np.nan and recall != np.nan:
            f1 = 2 * (precision * recall) / (precision + recall)

        return precision, recall, f1

    def get_normalised_confusion_matrix(self, data, columns):
        """ Returns the distribution of predictions (columns) belonging to a dialogue act (rows)

            Args:
                data            = a DataFrame consisting of columns containing predictions and labels
                column_names    = the columns in the DataFrame over which the precision and recall must be calculated

            Returns:
                A Dataframe containing the dialogue labels on the rows and the prediction distribution over the
                dialogue acts on the columns.
        """
        # Makes the confusion matrix according to the labels and predictions columns of the data.
        labels = columns[0]
        predictions = columns[1]
        confusion_matrix = pd.crosstab(data[labels], data[predictions])
        confusion_matrix = confusion_matrix.div(confusion_matrix.sum(axis=1), axis=0)

        # Reorders the index and columns according to the dialogue act distribution.
        distribution_order = pd.read_csv('analyses/dialogue_act_distribution.csv', index_col=[0], header=None)
        confusion_matrix = confusion_matrix.reindex(distribution_order.index)

        if set(distribution_order.index) == set(confusion_matrix.columns):
            confusion_matrix = confusion_matrix[distribution_order.index]
        return confusion_matrix

    def get_speaker_ratios(self, levels):
            ratios = pd.DataFrame(index=sorted(list(set(self.data.data['speaker']))))
            for level in levels:
                level_data = self.data.data[self.data.data['level'] == level]
                dialogue_ids = sorted(list(set(level_data['dialogue_id'])))
                number_of_dialogues = len(dialogue_ids)

                level_ratios = pd.DataFrame(index=sorted(list(set(self.data.data['speaker']))))
                for ID in dialogue_ids:
                    dialogue_data = level_data[level_data['dialogue_id'] == ID]

                    # The dialogue acts in the dialogue are counted and then the counts are normalised.
                    ratio = dialogue_data['speaker'].value_counts()
                    nd = (ratio / ratio.sum(axis=0, skipna=True)).round(3)
                    normalised_distribution = pd.DataFrame(nd.values, index=nd.index, columns=[ID])

                    # The normalised counts for the dialogue are added to the level DataFrame.
                    level_ratios = level_ratios.merge(normalised_distribution, how='left',
                                                                    left_index=True, right_index=True)

                # The average distribution is taken over all the dialogues in the level.
                level_ratios = (level_ratios.sum(axis=1, skipna=True) / number_of_dialogues).round(3)
                level_ratios = pd.DataFrame(level_ratios.values, index=level_ratios.index,
                                                   columns=['Level ' + str(level)])

                # The average distribution of the level is added to a DataFrame containing the other levels as well.
                ratios = ratios.merge(level_ratios, how='left', left_index=True,
                                                    right_index=True)

            # Saves the dialogue act distributions per level to a .csv file.
            ratios.to_csv('analyses/speaker_turn_ratios.csv', index=True, header=True)
