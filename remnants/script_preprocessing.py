""" This script preprocesses the data and saves it into files. """
from data import Preprocessing, Statistics

preprocessed = Preprocessing('data/DA_labeled_belc_2019.csv')
preprocessed.save_dialogue_ids()
preprocessed.save_class_representation()
# preprocessed.save_dialogues_as_matrices_old(sequence_length=7)
preprocessed.save_dialogues_as_matrices()