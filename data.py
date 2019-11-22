def preprocessing(filename, sequence_length):
    """ Reads in the data from a file and returns the preprocessed data as a list with a (X, L) tuple for each dialogue.
        The X stands for the input sequences matrix and the L for the corresponding label vector.

        Args:
            filename        = the name of the file containing the data to be preprocessed
            sequence_length = the length of the sequences the model is to train on

        Returns:
            preprocessed_data   = a list containing the data in matrix form with corresponding label vectors.
    """

    return 0

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