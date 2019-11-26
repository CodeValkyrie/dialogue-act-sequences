# coding: utf-8
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import precision_score, recall_score, f1_score
import model

# Global Variables initialisation
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def main():
    """ Runs the RNN algorithm. """

    model = model.RNNModel(n_input, hidden_nodes, n_layers, n_classes).to(device)
    return 0

###################################################################################
#                            HELPER FUNCTIONS                                     #
###################################################################################

def train(model, data, learning_rate, batch_size, epochs):
    """ Trains a given RNN model on a given preprocessed data set with a specified learning rate,
        batch size and number of epochs.

    Args
        model           = the RNN model that is to be trained
        data            = the preprocessed data set that the RNN model is going to train on
        learning_rate   = learning rate
        batch_size      = the number of data points used in training at the same time
        epochs          = the number of times the RNN trains on the same data points
    """
    criterion = nn.CrossEntropyLoss()

    # ADJUST FOR OTHER OPTIMISERS
    optimiser = optim.Adam(model.parameters(), lr=learning_rate)

    model.train()
    total_loss = 0
    for batch, labels in data:
        optimiser.zero_grad()
        batch = batch.to(device)
        labels = labels.to(device)
        output = model(data, None)
        loss = criterion(output.view(-1, size_classes), labels)
        loss.backward()
        optimiser.step()
        total_loss += loss.item()
        """if batch % args.log_interval == 0 and batch > 0:
            cur_loss = total_loss / args.log_interval
            elapsed = time.time() - start_time
            print('| epoch {:3d} | {:5d}/{:5d} batches | lr {:02.2f} | ms/batch {:5.2f} | '
                    'loss {:5.2f} | ppl {:8.2f}'.format(
                epoch, batch, len(train_data) // args.bptt, lr,
                elapsed * 1000 / args.log_interval, cur_loss, math.exp(cur_loss)))
            total_loss = 0
            start_time = time.time()"""

def evaluate(model, data, labels):
    """ Returns the prediction evaluation scores precision, recall and F1 of the RNN model
        on a data sequences of length x after 10-fold cross-validation

        Args:
            model                      = RNN model to be evaluated
            data                       = data on which the RNN is evaluated
            labels                     = the labels of the data

        Returns:
            (Precision, recall, F1)    = a tuple containing the scores of the precision, recall and F1 measures
    """
    prediction = predict(model, data)
    precision = precision_score(labels, prediction)
    recall = recall_score(labels, prediction)
    f1 = f1_score(labels, prediction)
    return precision, recall, f1



def predict(model, input):
    """ Returns the predicted output of the RNN model given the input.

        Args:
            model   = a trained RNN model
            input   = a data point/sequence or several independent data points/sequences

        Returns:
            output  = predicted next data point or data points given the input
    """
    return torch.softmax(model(input, None)[0])

def generate_sequence(model, input, x):
    """ Returns a generated sequence of length x given an input.

            Args:
                model   = a trained RNN model
                input   = a data point
                x       = the length of the generated sequence

            Returns:
                output  = generated data sequence of length x
    """
    hidden_state = None
    sequence = [input]
    for i in range(x):
        output, hidden_state = model(sequence[-1], hidden_state)
        sequence.append(output)
    return sequence

#####################################################################################

main()