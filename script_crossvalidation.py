import main

# Defines the k-fold cross-validation parameters.
LEVELS = [1, 2, 3]
K = 10

# Model Hyperparameters.
N_LAYERS = 1
HIDDEN_NODES = 64

# Training hyperparameters
LEARNING_RATE = 5e-3
BATCH_SIZE = 16
EPOCHS = 10

# Makes a Dataset object from the dataset and creates the k-fold cross-validation splits.
dataset = Dataset()
dataset.make_k_fold_cross_validation_split(LEVELS, K)

n_classes = dataset.get_number_of_classes()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

for i in range(K):
    dataset.set_k_iteration(i)
    rnn = LSTM(n_classes, hidden_nodes, n_layers).to(device)
    train(rnn, dataset, learning_rate, batch_size, epochs)

    return 0
