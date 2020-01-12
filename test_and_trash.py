import random
IDs = random.sample(range(40), 36)
k = 10

min_size = int(len(IDs)/k)
remainder = len(IDs)%k

test_samples = []
already_allotted = 0

for i in range(k):
    if i < remainder:
        test_samples.append(IDs[already_allotted:already_allotted + min_size + 1])
        already_allotted = already_allotted + min_size + 1
    else:
        test_samples.append(IDs[already_allotted:already_allotted + min_size])
        already_allotted = already_allotted + min_size
print(already_allotted)
print(test_samples)

chunk_size = int(len(IDs) / self.k)
for i in range(self.k):
    if i == self.k - 1:
        test_samples.append(IDs[i * chunk_size:])
    else:
        test_samples.append(IDs[i * chunk_size:i * chunk_size + chunk_size])

""" print(ID)
# Checking if the sequences make sense
for i in range(7):
    for turn, clas in self.class_dict.items():
        if np.array_equal(clas, sequence[i,0]):
            print(turn)
            print(sequence[i,0])"""

precision_total = 0
recall_total = 0
f1_score_total = 0
for dialogue in data:
    batches_labels = data.get_batch_labels(dialogue, batch_size=16)
    for batch, labels in batches_labels:
        labels = torch.argmax(labels, dim=2).numpy().reshape(-1)
        prediction = torch.argmax(predict(model, batch), dim=2).detach().numpy().reshape(-1)
        precision_total += precision_score(labels, prediction, average='macro')
        recall_total += recall_score(labels, prediction, average='macro')
        f1_score_total += f1_score(labels, prediction, average='macro')
        i += 1
return np.array([precision_total, recall_total, f1_score_total]) / i

for i in range(self.k):
    if i < remainder:
        test_samples.append(ids[already_allotted:already_allotted + min_size + 1])
        already_allotted = already_allotted + min_size + 1
    else:
        test_samples.append(ids[already_allotted:already_allotted + min_size])
        already_allotted = already_allotted + min_size

    # x, y  = tuple(top_n_bigrams.to_dict('list').values())
    # top_n_bigrams = dict(zip(x, y))
    # top_n[speaker_bigram] = top_n_bigrams
    # print(top_n)
# top_n_bigrams_per_speaker = pd.DataFrame(top_n)

def evaluate(model, data, save_labels_predictions=False):
    """ Returns the prediction evaluation scores precision, recall and F1 of the RNN model
        on a data sequences of length x after 10-fold cross-validation

        Args:
            model                      = RNN model to be evaluated
            data                       = data on which the RNN is evaluated
            labels                     = the labels of the data

        Returns:
            (Precision, recall, F1)    = a tuple containing the scores of the precision, recall and F1 measures
    """
    i = 0
    accuracy_total = 0
    model.eval()
    labels_predictions = None
    for dialogue in data:
        batches_labels = data.get_batch_labels(dialogue, batch_size=16)
        for batch, labels in batches_labels:

            if labels_predictions is None:
                labels_predictions = np.empty((batch.shape[0], 0, 3))

            # If the predictions and labels must be stored, stores the labels and predictions with their input's index.
            if save_labels_predictions:
                labels_to_store = np.expand_dims(labels, axis=2)
                index_to_store = np.expand_dims(batch[:, :, 4], axis=2)
                predictions = np.expand_dims(torch.argmax(predict(model, batch[:, :, :4]), dim=2).detach().numpy(), axis=2)
                labels_predictions_batch = np.concatenate((index_to_store, labels_to_store, predictions), axis=2)
                labels_predictions = np.concatenate((labels_predictions, labels_predictions_batch), axis=1)

            # Computes the accuracy score.
            labels = labels.numpy().reshape(-1)
            predictions = predictions.reshape(-1)
            accuracy_total += accuracy_score(labels, predictions)
            i += 1
    print('accuracy', accuracy_total / i)
    if save_labels_predictions:
        return labels_predictions, accuracy_total / i
    return accuracy_total / i


dialogue_ids = sorted(list(map(str, list(set(level_data[column])))))
if 'nan' in dialogue_ids:
    dialogue_ids.remove('nan')

# This is the shape of the predictions and labels after the folds (44595, 3)

    # (15219, 7)
    # print(preprocessed.data.shape)

    # 45539
    # print(preprocessed.data.shape[0] * 3 - preprocessed.number_of_dialogues)

possible_predictions = None

            # # Get possible bigrams in case of bigram model.
            # if n == 2:
            #     possible_predictions = model[[test_dialogue_inputs[i]]].items()
            #
            # # Get possible trigrams in case of trigram model.
            # elif n == 3:
            #     possible_predictions = model[[test_dialogue_inputs[i-1], test_dialogue_inputs[i]]].items()
            #
            # # In case another model is inputted.
            # else:
            #     print("Can only work with bigram and trigram models.")
            #     exit(1)

            # # Takes the n-gram with the input at the base and the highest count as the prediction of the input.
            # best_count = 0
            # best_prediction = ""
            # for prediction, count in possible_predictions:
            #     if count >= best_count:
            #         best_count = count
            #         best_prediction = prediction

# Performs cross validation on different subsets of classes as input parameters.
    for subsection in range(len(input_classes)):
        classes = input_classes[:subsection + 1]
        input_short = '_'.join([c[0] for c in classes])
        print("Cross-validation for input {}".format(classes))


# Initialises the DataFrame that will be used for the Seaborn graphs.
data_frame = pd.read_csv('analyses/dialogue_act_distribution.csv', index_col=[0], header=None)
data_frame = pd.DataFrame(index=data_frame.index)

# Gets the f1-scores over all levels for every sequence length the model was run on.
for sequence_length in sequence_lengths:
    filename = 'analyses/weighted_model_with_txt_sequence_length_' + str(sequence_length) + '_accuracy.csv'
    accuracies = pd.read_csv(filename, index_col=[0], header=[0, 1])
    data_frame = data_frame.merge(accuracies['all_levels']['f1'], how='left', left_index=True, right_index=True)
    data_frame = data_frame.rename(columns={"f1": "sequence length " + str(sequence_length)})

# Gets the f1-scores over all levels for every baseline model.
for baseline in baselines:
    accuracies = pd.read_csv('analyses/model_' + baseline + '_baseline_accuracy.csv', index_col=[0], header=[0, 1])
    data_frame = data_frame.merge(accuracies['all_levels']['f1'], how='left', left_index=True, right_index=True)
    data_frame = data_frame.rename(columns={"f1": baseline})
    data_frame = data_frame.set_index("Dialogue Act")