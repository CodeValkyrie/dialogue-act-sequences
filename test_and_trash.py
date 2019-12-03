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