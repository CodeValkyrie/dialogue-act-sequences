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