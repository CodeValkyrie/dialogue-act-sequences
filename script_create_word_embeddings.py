import bcolz
import numpy as np
import pickle
import pandas as pd
import json

''' This file creates the matrix that is needed to convert the words in the data set to word embedding vectors. The 
    word embedder used is GloVe.
'''

glove_path = "glove_6B"

words = []
idx = 0
word2idx = {}
vectors = bcolz.carray(np.zeros(1), rootdir=f'{glove_path}/6B.50.dat', mode='w')

with open(f'{glove_path}/glove.6B.50d.txt', 'rb') as f:
    for l in f:
        line = l.decode().split()
        word = line[0]
        words.append(word)
        word2idx[word] = idx
        idx += 1
        vect = np.array(line[1:]).astype(np.float)
        vectors.append(vect)
    
vectors = bcolz.carray(vectors[1:].reshape((400001, 50)), rootdir=f'{glove_path}/6B.50.dat', mode='w')
vectors.flush()
pickle.dump(words, open(f'{glove_path}/6B.50_words.pkl', 'wb'))
pickle.dump(word2idx, open(f'{glove_path}/6B.50_idx.pkl', 'wb'))

vectors = bcolz.open(f'{glove_path}/6B.50.dat')[:]
words = pickle.load(open(f'{glove_path}/6B.50_words.pkl', 'rb'))
word2idx = pickle.load(open(f'{glove_path}/6B.50_idx.pkl', 'rb'))

glove = {w: vectors[word2idx[w]] for w in words}
print(glove['the'])

df = pd.read_csv("DA_labeled_belc_2019.csv")
t = [str(s).split() for s in df['text'].values.tolist()]

words = {}
for sentence in t:
    for word in sentence:
        words[word] = word

vocabulary = []
for key, value in words.items():
    vocabulary.append(value)

print("done with vocab creation")

target_vocab = vocabulary

matrix_len = len(target_vocab)
emb_dim = 50
weights_matrix = np.zeros((matrix_len, emb_dim))
words_found = 0
word_map = {}
for i, word in enumerate(target_vocab):
    try:
        word_map[word] = i
        weights_matrix[i] = glove[word]
        words_found += 1
    except KeyError:
        weights_matrix[i] = np.random.normal(scale=0.6, size=(emb_dim, ))

with open('word_vector_mapping.json','w') as f:
    json.dump(word_map, f)
        
np.save("weights_matrix", weights_matrix)
print("Done with weights_matrix")
print(weights_matrix.shape)


def create_emb_layer(weights_matrix, non_trainable=False):
    num_embeddings, embedding_dim = weights_matrix.shape()
    emb_layer = nn.Embedding(num_embeddings, embedding_dim)
    emb_layer.load_state_dict({'weight': weights_matrix})
    if non_trainable:
        emb_layer.weight.requires_grad = False

    return emb_layer, num_embeddings, embedding_dim

class ToyNN(nn.Module):
    def __init__(self, weights_matrix, hidden_size, num_layers):
        super(self).__init__()
        self.embedding, num_embeddings, embedding_dim = create_emb_layer(weights_matrix, True)
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.gru = nn.GRU(embedding_dim, hidden_size, num_layers, batch_first=True)

    def forward(self, inp, hidden):
        return self.gru(self.embedding(inp), hidden)

    def init_hidden(self, batch_size):
        return Variable(torch.zeros(self.num_layers, batch_size, self.hidden_size))