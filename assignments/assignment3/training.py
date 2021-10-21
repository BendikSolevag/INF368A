import numpy as np
from model import FFNN
import re
from tqdm import tqdm

with open("gatsby.txt", "r", encoding='utf-8') as file:
    book = file.read()

gatsby_clean = book.replace('\n', ' ').replace('\t', ' ').replace('\r', ' ').replace(';', '').replace(':', '').replace('  ', ' ').replace('.', '').replace(',', '').replace('"', '').lower()
gatsby_clean = re.sub(r'[^\w\s]', '', gatsby_clean).split()


# We create a translation table, turning our vocabulary into one hot vectors.
vocab = list(set(gatsby_clean))
vocab.sort()
vocab_dict = {}
for i, word in enumerate(vocab):
    identity_vector = np.zeros(len(vocab))
    identity_vector[i] = 1
    vocab_dict[word] = identity_vector


reverse_dict = {}
for key in vocab_dict:
    reverse_dict[np.argmax(vocab_dict[key])] = key

words_to_view = 3
len_vocab = len(vocab)
len_dataset = len(gatsby_clean) - words_to_view
print("dataset length" + str(len_dataset))
model = FFNN(len_vocab, 64, learning_rate=0.001, words_to_view=words_to_view)


for epoch in range(1):
    # Initialise vectors to pass to our model
    input_vector = np.zeros((len(gatsby_clean) - words_to_view, words_to_view, len(vocab_dict)), dtype='float16')
    output_vector = np.zeros((len(gatsby_clean) - words_to_view, len(vocab_dict)), dtype='int16')
    for i in tqdm(range(len(gatsby_clean) - words_to_view)):
        # Iterate over book, collect words_to_view length vectors (amount of words to pass to our model)
        x = np.zeros((words_to_view, len(vocab_dict)))
        for j in range(words_to_view):
            x[j] = vocab_dict[gatsby_clean[i+j]]
        input_vector[i] = x
        # Connect true output to our input vector
        output_vector[i] = vocab_dict[gatsby_clean[i+words_to_view]]

    # Pass our data to our model, update weights as we go
    pbar = tqdm(input_vector.shape[0])
    correct_count = 0
    i = 0
    for x, y in zip(input_vector, output_vector):
        y_pred = model.forward(x)
        model.backprop(y, y_pred)
        pbar.update()
    pbar.close()


embed_weights = model.embedding_layer.weights
hidden_weights = model.hidden_layer.weights
output_weights = model.output_layer.weights

np.savetxt('gatsbyembed.tsv', embed_weights, delimiter='\t')
np.savetxt('gatsbyembed.csv', embed_weights, delimiter=',')
np.savetxt('gatsbyhidden.csv', hidden_weights, delimiter=',')
np.savetxt('gatsbyoutput.csv', output_weights, delimiter=',')




# Predict next words
input_vector = np.zeros((10, words_to_view, len(vocab_dict)), dtype='float16')
for i in tqdm(range(10)):
    # Iterate over book, collect words_to_view length vectors (amount of words to pass to our model)
    x = np.zeros((words_to_view, len(vocab_dict)))
    for j in range(words_to_view):
        x[j] = vocab_dict[gatsby_clean[i+j]]
    input_vector[i] = x

for x in output_vector:
    y_pred = model.forward(x)
    print(reverse_dict[y_pred])