import numpy as np

from parts.EmbedLayer import EmbedLayer
from parts.LinearLayer import LinearLayer
from parts.softmax import softmax
from parts import ReLU


def d_LCE(y, y_pred):
    return y_pred - y


def identity(x):
    return x


class FFNN:

    def __init__(self, len_vocab, embedding_size, learning_rate=0.01, words_to_view=3):
        self.words_to_view = words_to_view
        self.embedding_size = embedding_size
        self.vocab_size = len_vocab
        self.learning_rate = learning_rate

        self.embedding_layer = EmbedLayer(len_vocab, embedding_size)
        self.hidden_layer = LinearLayer(words_to_view * embedding_size, embedding_size)
        self.output_layer = LinearLayer(embedding_size, len_vocab)

        self.z = {}
        self.a = {}
        self.df = {'0': identity, '1': ReLU.backwards, '2': identity}

    def forward(self, x):
        self.a['-1'] = x
        e = np.array([self.embedding_layer.forward(x[i, :]) for i in range(self.words_to_view)])
        self.z['0'] = np.concatenate(e, axis=0)
        self.a['0'] = self.z['0']
        self.z['1'] = self.hidden_layer.forward(self.a['0'])
        self.a['1'] = ReLU.forward(self.z['1'])
        self.z['2'] = self.output_layer.forward(self.a['1'])
        self.a['2'] = softmax(self.z['2']).reshape(1, self.vocab_size)
        return self.a['2']

    def backprop(self, y_true, y_pred):
        dl_da2 = d_LCE(y_true, y_pred) * self.a['2']
        dl_dz2 = np.matmul(dl_da2.T, self.a['1'].reshape(1, -1))
        s = np.sum(np.matmul(dl_da2, self.output_layer.weights), axis=1,  keepdims=True)
        df_z1 = self.df['1'](self.z['1'])
        dl_da1 = np.matmul(s.reshape(-1, 1), df_z1.reshape(1, -1))
        dl_dz1 = np.matmul(dl_da1.reshape(-1, 1), self.a['0'].reshape(1, -1))
        s = np.sum(np.matmul(dl_da1, self.hidden_layer.weights), axis=1,  keepdims=True)
        df_z0 = self.df['0'](self.z['0'])
        delta = np.matmul(s.reshape(-1, 1), df_z0.reshape(1, -1))
        q = int((delta.shape[1] / self.words_to_view))
        dl_da0 = np.array([delta[:, i: i+q] for i in range(self.words_to_view)]).reshape(self.words_to_view, self.embedding_size)
        dl_dz0 = np.array([np.matmul(dl_da0[i].reshape(-1, 1), self.a['-1'][i].reshape(1, -1)) for i in range(self.words_to_view)])

        # Update weights, biases based on derivatives
        self.output_layer.weights -= self.learning_rate * dl_dz2
        self.hidden_layer.weights -= self.learning_rate * dl_dz1
        for i in range(self.words_to_view):
            self.embedding_layer.weights -= self.learning_rate * dl_dz0[i]
        self.output_layer.bias -= self.learning_rate * dl_da2.flatten()
        self.hidden_layer.bias -= self.learning_rate * dl_da1.flatten()



