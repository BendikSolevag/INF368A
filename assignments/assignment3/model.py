
import numpy as np

from parts.EmbedLayer import EmbedLayer
from parts.LinearLayer import LinearLayer
from parts import ReLU, softmax


def d_LCE(y, y_pred):
    """
    derivative of cross entropy
    :param y: true y value
    :param y_pred: predicted y value
    :return: the derivative of the cross entropy loss
    """
    return y_pred - y


class MyFFLM:

    def __init__(self, len_vocab, embedding_size, learning_rate=0.01, words_to_view=3):
        """
        :param len_vocab:
        :param embedding_size:
        :param words_to_view:
        """
        self.embedding_layer = EmbedLayer(len_vocab, embedding_size)
        self.hidden_layer = LinearLayer(words_to_view * embedding_size, embedding_size)
        self.ReLu = ReLU.forward
        self.output_layer = LinearLayer(embedding_size, len_vocab)
        self.softmax = softmax.forward
        self.words_to_view = words_to_view
        self.embedding_size = embedding_size
        self.vocab_size = len_vocab
        self.learning_rate = learning_rate
        self.z = {}
        self.a = {}
        self.dL_dw = {str(i): None for i in range(3)}
        self.dL_db = {str(i): None for i in range(3)}
        self.df = {'0': lambda x: x, '1': ReLU.backwards, '2': lambda x: x}
        self.dL_dw['0'] = [None for _ in range(words_to_view)]


    def forward(self, x):
        self.a['-1'] = x
        e = np.array([self.embedding_layer.forward(x[i, :]) for i in range(self.words_to_view)])
        self.z['0'] = np.concatenate(e, axis=0)
        self.a['0'] = self.z['0']
        self.z['1'] = self.hidden_layer.forward(self.a['0'])
        self.a['1'] = self.ReLu(self.z['1'])
        self.z['2'] = self.output_layer.forward(self.a['1'])
        self.a['2'] = self.softmax(self.z['2']).reshape(1, self.vocab_size)
        return self.a['2']


    def backprop(self, y_true, y_pred):
        delta = d_LCE(y_true, y_pred) * self.a['2']
        dl_da2 = delta
        dl_dz2 = np.matmul(delta.T, self.a['1'].reshape(1, -1))
        tmp = np.matmul(delta, self.output_layer.weights)
        s = np.sum(tmp, axis=1,  keepdims=True)
        df_z1 = self.df['1'](self.z['1'])
        delta = np.matmul(s.reshape(-1, 1), df_z1.reshape(1, -1))
        dl_da1 = delta
        dl_dz1 = np.matmul(dl_da1.reshape(-1, 1), self.a['0'].reshape(1, -1))
        tmp = np.matmul(delta, self.hidden_layer.weights)
        s = np.sum(tmp, axis=1,  keepdims=True)
        df_z0 = self.df['0'](self.z['0'])
        delta = np.matmul(s.reshape(-1, 1), df_z0.reshape(1, -1))
        q = int((delta.shape[1] / self.words_to_view))
        d = np.array([delta[:, i: i+q] for i in range(self.words_to_view)]).reshape(self.words_to_view, self.embedding_size)
        dl_da0 = d
        dl_dz0 = np.array([np.matmul(dl_da0[i].reshape(-1, 1), self.a['-1'][i].reshape(1, -1)) for i in range(self.words_to_view)])

        self.dL_dw['2'] = dl_dz2
        self.dL_dw['1'] = dl_dz1
        for i in range(self.words_to_view):
            self.dL_dw['0'][i] = dl_dz0[i]
        self.dL_db['2'] = dl_da2
        self.dL_db['1'] = dl_da1
        self.dL_db['0'] = dl_da0

    def update(self):
        for i in range(self.words_to_view):
            self.embedding_layer.weights -= self.learning_rate*self.dL_dw['0'][i]
        self.hidden_layer.weights -= self.learning_rate*self.dL_dw['1']
        self.hidden_layer.bias -= self.learning_rate*self.dL_db['1'].flatten()
        self.output_layer.weights -= self.learning_rate * self.dL_dw['2']
        self.output_layer.bias -= self.learning_rate * self.dL_db['2'].flatten()