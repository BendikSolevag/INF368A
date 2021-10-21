import numpy as np


class EmbedLayer:
    def __init__(self, vocab_size, embedding_size, random_seed=27):
        rand = np.random.default_rng(seed=random_seed)
        self.weights = rand.random((embedding_size, vocab_size))

    def forward(self, x):
        return np.matmul(self.weights, x.T)

    def get_weights(self):
        return self.weights

    def set_weights(self, w):
        self.weights = w