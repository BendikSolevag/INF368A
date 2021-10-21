import numpy as np


class LinearLayer:

    def __init__(self, input_size, output_size, random_seed=27):
        rand = np.random.default_rng(seed=random_seed)
        self.weights = rand.random((output_size, input_size))
        self.bias = rand.random(output_size)
        self.input_size = input_size
        self.output_size = output_size

    def forward(self, x):
        wx = np.matmul(self.weights, x.T)
        wx_b = wx + self.bias
        return wx_b

    def get_weights(self):
        return self.weights

    def set_weights(self, w):
        self.weights = w

    def get_bias(self):
        return self.bias

    def set_bias(self, b):
        self.bias = b

