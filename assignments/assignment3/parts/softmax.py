import numpy as np


def forward(x):
    max_val = np.max(x, axis=0, keepdims=True)
    e_x = np.exp(x - max_val)
    sum_val = np.sum(e_x, axis=0, keepdims=True)
    f_x = e_x / sum_val
    return f_x
