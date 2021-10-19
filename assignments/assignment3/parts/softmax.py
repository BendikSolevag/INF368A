import numpy as np


def forward(x):
    maxval = np.max(x, axis=0, keepdims=True)
    e_x = np.exp(x - maxval)
    sum_val = np.sum(e_x, axis=0, keepdims=True)
    f_x = e_x / sum_val
    return f_x

def backwards(x):
    pass
