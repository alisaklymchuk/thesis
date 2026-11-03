import numpy as np

data = np.load("experiments/ipla/data/synthetic_logistic_regression.npz")

for name in data.files:
    print(name, data[name].shape, data[name].dtype)