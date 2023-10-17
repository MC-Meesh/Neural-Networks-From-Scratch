import numpy as np

def mse(y, y_hat):
     return np.mean(np.power(y - y_hat, 2))

def mse_prime(y, y_hat):
     return 2 * (y- y_hat) / np.size(y)