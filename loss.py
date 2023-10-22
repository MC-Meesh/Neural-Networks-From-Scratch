import numpy as np

def mse(y, y_hat):
     return np.mean(np.power(y_hat - y, 2))

def mse_prime(y, y_hat):
     return 2 * (y_hat - y) / np.size(y)