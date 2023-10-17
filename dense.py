import numpy as np
from layer import Layer

class Dense(Layer):
    def __init__(self, x_size, y_size) -> None:
        '''
        input_size: number of specified rows for input
        output_size: number of specified neurons in output from layer

        weights: A 2d matrix of shape (m, n) = (input_size, ouptut_size)
        bias: A vector of shape (1, n) introduced to model to offset acitvations and break symmetry
        '''
        
        self.weights = np.random.randn(x_size, y_size)
        self.bias = np.random.randn(1, y_size)
    
    def forward(self, input):
        '''
        Forward propogation, taking in some tensor X as input
        Return the dot product between input and Weights + biases (Y = XW+B)
        '''
        self.input = input
        return  np.dot(self.input, self.weights) + self.bias
    
    def backward(self, y_gradient, learning_rate):
        weight_gradient = np.outer(self.input.T, y_gradient)
        self.weights -= learning_rate * weight_gradient #update weights
        self.bias -= learning_rate * y_gradient #update bias
        return np.dot(y_gradient, self.weights.T) #Return dL/dx for next layer