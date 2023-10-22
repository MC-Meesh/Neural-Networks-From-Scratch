import numpy as np
from layer import Layer

class Dense(Layer):
    def __init__(self, x_size, y_size) -> None:
        '''
        input_size: number of specified rows for input
        output_size: number of specified neurons in output from layer

        weights: A 2d matrix of shape (m, n) = (output_size, input_size)
        bias: A vector of shape (m, 1) introduced to model to offset acitvations and break symmetry
        '''
        
        self.weights = np.random.randn(y_size, x_size)
        self.bias = np.random.randn(y_size, 1)
    
    def forward(self, input):
        '''
        Forward propogation, taking in some tensor X as input
        Return the dot product between input and Weights + biases (Y = WX+B)
        '''
        self.input = input
        return  np.dot(self.weights, self.input) + self.bias
    
    def backward(self, y_gradient, learning_rate):
        weight_gradient = np.dot(y_gradient, self.input.T)
        self.weights -= learning_rate * weight_gradient #update weights
        self.bias -= learning_rate * y_gradient
        return np.dot(self.weights.T, y_gradient) #Return dL/dx for next layer