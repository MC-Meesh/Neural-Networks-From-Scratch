import numpy as np
from layer import Layer

class Dense(Layer):
    def __init__(self, input_size, output_size) -> None:
        '''
        input_size: number of specified rows for input
        output_size: number of specified neurons in output from layer

        weights: A 2d matrix of shape (m, n) = (output_size, input_size)
        bias: A 1d matrix of shape (m, 1) introduced to model to offset acitvations and break symmetry
        '''
        
        self.weights = np.random.randn(output_size, input_size)
        self.bias = np.random.randn(output_size, 1)
    
    def forward(self, input):
        '''
        Forward propogation, taking in some tensor X as input
        Return the dot product between input and Weights + biases (Y = W•X+b)
        '''
        self.input = input
        return  np.dot(self.weights, self.input) + self.bias
    
    def backward(self, gradient, learning_rate):
        #Backprop needs to be added once we have cost functions
        pass