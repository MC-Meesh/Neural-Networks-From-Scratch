class Layer:
    def __init__(self) -> None:
        self.input = None
        self.output = None

    def forward(self, input):
        '''
        Forward propogation: Takes in a tensor input and 
        returns some transformation as output
        '''
        pass

    def backward(self, gradient, learning_rate):
        '''
        Backward propogation: Takes in the gradient of error 
        with respect to output (gradient) then - (1) updates 
        params (2) computes derivative of error with respect 
        to input to pass to previous layer
        '''
        pass