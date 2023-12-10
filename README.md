# [Read the Paper Here (PDF)](Neural_Networks_from_Scratch.pdf)

# Neural Networks From Scratch

This project is an in-depth exploration of Deep Neural Networks (DNNs). 
By clearly defining the components of a DNN, and how they function, a much deeper understanding of these models can be cultivated. 
The full paper dives into the implementation in great detail, covering everything from definitions to proofs.
I hope this project can serve as a valuable reference for those looking to understand DNNs in a technical manner.
This project was inspired by [Omar Aflak's Implementation](https://github.com/TheIndependentCode/Neural-Network), and much of the functional code/structure was based on their repository.

## Getting Started
### Paper Review
If you wish to learn more about DNNs, start by reading the included paper to gain a better understanding.

### Implementation
Implement your own models by creating an array of layers,  note the final layer shape should be the number of features you wish to output. In this case, only 1 neuron output indicates a binary classifier.
```python
network = [
    Dense(2, 3),
    tanh(),
    Dense(3, 1),
    tanh()
]
```

Train the model using the following function, defined in the _network.py_ file:
```python
train(network, mse, mse_prime, X, Y, epochs=10000, learning_rate=0.1)
```
