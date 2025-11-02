import numpy as np 

# Step function 
def step_function(x): 
    """
    Step function is the activation function used in perceptrons
        - Its a binary classifier/decision maker 
        - Think of it as a decision switch that decides whether or not a neuron should be activate(1) or not(0)
    """
    return 1 if x >= 0 else 0 

class Perceptron:
    def __init__(self, learning_rate=0.1, epochs=10):
        self.lr = learning_rate
        self.epochs = epochs

    def fit(self, X, y):
        self.weights = np.zeros(X.shape[1]) # initialize weights to 0 
        self.bias = 0 # initialize bias = 0

        for _ in range(self.epochs): # Loop through the dataset multiple times to allow the perceptron refine its weights 
            for i in range(len(X)):
                # Compute linear function of inputs and weights 
                linear_output = np.dot(X[i], self.weights) + self.bias
                # Apply step function to the linear_output 
                y_pred = step_function(linear_output)

                # Calculate error 
                error = y[i] - y_pred

                # update weights and bias 
                self.weights += self.lr * error * X[i]
                self.bias += self.lr + error

