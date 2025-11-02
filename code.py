import numpy as np 

# Step function 
def step_function(x): 
    """
    Step function is the activation function used in perceptrons
        - Its a binary classifier/decision maker 
        - Think of it as a decision switch that decides whether or not a neuron should be activate(1) or not(0)
    """
    return 1 if x >= 0 else 0 

