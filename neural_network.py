# -*- coding: utf-8 -*-
"""
Created on Tue Mar  7 07:04:18 2017

@author: rpicatos
"""
#%%
import numpy as np

# We've built out the structure and the backwards pass. 
# You'll implement the forward pass through the network. 
# You'll also set the hyperparameters: the learning rate, the number of hidden units, and the number of training passes.
# The network has two layers, a hidden layer and an output layer. 
# The hidden layer will use the sigmoid function for activations. 
# The output layer has only one node and is used for the regression, the output
# of the node is the same as the input of the node. That is, the activation 
# function is  f(x)=xf(x)=x . 

#Below, you have these tasks:
#Implement the sigmoid function to use as the activation function. 
# Set self.activation_function in __init__ to your sigmoid function.
#Implement the forward pass in the train method.
#Implement the backpropagation algorithm in the train method, including calculating the output error.
#Implement the forward pass in the run method
class NeuralNetwork(object):
    def __init__(self, input_nodes, hidden_nodes, output_nodes, learning_rate):
        # Set number of nodes in input, hidden and output layers.
        self.input_nodes = input_nodes
        self.hidden_nodes = hidden_nodes
        self.output_nodes = output_nodes

        # Initialize weights
        self.weights_i_h = \
                    np.random.normal( 0.0, self.hidden_nodes**-0.5, 
                                      (self.hidden_nodes, self.input_nodes))

        self.weights_h_o = \
                    np.random.normal( 0.0, self.output_nodes**-0.5, 
                                      (self.output_nodes, self.hidden_nodes))
        
        self.lr = learning_rate
        
        self.activation_function = lambda x : 1 / (1 + np.exp(-x))  
                    
    
    def train(self, inputs_list, targets_list):
        # Convert inputs list to 2d array
        # inputs_list size: 56, inputs size: 56x1 
        inputs = np.array(inputs_list, ndmin=2).T
        # targets size: 1x1 
        targets = np.array(targets_list, ndmin=2).T
        
        #### Forward pass here ####
        final_outputs, final_inputs, hidden_outputs, hidden_inputs = \
            self.run(inputs_list)
 
        ### Backward pass ###
        # Output layer
        # e = (y - y_hat)
        output_error = targets - final_outputs
        # delta_e = e * f_prime(h)
        output_error_term = output_error * 1
        # delta_W = delta_e * input_for_the_layer
        output_grad = np.dot(output_error_term, hidden_outputs.T)
        
        # Hidden layer
        hidden_errors = np.dot(self.weights_h_o.T, output_error_term)
        hidden_error_term = hidden_errors * hidden_outputs * (1-hidden_outputs)
        hidden_grad = np.dot(hidden_error_term, inputs.T)
        
        # update hidden-to-output weights with gradient descent step
        self.weights_h_o += self.lr * output_grad
        # update input-to-hidden weights with gradient descent step
        self.weights_i_h += self.lr * hidden_grad 
 
        
    def run(self, inputs_list):
        # Run a forward pass through the network
        # inputs_list size: 56, inputs size: 56x1 
        inputs = np.array(inputs_list, ndmin=2).T
#        print("inputs " + str(inputs))
#        print("test_w_i_h " + str(self.weights_i_h))
        
        #### Implement the forward pass here ####
        hidden_inputs = np.dot(self.weights_i_h, inputs) 
#        print("hidden_inputs " + str(hidden_inputs))
        hidden_outputs = self.activation_function( hidden_inputs ) 
#        print("hidden_outputs " + str(hidden_outputs))
        
        # signals into final output layer
        final_inputs = np.dot(self.weights_h_o, hidden_outputs)
#        print("final_inputs " + str(final_inputs))
        # signals from final output layer with f(x) = x
        final_outputs = final_inputs 
#        print("final_outputs " + str(final_outputs), "\n")
        
        return final_outputs, final_inputs, hidden_outputs, hidden_inputs
    
def MSE(y, Y):
    return np.mean((y-Y)**2)
