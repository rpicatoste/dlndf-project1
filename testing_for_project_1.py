# -*- coding: utf-8 -*-
"""
Created on Thu Mar  9 08:35:20 2017

@author: rpicatos
"""
#%%
inputs = [0.5, -0.2, 0.1]
targets = [0.4]
test_w_i_h = np.array([[0.1, 0.4, -0.3], 
                       [-0.2, 0.5, 0.2]])
test_w_h_o = np.array([[0.3, -0.1]])


network = NeuralNetwork(3, 2, 1, 0.5)
network.weights_input_to_hidden = test_w_i_h.copy()
network.weights_hidden_to_output = test_w_h_o.copy()

print(network.run(inputs)[0])

print("inputs " + str(inputs))
a = np.dot(test_w_i_h, inputs)
print("hidden_inputs " + str(a))

b = 1 / (1 + np.exp(-a))
print("hidden_outputs " + str(b))

c = np.dot(test_w_h_o, b)
print("final_inputs " + str(c))


EL PROBLEMA ES QUE EN EL UNIT TESTING LE PASATMOS EL INPUT DE NX1 Y EN EL TRAINING 
SE LO PARAMOS COMO 1XN, PARA QUE ESE 1 PUEDAN SER LOS SAMPLES, SI SE HICIERA 
MATRICIALMENTE