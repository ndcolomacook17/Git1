# import necessary libraries
import tensorflow as tf
import tensorflow_quantum as tfq
import cirq
import sympy
import numpy as np

# visualization tools
import matplotlib.pyplot as plt
from cirq.contrib.svg import SVGCircuit

# supress warning for matplotlib, TFQ
import warnings
warnings.filterwarnings("ignore")
import networkx as nx
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# import new module
import QAOA_Project_functions_implementations as qaoa

# Create the graph
graph_1 = nx.random_regular_graph(n=5, d=2)

# Create the TFQ model components for graph_1 at p=1 depth
model, model_input, graph_1_params, qaoa_circuit = qaoa.maxcut_qaoa_TFQ_model(graph_1, 2)
print(model.summary(), model_input, graph_1_params)

# Define optimum for loss
optimum = [0]
optimum = np.array(optimum)

# Compile the model
model.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=0.001),
              loss=tf.keras.losses.mean_squared_error)

# Create an early stopping mechanism
callback_strat = tf.keras.callbacks.EarlyStopping(
    monitor='loss', min_delta=0.00001, patience=0, verbose=1)
#print(callback_strat)

# Train model
history = model.fit(model_input, optimum, epochs=100, callbacks=[callback_strat],
verbose=1)

# View performance
plt.plot(history.history['loss'])
plt.title("QAOA Parameter Optimization with TFQ")
plt.xlabel("Epoch")
plt.ylabel("Mean Squared Error")
plt.show()

# Use the model for prediction (model.predict())
# Read out the optimal parameters
opt_params = model.trainable_variables
print(opt_params)
# Create a circuit and sample bitstrings from the final state 1000 times, can create in Cirq, TFQ
add = tfq.layers.AddCircuit() 
output_circuit_tensor = add(model_input, append=qaoa_circuit)
sample_layer = tfq.layers.Sample()
output_bitstrings = sample_layer(output_circuit_tensor,  symbol_names= graph_1_params, symbol_values= opt_params, repetitions=1000)

# Converting the output_bitstrings from ragged tensor -> tensor -> np 2D array 
bitstrings = output_bitstrings[0].to_tensor()
bitstrings = np.array(bitstrings.numpy())

# Joining the bits for the 4 qubits, list now has str elements
new_bitstrings = []
for item in bitstrings:
    joined_bitstring = ''.join(map(str, item))
    new_bitstrings.append(joined_bitstring)
print("\nBitstrings joined (1D array) but now as strings: ", new_bitstrings[:100], type(new_bitstrings[0]))

# Converting elements in list from str to int
integer_list = list(map(int, new_bitstrings[:5]))
print("\nBitstrings converted int using map method:", integer_list, type(integer_list))

new_int_list = []
for element in new_bitstrings:
    bit_binary = int(element, 2)
    new_int_list.append(bit_binary)

# Could technically just use this for histogram
print("Converted binary strings to ints: ", new_int_list[:5])
print(max(new_int_list))

# Plotting results 
xticks = range(0, max(new_int_list)+1)
xtick_labels = list(map(lambda x: format(x, "0{}b".format(len(graph_1.nodes()))), xticks))
bins = np.arange(0, max(new_int_list)+2) - 0.5

plt.title("{} node Maxcut: p={}".format(len(graph_1.nodes()), int(len(graph_1_params) / 2)))
plt.xlabel("bitstrings")
plt.ylabel("counts/frequency")
plt.xticks(xticks, xtick_labels, rotation="vertical")
plt.hist(new_int_list, bins=bins, rwidth=0.5)
plt.show()

# TODO: fix optimal bistrings (C(x), where x = {00..0, 11..1}), look at/compare different optimizers, create modules (maybe packages, subpackages) 
# 