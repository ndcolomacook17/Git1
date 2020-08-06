import tensorflow as tf
import tensorflow_quantum as tfq
import cirq
import sympy
import numpy as np
import random 

# visualization tools
import matplotlib.pyplot as plt
from cirq.contrib.svg import SVGCircuit

# supress warning for matplotlib, TFQ
import warnings
warnings.filterwarnings("ignore")
import networkx as nx
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import qaoa_fxns_debug as qaoa
np.random.seed(123)
random.seed(123)

# Create graph 
graph_test = nx.random_regular_graph(n=4, d=2)

# Use module import
model_components = qaoa.maxcut_qaoa_TFQ_model(graph_test, 2)

#Minimize the energy
optimum = np.array([0])

#Fit model
model_components[0].compile(
        loss=tf.keras.losses.mean_absolute_error,
        optimizer=tf.keras.optimizers.Adam())

# Create an early stopping mechanism
callback_strat = tf.keras.callbacks.EarlyStopping(
    monitor='loss', min_delta=0.00001, patience=0, verbose=1)

history = model_components[0].fit(
        model_components[1],
        optimum,
        epochs=500,
        callbacks= [callback_strat],
        verbose=1)

# View performance
plt.plot(history.history['loss'])
plt.title("QAOA Parameter Optimization with TFQ")
plt.xlabel("Epoch")
plt.ylabel("Mean Squared Error")

opt_params = model_components[0].trainable_variables
sample_layer = tfq.layers.Sample()
#output_bitstrings = sample_layer(output_circuit_tensor, symbol_names= model_components[2], symbol_values= opt_params, repetitions=1000)
output_bitstrings = sample_layer(model_components[4] + model_components[3], 
    symbol_names= model_components[2], 
    symbol_values= opt_params, 
    repetitions=5000)

# Converting the output_bitstrings from ragged tensor -> tensor -> np 2D array 
bitstrings = output_bitstrings[0].to_tensor()
bitstrings = np.array(bitstrings.numpy())

# Joining the bits for the 4 qubits, list now has str elements
new_bitstrings = []
for item in bitstrings:
    joined_bitstring = ''.join(map(str, item))
    new_bitstrings.append(joined_bitstring)

new_int_list = []
for element in new_bitstrings:
    bit_binary = int(element, 2)
    new_int_list.append(bit_binary)

#for node in graph_test.nodes():
#    if node == 0 | 2:
        
# Could technically just use this for histogram
print("Converted binary strings to ints: ", new_int_list[:5])

# Plotting results 
#for _ in range(2):
xticks = range(0, max(new_int_list)+1)
xtick_labels = list(map(lambda x: format(x, "0{}b".format(len(graph_test.nodes()))), xticks))
bins = np.arange(0, max(new_int_list)+2) - 0.5

plt.subplots(nrows=1, ncols=3, figsize=(15, 6))
plt.subplot(1, 3, 1)
plt.title("{} node Maxcut: p={}".format(len(graph_test.nodes()), int(len(model_components[2]) / 2)))
plt.xlabel("bitstrings")
plt.ylabel("counts/frequency")
plt.xticks(xticks, xtick_labels, rotation="vertical")
plt.hist(new_int_list, bins=bins, rwidth=0.5)
plt.subplot(1, 3, 2)
nx.draw_networkx(graph_test, node_color=['red','blue','red', 'blue'])
plt.title("{} node MaxCut Graph Cut set for 0110".format(len(graph_test.nodes())))
plt.subplot(1, 3, 3)
nx.draw_networkx(graph_test, node_color=['blue','red','blue', 'red'])
plt.title("{} node MaxCut Graph Cut set for 1001".format(len(graph_test.nodes())))
plt.show()