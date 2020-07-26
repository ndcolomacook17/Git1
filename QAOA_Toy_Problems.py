import tensorflow as tf
import tensorflow_quantum as tfq
import cirq
import sympy
import numpy as np

# visualization tools
import matplotlib.pyplot as plt
from cirq.contrib.svg import SVGCircuit

#supress warning for matplotlib, TFQ
import warnings
warnings.filterwarnings("ignore")
import networkx as nx
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

def maxcut_qaoa_TFQ_model(graph, depth_tot):
    """Takes in a unweighted, n-node regular or irregular MaxCut graph, maps it to a QAOA circuit (assuming full hardware connectivity), 
    and creates TFQ model for it"""
    
    # Map nodes to qubits 1:1, crearte variables for circuit & total params
    qubits = cirq.GridQubit.rect(1, len(graph.nodes))
    qaoa_circuit = cirq.Circuit()
    qaoa_params = []
    
    # Create a parameter set
    for depth in range(depth_tot):
        qaoa_params.append(sympy.Symbol("gamma_{}".format(depth)))
    
        # Create {H_C, H_M} alternating depth_tot times 
        for edge in graph.edges():
            qaoa_circuit += cirq.CNOT(qubits[edge[0]], qubits[edge[1]])
            qaoa_circuit += cirq.rz(1 * qaoa_params[-1])(qubits[edge[1]])
            qaoa_circuit += cirq.CNOT(qubits[edge[0]], qubits[edge[1]]) 
            
        qaoa_params.append(sympy.Symbol("beta_{}".format(depth)))
        for node in graph.nodes():
            qaoa_circuit += cirq.rx(2 * qaoa_params[-1])(qubits[node])
            
    # Define the H_c to pass into tfq.layers.PQC as measurement operator?
    cost_op = None
    for edge in graph.edges():
        if cost_op is None:
            cost_op = cirq.Z(qubits[edge[0]])*cirq.Z(qubits[edge[1]])
        else:
            cost_op += cirq.Z(qubits[edge[0]])*cirq.Z(qubits[edge[1]])
        
    """Use qaoa_circuit, cost_op, hadamard_circuit for model to create the TFQ model"""
    
    # Input will be initial superposition (hadamard transform)
    hadamard_transform = cirq.Circuit()
    for q in qubits:
        hadamard_transform += cirq.H(q)
    model_input = tfq.convert_to_tensor([hadamard_transform])

    
    # Construct model layers/architecture
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(), dtype=tf.dtypes.string),
        tfq.layers.PQC(qaoa_circuit, cost_op)])
    
    # Provide elements needed to train/fit model
    return model, model_input, qaoa_params, qaoa_circuit

# Create the graph
graph_1 = nx.random_regular_graph(n=4, d=2)

# Create the TFQ model components for graph_1 at p=1 depth
model, model_input, graph_1_params, qaoa_circuit = maxcut_qaoa_TFQ_model(graph_1, 2)
print(model.summary(), model_input, graph_1_params)

# Store the QAOA PQC for future sampling
#qaoa_circuit_new, vis_1 = QAOA_circuit_vis(graph_1, 1)

# Define optimum for loss
optimum = [0]
optimum = np.array(optimum)

# Compile the model
model.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=0.001),
              loss=tf.keras.losses.mean_squared_error)

# Create an early stopping mechanism
callback_strat = tf.keras.callbacks.EarlyStopping(
    monitor='loss', min_delta=0.0000001, patience=0, verbose=1)
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
out1 = output_bitstrings[0].to_tensor()
bitstrings_all = np.array(out1.numpy())
print("NumPy array of sampled output:\n", bitstrings_all[:5])

# Joining the bits for the 4 qubits, list now has str elements
new_bitstrings = []
for item in bitstrings_all:
    joined_bitstring = ''.join(map(str, item))
    new_bitstrings.append(joined_bitstring)

print("\nBitstrings joined (1D array) but now as strings: ", new_bitstrings[:5], type(new_bitstrings[0]))

# Converting elements in list from str to int
# Note: Issue is that you need to figure out how to keep ALL significant figures 
bit_list_2 = list(map(int, new_bitstrings[:5]))
print("\nBitstrings converted int using map method:", bit_list_2, type(bit_list_2))

new_int_list = []
for element in new_bitstrings:
    bit_binary = int(element, 2)
    new_int_list.append(bit_binary)

# Could technically just use this for histogram
print("Converted binary strings to ints: ", new_int_list[:5])

# Plotting results 
xticks = range(0, 16)
xtick_labels = list(map(lambda x: format(x, "04b"), xticks))
bins = np.arange(0, 17) - 0.5

plt.title("4 node, u2R Maxcut: p=1")
plt.xlabel("bitstrings")
plt.ylabel("freq.")
plt.xticks(xticks, xtick_labels, rotation="vertical")
plt.hist(new_int_list, bins=bins, rwidth=.5)
plt.margins(1, 0)
plt.show()