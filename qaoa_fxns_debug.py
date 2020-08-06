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
np.random.seed(123)
random.seed(123)

__all__ = ['maxcut_qaoa_TFQ_model', 'QAOA_circuit_vis', 'create_graphs']

def maxcut_qaoa_TFQ_model(graph, depth_tot=1):
    """Takes in a unweighted, n-node regular or irregular MaxCut graph, maps it to a QAOA circuit (assuming full hardware connectivity), 
    and creates TFQ model for it"""

    # Map nodes to qubits 1:1, crearte variables for circuit & total params
    cirq_qubits = cirq.GridQubit.rect(1, len(graph.nodes))
    qaoa_params = []

    # Input will be initial superposition (hadamard transform)
    hadamard_circuit = cirq.Circuit()
    for q in graph.nodes():
        qubit = cirq_qubits[q]  # mapping each node in MaxCut graph to qubits
        hadamard_circuit.append(cirq.H.on(qubit)) 
    #model_input = tfq.convert_to_tensor([hadamard_transform])
    
    # Create a parameter set
    total_ops = []
    for depth in range(depth_tot):
        qaoa_params.append(sympy.Symbol("gamma_{}".format(depth)))
    
        """ Create {H_C, H_M} alternating depth_tot times """
        # Cost Ham - coupled with gamma
        cost_ham = graph.number_of_edges()/2 
        for edge in graph.edges():
            cost_ham += cirq.PauliString(0.5*(cirq.Z(cirq_qubits[edge[0]])*cirq.Z(cirq_qubits[edge[1]])))
        
        qaoa_params.append(sympy.Symbol("zbeta_{}".format(depth)))

        # Mixing ham - coupled with zbeta
        mixing_ham = 0
        for node in graph.nodes():
            mixing_ham += cirq.PauliString(cirq.X(cirq_qubits[node]))
        
        total_ops.append(cost_ham)
        total_ops.append(mixing_ham)

    # Generating circuit unitaries
    qaoa_circuit = tfq.util.exponential(
    operators=total_ops,
    coefficients= qaoa_params)

    """ML side: Use qaoa_circuit, cost_op, hadamard_circuit for model to create the TFQ model"""

    input_hadamard = tfq.convert_to_tensor([hadamard_circuit])
    # Construct model layers/architecture
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Input(shape=(), dtype=tf.dtypes.string))
    model.add(tfq.layers.PQC(qaoa_circuit, cost_ham))
    
    # Provide elements needed to train/fit model
    return [model, input_hadamard, qaoa_params, qaoa_circuit, hadamard_circuit]

    # TODO: FIX p > 1 ISSUE -> doesn't yield correct bitstr/result for trivial maxcut graphs
    # TODO: Look at git .py script/thread and understand how cost op mins energy AND how RI params get optimized

def QAOA_circuit_vis(graph, depth_tot, ZZ=False):
    """Provides the QAOA Circuit ansatz of any depth (assuming full connectivity)
    for a MaxCut instance and the MaxCut graph itself, purely for visualization"""
    
    # Map nodes to qubits 1:1, crearte variables for circuit & total params
    qubits = cirq.GridQubit.rect(1, len(graph.nodes))
    qaoa_circuit = cirq.Circuit()
    qaoa_params = []

    # Initial Hadamard transform 
    for qubit in qubits:
        qaoa_circuit += cirq.H(qubit)
    
    # Create a parameter set
    for depth in range(depth_tot):
        qaoa_params.append(sympy.Symbol("gamma_{}".format(depth)))
    
        if ZZ:

            for edge in graph.edges():
                qaoa_circuit += cirq.ZZ(qubits[edge[0]], qubits[edge[1]])**qaoa_params[-1]

            qaoa_params.append(sympy.Symbol("beta_{}".format(depth)))
            cost_ham = cirq.Moment([])
            for node in graph.nodes():
                cost_ham += (cirq.X(qubits[node])**qaoa_params[-1])
            qaoa_circuit.append(cost_ham)
        else:
            # Create {H_C, H_M} alternating depth_tot times 
            for edge in graph.edges():
                qaoa_circuit += cirq.CNOT(qubits[edge[0]], qubits[edge[1]])
                qaoa_circuit += cirq.rz(1 * qaoa_params[-1])(qubits[edge[1]])
                qaoa_circuit += cirq.CNOT(qubits[edge[0]], qubits[edge[1]])

            qaoa_params.append(sympy.Symbol("beta_{}".format(depth)))
            cost_ham = cirq.Moment([])
            for node in graph.nodes():
                cost_ham += cirq.rx(2 * qaoa_params[-1])(qubits[node])
            qaoa_circuit.append(cost_ham)

    #Draw physical graph w/ nodes relabelled to agree with QAOA circuit 
    sorted(graph)
    mapping = {}
    for node in sorted(graph.nodes()):
        mapping[node] = '(0,' + str(node) + ')'
    graph_new = nx.relabel_nodes(graph, mapping)
    plt.figure()
    plt.title('{}-node, MaxCut Graph'.format(len(graph.nodes())))
    graph_plot = nx.draw_networkx(graph_new, node_size=800)
    print(graph_plot)

    return SVGCircuit(qaoa_circuit)

def create_graphs(node_num, deg_num, depth = []):
    """Creates an arbitrary amount (graph_num) of a desired n-node udR MaxCut graph of different depths,
    can be used w/ loop to create graph batches"""
    dataset = []
    for i in depth:
        graph_rand_model_components = []

        graph_rand = nx.random_regular_graph(n=node_num, d=deg_num)
        graph_rand_model_components = (maxcut_qaoa_TFQ_model(graph_rand, i))  # nested list
        dataset.append(graph_rand_model_components)

    return dataset, graph_rand 

def main():
    # TODO: Color nodes to distinguish/visualize cut sets (bitstrings) with most counts
    graph_test = nx.random_regular_graph(n=4, d=2)
    model_components = maxcut_qaoa_TFQ_model(graph_test, 1)

    #Minimize the energy
    optimum = np.array([0])

    #Fit model
    model_components[0].compile(
            loss=tf.keras.losses.mean_absolute_error,
            optimizer=tf.keras.optimizers.Adam())

    history = model_components[0].fit(
            model_components[1],
            optimum,
            epochs=900,
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
    plt.title("{} node MaxCut Graph Cut set for 0011".format(len(graph_test.nodes())))
    plt.subplot(1, 3, 3)
    nx.draw_networkx(graph_test, node_color=['blue','red','blue', 'red'])
    plt.title("{} node MaxCut Graph Cut set for 1100".format(len(graph_test.nodes())))
    plt.show()

    
if __name__ == "__main__":
    main()
    