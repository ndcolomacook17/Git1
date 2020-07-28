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

__all__ = ['maxcut_qaoa_TFQ_model', 'QAOA_circuit_vis']

def maxcut_qaoa_TFQ_model(graph, depth_tot):
    """Takes in a udR, n-node MaxCut graph, maps it to a QAOA circuit (assuming full connectivity), 
    and creates TFQ model for it  """
        
    # Map nodes to qubits 1:1, crearte variables for circuit & total params
    qubits = cirq.GridQubit.rect(1, len(graph.nodes))
    qaoa_circuit = cirq.Circuit()
    qaoa_params = []
        
    # Create a parameter set
    for depth in range(depth_tot):
        qaoa_params.append(sympy.Symbol("gamma_{}".format(depth)))
        
        # Create {H_C, H_M} alternating depth_tot times 
        for edge in graph.edges():
            qaoa_circuit += cirq.ZZ(qubits[edge[0]], qubits[edge[1]])**qaoa_params[-1]
                
        qaoa_params.append(sympy.Symbol("beta_{}".format(depth)))
        for node in graph.nodes():
            qaoa_circuit += cirq.X(qubits[node])**qaoa_params[-1]
                
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

def QAOA_circuit_vis(graph, depth_tot):
    """Provides the QAOA Circuit ansatz of any depth (assuming full connectivity)
    for a MaxCut instance and the MaxCut graph itself, purely for visualization"""
    
    qubits = cirq.GridQubit.rect(1, len(graph.nodes))
    qaoa_circuit = cirq.Circuit()
    qaoa_params = []
    
    # Initial Hadamard transform 
    for qubit in qubits:
        qaoa_circuit += cirq.H(qubit)
    
    # Create a parameter set
    for depth in range(depth_tot):
        qaoa_params.append(sympy.Symbol("gamma_{}".format(depth)))
    
        # Create {H_C, H_M} alternating depth_tot times 
        for edge in graph.edges():
            qaoa_circuit += cirq.ZZ(qubits[edge[0]], qubits[edge[1]])**qaoa_params[-1]
            
        qaoa_params.append(sympy.Symbol("beta_{}".format(depth)))
        cost_ham = cirq.Moment([])
        for node in graph.nodes():
            cost_ham += (cirq.X(qubits[node])**qaoa_params[-1])
        qaoa_circuit.append(cost_ham)
        
    # Draw physical graph w/ nodes relabelled to agree with QAOA circuit 
    sorted(graph)
    mapping = {}
    for node in sorted(graph.nodes()):
        mapping[node] = '(0,' + str(node) + ')'
    graph_new = nx.relabel_nodes(graph, mapping)
    plt.figure()
    plt.title('{}-node, regular MaxCut Graph'.format(len(graph.nodes())))
    graph_plot = nx.draw_networkx(graph_new, node_size=800)
    print(graph_plot)
    
    return SVGCircuit(qaoa_circuit)


# TODO: 
# Use qaoa_circuit, cost_op, hadamard_circuit for model for this def
# Working through:
# Uses maxcut_qaoa_TFQ_model function but is only limited to creating several instance of TFQ models for depth p=1, 
## trying to generalize this for any depth p
# Extension: once depth p can be varied, this functions could potentially return TFQ models for a specific n-node 
## udR MaxCut graph instance SWEEPING p for a desired interval
#This can easily provide the side by side differences in parameter optimization paths as p varies

def create_graphs(node_num, deg_num, graph_num):
    """Creates an arbitrary amount (graph_num) of a desired n-node udR MaxCut graph, 
    can be used to create many models at once corresponding to these graph instances"""
    dataset = []
    
    for i in range(graph_num):
        graph_rand = nx.random_regular_graph(n=node_num, d=deg_num)
        
        graph_rand_model_components = []
        model_components = maxcut_qaoa_TFQ_model(graph_rand, 1)
        graph_rand_model_components.append(model_components)
        dataset.append(graph_rand_model_components)

    return dataset

def main():
    # Testing out the visualization function
    graph_b = nx.random_regular_graph(n=8, d=3)
    QAOA_circuit_vis(graph_b, 2)

    # Testing out multiple model fxn
    all_data = create_graphs(6, 3, 3)

    # Model returns the tfq model for 3 different graph instances 
    print("TFQ Model for graph 1:\n\n", all_data[0])
    print("\nTFQ Model for graph 2:\n\n", all_data[1])
    print("\nTFQ Model for graph 2:\n\n", all_data[2])

if __name__ == "__main__":
    main()
    