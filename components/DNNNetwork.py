from typing import List, Tuple
from components import default_dnn_max_chain_depth, default_dnn_chains, \
    default_dnn_input_node_connectivity, default_dnn_output_node_connectivity


class DynamicNeuralNetwork:

    def __init__(self, input_shapes: List[Tuple[int, int]], output_shapes: List[Tuple[int, int]],
                 num_chains: int = None, max_chain_depth: int = None, input_node_connectivity: float = None,
                 output_node_connectivity: float = None):
        self.num_inputs = len(input_shapes)
        self.num_outputs = len(output_shapes)
        self.num_chains = num_chains \
            if num_chains is not None \
            else default_dnn_chains
        self.chain_depth = max_chain_depth \
            if max_chain_depth is not None \
            else default_dnn_max_chain_depth
        self.input_node_connectivity = input_node_connectivity \
            if input_node_connectivity is not None \
            else default_dnn_input_node_connectivity
        self.output_node_connectivity = output_node_connectivity \
            if output_node_connectivity is not None \
            else default_dnn_output_node_connectivity
        self.__construct_network()

    def __construct_network(self):
        pass
