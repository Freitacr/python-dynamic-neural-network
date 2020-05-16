from random import randint
from typing import List, Tuple, Set

from numpy import array_equiv
from numpy import ndarray

from components import default_dnn_max_chain_depth, default_dnn_chains, \
    default_dnn_input_node_connectivity, default_dnn_output_node_connectivity, \
    dnn_shape_max_cols, dnn_shape_max_rows, dnn_shape_min_cols, dnn_shape_min_rows
from components.DNNInputNode import DNNInputNode
from components.DNNNode import DNNNode
from components.DNNOutputNode import DNNOutputNode


class DynamicNeuralNetwork:

    def __init__(self, input_shapes: List[Tuple[int, int]], output_shapes: List[Tuple[int, int]],
                 num_chains: int = None, max_chain_depth: int = None, input_node_connectivity: float = None,
                 output_node_connectivity: float = None):
        self.input_shapes = input_shapes
        self.output_shapes = output_shapes
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
        self.input_nodes: List['DNNInputNode'] = []
        self.output_nodes: List['DNNOutputNode'] = []
        self.active_nodes: List['DNNNode'] = []
        self.active_nodes_set: Set['DNNNode'] = set()
        self.__construct_network()

    @staticmethod
    def __generate_random_shape() -> Tuple[int, int]:
        n_rows = randint(dnn_shape_min_rows, dnn_shape_max_rows)
        n_cols = randint(dnn_shape_min_cols, dnn_shape_max_cols)
        return n_rows, n_cols

    def __form_chain(self) -> Tuple['DNNNode', 'DNNNode']:
        num_nodes = randint(1, default_dnn_max_chain_depth)
        starting_node = DNNNode(self.__generate_random_shape())
        current_node = starting_node
        for _ in range(num_nodes-1):
            next_node = DNNNode(self.__generate_random_shape())
            current_node.add_outgoing_connection(next_node)
            current_node = next_node
        return starting_node, current_node

    def __construct_network(self):
        # generate input nodes
        for in_shape in self.input_shapes:
            self.input_nodes.append(DNNInputNode(in_shape))

        # generate output nodes
        for out_shape in self.output_shapes:
            self.output_nodes.append(DNNOutputNode(out_shape))

        # generate chains
        starting_chains: List[Tuple['DNNNode', 'DNNNode']] = []
        for _ in range(self.num_chains):
            starting_chains.append(self.__form_chain())

        # generate initial connections from inputs to primary chain nodes
        potential_connections = self.num_inputs * len(starting_chains)
        connections_to_form = int(potential_connections * self.input_node_connectivity)
        self.__construct_non_excluding_connections(connections_to_form,
                                                   self.input_nodes,
                                                   [x[0] for x in starting_chains])

        # generate initial connections from ending chain nodes to outputs
        potential_connections = self.num_outputs * len(starting_chains)
        connections_to_form = int(potential_connections * self.output_node_connectivity)
        self.__construct_non_excluding_connections(connections_to_form,
                                                   [x[1] for x in starting_chains],
                                                   self.output_nodes)

    @staticmethod
    def __construct_non_excluding_connections(num_connections: int, node_set_a: List['DNNNode'],
                                              node_set_b: List['DNNNode']):

        def check_for_violations():
            violations = False
            for input_node in node_set_a:
                if len(input_node.outgoing_connections) == 0:
                    violations = True
            for output_node in node_set_b:
                if len(output_node.incoming_connections) == 0:
                    violations = True
            return violations

        # The goal here is construct connections such that:
        # Every node in node_set_a has a connection going out
        # Every node in node_set_b has a connection coming in
        # There are no more than num_connections connections formed
        # If the third condition cannot be satisfied, an error must be thrown.

        if num_connections < max(len(node_set_a), len(node_set_b)):
            raise ValueError("num_connections must be greater than or equal "
                             "to the number of nodes in, and the number of nodes out")
        # First form random connections
        for _ in range(num_connections):
            while True:
                curr_a_node = node_set_a[randint(0, len(node_set_a))-1]
                curr_b_node = node_set_b[randint(0, len(node_set_b))-1]
                success = curr_a_node.add_outgoing_connection(curr_b_node)
                if success:
                    break
        # Check to see if there are any violations to the first two conditions
        if not check_for_violations():
            return
        # Then fix any violations
        # Violations will temporarily be fixed in such a way that will guarantee that
        # there will be more than num_connections formed.
        # todo: Fix connection forming violations so there are no more than num_connections formed
        # Fix violations in input nodes
        for input_node in node_set_a:
            if len(input_node.outgoing_connections) == 0:
                rand_b_node = node_set_b[randint(0, len(node_set_b))-1]
                input_node.add_outgoing_connection(rand_b_node)
        for output_node in node_set_b:
            if len(output_node.incoming_connections) == 0:
                rand_a_node = node_set_a[randint(0, len(node_set_a))-1]
                rand_a_node.add_outgoing_connection(output_node)

    def add_input_data(self, input_data: List['ndarray']):
        if not len(input_data) == self.num_inputs:
            raise ValueError("Incorrect number of inputs supplied to network. Expected " +
                             str(self.num_inputs) + " but received " +
                             str(len(input_data))
                             )
        for i in range(self.num_inputs):
            in_data = input_data[i]
            in_node = self.input_nodes[i]
            if not array_equiv(in_data.shape, in_node.internal_shape):
                self.active_nodes.clear()
                raise ValueError("Incorrect shape in position " + str(i) +
                                 " expected (" + str(in_node.internal_shape[0]) +
                                 ", " + str(in_node.internal_shape[1]) + ") but " +
                                 "received (" + str(in_data.shape[0]) + ', ' +
                                 str(in_data.shape[1]) + ")"
                                 )
            in_node.add_input_data(in_data)
            self.active_nodes.append(in_node)
            self.active_nodes_set.add(in_node)

    def extract_output_data(self):
        ret_outputs: List['ndarray'] = []
        for out_node in self.output_nodes:
            ret_outputs.append(out_node.extract_output())
        return ret_outputs

    def propagate_inputs(self):
        while not len(self.active_nodes) == 0:
            curr_node = self.active_nodes.pop(0)
            self.active_nodes_set.remove(curr_node)
            curr_node.transmit_data()
            for connection in curr_node.outgoing_connections:
                if not isinstance(connection.node_out, DNNOutputNode):
                    if connection.node_out in self.active_nodes_set:
                        continue
                    self.active_nodes.append(connection.node_out)
                    self.active_nodes_set.add(connection.node_out)

    def perform_backpropagation(self, input_data: List['ndarray'], expected_outputs: List['ndarray']):
        self.add_input_data(input_data)
        self.propagate_inputs()
        outputs = self.extract_output_data()
        differences = [expected_outputs[i] - outputs[i] for i in range(len(expected_outputs))]
        self.clear_incoming_messages()

        for i in range(len(differences)):
            self.output_nodes[i].outgoing_buffer.contents = differences[i]
            self.output_nodes[i].receive_incoming_message(self.output_nodes[i].outgoing_buffer)
            self.output_nodes[i].outgoing_buffer = None
            self.active_nodes.append(self.output_nodes[i])
            self.active_nodes_set.add(self.active_nodes[-1])
        while not len(self.active_nodes) == 0:
            curr_node = self.active_nodes.pop(0)
            self.active_nodes_set.remove(curr_node)
            curr_node.transmit_error()
            for connection in curr_node.incoming_connections:
                if not isinstance(connection.node_in, DNNInputNode):
                    if connection.node_in in self.active_nodes_set:
                        continue
                    self.active_nodes.append(connection.node_in)
                    self.active_nodes_set.add(connection.node_in)
        self.clear_incoming_messages()
        self.update_weights()
        pass

    def clear_incoming_messages(self):
        for in_node in self.input_nodes:
            self.active_nodes.append(in_node)
            self.active_nodes_set.add(in_node)
        while len(self.active_nodes) > 0:
            curr_node = self.active_nodes.pop(0)
            self.active_nodes_set.remove(curr_node)
            curr_node.incoming_messages.clear()
            for connection in curr_node.outgoing_connections:
                if connection.node_out in self.active_nodes_set:
                    continue
                self.active_nodes.append(connection.node_out)
                self.active_nodes_set.add(connection.node_out)

    def update_weights(self):
        for in_node in self.input_nodes:
            self.active_nodes.append(in_node)
            self.active_nodes_set.add(in_node)
        while len(self.active_nodes) > 0:
            curr_node = self.active_nodes.pop(0)
            self.active_nodes_set.remove(curr_node)
            for connection in curr_node.outgoing_connections:
                connection.update_weights()
                if connection.node_out in self.active_nodes_set:
                    continue
                self.active_nodes.append(connection.node_out)
                self.active_nodes_set.add(connection.node_out)
