import unittest
from components.DNNNode import DNNNode
from components.DNNInputNode import DNNInputNode
import numpy as np


class DNNConnectionTest(unittest.TestCase):

    def setUp(self) -> None:
        self.input_node = DNNInputNode((5, 4))
        self.internal_node = DNNNode((3, 2))
        self.input_node.add_outgoing_connection(self.internal_node)

    def test_form_connection(self):
        # nodes are already created, test the link formation
        self.assertTrue(len(self.input_node.outgoing_connections) == 1)
        self.assertTrue(len(self.internal_node.incoming_connections) == 1)
        self.assertEqual(
            self.input_node.outgoing_connections[0],
            self.internal_node.incoming_connections[0]
        )

    def test_transmit_data_singular_link(self):
        # Add starting data
        starting_data = np.random.random(self.input_node.internal_shape)
        self.input_node.add_input_data(starting_data)

        # calculate expected result
        inspected_connection = self.input_node.outgoing_connections[0]
        conn_weight_a = inspected_connection.weight_a
        conn_weight_b = inspected_connection.weight_b
        expected_message_contents: 'np.ndarray' = conn_weight_a @ starting_data @ conn_weight_b

        # ensure data was transmitted
        self.input_node.transmit_data()
        self.assertTrue(len(self.internal_node.incoming_messages) == 1)
        transmitted_data = self.internal_node.incoming_messages[0]
        self.assertTrue(len(transmitted_data.message_history) == 1)

        # compare transmitted result and expected result
        eq_res: 'np.ndarray' = np.equal(transmitted_data.contents, expected_message_contents)
        self.assertTrue(
            eq_res.all()
        )
        self.assertEqual(transmitted_data.message_history.contained_history[0], inspected_connection)

    def test_transmit_differing_shapes(self):
        # create secondary internal node with a differing shape from self.internal_node
        internal_node2 = DNNNode((8, 4))
        self.input_node.add_outgoing_connection(internal_node2)

        # Add starting data
        starting_data = np.random.random(self.input_node.internal_shape)
        self.input_node.add_input_data(starting_data)

        # Calculate expected result for first internal node
        inspected_connection_a = self.input_node.outgoing_connections[0]
        conn_weight_a_a = inspected_connection_a.weight_a
        conn_weight_a_b = inspected_connection_a.weight_b
        expected_message_contents_a: 'np.ndarray' = conn_weight_a_a @ starting_data @ conn_weight_a_b

        # Calculate expected result for second internal node
        inspected_connection_b = self.input_node.outgoing_connections[1]
        conn_weight_b_a = inspected_connection_b.weight_a
        conn_weight_b_b = inspected_connection_b.weight_b
        expected_message_contents_b: 'np.ndarray' = conn_weight_b_a @ starting_data @ conn_weight_b_b

        # Transmit data
        self.input_node.transmit_data()
        transmitted_data_a = self.internal_node.incoming_messages[0]
        transmitted_data_b = internal_node2.incoming_messages[0]

        # Compare transmitted result in first internal node to expected
        eq_res: 'np.ndarray' = np.equal(transmitted_data_a.contents, expected_message_contents_a)
        self.assertTrue(eq_res.all())

        # Compare transmitted result in second internal node to expected
        eq_res: 'np.ndarray' = np.equal(transmitted_data_b.contents, expected_message_contents_b)
        self.assertTrue(eq_res.all())

    def test_transmit_data_convergent(self):
        # Create second input node and add connection to internal node
        input_node_b = DNNInputNode((2, 2))
        input_node_b.add_outgoing_connection(self.internal_node)

        # Add input data to both input nodes
        starting_data_a = np.random.random(self.input_node.internal_shape)
        starting_data_b = np.random.random(input_node_b.internal_shape)
        self.input_node.add_input_data(starting_data_a)
        input_node_b.add_input_data(starting_data_b)

        # Calculate expected result
        # Calculate expected result from self.input_node
        connection_a = self.input_node.outgoing_connections[0]
        weight_a_a = connection_a.weight_a
        weight_a_b = connection_a.weight_b
        expected_a = weight_a_a @ starting_data_a @ weight_a_b
        # Calculate expected result from input_node_b
        connection_b = input_node_b.outgoing_connections[0]
        weight_b_a = connection_b.weight_a
        weight_b_b = connection_b.weight_b
        expected_b = weight_b_a @ starting_data_b @ weight_b_b

        # Transmit data
        self.input_node.transmit_data()
        input_node_b.transmit_data()

        # Compare expected value to potential transmitted
        incoming_buffer = self.internal_node.incoming_messages
        res = np.zeros(self.internal_node.internal_shape)
        for msg_in in incoming_buffer:
            res += msg_in.contents
        eq_res = np.equal(res, expected_a + expected_b)
        self.assertTrue(eq_res.all())

        # Test message combination upon transmittal
        # Create catching internal node
        catch_node = DNNNode(self.internal_node.internal_shape)
        self.internal_node.add_outgoing_connection(catch_node)
        self.internal_node.transmit_data()
        rec_msg = catch_node.incoming_messages[0]
        msg_history = rec_msg.message_history
        # the length should be two as the transmittal crosses a connection
        self.assertTrue(len(msg_history) == 2)
        self.assertTrue(isinstance(msg_history.contained_history[0], list))
        self.assertTrue(len(msg_history.contained_history[0]) == 2)


if __name__ == '__main__':
    unittest.main()
