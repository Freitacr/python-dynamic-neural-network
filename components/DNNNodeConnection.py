from copy import copy
from typing import TYPE_CHECKING

import numpy as np
from numpy.random import uniform

import matrix.MatrixMultiplicationSolver as mms

if TYPE_CHECKING:
    from components.DNNNode import DNNNode
    from components.DNNMessage import DNNMessage


class DNNConnection:

    def __init__(self, node_in: "DNNNode", node_out: "DNNNode"):
        self.node_in = node_in
        self.node_out = node_out
        self.weight_a = uniform(-1, 1, (node_out.internal_shape[0], node_in.internal_shape[0]))
        self.weight_b = uniform(-1, 1, (node_in.internal_shape[1], node_out.internal_shape[1]))
        self.change_weight_a = np.zeros(self.weight_a.shape, float)
        self.change_weight_b = np.zeros(self.weight_b.shape, float)
        # Temporary, in a full system this would be modified to pass more or less error back
        self.weight_change_ratio = .5

    def perform_transmit(self):
        if self.node_in.outgoing_buffer is None:
            raise ValueError("Cannot transmit data, incoming node did not contain data to transmit")
        trans_msg = copy(self.node_in.outgoing_buffer)
        original_input = trans_msg.contents.copy()
        row_res = self.weight_a @ trans_msg.contents
        trans_msg.contents = row_res @ self.weight_b
        trans_msg.message_history.add_to_history((self, original_input, row_res))
        self.node_out.receive_incoming_message(trans_msg)

    def perform_err_transmit(self, err_msg: "DNNMessage"):
        # Todo explore having nodes deeper in the network send more error backwards
        connection_history = err_msg.message_history.last()
        row_res = connection_history[-1]
        original_input = connection_history[-2]

        err_weight_b = self.weight_change_ratio * err_msg.contents
        err_row_res = err_msg.contents - err_weight_b

        # Calculate needed change in weight_b (half of err_msg.contents)
        # weight_b is on the right hand side, no transposition needed
        d_weight_b = mms.solveEquation(row_res, err_weight_b)

        # Calculate needed change to row_res (half of err_msg.contents)
        # row_res on left hand side, transposition is required
        d_row_res = mms.solveEquation(self.weight_b.T, err_row_res.T).T

        # needed change to row_res is the required change to weight_a and the original input

        is_incoming_input = hasattr(self.node_in, "is_input")

        err_weight_a = d_row_res if is_incoming_input else self.weight_change_ratio * d_row_res
        err_original_input = d_row_res - err_weight_a

        # Calculate needed change in weight_a (either row_res (if node_in is InputNode) or half of row_res)
        # weight_a is on the left hand side, transposition is required
        d_weight_a = mms.solveEquation(original_input.T, err_weight_a.T).T

        if not is_incoming_input:
            # Calculate needed change in trans_msg.contents (if node_in is InputNode)
            # original_input is on the right hand side, transposition is not required
            d_original_input = mms.solveEquation(self.weight_a, err_original_input)

            # Send error backward
            err_msg.contents = d_original_input
            err_msg.message_history.contained_history.pop()
            self.node_in.receive_incoming_message(err_msg)

        # Get ready to update weights
        self.change_weight_a += d_weight_a
        self.change_weight_b += d_weight_b

    def update_weights(self):
        self.weight_b += self.change_weight_b
        self.weight_a += self.change_weight_a
        self.change_weight_b = np.zeros(self.weight_b.shape, float)
        self.change_weight_a = np.zeros(self.weight_a.shape, float)
