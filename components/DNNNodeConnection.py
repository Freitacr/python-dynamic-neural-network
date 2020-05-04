import numpy as np
from numpy.random import random
from copy import copy
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from components.DNNNode import DNNNode


class DNNConnection:

    def __init__(self, node_in: "DNNNode", node_out: "DNNNode"):
        self.node_in = node_in
        self.node_out = node_out
        self.weight_a = random((node_out.internal_shape[0], node_in.internal_shape[0]))
        self.weight_b = random((node_in.internal_shape[1], node_out.internal_shape[1]))

    def perform_transmit(self):
        if self.node_in.outgoing_buffer is None:
            raise ValueError("Cannot transmit data, incoming node did not contain data to transmit")
        trans_msg = copy(self.node_in.outgoing_buffer)
        trans_msg.contents = self.weight_a @ trans_msg.contents @ self.weight_b
        trans_msg.message_history.add_to_history(self)
        self.node_out.receive_incoming_message(trans_msg)
