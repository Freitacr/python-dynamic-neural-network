from typing import Tuple, List

import numpy as np

from components.DNNMessage import DNNMessage
from components.DNNNodeConnection import DNNConnection


class DNNNode:

    def __init__(self, internal_shape: Tuple[int, int]):
        self.incoming_connections: List['DNNConnection'] = []
        self.outgoing_connections: List['DNNConnection'] = []
        self.internal_shape = internal_shape
        self.outgoing_buffer: 'DNNMessage' = None
        self.incoming_messages: List['DNNMessage'] = []
        self.__connected_nodes_out = set()

    def add_outgoing_connection(self, node_to_connect: "DNNNode") -> bool:
        if node_to_connect in self.__connected_nodes_out:
            return False
        shared_connection = DNNConnection(self, node_to_connect)
        self.outgoing_connections.append(shared_connection)
        node_to_connect.incoming_connections.append(shared_connection)
        self.__connected_nodes_out.add(node_to_connect)
        return True

    def receive_incoming_message(self, msg):
        self.incoming_messages.append(msg)

    def combine_incoming_messages(self):
        self.outgoing_buffer = DNNMessage(np.zeros(self.internal_shape))
        for msg_in in self.incoming_messages:
            self.outgoing_buffer.contents += msg_in.contents
            self.outgoing_buffer.add_history(msg_in.message_history)

    def transmit_data(self):
        if self.outgoing_buffer is None and len(self.incoming_messages) == 0:
            raise ValueError("Cannot transmit data, no data ready for transmittal")
        elif self.outgoing_buffer is None:
            self.combine_incoming_messages()
        for out_connection in self.outgoing_connections:
            out_connection.perform_transmit()
        pass
