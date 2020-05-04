from typing import TYPE_CHECKING, Tuple, List
from components.DNNNodeConnection import DNNConnection
from components.DNNMessage import DNNMessage
import numpy as np


class DNNNode:

    def __init__(self, internal_shape: Tuple[int, int]):
        self.incoming_connections: List['DNNConnection'] = []
        self.outgoing_connections: List['DNNConnection'] = []
        self.internal_shape = internal_shape
        self.outgoing_buffer: 'DNNMessage' = None
        self.incoming_messages: List['DNNMessage'] = []

    def add_outgoing_connection(self, node_to_connect: "DNNNode"):
        shared_connection = DNNConnection(self, node_to_connect)
        self.outgoing_connections.append(shared_connection)
        node_to_connect.incoming_connections.append(shared_connection)

    def receive_incoming_message(self, msg):
        self.incoming_messages.append(msg)

    def transmit_data(self):
        if self.outgoing_buffer is None and len(self.incoming_messages) == 0:
            raise ValueError("Cannot transmit data, no data ready for transmittal")
        elif self.outgoing_buffer is None:
            self.outgoing_buffer = DNNMessage(np.zeros(self.internal_shape))
            for msg_in in self.incoming_messages:
                self.outgoing_buffer.contents += msg_in.contents
                self.outgoing_buffer.add_history(msg_in.message_history)
        for out_connection in self.outgoing_connections:
            out_connection.perform_transmit()
        pass
