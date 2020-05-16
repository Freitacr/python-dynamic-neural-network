from typing import Tuple, List

import numpy as np

from components.DNNMessage import DNNMessage
from components.DNNMessageHistory import DNNMessageHistory
from components.DNNNodeConnection import DNNConnection


class TransmissionError(ValueError):
    def __init__(self, message):
        super().__init__(message)


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
        self.incoming_messages.clear()
        self.outgoing_buffer = None

    def transmit_error(self):
        if len(self.incoming_messages) == 0:
            raise TransmissionError("Cannot transmit error, no errors ready for transmittal")
        for msg in self.incoming_messages:
            msg_history = msg.message_history
            last_entry = msg_history.last()
            if last_entry is None:
                raise ValueError("Message's history was empty, this should not happen")
            elif isinstance(last_entry, tuple):
                last_entry[0].perform_err_transmit(msg)
            elif isinstance(last_entry, list):
                err_contents = msg.contents / len(last_entry)
                for hist in last_entry:
                    hist: "DNNMessageHistory" = hist
                    err_msg = DNNMessage(err_contents.copy())
                    err_msg.message_history = hist
                    last_connection: "DNNConnection" = hist.last()[0]
                    last_connection.perform_err_transmit(err_msg)
            else:
                raise ValueError("Unrecognized entry in message history " + str(last_entry))
        self.incoming_messages.clear()
        self.outgoing_buffer = None
