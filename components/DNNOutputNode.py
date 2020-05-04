from typing import Tuple

from numpy import ndarray

from components.DNNNode import DNNNode


class DNNOutputNode(DNNNode):

    def __init__(self, internal_shape: Tuple[int, int]):
        super().__init__(internal_shape)
        pass

    def extract_output(self) -> 'ndarray':
        if self.outgoing_buffer is None and len(self.incoming_messages) == 0:
            raise ValueError("Cannot extract data, no data ready for extraction")
        elif self.outgoing_buffer is None:
            self.combine_incoming_messages()
        return self.outgoing_buffer.contents
