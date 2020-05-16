from typing import Tuple

from numpy import ndarray

from components.DNNMessage import DNNMessage
from components.DNNNode import DNNNode


class DNNInputNode(DNNNode):

    def __init__(self, internal_shape: Tuple[int, int]):
        super().__init__(internal_shape)
        self.is_input = True

    def add_input_data(self, input_data: 'ndarray'):
        self.outgoing_buffer = DNNMessage(input_data)
