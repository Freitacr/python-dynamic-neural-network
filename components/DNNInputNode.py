from typing import TYPE_CHECKING, Tuple
from components.DNNNode import DNNNode
from components.DNNMessage import DNNMessage
from numpy import ndarray


class DNNInputNode(DNNNode):

    def __init__(self, internal_shape: Tuple[int, int]):
        super().__init__(internal_shape)

    def add_input_data(self, input_data: 'ndarray'):
        self.outgoing_buffer = DNNMessage(input_data)
