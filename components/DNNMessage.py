from copy import copy
from typing import TYPE_CHECKING
from typing import Tuple

import numpy as np

from components.DNNMessageHistory import DNNMessageHistory

if TYPE_CHECKING:
    from components.DNNNodeConnection import DNNConnection
    from numpy import ndarray


class DNNMessage:

    def __init__(self, message_contents: "np.ndarray"):
        self.contents = message_contents
        self.message_history = DNNMessageHistory()

    def __copy__(self):
        ret = DNNMessage(self.contents.copy())
        ret.message_history = copy(self.message_history)
        return ret

    def add_history(self, history: "DNNMessageHistory"):
        self.message_history.add_history(history)

    def add_to_history(self, connection: Tuple["DNNConnection", "ndarray", "ndarray"]):
        self.message_history.add_to_history(connection)
