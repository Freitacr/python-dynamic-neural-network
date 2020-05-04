from typing import TYPE_CHECKING, Union, List
if TYPE_CHECKING:
    from components.DNNNodeConnection import DNNConnection
from copy import copy


class DNNMessageHistory:

    def __init__(self):
        self.contained_history = []

    def add_to_history(self, connection: Union["DNNConnection", List["DNNMessageHistory"]]):
        self.contained_history.append(connection)

    def add_history(self, history: "DNNMessageHistory"):
        if len(self.contained_history) == 0 or not isinstance(self.contained_history[-1], list):
            self.contained_history.append([])
        self.contained_history[-1].append(history)

    def __copy__(self):
        ret = DNNMessageHistory()
        for history_element in self.contained_history:
            ret.contained_history.append(history_element)
        return ret

    def copy(self):
        return copy(self)

    def __len__(self):
        return len(self.contained_history)
