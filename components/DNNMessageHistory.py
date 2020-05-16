from typing import TYPE_CHECKING, Union, List, Tuple
if TYPE_CHECKING:
    from components.DNNNodeConnection import DNNConnection
    from numpy import ndarray
from copy import copy


class DNNMessageHistory:

    def __init__(self):
        self.contained_history = []

    def add_to_history(self, connection: Union[
                Tuple["DNNConnection", "ndarray", "ndarray"],
                List["DNNMessageHistory"]
            ]):
        self.contained_history.append(connection)

    def add_history(self, history: "DNNMessageHistory"):
        if len(self.contained_history) == 0 or not isinstance(self.contained_history[-1], list):
            self.contained_history.append([])
        self.contained_history[-1].append(history)

    def last(self) -> Union[
                "None",
                List["DNNMessageHistory"],
                Tuple["DNNConnection", "ndarray", "ndarray"]
            ]:
        if len(self.contained_history) == 0:
            return None
        return self.contained_history[-1]

    def __copy__(self):
        ret = DNNMessageHistory()
        for history_element in self.contained_history:
            ret.contained_history.append(history_element)
        return ret

    def copy(self):
        return copy(self)

    def __len__(self):
        return len(self.contained_history)
