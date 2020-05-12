from multiprocessing import Pool
from typing import List, Tuple, Callable, Iterable

import numpy as np

from matrix.MatrixUtils import getPivotPositions, swap_nonzero, scale_and_subtract_rows, scale_row

multipool: "Pool" = None


def enableMultiprocessing(pool: 'Pool'):
    global multipool
    multipool = pool


class ThreadBlock:

    def __init__(self):
        self.history: List[Tuple[Callable, Iterable[object]]] = []

    def addFunctionCall(self, f: "Callable", args: Iterable[object]):
        self.history.append((f, args))

    def __len__(self):
        return len(self.history)


def execute_history(history: List['ThreadBlock'], matrix: 'np.ndarray', use_multiprocessing: bool):
    global multipool
    if use_multiprocessing and multipool is None:
        raise ValueError("Multiprocessing has not been enabled for gaussian elimination. "
                         "Did you call enableMultiprocessing?")
    for curr_block in history:
        async_results = []
        for call, args in curr_block.history:
            if not use_multiprocessing or len(curr_block.history) == 1:
                call(matrix, *args)
            else:
                async_results.append(multipool.apply_async(call, (matrix, *args)))
        if use_multiprocessing and len(curr_block.history) > 1:
            for async_res in async_results:
                result = async_res.get()
                matrix[result[1]] = result[0]


def _forward_eliminate(matrix: 'np.ndarray', use_multiprocessing: bool = False) -> List['ThreadBlock']:
    gauss_history: List['ThreadBlock'] = []
    max_col = min(matrix.shape)
    curr_row = 0
    for col in range(0, max_col):
        if matrix[curr_row][col] == 0:
            _, ret_action = swap_nonzero(matrix, col, col)
            if ret_action is None:
                continue
            gauss_history.append(ThreadBlock())
            gauss_history[-1].addFunctionCall(ret_action[0], ret_action[1])

        # If we're here then we have a non-zero row entry.
        curr_block = ThreadBlock()
        mat_val = matrix[curr_row][col]
        div_precompute = 1.0/mat_val

        for row in range(curr_row+1, matrix.shape[0]):
            if matrix[row][col] == 0:
                continue
            curr_block.addFunctionCall(scale_and_subtract_rows, (curr_row, row, matrix[row][col] * div_precompute))

        execute_history([curr_block], matrix, use_multiprocessing)
        curr_row += 1
        if not len(curr_block) == 0:
            gauss_history.append(curr_block)
    # end
    return gauss_history


def gaussian_elimination(matrix: 'np.ndarray', use_multiprocessing: bool = False) -> List['ThreadBlock']:
    global multipool
    if use_multiprocessing and multipool is None:
        raise ValueError("Multiprocessing has not been enabled for gaussian elimination. "
                         "Did you call enableMultiprocessing?")
    gauss_history = _forward_eliminate(matrix, use_multiprocessing)

    scale_block = ThreadBlock()
    pivot_positions = reversed(getPivotPositions(matrix))
    for piv_row, piv_col in pivot_positions:
        pre_div = 1.0 / matrix[piv_row][piv_col]
        if not matrix[piv_row][piv_col] == 1:
            scale_block.addFunctionCall(scale_row, (piv_row, pre_div))
        curr_block = ThreadBlock()
        for row in reversed(range(0, piv_row)):
            mat_val = matrix[row][piv_col]
            if mat_val == 0:
                continue
            curr_block.addFunctionCall(scale_and_subtract_rows, (piv_row, row, mat_val * pre_div))
        if not len(curr_block) == 0:
            gauss_history.append(curr_block)
        execute_history([curr_block], matrix, use_multiprocessing)
    execute_history([scale_block], matrix, use_multiprocessing)
    if not len(scale_block) == 0:
        gauss_history.append(scale_block)
    return gauss_history
