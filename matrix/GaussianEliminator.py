from multiprocessing import Pool, Array
from typing import List, Tuple, Callable, Iterable

import numpy as np

multipool: "Pool" = None


def enableMultiprocessing(pool: 'Pool'):
    raise NotImplementedError("Multiprocessing is not supported at this time.")
    # global multipool
    # multipool = pool


class ThreadBlock:

    def __init__(self):
        self.history: List[Tuple[Callable, Iterable[object]]] = []

    def addFunctionCall(self, f: "Callable", args: Iterable[object]):
        self.history.append((f, args))

    def __len__(self):
        return len(self.history)


def _swap_row(matrix: 'np.ndarray', row1: int, row2: int):
    temp = matrix[row1].copy()
    matrix[row1] = matrix[row2]
    matrix[row2] = temp
    return matrix


def _scale_and_subtract_rows(matrix: 'np.ndarray', row1: int, row2: int, scale: float, scale_row_1: bool = False):
    if not scale_row_1:
        matrix[row2] = matrix[row2] - (matrix[row1] * scale)
    else:
        matrix[row2] = (matrix[row2] * scale) - matrix[row1]
    return matrix


def _scale_row(matrix: 'np.ndarray', row: int, scale: float):
    matrix[row] = matrix[row] * scale
    return matrix


def _swap_nonzero(matrix: 'np.ndarray', start_row: int, col: int):
    action = None
    for r in range(start_row + 1, matrix.shape[0]):
        if not matrix[r][col] == 0:
            _swap_row(matrix, start_row, r)
            action = _swap_row, (start_row, r)
            break
    return matrix, action


def execute_history(history: List['ThreadBlock'], matrix: 'np.ndarray', use_multiprocessing: bool):
    global multipool
    if use_multiprocessing and multipool is None:
        raise ValueError("Multiprocessing has not been enabled for gaussian elimination. "
                         "Did you call enableMultiprocessing?")
    for curr_block in history:
        for call, args in curr_block.history:
            if not use_multiprocessing:
                call(matrix, *args)
            else:
                multipool.apply_async(call, (Array(np.float64, matrix), *args))


def _forward_eliminate(matrix: 'np.ndarray', use_multiprocessing: bool = False):
    gauss_history: List['ThreadBlock'] = []
    max_col = min(matrix.shape)
    curr_row = 0
    for col in range(0, max_col):
        if matrix[curr_row][col] == 0:
            _, ret_action = _swap_nonzero(matrix, col, col)
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
            curr_block.addFunctionCall(_scale_and_subtract_rows, (curr_row, row, matrix[row][col] * div_precompute))

        execute_history([curr_block], matrix, use_multiprocessing)
        curr_row += 1
        if not len(curr_block) == 0:
            gauss_history.append(curr_block)
    # end
    return gauss_history


def getPivotPositions(matrix: 'np.ndarray') -> List[Tuple[int, int]]:
    pivots: List[Tuple[int, int]] = []
    last_col = 0
    for row in range(matrix.shape[0]):
        for col in range(last_col, matrix.shape[1]):
            mat_val = matrix[row][col]
            if not mat_val == 0:
                pivots.append((row, col))
                last_col = col+1
                break
    return pivots


def gaussian_elimination(matrix: 'np.ndarray', use_multiprocessing: bool = False):
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
            scale_block.addFunctionCall(_scale_row, (piv_row, pre_div))
        curr_block = ThreadBlock()
        for row in reversed(range(0, piv_row)):
            mat_val = matrix[row][piv_col]
            if mat_val == 0:
                continue
            curr_block.addFunctionCall(_scale_and_subtract_rows, (piv_row, row, mat_val * pre_div))
        if not len(curr_block) == 0:
            gauss_history.append(curr_block)
        execute_history([curr_block], matrix, use_multiprocessing)
    execute_history([scale_block], matrix, use_multiprocessing)
    if not len(scale_block) == 0:
        gauss_history.append(scale_block)
    return gauss_history
