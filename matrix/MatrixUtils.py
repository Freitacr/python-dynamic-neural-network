from typing import List, Tuple

import numpy as np


def swap_row(matrix: 'np.ndarray', row1: int, row2: int):
    temp = matrix[row1].copy()
    matrix[row1] = matrix[row2]
    matrix[row2] = temp
    return matrix


def scale_and_subtract_rows(matrix: 'np.ndarray', row1: int, row2: int, scale: float, scale_row_1: bool = False):
    if not scale_row_1:
        matrix[row2] = matrix[row2] - (matrix[row1] * scale)
    else:
        matrix[row2] = (matrix[row2] * scale) - matrix[row1]
    return matrix[row2], row2


def scale_row(matrix: 'np.ndarray', row: int, scale: float):
    matrix[row] = matrix[row] * scale
    return matrix[row], row


def swap_nonzero(matrix: 'np.ndarray', start_row: int, col: int):
    action = None
    for r in range(start_row + 1, matrix.shape[0]):
        if not matrix[r][col] == 0:
            swap_row(matrix, start_row, r)
            action = swap_row, (start_row, r)
            break
    return matrix, action


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
