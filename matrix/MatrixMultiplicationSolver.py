from multiprocessing import Pool
from typing import List, Tuple

import numpy as np

from matrix import GaussianEliminator as ge
from matrix.MatrixUtils import getPivotPositions


class PvfSolution:

    def __init__(self, matrix: 'np.ndarray', augmented_column: 'np.ndarray', pivot_positions: List[Tuple[int, int]]):
        non_piv_cols = set(range(matrix.shape[1]))
        non_piv_cols = non_piv_cols.difference([x[1] for x in pivot_positions])
        self.dependent_coeffs: List['np.ndarray'] = []

        for non_piv_col in non_piv_cols:
            col = matrix[:, [non_piv_col]]
            are_zero = np.apply_along_axis(lambda x: x == 0, 0, col)
            if np.all(are_zero):
                continue
            self.dependent_coeffs.append(col[:matrix.shape[1], ] * -1)
            self.dependent_coeffs[-1][non_piv_col][0] = 1
        self.constants = augmented_column[:matrix.shape[1], ]

    def insert_dependents(self, *args) -> 'np.ndarray':
        ret = self.constants
        for i in range(min(len(args), len(self.dependent_coeffs))):
            ret += self.dependent_coeffs[i] * args[i]
        return ret

    def __len__(self):
        return len(self.dependent_coeffs)


def systemSolveStatus(matrix: 'np.ndarray', augmented_column: 'np.ndarray'):
    """
        Returns whether the system of equations represented by the inputs has one solution,
        infinite solutions, or no solutions
        The return values are as follows:
        0: one solution
        1: Infinitely many solutions
        2: No solutions
    """

    pivot_positions = getPivotPositions(matrix)
    non_piv_rows = set()
    for i in range(0, matrix.shape[0]):
        non_piv_rows.add(i)
    non_piv_rows = non_piv_rows.difference([x[0] for x in pivot_positions])
    for non_piv_row in non_piv_rows:
        if not augmented_column[non_piv_row] == 0:
            return 2, pivot_positions
    if not len(pivot_positions) == augmented_column.shape[0]:
        return 1, pivot_positions
    return 0, pivot_positions


def extractSingleSolution(matrix: 'np.ndarray', augmentation: 'np.ndarray', pivot_positions: List[Tuple[int, int]]):
    ret_arr = np.zeros((matrix.shape[1], 1), float)
    for pivot_position in pivot_positions:
        ret_arr[pivot_position[1]] = augmentation[pivot_position[1]][0]
    return ret_arr


def extractInfSolution(matrix: 'np.ndarray', augmentation: 'np.ndarray', pivot_positions: List[Tuple[int, int]]):
    pvf_sol = PvfSolution(matrix, augmentation, pivot_positions)
    return pvf_sol.insert_dependents(np.random.random((len(pvf_sol))))


def solveEquation(original_matrix: 'np.ndarray', resulting_matrix: 'np.ndarray',
                  known_history: List['ge.ThreadBlock'] = None,
                  multiprocessing_pool: "Pool" = None):
    use_multiprocessing = multiprocessing_pool is not None
    if use_multiprocessing:
        ge.enableMultiprocessing(Pool)

    # If we have a known change history, then assume original matrix is in gauss eliminated form
    operating_history = known_history
    working_original_matrix = original_matrix.copy()
    working_resulting_matrix = resulting_matrix.copy()

    # If matrix multiplication is on the left, assume that the arguments are passed in transposed already

    if operating_history is None:
        operating_history = ge.gaussian_elimination(working_original_matrix, use_multiprocessing)
    ge.execute_history(operating_history, working_resulting_matrix, use_multiprocessing)

    ret_matrix = np.zeros((original_matrix.shape[1], resulting_matrix.shape[1]))
    for i in range(ret_matrix.shape[1]):
        curr_res_col = working_resulting_matrix[:, [i]]
        solve_status, pivot_positions = systemSolveStatus(working_original_matrix, curr_res_col)
        if solve_status == 0:
            ret_matrix[:, [i]] = extractSingleSolution(working_original_matrix, curr_res_col, pivot_positions)
        elif solve_status == 1:
            # construct a pvf of the solution and do a small, random solution
            ret_matrix[:, [i]] = extractInfSolution(working_original_matrix, curr_res_col, pivot_positions)
        else:
            # left multiply be the transpose for both and re-solve.
            working_orig_matrix_trans = working_original_matrix.T
            reval_org_matrix = working_orig_matrix_trans @ working_original_matrix
            reval_res_col = working_orig_matrix_trans @ curr_res_col
            reval_hist = ge.gaussian_elimination(reval_org_matrix, use_multiprocessing)
            ge.execute_history(reval_hist, reval_res_col, use_multiprocessing)
            solve_status, pivot_positions = systemSolveStatus(reval_org_matrix, reval_res_col)
            if solve_status == 0:
                ret_matrix[:, [i]] = extractSingleSolution(reval_org_matrix, reval_res_col, pivot_positions)
            elif solve_status == 1:
                # construct pvf of the solution, do a small random solution
                ret_matrix[:, [i]] = extractInfSolution(reval_org_matrix, reval_res_col, pivot_positions)
            else:
                raise ValueError("System solving failed. This really shouldn't happen in theory.")
            pass
    return ret_matrix
