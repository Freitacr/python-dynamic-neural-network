import unittest

import numpy as np

from matrix import GaussianEliminator as ge
from matrix import MatrixMultiplicationSolver as mms


class LinearEquationSolverTest(unittest.TestCase):

    def test_identifySingleSolution(self):
        test_mat = np.array([[1, 1, 1], [0, 2, 5], [2, 5, -1]], float)
        res_col = np.array([[6], [-4], [27]], float)
        hist = ge.gaussian_elimination(test_mat)
        ge.execute_history(hist, res_col, False)
        self.assertEqual(mms.systemSolveStatus(test_mat, res_col)[0], 0)

    def test_identifySingleSolutionMoreComplicated(self):
        test_mat = np.array([[5, 3, 9], [-2, 3, -1], [-1, -4, 5]], float)
        res_col = np.array([[-1], [-4], [1]], float)
        hist = ge.gaussian_elimination(test_mat)
        ge.execute_history(hist, res_col, False)
        self.assertEqual(mms.systemSolveStatus(test_mat, res_col)[0], 0)

    def test_identifyInfiniteSolution(self):
        test_mat = np.array([[-1, -2, 1], [2, 3, 0], [0, 1, -2]], float)
        res_col = np.array([[-1], [2], [0]], float)
        hist = ge.gaussian_elimination(test_mat)
        ge.execute_history(hist, res_col, False)
        self.assertEqual(mms.systemSolveStatus(test_mat, res_col)[0], 1)

    def test_identifyNoSolution(self):
        test_mat = np.array([[2, 3], [2, 3]], float)
        res_col = np.array([[10], [12]], float)
        hist = ge.gaussian_elimination(test_mat)
        ge.execute_history(hist, res_col, False)
        self.assertEqual(mms.systemSolveStatus(test_mat, res_col)[0], 2)

    def test_solveSingularSolution(self):
        test_mat = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]], float)
        res_mat = np.array([[2, 3, 4], [5, 6, 7], [8, 9, 10]], float)
        # expecting the resulting matrix to be the same as the multiplicand
        multiplicand = mms.solveEquation(test_mat, res_mat)
        self.assertTrue(np.array_equiv(res_mat, multiplicand))

    def test_solveSingularSolutionLeftMultiply(self):
        test_mat = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]], float)
        res_mat = np.array([[2, 3, 4], [5, 6, 7], [8, 9, 10]], float)
        # expecting the resulting matrix to be the same as the multiplicand
        multiplicand = mms.solveEquation(test_mat.T, res_mat.T)
        self.assertTrue(np.array_equiv(res_mat, multiplicand.T))

    def test_solveSingularSolutionNonIdentity(self):
        test_mat = np.array([[1, 2, 1], [4, 5, 6], [7, 8, 9]], float)
        res_mat = np.array([[24, 20, 16], [84, 69, 54], [138, 114, 90]], float)
        expected = np.array([[9, 8, 7], [6, 5, 4], [3, 2, 1]], float)
        multiplicand = mms.solveEquation(test_mat, res_mat)
        self.assertTrue(np.array_equiv(expected, multiplicand))

    def test_solveInfiniteSolution(self):
        test_mat = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], float)
        res_mat = np.array([[14, 32, 50], [32, 77, 122], [50, 122, 194]], float)
        multiplicand = mms.solveEquation(test_mat, res_mat)
        mul_res = test_mat @ multiplicand
        mul_res = np.around(mul_res, 5)  # rounding is necessary due to floating point arithmetic inaccuracies
        self.assertTrue(np.array_equiv(res_mat, mul_res))


if __name__ == '__main__':
    unittest.main()
