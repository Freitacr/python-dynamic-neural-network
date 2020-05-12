import unittest
from multiprocessing import Pool

import numpy as np

import matrix.GaussianEliminator as ge


class GaussianEliminationTest(unittest.TestCase):

    def test_SwapZeroColumn(self):
        test_arr = np.array([[0, 1, 0], [1, 0, 0], [0, 0, 0]], float)
        arr_cpy = test_arr.copy()
        res = ge.swap_nonzero(arr_cpy, 0, 0)
        self.assertTrue(res is not None)
        self.assertTrue(isinstance(res, tuple))
        self.assertTrue(len(res) == 2)
        self.assertTrue(callable(res[1][0]))
        equiv_arr = np.array_equiv(arr_cpy, np.array([[1, 0, 0], [0, 1, 0], [0, 0, 0]]))
        self.assertTrue(np.all(equiv_arr))
        arr_cpy = test_arr.copy()
        res[1][0](arr_cpy, *res[1][1])
        equiv_arr = np.array_equiv(test_arr, arr_cpy)
        self.assertFalse(np.all(equiv_arr))

    def test_SimpleElimination(self):
        test_arr = np.array([[3, 0, 1], [3, 1, 1]], float)
        arr_cpy = test_arr.copy()
        res = ge.gaussian_elimination(arr_cpy)
        self.assertTrue(res is not None)
        self.assertTrue(len(res) == 2)
        self.assertTrue(np.array_equiv(arr_cpy, np.array([[1, 0, 1/3], [0, 1, 0]], float)))
        pass

    def test_ForwardEliminationZeroColumn(self):
        test_arr = np.array([[3, 0, 0, 1], [3, 0, 1, 1], [0, 0, 1, 1]], float)
        expected = np.array([[1, 0, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]], float)
        arr_cpy = test_arr.copy()
        res = ge.gaussian_elimination(arr_cpy)
        self.assertTrue(len(res) == 4)
        self.assertTrue(np.array_equiv(arr_cpy, expected))

    def test_FullElimination(self):
        test_arr = np.array([[21, 16, 13], [21, 18, 38], [24, 24, 6]], float)
        expected = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]], float)
        arr_cpy = test_arr.copy()
        res = ge.gaussian_elimination(arr_cpy)
        expected = np.around(expected, 13)
        arr_cpy = np.around(arr_cpy, 13)
        self.assertEqual(len(res), 5)
        self.assertTrue(np.array_equiv(arr_cpy, expected),
                        "np.array_equiv was false for arrays: " + str(arr_cpy) + "\n" + str(expected)
                        )

    def test_FullEliminationMulti(self):
        multi_pool = Pool(3)
        ge.enableMultiprocessing(multi_pool)
        test_arr = np.array([[21, 16, 13], [21, 18, 38], [24, 24, 6]], float)
        expected = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]], float)
        arr_cpy = test_arr.copy()
        res = ge.gaussian_elimination(arr_cpy, True)
        expected = np.around(expected, 13)
        arr_cpy = np.around(arr_cpy, 13)
        self.assertEqual(len(res), 5)
        self.assertTrue(np.array_equiv(arr_cpy, expected),
                        "np.array_equiv was false for arrays: " + str(arr_cpy) + "\n" + str(expected)
                        )
        multi_pool.close()

    def test_ForwardEliminationZeroColumnMulti(self):
        multi_pool = Pool(3)
        ge.enableMultiprocessing(multi_pool)
        test_arr = np.array([[3, 0, 0, 1], [3, 0, 1, 1], [0, 0, 1, 1]], float)
        expected = np.array([[1, 0, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]], float)
        arr_cpy = test_arr.copy()
        res = ge.gaussian_elimination(arr_cpy)
        self.assertTrue(len(res) == 4)
        self.assertTrue(np.array_equiv(arr_cpy, expected))
        multi_pool.close()

    def test_SimpleEliminationMulti(self):
        multi_pool = Pool(3)
        ge.enableMultiprocessing(multi_pool)
        test_arr = np.array([[3, 0, 1], [3, 1, 1]], float)
        arr_cpy = test_arr.copy()
        res = ge.gaussian_elimination(arr_cpy)
        self.assertTrue(res is not None)
        self.assertTrue(len(res) == 2)
        self.assertTrue(np.array_equiv(arr_cpy, np.array([[1, 0, 1/3], [0, 1, 0]], float)))
        multi_pool.close()


if __name__ == '__main__':
    unittest.main()
