import random

import numpy as np

import matrix.MatrixMultiplicationSolver as mms

use_tqdm = True

try:
    import tqdm
except ImportError:
    use_tqdm = False


def calc_total_error(expected, actual):
    tosum = expected - actual
    max_err = np.max(np.abs(tosum))
    ret_err = 0
    for i in range(tosum.shape[0]):
        for j in range(tosum.shape[1]):
            ret_err += abs(tosum[i][j])
    return ret_err, max_err


def print_arr(arr_in):
    row_strs = []
    for row in arr_in:
        elements_formatted = []
        for ele in row:
            elements_formatted.append(format(ele, ".15f"))
        row_strs.append("[" + ", ".join(elements_formatted) + "]")
    print("[" + ", ".join(row_strs) + "]")


if __name__ == "__main__":
    # multi_pool = Pool(4)
    iter_obj = None
    if use_tqdm:
        iter_obj = tqdm.tqdm(range(100000))
    else:
        iter_obj = range(100000)
    avg_err = 0
    for i in iter_obj:
        a_scalar = random.randint(10, 50)
        # a_scalar = 1
        b_scalar = random.randint(10, 50)
        # b_scalar = 1
        a_shape = (random.randint(3, 8), random.randint(3, 8))
        b_shape = (a_shape[1], random.randint(3, 8))
        a_mat = np.random.uniform(low=-1.0, high=1.0, size=a_shape) * a_scalar
        b_mat = np.random.uniform(low=-1.0, high=1.0, size=b_shape) * b_scalar
        res = a_mat @ b_mat
        predicted_val = mms.solveEquation(a_mat, res)
        compare_mat = a_mat @ predicted_val
        err, max_err = calc_total_error(res, compare_mat)
        avg_err += err
        # print(err)
        if err > .00001:
            print_arr(a_mat)
            print()
            print_arr(b_mat)
            print()
            print_arr(predicted_val)
            print()
            print_arr(res)
            print()
            print_arr(compare_mat)
            print()
            print("err:", err)
            print()
            print("Max element error:", max_err)
            print()
    # multi_pool.close()
    avg_err /= 100000
    print("Average Error:", avg_err)
