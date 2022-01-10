# cython: infer_types=True
import numpy as np
cimport cython
from cython.parallel import prange
from libc.math cimport exp as c_exp
from libc.math cimport pow as c_pow
# from libc.math cimport log as c_log

ctypedef fused my_type:
    cython.double

@cython.boundscheck(False)  # Deactivate bounds checking
@cython.wraparound(False)   # Deactivate negative indexing.
@cython.cdivision(True) # Do division using C?
def gaussian_kernel(my_type[:, :] X, my_type[:, :] Y, double sigma=1.):
    """
    https://math.stackexchange.com/questions/434629/3-d-generalization-of-the-gaussian-point-spread-function
    :param X:
    :param Y:
    :param sigma:
    :return:
    """
    cdef Py_ssize_t x_i_shape, x_j_shape, y_i_shape, y_j_shape
    cdef my_type tmp_total, gaussian_multiplier, two_sigma2

    if my_type is double:
        dtype = np.double
    # elif my_type is int:
    #     dtype = np.intc
    # elif my_type is cython.longlong:
    #     dtype = np.longlong

    x_i_shape = X.shape[0]
    x_j_shape = X.shape[1]
    y_i_shape = Y.shape[0]
    y_j_shape = Y.shape[1]

    assert x_j_shape == y_j_shape

    gaussian_multiplier = 1/(c_pow(sigma, x_j_shape) * c_pow((2 * np.pi), x_j_shape/2))
    two_sigma2 = 2 * c_pow(sigma, 2.)

    kernel = np.zeros((x_i_shape, y_i_shape), dtype=dtype)
    cdef my_type[:,:] kernel_view = kernel

    den = np.zeros(x_i_shape, dtype=dtype)
    cdef my_type[:] den_view = den

    cdef my_type[:, :] X_view = X
    cdef my_type[:, :] Y_view = Y

    for i in range(x_i_shape):
        for j in range(y_i_shape):
            tmp_total = 0
            for k in range(x_j_shape):
                tmp_total += c_pow(X_view[i,k] - Y_view[j,k], 2.)
            kernel_view[i, j] = c_exp(-tmp_total / two_sigma2)
            kernel_view[i, j] *=  gaussian_multiplier

    for i in range(x_i_shape):
        for j in range(y_i_shape):
            den_view[i] += kernel_view[i, j]

    for i in range(x_i_shape):
        for j in range(y_i_shape):
            kernel_view[i, j] = kernel_view[i, j] / den_view[i]

    return kernel