# cython: infer_types=True
import numpy as np
cimport cython
from cython.parallel import prange
from libc.math cimport exp as c_exp
from libc.math cimport pow as c_pow
# from libc.math cimport log as c_log

ctypedef fused my_type:
    cython.double
    cython.float

@cython.boundscheck(False)  # Deactivate bounds checking
@cython.wraparound(False)   # Deactivate negative indexing.
@cython.cdivision(True) # Do division using C?
def gaussian_kernel(
    my_type[:, :] X, 
    my_type[:, :] Y, 
    double sigma=1.,
    # bool normalize=True
):
    """
    Get gaussian kernal for every point in array X to every point in array Y.
    If X/Y are the same, then this will just smooth array X. 
    If X/Y are different, can be used to smooth one onto the other. 

    Parameters
    ----------
    X : numpy.ndarray
        First array to compute gaussian kernel for
    Y : numpy.ndarray
        Second array to compute gaussian kernel for
    sigma : float, optional
        Standard deviation (sigma) for gaussian kernel, by default 1.
    # normalize: bool, optional
    #     Whether or not to normalize the scalar values. Normalizing will ensure
    #     that each point in x is a weighted sum of all points in Y with those weightings
    #     totalling 1.0. Therefore, 

    Returns
    -------
    numpy.ndarray
        Array that can be multiple by scalar values to smooth them. 
        Smoothing can be done inherently, or from one surface onto another. 
    
    Notes
    -----
    https://math.stackexchange.com/questions/434629/3-d-generalization-of-the-gaussian-point-spread-function
    """
    cdef Py_ssize_t x_i_shape, x_j_shape, y_i_shape, y_j_shape
    cdef my_type tmp_total, gaussian_multiplier, two_sigma2

    if my_type is double:
        dtype = np.double
    elif my_type is float:
        dtype = np.float32
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

    # The following normalizes all of the values. This ensures that the points in an image won't be darkened with smoothing
    # Otherwise, all of the kernel values will be < 1.0 so the image will get darker.
    # If we had a continuous surface, then the sum of the kernel on each point would be 1.0
    # Since we have a discrete surface, when we calculate the kernal at finite points we lose data between those points
    # Normalizing in this way helps preserve the scale of the data. E.g., the mean of all of the points will be less
    # than the original mean, but it will be much closer than if we did not normalize.  

    # https://en.wikipedia.org/wiki/Gaussian_blur#Implementation


    for i in range(x_i_shape):
        for j in range(y_i_shape):
            den_view[i] += kernel_view[i, j]

    for i in range(x_i_shape):
        for j in range(y_i_shape):
            kernel_view[i, j] = kernel_view[i, j] / den_view[i]

    return kernel