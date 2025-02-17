import errno
import os
import time

import numpy as np
import vtk


def l2n(l):
    """
    convert list to numpy array

    Parameters
    ----------
    l : list
        list to convert to numpy array

    Returns
    -------
    numpy.ndarray
        array created from inputted list `l`
    """
    return np.array(l)


def n2l(n):
    """
    convert numpy array into list

    Parameters
    ----------
    n : numpy.ndarray
        array to convert into list.

    Returns
    -------
    list
        list created from inputted numpy array `n`.
    """
    return list(n)


def create_4x4_from_3x3(three_by_three, translation=None):
    """
    Create a 4x4 transformation matrix from a 3x3 transformation matrix.

    Parameters
    ----------
    three_by_three : numpy.ndarray
        3x3 transformation matrix to convert into a 4x4
    translation : numpy.ndarray, optional
        translation to include in the 4x4, by default None

    Returns
    -------
    numpy.ndarray
        4x4 transformation matrix created from inputted
        `three_by_three` and `translation`.
    """
    if len(three_by_three) == 9:
        three_by_three = np.reshape(three_by_three, (3, 3))
    four_by_four = np.identity(4)
    four_by_four[:3, :3] = three_by_three

    # if translation is provided, put it in the last column
    if translation is not None:
        four_by_four[:3, 3] = translation
    return four_by_four


def copy_image_transform_to_mesh(mesh, image, verbose=False):
    """
    Copy the transformation matrix from image to mesh.
    This is the same as the function `meshTransform.copy_image_transform_to_mesh` and
    `meshTransform.apply_transform`.

    Parameters
    ----------
    mesh : vtk.vtkPolyData
        Surface mesh to transform.
    image : SimpleITK.Image
        Image who's transform to apply to the `mesh`.
    verbose : bool, optional
        Whether or not to print the transform to console, by default False

    Returns
    -------
    [type]
        [description]
    """
    transform_array = create_4x4_from_3x3(image.GetDirection())
    transform_array[:3, 3] = image.GetOrigin()
    transform = vtk.vtkTransform()
    transform.SetMatrix(transform_array.flatten())
    #     transform.Translate(image.GetOrigin())

    if verbose is True:
        print(transform)

    transformer = vtk.vtkTransformPolyDataFilter()
    transformer.SetTransform(transform)
    transformer.SetInputData(mesh)
    transformer.Update()
    return transformer.GetOutput()


def sigma2fwhm(sigma):
    """
    Converting sigma to Full Width Half Maximum (FWHM).

    Parameters
    ----------
    sigma : float
        The standard deviation (sigma) to convert to FWHM

    Returns
    -------
    float
        The FWHM that is equivalent to the inputted sigma.
    """
    return sigma * np.sqrt(8 * np.log(2))


def fwhm2sigma(fwhm):
    """
    Convert a Full Width Half Maximum into sigma.

    Parameters
    ----------
    fwhm : float
        The FWHM to convert to sigma.

    Returns
    -------
    float
        The sigma that is equivalent to the inputted FWHM.
    """
    return fwhm / np.sqrt(8 * np.log(2))


# def safely_delete_tmp_file(location, filename):
#     """
#     Function to safely remove a temporary file.

#     Parameters
#     ----------
#     location : str
#         location of the temporary file to remove
#     filename : str
#         the filename of the temporary file to delete
#     """

#     if os.path.exists(location):
#         try:
#             os.remove(os.path.join(location, filename))
#         except OSError as exc:
#             if exc.errno != errno.ENOENT:
#                 raise
#             pass

#     else:
#         print("File does not exist.")


def safely_delete_tmp_file(location, filename):
    """
    Function to safely remove a temporary file.

    Parameters
    ----------
    location : str
        Location of the temporary file to remove.
    filename : str
        The filename of the temporary file to delete.
    """
    file_path = os.path.join(location, filename)

    if os.path.exists(file_path):
        for attempt in range(5):  # Retry up to 5 times
            try:
                os.remove(file_path)
                # print(f"Successfully deleted {file_path}.")
                break  # Exit the loop if successful
            except PermissionError:
                # print(f"PermissionError: Unable to delete {file_path}. Attempt {attempt + 1} of 5.")
                time.sleep(1)  # Wait before retrying
            except OSError as exc:
                if exc.errno != errno.ENOENT:
                    raise  # Re-raise if it's not a "file not found" error
                pass
        else:
            print(f"Failed to delete {file_path} after multiple attempts.")


def gaussian_kernel(X, Y, sigma=1.0):
    """
    Compute a Gaussian kernel between all rows of X and Y.
    If the Cython implementation is available (from pymskt.cython.cython_functions), it will be used.
    Otherwise, a pure Python implementation is used.

    Parameters:
        X (numpy.ndarray): Input array of shape (n, d)
        Y (numpy.ndarray): Input array of shape (m, d)
        sigma (float, optional): Standard deviation for the Gaussian kernel. Default is 1.0.

    Returns:
        numpy.ndarray: A (n x m) kernel matrix where each row is normalized to sum to 1.
    """
    try:
        # Attempt to use the Cython version if available
        import pymskt.cython_functions as cython_functions

        return cython_functions.gaussian_kernel(X, Y, sigma)
    except (ImportError, AttributeError):
        # Fall back to a pure Python implementation
        print("Using pure Python implementation of gaussian_kernel")
        print("if slow performance, try installing with Cython version")
    n, d = X.shape
    m, d2 = Y.shape
    if d != d2:
        raise ValueError("X and Y must have the same number of columns")

    # Compute the Gaussian multiplier
    multiplier = 1 / (sigma**d * ((2 * np.pi) ** (d / 2)))
    two_sigma2 = 2 * sigma**2

    kernel = np.zeros((n, m))
    for i in range(n):
        for j in range(m):
            diff = X[i] - Y[j]
            sqnorm = np.dot(diff, diff)
            kernel[i, j] = multiplier * np.exp(-sqnorm / two_sigma2)

    # Normalize each row so that the entries sum to 1
    row_sums = kernel.sum(axis=1, keepdims=True)
    # Avoid division by zero
    row_sums[row_sums == 0] = 1
    return kernel / row_sums
