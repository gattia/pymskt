import vtk
import numpy as np
import os
import errno



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

def safely_delete_tmp_file(location,
                           filename):
    """
    Function to safely remove a temporary file. 

    Parameters
    ----------
    location : str
        location of the temporary file to remove
    filename : str
        the filename of the temporary file to delete
    """

    if os.path.exists(location):        
        try:
            os.remove(os.path.join(location, filename))
        except OSError as exc:
            if exc.errno != errno.ENOENT:
                raise
            pass
    
    else:
        print('File does not exist.')