import json
import vtk
import numpy as np
import SimpleITK as sitk
from pymskt.mesh.utils import vtk_deep_copy
from pymskt.utils import create_4x4_from_3x3
import math

# import pyfocusr

def separate_scale_and_transform(transform, tolerance=1e-5):
    """
    Separate a 3x3 or 4x4 transform into a scale and a unit transform.

    Parameters
    ----------
    transform : np.ndarray
        The transform to separate.
    tolerance : float, optional
        The tolerance to use when checking if the scales are uniform, by default 1e-5
    
    Returns
    -------
    float
        The scale of the transform.
    np.ndarray
        The unit transform of the transform.
    
    Raises
    ------
    Exception
        If the transform is not 3x3 or 4x4.
    Exception
        If the scales are not uniform.
    
    Notes
    -----
    The scale is the average of the norms of the rows of the transform.
    The unit transform is the transform divided by the scale.
    """
    if transform.shape[0] == 3:
        pass
    elif transform.shape[0] == 4:
        transform = transform[:3, :3]
    else:
        raise Exception('Transform should be 3x3 or 4x4 np array')
    
    scales = []
    for row in range(3):
        scales.append(np.linalg.norm(transform[row, :]))
    
    for i in range(1, 3):
        if not math.isclose(scales[0], scales[i], abs_tol=tolerance):
            raise Exception('Scales along axes ' +
                            'do not appear to be uniform: ' +
                            f'{scales[0]} & {scales[i]}; ' +
                            f'tolerance is {tolerance}')
        else:
            pass
    # There might be very small differences b/w norms - so, average them.
    scale = np.mean(scales) 
    
    unit_transform = np.eye(3)
    for row in range(3):
        unit_transform[row,:] = transform[row, :] / np.linalg.norm(transform[row, :])
    
    return scale, unit_transform

# def create_linear_transform(transform_matrix):
#     matrix = vtk.vtkMatrix4x4()
#     for row in range(4):
#         for col in range(4):
#             matrix.SetElement(row, col, transform_matrix[row, col])
    
#     transform = vtk.vtkTransform()
#     transform.SetMatrix(matrix)
#     return transform


def get_linear_transform_matrix(transform):
    """
    Get the 4x4 matrix of a linear transform.

    Parameters
    ----------
    transform : vtk.vtkTransform (or subclass)
        The transform to get the matrix of.

    Returns
    -------
    np.ndarray
        The 4x4 matrix of the transform.
    """

    four_by_four = np.eye(4)
    transform_matrix = transform.GetMatrix()
    for row in range(4):
        for col in range(4):
            four_by_four[row, col] = transform_matrix.GetElement(row, col)
    
    return four_by_four


def write_linear_transform(transform, filepath):
    """
    Write a linear transform to a file. 

    Parameters
    ----------
    transform : vtk.vtkTransform (or subclass)
        The transform to write to file. 
    filepath : str
        The filepath to write the transform to.

    Returns
    -------
    None
    """    
    
    four_by_four = get_linear_transform_matrix(transform)
    
    scale, unit_transform = separate_scale_and_transform(four_by_four)

    transform_dict = {
        'transform_matrix': four_by_four.tolist(),
        'scale': scale,
        'unit_transform': unit_transform.tolist()
    }

    with open(filepath, 'w') as f:
        json.dump(transform_dict, f, indent=4)

def read_linear_transform(filepath, return_scale=False, return_unit_transform=False):
    """
    Read a linear transform from a file. 

    Parameters
    ----------
    filepath : str
        The filepath to read the transform from.
    return_scale : bool, optional
        Whether to return the scale of the transform, by default False
    return_unit_transform : bool, optional
        Whether to return the unit transform of the transform, by default False

    Returns
    -------
    vtk.vtkTransform
        The transform that was read from the file.
    float, optional
        The scale of the transform, if `return_scale` is True.
    np.ndarray, optional
        The unit transform of the transform, if `return_unit_transform` is True.

    """    
    with open(filepath, 'r') as f:
        transform_dict = json.load(f)
    
    matrix = vtk.vtkMatrix4x4()
    for row in range(4):
        for col in range(4):
            # idx = row * 4 + col
            matrix.SetElement(row, col, transform_dict['transform_matrix'][row][col])
    
    transform = vtk.vtkTransform()
    transform.SetMatrix(matrix)

    if return_scale:
        if return_unit_transform:
            return transform, transform_dict['scale'], transform_dict['unit_transform']
        else:
            return transform, transform_dict['scale']
    else:
        if return_unit_transform:
            return transform, transform_dict['unit_transform']
        else:
            return transform


def create_transform(transform_matrix):
    """
    Turn 4x4 matrix into a vtkTransform

    Parameters
    ----------
    transform_matrix : numpy.ndarray
        4x4 transformation matrix to be converted to transform
    """

    if np.allclose(transform_matrix.shape, [4, 4]):
        pass
    else:
        raise Exception('transform matrix should be 4x4 matrix')
    transform = vtk.vtkTransform()
    transform.SetMatrix(transform_matrix.flatten())
    return transform


def apply_transform(source, transform):
    """
    Apply transform to surface mesh

    Parameters
    ----------
    source : vtk.vtkPolyData
        The source surface mesh to apply transformation (`transform`) to. 
    transform : vtk.vtkTransform
        The transofrm to apply to the surface mesh (`source`)

    Returns
    -------
    vtk.vtkPolyData
        The transformed surface mesh. 
    """    
    transform_filter = vtk.vtkTransformPolyDataFilter()
    transform_filter.SetInputData(source)
    transform_filter.SetTransform(transform)
    transform_filter.Update()
    return transform_filter.GetOutput()

def copy_image_transform_to_mesh(mesh, image, verbose=False):
    """
    Copy the transformation matrix from a SimpleITK image onto a
    vtk.vtkPolyData

    Parameters
    ----------
    mesh : vtk.vtkPolyData
        Mesh to transform
    image : SimpleITK.Image
        Image that contains the transformation matrix of interest. 
    verbose : bool, optional
        Should we print the transform to the console?, by default False

    Returns
    -------
    vtk.vtkPolyData
        The surface `mesh` after apply the transformation matrix from the `image` to it. 
    """    
    transform_array = create_4x4_from_3x3(image.GetDirection(), translation=image.GetOrigin())

    transform = create_transform(transform_array)

    if verbose is True:
        print(transform)

    return apply_transform(source=mesh, transform=transform)

class SitkVtkTransformer:
    """
    An class to helping apply SimpleITK image transformation to vtk.vtkPolyData

    Parameters
    ----------
    sitk_image : SimpleITK.Image, optional
        Image whos transformation matrix should be used, by default None
    three_by_three : numpy.ndarray, optional
        3x3 transformation matrix to build the transformer based on, by default None
    translation : numpy.ndarray, optional
        1x3 translation to apply, by default None
    center : numpy.ndarray, optional
        center of the image volume to enable centering before applying transform, by default None


    Attributes
    ----------
    transform_array : numpy.ndarray
        4x4 transformation matrix
    inverse_transform_array : np.ndarray
        Inverse of transform array to allow "undoing" of transformation. 

    Methods
    ----------

    Notes
    -----
    There are remmnants of `three_by_three`, `translation` etc. to allow manual prescription of
    this information. 

    """    
    def __init__(self,
                 sitk_image=None,
                 three_by_three=None,
                 translation=None,
                 center=None
                 ):
        """
        Class to enable transformation of vtk.vtkPolyData based on SimpleITK.Image.

        Parameters
        ----------
        sitk_image : SimpleITK.Image, optional
            Image whos transformation matrix should be used, by default None
        three_by_three : numpy.ndarray, optional
            3x3 transformation matrix to build the transformer based on, by default None
        translation : numpy.ndarray, optional
            1x3 translation to apply, by default None
        center : numpy.ndarray, optional
            center of the image volume to enable centering before applying transform, by default None
        """        
        if sitk_image is not None:
            self.transform_array = create_4x4_from_3x3(sitk_image.GetDirection(), translation=sitk_image.GetOrigin())
        # Need to setup below code to receive 3x3, translation, and center to apply a transform. 

        # elif three_by_three is not None:
        #     self.transform_array = create_4x4_from_3x3(three_by_three)
        #     # if translation is not None:
        #     #     # self.transform_array[:3, 3] = translation
        # self.translation = translation

        # # Center is in this, but is not currently used. Below is commented out to show how it could be automatically applied. 
        # self.center = center 


        # Easier to rotate mesh to image instead of image to mesh. So take inverse of transform.
        self.inverse_transform_array = np.linalg.inv(self.transform_array)

    def get_transform(self):
        """
        Get vtkTransform of the `transform_array` created in `__init__`

        Returns
        -------
        vtk.vtkTransform
            VTK transform object that can be used for transforming a vtk object (e.g., vtk.vtkPolyData)
        """        
        # create vtk transform.
        transform = vtk.vtkTransform()
        # if self.center is not None:
        #     transform.PostMultiply()
        #     transform.Translate(-self.center[0], -self.center[1], -self.center[2])
        transform.SetMatrix(self.transform_array.flatten())
        # if self.translation is not None:
        #     transform.Translate(self.translation)
        # if self.center is not None:
        #     transform.Translate(self.center[0], self.center[1], self.center[2])
        return transform

    def get_inverse_transform(self):
        """
        Get vtkTransform of the `inverse_transform_array` created in `__init__`

        Returns
        -------
        vtk.vtkTransform
            VTK transform object that can be used for transforming a vtk object (e.g., vtk.vtkPolyData)
        """        
        transform = vtk.vtkTransform()
        transform.SetMatrix(self.inverse_transform_array.flatten())
        # if self.translation is not None:
        #     transform.Translate(tuple([-x for x in self.translation]))
        return transform

    def get_transformer(self):
        """
        Get `vtkTransformPolyDataFilter` using the `vtkTransform` of `transform_array`

        Returns
        -------
        vtk.vtkTransformPolyDataFilter
            Filter that can be applied to a vtk.vtkPolyData to transform it. 
        """        
        # Get transform filter and apply transform to bone mesh.
        transformer = vtk.vtkTransformPolyDataFilter()
        transformer.SetTransform(self.get_transform())
        # Now, can use this later to transform the bone/
        return transformer

    def get_inverse_transformer(self):
        """
        Get `vtkTransformPolyDataFilter` using the `vtkTransform` of `inverse_transform_array`

        Returns
        -------
        vtk.vtkTransformPolyDataFilter
            Filter that can be applied to a vtk.vtkPolyData to transform it. 
        """        
        inverse_transformer = vtk.vtkTransformPolyDataFilter()
        inverse_transformer.SetTransform(self.get_inverse_transform())
        return inverse_transformer

    def apply_transform_to_mesh(self,
                                mesh):
        """
        Transform surface mesh using transformation matrix from the image data. 

        Parameters
        ----------
        mesh : vtk.vtkPolyData
            surface mesh to apply transform to. 

        Returns
        -------
        vtk.vtkPolyData
            copy of the `mesh` object after being transformed using image data transformation matrix. 
        """        
        transformer = self.get_transformer()
        transformer.SetInputData(mesh)
        transformer.Update()
        return vtk_deep_copy(transformer.GetOutput())

    def apply_inverse_transform_to_mesh(self,
                                        mesh):
        """
        Transform surface mesh using the inverse of the transformation matrix from the image data. 

        Parameters
        ----------
        mesh : vtk.vtkPolyData
            surface mesh to apply transform to. 

        Returns
        -------
        vtk.vtkPolyData
            copy of the `mesh` object after being transformed using the inverse of the image data transformation matrix. 
        """                                        
        inverse_transformer = self.get_inverse_transformer()
        inverse_transformer.SetInputData(mesh)
        inverse_transformer.Update()
        return vtk_deep_copy(inverse_transformer.GetOutput())

def get_versor_from_transform(transform):
    """
    Get a `sitk.VersorRigid3DTransform` from a regular transform. 

    Parameters
    ----------
    transform : sitk.sitkTransform
        Transformation to turn into a versor transform. 

    Returns
    -------
    sitk.VersorRigid3DTransform
        The transform converted into a versor transform.

    Raises
    ------
    Exception
        If the transform is a composite of > 1 transformation.
    Exception
        If other error... ?? 
    """    
    try:
        versor = sitk.VersorRigid3DTransform(transform)
    except RuntimeError:
        try:
            composite = sitk.CompositeTransform(transform)
            if composite.GetNumberOfTransforms() == 1:
                versor = sitk.VersorRigid3DTransform(composite.GetBackTransform())
            else:
                raise Exception ('There is {} transforms in the composite transform, excpected 1!'.format(composite.GetNumberOfTransforms()))
        except BaseException as error:
            raise Exception(error)
    
    return versor

def break_versor_into_center_rotate_translate_transforms(versor):
    """
    Convert a sitk.VersorRigid3DTransform into a 3 vtk.vtkTransform objects: 
    1. center transformation, 2. rotation transform, 3. translation transform. 

    Parameters
    ----------
    versor : sitk.VersorRigid3DTransform
        The sitk transformation matrix that we want to break into its constituent parts. 

    Returns
    -------
    tuple of 3x vtk.vtkTransform
        Tuple of transformations to be applied to surface mesh to align it with the original image(s). 
    """    
    center_of_rotation = versor.GetCenter()
    center_transform = vtk.vtkTransform()
    center_transform.Translate(center_of_rotation[0], center_of_rotation[1], center_of_rotation[2])

    rotate_transform = vtk.vtkTransform()
    four_by_four = np.identity(4)
    four_by_four[:3,:3] = np.reshape(versor.GetMatrix(), (3,3))
    rotate_transform.SetMatrix(four_by_four.flatten())

    translate_transform = vtk.vtkTransform()
    translate_transform.Translate(versor.GetTranslation())

    return center_transform, rotate_transform, translate_transform