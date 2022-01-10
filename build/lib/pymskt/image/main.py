import os
import vtk
import SimpleITK as sitk
import numpy as np

def set_vtk_image_origin(vtk_image, new_origin=(0, 0, 0)):
    """
    Reset the origin of a `vtk_image`

    Parameters
    ----------
    vtk_image
    new_origin

    Returns
    -------

    """
    change_origin = vtk.vtkImageChangeInformation()
    change_origin.SetInputConnection(vtk_image.GetOutputPort())
    change_origin.SetOutputOrigin(new_origin)
    change_origin.Update()
    return change_origin

def read_nrrd(path, set_origin_zero=False):
    """
    Read NRRD image file into vtk. Enables usage of marching cubes
    and other functions that work on image data.

    Parameters
    ----------
    location_image
    image_name
    set_origin_zero

    Returns
    -------

    """
    image_reader = vtk.vtkNrrdReader()
    image_reader.SetFileName(path)
    image_reader.Update()
    if set_origin_zero is True:
        change_origin = set_vtk_image_origin(image_reader, new_origin=(0, 0, 0))
        return change_origin
    elif set_origin_zero is False:
        return image_reader


def set_seg_border_to_zeros(seg_image,
                            border_size=1):
    """
    Utility function to ensure that all segmentations are "closed" after marching cubes. 
    If the segmentation extends to the edges of the image then the surface wont be closed at the places it touches the edges. 


    """
    seg_array = sitk.GetArrayFromImage(seg_image)
    new_seg_array = np.zeros_like(seg_array)
    new_seg_array[border_size:-border_size, border_size:-border_size, border_size:-border_size] = seg_array[border_size:-border_size, border_size:-border_size, border_size:-border_size]
    new_seg_image = sitk.GetImageFromArray(new_seg_array)
    new_seg_image.CopyInformation(seg_image)
    return new_seg_image


def smooth_image(image, bone_idx, variance=1.0):
    array = sitk.GetArrayFromImage(image)
    bone_array = np.zeros_like(array)
    bone_array[array == bone_idx] = 1.
    bone_image = sitk.GetImageFromArray(bone_array)
    bone_image.CopyInformation(image)
    bone_image = sitk.Cast(bone_image, sitk.sitkFloat32)

    gauss_filter = sitk.DiscreteGaussianImageFilter()
    gauss_filter.SetVariance(variance)
    #     gauss_filter.SetUseImageSpacingOn
    gauss_filter.SetUseImageSpacing(True)
    filtered_bone_image = gauss_filter.Execute(bone_image)

    return filtered_bone_image