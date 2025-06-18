import math
import os
from typing import Optional

import numpy as np
import SimpleITK as sitk
import vtk
from vtk.util.numpy_support import numpy_to_vtk


def set_vtk_image_origin(vtk_image, new_origin=(0, 0, 0)):
    """
    Reset the origin of a `vtk_image`

    Parameters
    ----------
    vtk_image : vtk.image
        VTK image that we want to change the origin of.
    new_origin : tuple, optional
        New origin to asign to `vtk_image`, by default (0, 0, 0)

    Returns
    -------
    vtk.Filter
        End of VTK filter pipeline after applying origin change.
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
    path : str
        Path to `.nrrd` medical image to read in.
    set_origin_zero : bool, optional
        Bool to determine if origin should be set to zeros, by default False

    Returns
    -------
    vtk.Filter
        End of VTK filter pipeline.
    """

    image_reader = vtk.vtkNrrdReader()
    image_reader.SetFileName(path)
    image_reader.Update()
    if set_origin_zero is True:
        change_origin = set_vtk_image_origin(image_reader, new_origin=(0, 0, 0))
        return change_origin
    elif set_origin_zero is False:
        return image_reader


def set_seg_border_to_zeros(seg_image, border_size=1):
    """
    Utility function to ensure that all segmentations are "closed" after marching cubes.
    If the segmentation extends to the edges of the image then the surface wont be closed
    at the places it touches the edges.

    Parameters
    ----------
    seg_image : SimpleITK.Image
        Image of a segmentation.
    border_size : int, optional
        The size of the border to set around the edges of the 3D image, by default 1

    Returns
    -------
    SimpleITK.Image
        The image with border set to 0 (background).
    """

    seg_array = sitk.GetArrayFromImage(seg_image)
    new_seg_array = np.zeros_like(seg_array)
    new_seg_array[border_size:-border_size, border_size:-border_size, border_size:-border_size] = (
        seg_array[border_size:-border_size, border_size:-border_size, border_size:-border_size]
    )
    new_seg_image = sitk.GetImageFromArray(new_seg_array)
    new_seg_image.CopyInformation(seg_image)
    return new_seg_image


def smooth_image(image, label_idx, variance=1.0):
    """
    Smooth a single label in a SimpleITK image. Used as pre-processing for
    bones/cartilage before applying marching cubes. Helps obtain smooth surfaces.

    Parameters
    ----------
    image : SimpleITK.Image
        Image to be smoothed.
    label_idx : int
        Integer of the tissue of interest to be smoothed in the image.
    variance : float, optional
        The size of the smoothing, by default 1.0

    Returns
    -------
    SimpleITK.Image
        Image of only the label (tissue) of interest after being smoothed.
    """
    new_image = binarize_segmentation_image(image, label_idx)

    new_image = sitk.Cast(new_image, sitk.sitkFloat32)

    gauss_filter = sitk.DiscreteGaussianImageFilter()
    gauss_filter.SetVariance(variance)
    #     gauss_filter.SetUseImageSpacingOn
    gauss_filter.SetUseImageSpacing(True)
    filtered_new_image = gauss_filter.Execute(new_image)

    return filtered_new_image


def binarize_segmentation_image(seg_image, label_idx):
    """
    Return segmentation that is only 0s/1s, with 1s where label_idx is
    located in the image.

    Parameters
    ----------
    seg_image : SimpleITK.Image
        Segmentation image that contains data we want to binarize
    label_idx : int
        Integer/label that we want to extract (binarize) from the
        `seg_image`.

    Returns
    -------
    SimpleITK.Image
        New segmentation image that is binarized.
    """
    array = sitk.GetArrayFromImage(seg_image)
    array_ = np.zeros_like(array)
    array_[array == label_idx] = 1
    new_seg_image = sitk.GetImageFromArray(array_)
    new_seg_image.CopyInformation(seg_image)
    return new_seg_image


def crop_bone_based_on_width(
    seg_image,
    bone_idx,
    np_med_lat_axis=0,
    np_inf_sup_axis=1,
    bone_crop_distal=True,
    value_to_reassign=0,
    percent_width_to_crop_height=1.0,
    idx_crop_on=None,
):
    """
    Crop the bone labelmap of a SimpleITK.Image so that it is proportional to the
    bones medial/lateral width.

    Parameters
    ----------
    seg_image : SimpleITK.Image
        Image to be cropped.
    bone_idx : int
        Label_index of the bone to be cropped.
    np_med_lat_axis : int, optional
        Medial/lateral axis, by default 0
    np_inf_sup_axis : int, optional
        Inferior/superir axis, by default 1
    bone_crop_distal : bool, optional
        Boolean of cropping should occur distal or proximally, by default True
    value_to_reassign : int, optional
        Value to replace bone label with, by default 0
    percent_width_to_crop_height : float, optional
        Bone length as a proportion of width, by default 1.0

    Returns
    -------
    SimpleITK.Image
        Image after bone is cropped as a proportion of the bone's width.
    """
    seg_array = sitk.GetArrayFromImage(seg_image)
    if idx_crop_on is not None:
        loc_bone = np.where(seg_array == idx_crop_on)
    else:
        loc_bone = np.where(seg_array == bone_idx)

    # compute med/lat width in mm
    med_lat_width_bone_mm = (
        np.max(loc_bone[np_med_lat_axis]) - np.min(loc_bone[np_med_lat_axis])
    ) * seg_image.GetSpacing()[::-1][np_med_lat_axis]

    # compute inf/sup crop in pixels
    inf_sup_crop_in_pixels = (
        med_lat_width_bone_mm / seg_image.GetSpacing()[::-1][np_inf_sup_axis]
    ) * percent_width_to_crop_height

    # inf_sup_crop_in_pixels = int(round(inf_sup_crop_in_pixels))

    # determine distal/proximal crop in pixels depending on if
    # cropping distal or proximal (tibia/femur)
    if bone_crop_distal is True:
        bone_distal_idx = seg_array.shape[np_inf_sup_axis] - 1
        bone_distal_idx_ = np.max(loc_bone[np_inf_sup_axis])
        bone_proximal_idx = max(bone_distal_idx_ - inf_sup_crop_in_pixels, 1)

    elif bone_crop_distal is False:
        bone_proximal_idx = 1
        bone_proximal_idx_ = np.min(loc_bone[np_inf_sup_axis])
        bone_distal_idx = min(
            bone_proximal_idx_ + inf_sup_crop_in_pixels, seg_array.shape[np_inf_sup_axis] - 1
        )

    # if cropping idx_crop_on not none... then change loc_bone to use bone_idx
    # the idea is that we determined the cropping (above) using the idx_crop_on
    # parameter. Now, we are applying that cropping to the bone_idx.
    if idx_crop_on is not None:
        loc_bone = np.where(seg_array == bone_idx)

    inside = (loc_bone[np_inf_sup_axis] >= bone_proximal_idx) & (
        loc_bone[np_inf_sup_axis] <= bone_distal_idx
    )

    loc_bone_to_remove = tuple(idx[~inside] for idx in loc_bone)

    seg_array[loc_bone_to_remove] = value_to_reassign

    new_seg_image = sitk.GetImageFromArray(seg_array)
    new_seg_image.CopyInformation(seg_image)

    return new_seg_image


def apply_transform_retain_array(image, transform, interpolator=sitk.sitkNearestNeighbor):
    """
    This function will move the actual image in space but keep the underlying array the same.
    So, in x/y/z land the pixels are in a new location, but the actual underlying data array
    is the same.

    Parameters
    ----------
    image : SimpleITK.Image
        Image to be transformed.
    transform : SimpleITK.Transform
        Transform to apply
    interpolator : SimpleITK.Interpolator, optional
        Interpolator type to use, by default sitk.sitkNearestNeighbor

    Returns
    -------
    SimpleITK.Image
        New image after applying the appropriate transform.

    Notes
    -----
    I have a feeling that this is overkill.
    """

    inverse_transform = transform.GetInverse()
    new_origin = inverse_transform.TransformPoint(image.GetOrigin())

    new_x = inverse_transform.TransformPoint(
        image.TransformIndexToPhysicalPoint((image.GetSize()[0], 0, 0))
    )
    new_y = inverse_transform.TransformPoint(
        image.TransformIndexToPhysicalPoint((0, image.GetSize()[1], 0))
    )
    new_z = inverse_transform.TransformPoint(
        image.TransformIndexToPhysicalPoint((0, 0, image.GetSize()[2]))
    )

    # Create x-axis vector
    new_x_vector = np.asarray(new_x) - np.asarray(new_origin)
    new_x_vector /= np.sqrt(np.sum(np.square(new_x_vector)))
    # Create y-axis vector
    new_y_vector = np.asarray(new_y) - np.asarray(new_origin)
    new_y_vector /= np.sqrt(np.sum(np.square(new_y_vector)))
    # Create z-axis vector
    new_z_vector = np.asarray(new_z) - np.asarray(new_origin)
    new_z_vector /= np.sqrt(np.sum(np.square(new_z_vector)))
    # New image size (shape)
    new_size = image.GetSize()
    # New image spacing
    new_spacing = image.GetSpacing()
    # Create 3x3 transformation matrix from the x/y/z unit vectors.
    new_three_by_three = np.zeros((3, 3))
    new_three_by_three[:, 0] = new_x_vector
    new_three_by_three[:, 1] = new_y_vector
    new_three_by_three[:, 2] = new_z_vector

    new_image = sitk.Resample(
        image,
        new_size,
        transform,
        interpolator,
        new_origin,
        new_spacing,
        new_three_by_three.flatten().tolist(),
    )
    return new_image


def create_vtk_image(
    origin: Optional[int] = [0, 0, 0],
    dimensions: Optional[list] = [20, 20, 20],
    spacing: Optional[float] = [1.0, 1.0, 1.0],
    scalar: Optional[float] = 20.0,
    data: Optional[np.ndarray] = None,
):
    """
    Function to create a 3D vtkimage from a numpy array
    OR to create a uniform image (all same value)

    Parameters
    ----------
    origin : Optional[int], optional
        X/Y/Z origin of the image, by default [0, 0, 0]
    dimensions : Optional[list], optional
        Size of the image along each dimension, by default [20, 20, 20]
    spacing : Optional[float], optional
        Image spacing along each dimension, by default [1., 1., 1.]
    scalar : Optional[float], optional
        Scalar value to use for a uniform image, by default 20.
    data : Optional[np.ndarray], optional
        Data for a non-uniform image, by default None
    """

    if data is None:
        data = np.ones(dimensions) * scalar
    else:
        if len(data.shape) == 3:
            dimensions = data.shape
        else:
            dimensions = [1, 1, 1]
            for idx, dim_size in enumerate(data.shape):
                dimensions[idx] = dim_size
    vtk_array = numpy_to_vtk(data.flatten(order="F"))
    vtk_array.SetName("test")

    # points = vtk.vtkDoubleArray()
    # points.SetName('test')
    # points.SetNumberOfComponents(1)
    # points.SetNumberOfTuples(np.product(dimensions))
    # for x in range(dimensions[0]):
    #   for y in range(dimensions[1]):
    #      for z in range(dimensions[2]):
    #         points.SetValue(
    #            (z * dimensions[0] * dimensions[1]) + (x * dimensions[1]) + y,
    #            array[x, y, z]
    #         )

    vtk_image = vtk.vtkImageData()
    vtk_image.SetOrigin(*origin)
    vtk_image.SetDimensions(*dimensions)
    vtk_image.SetSpacing(*spacing)
    vtk_image.GetPointData().SetScalars(vtk_array)

    #
    return vtk_image


def get_largest_component_binary(
    seg,
):
    if isinstance(seg, np.ndarray):
        input_type = "array"
        seg = sitk.GetImageFromArray(seg)
    else:
        input_type = "sitk"

    labelled_regions_image = sitk.RelabelComponent(
        sitk.ConnectedComponent(seg == 1), sortByObjectSize=True
    )
    # labelled_regions = sitk.GetArrayFromImage(labelled_regions_image)
    largest_component = labelled_regions_image == 1

    if input_type == "array":
        largest_component = sitk.GetArrayFromImage(largest_component)

    return largest_component


def get_largest_connected_components(seg, labels):
    if isinstance(labels, int):
        labels = [labels]

    if isinstance(seg, np.ndarray):
        input_type = "array"
        seg_array = seg
        seg = sitk.GetImageFromArray(seg)
    else:
        seg_array = sitk.GetArrayFromImage(seg)
        input_type = "sitk"

    result = np.zeros_like(seg_array)
    other_labels = [label_idx for label_idx in np.unique(seg_array) if label_idx not in labels]
    for label_idx in other_labels:
        result[seg_array == label_idx] = label_idx

    for label in labels:
        binary_seg_array = (seg_array == label).astype(int)
        largest_connected = get_largest_component_binary(binary_seg_array)
        result[largest_connected == 1] = label

    if input_type == "sitk":
        result = sitk.GetImageFromArray(result)
        result.CopyInformation(seg)

    return result
