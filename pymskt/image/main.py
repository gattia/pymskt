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
    new_seg_array[border_size:-border_size, border_size:-border_size, border_size:-border_size] = seg_array[border_size:-border_size, border_size:-border_size, border_size:-border_size]
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
    array = sitk.GetArrayFromImage(image)
    new_array = np.zeros_like(array)
    new_array[array == label_idx] = 1.
    new_image = sitk.GetImageFromArray(new_array)
    new_image.CopyInformation(image)
    new_image = sitk.Cast(new_image, sitk.sitkFloat32)

    gauss_filter = sitk.DiscreteGaussianImageFilter()
    gauss_filter.SetVariance(variance)
    #     gauss_filter.SetUseImageSpacingOn
    gauss_filter.SetUseImageSpacing(True)
    filtered_new_image = gauss_filter.Execute(new_image)

    return filtered_new_image

def crop_bone_based_on_width(seg_image,
                             bone_idx,
                             np_med_lat_axis=0,
                             np_inf_sup_axis=1,
                             np_ant_post_axis=2,
                             bone_crop_distal=True,
                             new_seg_array=None,
                             value_to_reassign=0,
                             percent_width_to_crop_height=1.0):
    seg_array = sitk.GetArrayFromImage(seg_image)
    loc_bone = np.where(seg_array == bone_idx)
    med_lat_width_bone_mm = (np.max(loc_bone[np_med_lat_axis]) - np.min(loc_bone[np_med_lat_axis])) * \
                            seg_image.GetSpacing()[::-1][np_med_lat_axis]
    inf_sup_crop_in_pixels = (med_lat_width_bone_mm / seg_image.GetSpacing()[::-1][
        np_inf_sup_axis]) * percent_width_to_crop_height
    if bone_crop_distal is True:
        bone_distal_idx = np.max(loc_bone[np_inf_sup_axis])
        bone_proximal_idx = bone_distal_idx - inf_sup_crop_in_pixels
        if bone_proximal_idx < 1:
            bone_proximal_idx = 1

    elif bone_crop_distal is False:
        bone_proximal_idx = np.min(loc_bone[np_inf_sup_axis])
        bone_distal_idx = bone_proximal_idx + inf_sup_crop_in_pixels
        if bone_distal_idx > seg_array.shape[np_inf_sup_axis]:
            bone_distal_idx = seg_array.shape[np_inf_sup_axis] - 1

    max_inf_sup_idx = max(bone_distal_idx, bone_proximal_idx)
    min_inf_sup_idx = min(bone_distal_idx, bone_proximal_idx)

    idx_bone_to_keep = np.where(
        (loc_bone[np_inf_sup_axis] > min_inf_sup_idx) & (loc_bone[np_inf_sup_axis] < max_inf_sup_idx))
    loc_bone_to_remove = tuple([np.delete(x, idx_bone_to_keep) for x in loc_bone])

    seg_array[loc_bone_to_remove] = value_to_reassign

    new_seg_image = sitk.GetImageFromArray(seg_array)
    new_seg_image.CopyInformation(seg_image)

    return new_seg_image

def apply_transform_retain_array(image, transform, interpolator=sitk.sitkNearestNeighbor):
    """
    This function will move the actual image in space but keep the underlying array the same. 
    So, in x/y/z land the pixels are in a new location, but the actual underlying data array 
    is the same. 

    I have a feeling that this is overkill. 
    """
    inverse_transform = transform.GetInverse()
    new_origin = inverse_transform.TransformPoint(image.GetOrigin())

    new_x = inverse_transform.TransformPoint(image.TransformIndexToPhysicalPoint((image.GetSize()[0], 0, 0)))
    new_y = inverse_transform.TransformPoint(image.TransformIndexToPhysicalPoint((0, image.GetSize()[1], 0)))
    new_z = inverse_transform.TransformPoint(image.TransformIndexToPhysicalPoint((0, 0, image.GetSize()[2])))

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
    new_three_by_three = np.zeros((3,3))
    new_three_by_three[:,0] = new_x_vector
    new_three_by_three[:,1] = new_y_vector
    new_three_by_three[:,2] = new_z_vector

    new_image = sitk.Resample(image, 
                              new_size,
                              transform,
                              interpolator,
                              new_origin, 
                              new_spacing,
                              new_three_by_three.flatten().tolist())
    return new_image
