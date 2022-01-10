import os
import time
import errno

import vtk
import pyvista as pv
import pyacvd
from vtk.util.numpy_support import numpy_to_vtk, vtk_to_numpy
import SimpleITK as sitk
import numpy as np
from scipy.spatial.distance import cdist
from scipy.linalg import svd

import string
import random

from neuralsegProcessingTools import quantitativeCartilage as qc
from neuralseg_vtk.cython_functions import gaussian_kernel

from neuralseg import safely_delete_tmp_file





def get_smoothed_cartilage_thickness_values(loc_nrrd_images,
                                            seg_image_name,
                                            bone_label,
                                            list_cart_labels,
                                            image_smooth_var=1.0,
                                            smooth_cart=False,
                                            image_smooth_var_cart=1.0,
                                            ray_cast_length=10.,
                                            percent_ray_len_opposite_dir=0.2,
                                            smooth_surface_scalars=True,
                                            smooth_only_cartilage_values=True,
                                            scalar_gauss_sigma=1.6986436005760381,  # This is a FWHM = 4
                                            bone_pyacvd_subdivisions=2,
                                            bone_pyacvd_clusters=20000,
                                            crop_bones=False,
                                            crop_percent=0.7,
                                            bone=None,
                                            loc_t2_map_nrrd=None,
                                            t2_map_filename=None,
                                            t2_smooth_sigma_multiple_of_thick=3,
                                            assign_seg_label_to_bone=False,
                                            mc_threshold=0.5,
                                            bone_label_threshold=5000,
                                            path_to_seg_transform=None,
                                            reverse_seg_transform=True,
                                            verbose=False):
    """

    :param loc_nrrd_images:
    :param seg_image_name:
    :param bone_label:
    :param list_cart_labels:
    :param image_smooth_var:
    :param loc_tmp_save:
    :param tmp_bone_filename:
    :param smooth_cart:
    :param image_smooth_var_cart:
    :param tmp_cart_filename:
    :param ray_cast_length:
    :param percent_ray_len_opposite_dir:
    :param smooth_surface_scalars:
    :param smooth_surface_scalars_gauss:
    :param smooth_only_cartilage_values:
    :param scalar_gauss_sigma:
    :param scalar_smooth_max_dist:
    :param scalar_smooth_order:
    :param bone_pyacvd_subdivisions:
    :param bone_pyacvd_clusters:
    :param crop_bones:
    :param crop_percent:
    :param bone:
    :param tmp_cropped_image_filename:
    :param loc_t2_map_nrrd:.
    :param t2_map_filename:
    :param t2_smooth_sigma_multiple_of_thick:
    :param assign_seg_label_to_bone:
    :param multiple_cart_labels_separate:
    :param mc_threshold:
    :return:

    Notes:
    multiple_cart_labels_separate REMOVED from the function.
    """
    # Read segmentation image
    seg_image = sitk.ReadImage(os.path.join(loc_nrrd_images, seg_image_name))
    seg_image = set_seg_border_to_zeros(seg_image, border_size=1)

    seg_view = sitk.GetArrayViewFromImage(seg_image)
    n_pixels_labelled = sum(seg_view[seg_view == bone_label])

    if n_pixels_labelled < bone_label_threshold:
        raise Exception('The bone does not exist in this segmentation!, only {} pixels detected, threshold # is {}'.format(n_pixels_labelled, 
                                                                                                                           bone_label_threshold))
    
    # Read segmentation in vtk format if going to assign labels to surface.
    # Also, if femur break it up into its parts.
    if assign_seg_label_to_bone is True:
        tmp_filename = ''.join(random.choice(string.ascii_lowercase) for i in range(10)) + '.nrrd'
        if bone == 'femur':
            new_seg_image = qc.get_knee_segmentation_with_femur_subregions(seg_image,
                                                                           fem_cart_label_idx=1)
            sitk.WriteImage(new_seg_image, os.path.join('/tmp', tmp_filename))
        else:
            sitk.WriteImage(seg_image, os.path.join('/tmp', tmp_filename))
        vtk_seg_reader = read_nrrd('/tmp',
                                   tmp_filename,
                                   set_origin_zero=True
                                   )
        vtk_seg = vtk_seg_reader.GetOutput()

        seg_transformer = SitkVtkTransformer(seg_image)

        # Delete tmp files
        safely_delete_tmp_file('/tmp',
                               tmp_filename)

    # Crop the bones if that's an option/thing.
    if crop_bones is True:
        if 'femur' in bone:
            bone_crop_distal = True
        elif 'tibia' in bone:
            bone_crop_distal = False
        else:
            raise Exception('var bone should be "femur" or "tiba" got: {} instead'.format(bone))

        seg_image = crop_bone_based_on_width(seg_image,
                                             bone_label,
                                             percent_width_to_crop_height=crop_percent,
                                             bone_crop_distal=bone_crop_distal)

    if verbose is True:
        tic = time.time()

    # Create bone mesh/smooth/resample surface points.
    ns_bone_mesh = BoneMesh(seg_image=seg_image,
                            label_idx=bone_label)
    if verbose is True:
        print('Loaded mesh')
    ns_bone_mesh.create_mesh(smooth_image=True,
                             smooth_image_var=image_smooth_var,
                             marching_cubes_threshold=mc_threshold
                             )
    if verbose is True:
       print('Smoothed bone surface')
    ns_bone_mesh.resample_surface(subdivisions=bone_pyacvd_subdivisions,
                                  clusters=bone_pyacvd_clusters)
    if verbose is True:
       print('Resampled surface')
    n_bone_points = ns_bone_mesh._mesh.GetNumberOfPoints()

    if verbose is True:
        toc = time.time()
        print('Creating bone mesh took: {}'.format(toc - tic))
        tic = time.time()

    # Pre-allocate empty arrays for t2/labels if they are being placed on surface.
    if assign_seg_label_to_bone is True:
        # Apply inverse transform to get it into the space of the image.
        # This is easier than the reverse function.
        if assign_seg_label_to_bone is True:
            ns_bone_mesh.apply_transform_to_mesh(transform=seg_transformer.get_inverse_transform())

            labels = np.zeros(n_bone_points, dtype=np.int)

    thicknesses = np.zeros(n_bone_points, dtype=np.float)
    if verbose is True:
       print('Number bone mesh points: {}'.format(n_bone_points))

    # Iterate over cartilage labels
    # Create mesh & store thickness + cartilage label + t2 in arrays
    for cart_label_idx in list_cart_labels:
        # Test to see if this particular cartilage label even exists in the label :P
        # This is important for people that may have no cartilage (of a particular type)
        seg_array_view = sitk.GetArrayViewFromImage(seg_image)
        n_pixels_with_cart = np.sum(seg_array_view == cart_label_idx)
        if n_pixels_with_cart == 0:
            print("Not analyzing cartilage for label {} because it doesnt have any pixels!".format(cart_label_idx))
            continue

        ns_cart_mesh = CartilageMesh(seg_image=seg_image,
                                     label_idx=cart_label_idx)
        ns_cart_mesh.create_mesh(smooth_image=smooth_cart,
                                 smooth_image_var=image_smooth_var_cart,
                                 marching_cubes_threshold=mc_threshold)

        # Perform Thickness & label simultaneously. 

        if assign_seg_label_to_bone is True:
            ns_cart_mesh.apply_transform_to_mesh(transform=seg_transformer.get_inverse_transform())

        node_data = get_thickness_cartilage_at_points(ns_bone_mesh._mesh,
                                                      ns_cart_mesh._mesh,
                                                      t2_vtk_image=None,
                                                      seg_vtk_image=vtk_seg if assign_seg_label_to_bone is True else None,
                                                      ray_cast_length=ray_cast_length,
                                                      percent_ray_length_opposite_direction=percent_ray_len_opposite_dir
                                                      )
        if assign_seg_label_to_bone is False:
            thicknesses += node_data
        else:
            thicknesses += node_data[0]
            labels += node_data[1]

        if verbose is True:
            print('Cartilage label: {}'.format(cart_label_idx))
            print('Mean thicknesses (all): {}'.format(np.mean(thicknesses)))

    if verbose is True:
        toc = time.time()
        print('Calculating all thicknesses: {}'.format(toc - tic))
        tic = time.time()

    # Assign thickness & T2 data (if it exists) to bone surface.
    thickness_scalars = numpy_to_vtk(thicknesses)
    thickness_scalars.SetName('thickness (mm)')
    ns_bone_mesh._mesh.GetPointData().SetScalars(thickness_scalars)
    
    # Smooth surface scalars
    if smooth_surface_scalars is True:
        if smooth_only_cartilage_values is True:
            loc_cartilage = np.where(vtk_to_numpy(ns_bone_mesh._mesh.GetPointData().GetScalars())>0.01)[0]
            ns_bone_mesh.mesh = gaussian_smooth_surface_scalars(ns_bone_mesh.mesh,
                                                                sigma=scalar_gauss_sigma,
                                                                idx_coords_to_smooth=loc_cartilage)
        else:
            ns_bone_mesh.mesh = gaussian_smooth_surface_scalars(ns_bone_mesh.mesh, sigma=scalar_gauss_sigma)

        if verbose is True:
            toc = time.time()
            print('Smoothing scalars took: {}'.format(toc - tic))

    # Add the label values to the bone after smoothing is finished.
    if assign_seg_label_to_bone is True:
        label_scalars = numpy_to_vtk(labels)
        label_scalars.SetName('Cartilage Region')
        ns_bone_mesh._mesh.GetPointData().AddArray(label_scalars)

    if assign_seg_label_to_bone is True:
        # Transform bone back to the position it was in before rotating it (for the t2 analysis)
        ns_bone_mesh.reverse_all_transforms()

    return ns_bone_mesh.mesh


def get_smoothed_cartilage_t2_values(list_cart_labels,
                                     ns_bone_mesh=None,
                                     bone_mesh=None,
                                     loc_nrrd_images=None,
                                     seg_image_name=None,
                                     seg_image=None,
                                     smooth_cart=False,
                                     image_smooth_var_cart=1.0,
                                     ray_cast_length=10.,
                                     percent_ray_len_opposite_dir=0.2,
                                     smooth_surface_scalars=True,
                                     smooth_only_cartilage_values=True,
                                     scalar_gauss_sigma=1.6986436005760381,  # This is a FWHM = 4
                                     loc_t2_map_nrrd=None,
                                     t2_map_filename=None,
                                     mc_threshold=0.5,
                                     path_to_seg_transform=None,
                                     reverse_seg_transform=True,
                                     verbose=False,
                                     scalar_name='Mean T2 (ms)'):
    """

    :param loc_nrrd_images:
    :param seg_image_name:
    :param bone_label:
    :param list_cart_labels:
    :param image_smooth_var:
    :param loc_tmp_save:
    :param tmp_bone_filename:
    :param smooth_cart:
    :param image_smooth_var_cart:
    :param tmp_cart_filename:
    :param ray_cast_length:
    :param percent_ray_len_opposite_dir:
    :param smooth_surface_scalars:
    :param smooth_surface_scalars_gauss:
    :param smooth_only_cartilage_values:
    :param scalar_gauss_sigma:
    :param scalar_smooth_max_dist:
    :param scalar_smooth_order:
    :param bone_pyacvd_subdivisions:
    :param bone_pyacvd_clusters:
    :param crop_bones:
    :param crop_percent:
    :param bone:
    :param tmp_cropped_image_filename:
    :param loc_t2_map_nrrd:.
    :param t2_map_filename:
    :param t2_smooth_sigma_multiple_of_thick:
    :param assign_seg_label_to_bone:
    :param multiple_cart_labels_separate:
    :param mc_threshold:
    :return:

    Notes:
    multiple_cart_labels_separate REMOVED from the function.
    """
    # Read segmentation image
    # Need the segmentation image to apply inverse transform so that mesh
    # will be aligned with the un-rotated T2 data. 
    if (seg_image is None) & (seg_image_name is not None) & (loc_nrrd_images is not None):
        seg_image = sitk.ReadImage(os.path.join(loc_nrrd_images, seg_image_name))
    elif seg_image is None:
        raise Exception ('Must provide seg_image (sitk image) or location & name of nrrd file to read!')


    if (ns_bone_mesh is None) & (bone_mesh is not None):
        ns_bone_mesh = BoneMesh(mesh=bone_mesh)
    # Get seg_transformer before potentially applying any transformations to the image
    # This way, can apply a transformation to get seg to align with T2, and then can apply
    # inverse of this transformation to get the seg into the same space as the T2 image. 

    seg_transformer = SitkVtkTransformer(seg_image)

    # If transformation for segmentation provided, then apply it to the segmentation. 
    if path_to_seg_transform is not None:
        seg_transform = sitk.ReadTransform(path_to_seg_transform)
        seg_image = apply_transform_retain_array(seg_image, 
                                                 seg_transform, 
                                                 interpolator=sitk.sitkNearestNeighbor)

        try:
            versor = sitk.VersorRigid3DTransform(seg_transform)
        except RuntimeError:
            try:
                composite = sitk.CompositeTransform(seg_transform)
                if composite.GetNumberOfTransforms() == 1:
                    versor = sitk.VersorRigid3DTransform(composite.GetBackTransform())
                else:
                    raise Exception ('There is {} transforms in the composite transform, excpected 1!'.format(composite.GetNumberOfTransforms()))
            except error:
                raise Exception(error)

        center_of_rotation = versor.GetCenter()
        center_transform = vtk.vtkTransform()
        center_transform.Translate(center_of_rotation[0], center_of_rotation[1], center_of_rotation[2])

        seg_to_t2_rotate = vtk.vtkTransform()
        four_by_four = np.identity(4)
        four_by_four[:3,:3] = np.reshape(versor.GetMatrix(), (3,3))
        seg_to_t2_rotate.SetMatrix(four_by_four.flatten())

        seg_to_t2_translate = vtk.vtkTransform()
        seg_to_t2_translate.Translate(versor.GetTranslation())

        # If there is a transformation of the segmentation (to align with the t2) then apply
        # if to the bone mesh. 

        # First, apply negative of center of rotation to mesh: 
        ns_bone_mesh.apply_transform_to_mesh(transform=center_transform.GetInverse())
        # Then apply the transformation itself. 
        ns_bone_mesh.apply_transform_to_mesh(transform=seg_to_t2_rotate.GetInverse())
        ns_bone_mesh.apply_transform_to_mesh(transform=seg_to_t2_translate.GetInverse())
        # Then undo the center of rotation stuff
        ns_bone_mesh.apply_transform_to_mesh(transform=center_transform)

    # Read t2 map (vtk format) if going to assign T2 to surface.
    vtk_t2map_reader = read_nrrd(loc_t2_map_nrrd,
                                 t2_map_filename,
                                 set_origin_zero=True)
    vtk_t2map = vtk_t2map_reader.GetOutput()
    sitk_t2map = sitk.ReadImage(os.path.join(loc_t2_map_nrrd, t2_map_filename))

    t2_transformer = SitkVtkTransformer(sitk_t2map)

    tic = time.time()

    # Pre-allocate empty arrays for t2
    # Apply inverse transform to get it into the space of the image.
    # This is easier than the reverse function (transforming vtkImageData - this is because probefilter
    # is slow on vtkStructuredGrid, which is what is returned after applying transform to vtkImageData)

    # now undoing the t2 transform - seg "should" now be aligned with t2 image data
    ns_bone_mesh.apply_transform_to_mesh(transform=t2_transformer.get_inverse_transform())
    n_bone_points = ns_bone_mesh._mesh.GetNumberOfPoints()
    t2s = np.zeros(n_bone_points, dtype=np.float)

    # Iterate over cartilage labels
    # Create mesh & store thickness + cartilage label + t2 in arrays
    # ADD OPTION TO ITERATE OVER LIST OF CARTILAGE MESHES, INSTEAD OF MAKING NEW ONES! - if they are provided (optional?)
    for cart_label_idx in list_cart_labels:
        # Test to see if this particular cartilage label even exists in the label :P
        # This is important for people that may have no cartilage (of a particular type)
        seg_array_view = sitk.GetArrayViewFromImage(seg_image)
        n_pixels_with_cart = np.sum(seg_array_view == cart_label_idx)
        if n_pixels_with_cart == 0:
            print("Not analyzing cartilage for label {} because it doesnt have any pixels!".format(cart_label_idx))
            continue

        ns_cart_mesh = CartilageMesh(seg_image=seg_image,
                                     label_idx=cart_label_idx)
        ns_cart_mesh.create_mesh(smooth_image=smooth_cart,
                                 smooth_image_var=image_smooth_var_cart,
                                 marching_cubes_threshold=mc_threshold)

        # Perform Thickness & label simultaneously. 

        # rotate cartilage back to origin area to be able to extract t2 data properly. 
        # Instead of applying known rotation to T2, apply opposite to the stuff it is being rotated towards.
        ns_cart_mesh.apply_transform_to_mesh(transform=t2_transformer.get_inverse_transform())
        t2_data = get_t2_cartilage_at_points(surface_bone=ns_bone_mesh._mesh,
                                             surface_cartilage=ns_cart_mesh._mesh,
                                             t2_vtk_image=vtk_t2map,
                                             ray_cast_length=ray_cast_length,
                                             percent_ray_length_opposite_direction=percent_ray_len_opposite_dir)

        t2s += t2_data

        # Perform T2 after (or separate) from thickness/label. 
        if verbose is True:
            print('Cartilage label: {}'.format(cart_label_idx))
            print('Mean T2 (all): {}'.format(np.mean(t2s)))
    
    toc = time.time()
    if verbose is True:
        print('Calculating all T2: {}'.format(toc - tic))
    tic = time.time()

    # Assign thickness & T2 data (if it exists) to bone surface.
    t2_scalars = numpy_to_vtk(t2s)
    t2_scalars.SetName(scalar_name)

    bone_mesh_copy = vtk_deep_copy(ns_bone_mesh._mesh)

    # Used to provide multiple sigma values to smoothing operaton b/c smoothed multiple things (thickness & t2)
    # Instead, just remove all other scalars from this mesh before doing smoothing & provide single value.  
    for idx in list(range(bone_mesh_copy.GetPointData().GetNumberOfArrays()))[::-1]:
        bone_mesh_copy.GetPointData().RemoveArray(idx)

    bone_mesh_copy.GetPointData().AddArray(t2_scalars)

    # Smooth surface scalars
    if smooth_surface_scalars is True:
        if smooth_only_cartilage_values is True:
            # The next line assumes that the active scalars on the bone_mesh_copy are either: 
            #   thickness data, t2 data, or cartialge regions. All of which should have values >0 where 
            #   there is cartilage, or 0s elsewhere. 
            loc_cartilage = np.where(t2s > 0.01)[0]
            bone_mesh_copy = gaussian_smooth_surface_scalars(bone_mesh_copy,
                                                             sigma=scalar_gauss_sigma,
                                                             idx_coords_to_smooth=loc_cartilage)
        else:
            bone_mesh_copy = gaussian_smooth_surface_scalars(bone_mesh_copy, 
                                                             sigma=scalar_gauss_sigma)
        toc = time.time()
        if verbose is True:
            print('Smoothing scalars took: {}'.format(toc - tic))

    ns_bone_mesh._mesh.GetPointData().AddArray(bone_mesh_copy.GetPointData().GetArray(scalar_name))

    if reverse_seg_transform is True:
        ns_bone_mesh.reverse_all_transforms()

    return ns_bone_mesh.mesh
