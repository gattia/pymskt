import os
import vtk
import SimpleITK as sitk

import pymskt.image as msktimage
import pymskt.mesh.meshTransform as meshTransform
from pymskt.utils import safely_delete_tmp_file

def discrete_marching_cubes(vtk_image_reader,
                            n_labels=1,
                            start_label=1,
                            end_label=1,
                            compute_normals_on=True,
                            return_polydata=True
                            ):
    """
    Compute dmc on segmentation image.
    Creates a surface mesh (polydata) that closely covers binary (discrete) segmentations.

    Parameters
    ----------
    vtk_image_reader : vtk.Filter
        VTK pipeline to apply discrete marching cubes to. 
    n_labels : int, optional
        Number of labes to create mesh for, by default 1
    start_label : int, optional
        Starting index of labels to mesh, by default 1
    end_label : int, optional
        Ending index of labels to mesh, by default 1
    compute_normals_on : bool, optional
        Calculate normals to surface, by default True
    return_polydata : bool, optional
        Whether to return a vtk.polydata or not (VTK filter pipeline instead e.g., `dmc`), by default True

    Returns
    -------
    vtk.Filter Pipeline
        Returns a pipeline which more functions can be chained too - this improves performance.
    
    OR

    vtk.Polydata
        Returns a polydata (surface mesh). 

    """    

    dmc = vtk.vtkDiscreteMarchingCubes()
    dmc.SetInputConnection(vtk_image_reader.GetOutputPort())
    if compute_normals_on is True:
        dmc.ComputeNormalsOn()
    dmc.GenerateValues(n_labels, start_label, end_label)
    dmc.Update()

    if return_polydata is True:
        return dmc.GetOutput()
    elif return_polydata is False:
        return dmc


def continuous_marching_cubes(vtk_image_reader, 
                              threshold=0.5,
                              compute_normals_on=True,
                              compute_gradients_on=True,
                              return_polydata=True):
    """
    - Compute a continuous marching cubes on a segmentation mask. 
    - Enables defining the surface based on a contour set to a floating point cutoff. 


    Parameters
    ----------
    vtk_image_reader : vtk.Filter
        This is the output of a previous vtk filter from a previous step. E.g., output of pymskt.image.read_nrrd().
        
    threshold : float, optional
        Floating point value to create surface mesh, by default 0.5
    compute_normals_on : bool, optional
        Whether or not to compute surface normals for mesh, by default True
    compute_gradients_on : bool, optional
        Whether or not to compute gradients over mesh surface, by default True
    return_polydata : bool, optional
        Whether to return a vtk.polydata or not (VTK filter pipeline instead e.g., `mc`), by default True

    Returns
    -------
    vtk.Filter Pipeline
        Returns a pipeline which more functions can be chained too - this improves performance.
    
    OR

    vtk.Polydata
        Returns a polydata (surface mesh). 
    """    
    mc = vtk.vtkMarchingContourFilter()
    mc.SetInputConnection(vtk_image_reader.GetOutputPort())
    if compute_normals_on is True:
        mc.ComputeNormalsOn()
    elif compute_normals_on is False:
        mc.ComputeNormalsOff()
    
    if compute_gradients_on is True:
        mc.ComputeGradientsOn()
    elif compute_gradients_on is False:
        mc.ComputeGradientsOff()
    mc.SetValue(0, threshold)
    mc.Update()
    
    if return_polydata is True:
        mesh = mc.GetOutput()
        return mesh
    elif return_polydata is False:
        return mc

def create_surface_mesh_smoothed(seg_image,
                                 label_idx,
                                 image_smooth_var,
                                 loc_tmp_save='/tmp',
                                 tmp_filename='temp_smoothed_bone.nrrd',
                                 copy_image_transform=True,
                                 mc_threshold=0.5):
    """
    Create surface mesh based on a filtered binary image to try and get smoother surface representation

    Parameters
    ----------
    seg_image : SimpleITK.Image
        Segmentation image to be filtered and meshed with marching cubes. 
    label_idx : int
        What anatomical label to be meshed.
    image_smooth_var : float
        Variance to apply a gaussian smoothing function to. 
    loc_tmp_save : str, optional
        Location to save temporary files for passing SimpleITK.Image to vtk functions, by default '/tmp'
    tmp_filename : str, optional
        Filename of saved temporary file, by default 'temp_smoothed_bone.nrrd'
    copy_image_transform : bool, optional
        Whether or not to apply image transform to final mesh or to leave it at origin, by default True
    mc_threshold : float, optional
        What floating point value to create surface mesh at, by default 0.5

    Returns
    -------
    vtk.Polydata
        Surface mesh created using a continuous cutoff `mc_threshold` after applying 
        gaussian smoothing with variance = `image_smooth_var`.
    """    

    # Set border of segmentation to 0 so that segs are all closed.
    seg_image = msktimage.set_seg_border_to_zeros(seg_image, border_size=1)

    # smooth/filter the image to get a better surface. 
    filtered_image = msktimage.smooth_image(seg_image, label_idx, image_smooth_var)
    # save filtered image to disk so can read it in using vtk nrrd reader
    sitk.WriteImage(filtered_image, os.path.join(loc_tmp_save, tmp_filename))
    smoothed_nrrd_reader = msktimage.read_nrrd(os.path.join(loc_tmp_save, tmp_filename),
                                               set_origin_zero=True)
    # create the mesh using continuous marching cubes applied to the smoothed binary image. 
    smooth_mesh = continuous_marching_cubes(smoothed_nrrd_reader, threshold=mc_threshold)
    
    if copy_image_transform is True:
        # copy image transofrm to the image to the mesh so that when viewed (e.g. in 3D Slicer) it is aligned with image
        smooth_mesh = meshTransform.copy_image_transform_to_mesh(smooth_mesh, seg_image)

    # Delete tmp files
    safely_delete_tmp_file(loc_tmp_save,
                           tmp_filename)
    return smooth_mesh