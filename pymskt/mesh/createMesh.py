import logging
import os
import tempfile

import numpy as np
import SimpleITK as sitk
import vtk

import pymskt.image as msktimage
import pymskt.mesh.meshTransform as meshTransform
from pymskt.utils import safely_delete_tmp_file

logger = logging.getLogger(__name__)

# Smallest x extent (voxels) that vtkNrrdReader reads back as a 3D scalar image; see `_crop_to_label`.
_MIN_CROP_WIDTH_X = 10


def discrete_marching_cubes(
    vtk_image_reader,
    n_labels=1,
    start_label=1,
    end_label=1,
    compute_normals_on=True,
    return_polydata=True,
):
    """
    Compute dmc on segmentation image.
    Creates a surface mesh (polydata) that closely covers binary (discrete) segmentations.

    Parameters
    ----------
    vtk_image_reader : vtk.Filter
        VTK Filter pipeline to apply discrete marching cubes to.
    n_labels : int, optional
        Number of labes to create mesh for, by default 1
    start_label : int, optional
        Starting index of labels to mesh, by default 1
    end_label : int, optional
        Ending index of labels to mesh, by default 1
    compute_normals_on : bool, optional
        Calculate normals to surface, by default True
    return_polydata : bool, optional
        Whether to return a vtk.polydata or not (`vtk.Filter` pipeline instead), by default True

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


def continuous_marching_cubes(
    vtk_image_reader,
    threshold=0.5,
    compute_normals_on=True,
    compute_gradients_on=True,
    return_polydata=True,
):
    """
    - Compute a continuous marching cubes on a segmentation mask.
    - Enables defining the surface based on a contour set to a floating point cutoff.


    Parameters
    ----------
    vtk_image_reader : vtk.Filter
        This is the output of a `vtk.Filter` from a previous step. E.g., output of pymskt.image.read_nrrd().

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


def flying_edges_surface_extraction(vtk_image_reader, threshold=0.5):
    """
    Extract surface using flying edges method.
    """
    fe = vtk.vtkFlyingEdges3D()
    fe.SetInputConnection(vtk_image_reader.GetOutputPort())
    fe.SetValue(0, threshold)
    fe.Update()
    return fe.GetOutput()


def _crop_to_label(image, label_idx, image_smooth_var=None, extra=2):
    """
    Crop `image` to the bounding box of `label_idx` plus a margin that covers the
    Gaussian kernel used by `smooth_image`.

    Meshing the crop gives the same surface as meshing the whole volume. The binarized
    label is zero outside its bounding box, and ITK's `DiscreteGaussianImageFilter` uses
    zero-flux Neumann boundaries, so with a margin of at least one voxel the crop's outer
    layers are zero and every smoothed value inside the crop equals the full-volume value.
    Voxels outside the crop are further than the kernel's support from the label, so their
    smoothed value is exactly zero and no iso-surface can pass through them for any
    threshold above zero. Where the box is clipped to the image, the crop edge is the
    image edge, so the boundary handling is the same in both runs.

    Parameters
    ----------
    image : SimpleITK.Image
        Segmentation image.
    label_idx : int
        Label to crop around.
    image_smooth_var : float, optional
        Variance (mm^2) of the Gaussian that will be applied to the crop, or ``None`` if
        the image will not be smoothed. Sets the margin to ``ceil(4 * sigma)`` voxels per
        axis, which exceeds the filter's kernel truncation (max error 0.01, max width 32).
    extra : int, optional
        Voxels added to the margin on every side, by default 2.

    Returns
    -------
    SimpleITK.Image
        The cropped image, at least `_MIN_CROP_WIDTH_X` voxels wide along x; SimpleITK
        slicing keeps the sub-volume's physical origin, spacing and direction.
    """
    array = sitk.GetArrayViewFromImage(image)  # (z, y, x)
    mask = array == label_idx
    lo = np.empty(3, dtype=int)
    hi = np.empty(3, dtype=int)
    for axis in range(3):
        other_axes = tuple(a for a in range(3) if a != axis)
        occupied = np.flatnonzero(mask.any(axis=other_axes))
        if occupied.size == 0:
            raise ValueError(f"label {label_idx} is not present in the image")
        lo[axis], hi[axis] = occupied[0], occupied[-1] + 1

    margin = np.full(3, extra, dtype=int)
    if image_smooth_var is not None and image_smooth_var > 0:
        spacing_zyx = np.asarray(image.GetSpacing(), dtype=float)[::-1]
        sigma_vox = np.sqrt(image_smooth_var) / spacing_zyx
        margin += np.ceil(4 * sigma_vox).astype(int)

    lo = np.maximum(lo - margin, 0)
    hi = np.minimum(hi + margin, array.shape)

    # vtkNrrdReader reads the first (x) axis of a NRRD as vector components when it has
    # fewer than 10 samples, which would turn a thin crop into a 2D image. Widening the
    # crop never changes the mesh, so keep x at least that wide (clipped to the image).
    if hi[2] - lo[2] < _MIN_CROP_WIDTH_X:
        lo[2] = max(lo[2] - (_MIN_CROP_WIDTH_X - (hi[2] - lo[2])) // 2, 0)
        hi[2] = min(lo[2] + _MIN_CROP_WIDTH_X, array.shape[2])
        lo[2] = max(hi[2] - _MIN_CROP_WIDTH_X, 0)

    logger.debug(
        "crop_to_label: label %s box (z, y, x) %s:%s = %.1f%% of the volume",
        label_idx,
        lo.tolist(),
        hi.tolist(),
        100 * np.prod(hi - lo) / array.size,
    )
    # SimpleITK slices in (x, y, z) order.
    return image[int(lo[2]) : int(hi[2]), int(lo[1]) : int(hi[1]), int(lo[0]) : int(hi[0])]


def create_surface_mesh(
    seg_image,
    label_idx,
    image_smooth_var,
    loc_tmp_save=tempfile.gettempdir(),
    tmp_filename="temp_smoothed_bone.nrrd",
    copy_image_transform=True,
    mc_threshold=0.5,
    filter_binary_image=True,
    set_seg_border_to_zeros=True,
    surface_extraction_method="continuous_marching_cubes",
    crop_to_label=True,
    # use_discrete_marching_cubes=False,
):
    """
    Create surface mesh.
    Option to filter binary image to get smoother surface representation.

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
    filter_binary_image : bool, optional
        Should the binary image be filtered (smoothed) or not.
    crop_to_label : bool, optional
        Smooth and mesh only the bounding box of `label_idx` (plus a margin covering the
        Gaussian kernel) instead of the whole volume, by default True. The mesh is the same
        (see `_crop_to_label`); this only avoids filtering and contouring the empty part of
        the image, which is most of it for a single knee structure.

    Returns
    -------
    vtk.Polydata
        Surface mesh created using a continuous cutoff `mc_threshold` after applying
        gaussian smoothing with variance = `image_smooth_var`.
    """

    # Set border of segmentation to 0 so that segs are all closed.
    if set_seg_border_to_zeros is True:
        seg_image = msktimage.set_seg_border_to_zeros(seg_image, border_size=1)

    smooth = (surface_extraction_method != "discrete_marching_cubes") and (
        filter_binary_image is not False
    )

    if crop_to_label is True:
        # The crop keeps its physical origin/direction, so `copy_image_transform_to_mesh`
        # below places the mesh in world coordinates without further bookkeeping.
        seg_image = _crop_to_label(
            seg_image, label_idx, image_smooth_var=image_smooth_var if smooth else None
        )

    if smooth is False:
        seg_image = msktimage.binarize_segmentation_image(seg_image, label_idx)
    else:
        seg_image = msktimage.smooth_image(seg_image, label_idx, image_smooth_var)

    # save filtered image to disk so can read it in using vtk nrrd reader
    sitk.WriteImage(seg_image, os.path.join(loc_tmp_save, tmp_filename))
    nrrd_reader = msktimage.read_nrrd(
        os.path.join(loc_tmp_save, tmp_filename), set_origin_zero=True
    )
    # create the mesh using continuous marching cubes applied to the smoothed binary image.
    if surface_extraction_method == "discrete_marching_cubes":
        mesh = discrete_marching_cubes(
            nrrd_reader,
            n_labels=1,
            start_label=1,
            end_label=1,
            compute_normals_on=True,
            return_polydata=True,
        )
    elif surface_extraction_method == "continuous_marching_cubes":
        mesh = continuous_marching_cubes(nrrd_reader, threshold=mc_threshold)
    elif surface_extraction_method == "flying_edges":
        mesh = flying_edges_surface_extraction(nrrd_reader, threshold=mc_threshold)
    else:
        raise ValueError(f"Invalid surface extraction method: {surface_extraction_method}")

    if copy_image_transform is True:
        # copy image transofrm to the image to the mesh so that when viewed (e.g. in 3D Slicer) it is aligned with image
        mesh = meshTransform.copy_image_transform_to_mesh(mesh, seg_image)

    # Delete vtk reader - to ensure we can delete the tmp file
    del nrrd_reader

    # Delete tmp files
    safely_delete_tmp_file(loc_tmp_save, tmp_filename)
    return mesh
