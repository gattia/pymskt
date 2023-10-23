import pytest
import numpy as np
import pymskt as mskt
from pymskt.utils import testing
import SimpleITK as sitk

SEG_IMAGE_PATH = 'data/right_knee_example.nrrd'
SEG_IMAGE = sitk.ReadImage(SEG_IMAGE_PATH)
MESH_NO_SMOOTHING = mskt.mesh.io.read_vtk('data/femur_orig_no_image_smoothing.vtk')
MESH_WITH_SMOOTHING = mskt.mesh.io.read_vtk('data/femur_mesh_orig.vtk')

from pymskt import RTOL, ATOL

# Testing create mesh no smoothing
def test_pass_seg_image_directly_no_smoothing_pass_label_idx_to_init(
    seg_image=SEG_IMAGE,
    ref_mesh=MESH_NO_SMOOTHING,
    label_idx=5
):
    mesh = mskt.mesh.Mesh(seg_image=seg_image, label_idx=label_idx)
    mesh.create_mesh(smooth_image=False)

    testing.assert_mesh_coordinates_same(ref_mesh, mesh.mesh, rtol=RTOL, atol=ATOL)

def test_pass_seg_image_directly_no_smoothing_pass_label_idx_to_create_mesh_function(
    seg_image=SEG_IMAGE,
    ref_mesh=MESH_NO_SMOOTHING,
    label_idx=5
):
    mesh = mskt.mesh.Mesh(seg_image=seg_image)
    mesh.create_mesh(smooth_image=False, label_idx=label_idx)

    testing.assert_mesh_coordinates_same(ref_mesh, mesh.mesh, rtol=RTOL, atol=ATOL)

# Testing create mesh with smoothing
def test_pass_seg_image_directly_with_smoothing_and_pass_label_idx_to_init(
    seg_image=SEG_IMAGE,
    ref_mesh=MESH_WITH_SMOOTHING,
    label_idx=5,
    smooth_image_var=0.3125 / 2
):
    mesh = mskt.mesh.Mesh(seg_image=seg_image, label_idx=label_idx)
    mesh.create_mesh(smooth_image=True, smooth_image_var=smooth_image_var)

    testing.assert_mesh_coordinates_same(ref_mesh, mesh.mesh, rtol=RTOL, atol=ATOL)

def test_pass_seg_image_directly_with_smoothing_and_pass_label_idx_to_create_mesh_function(
    seg_image=SEG_IMAGE,
    ref_mesh=MESH_WITH_SMOOTHING,
    label_idx=5,
    smooth_image_var=0.3125 / 2
):
    mesh = mskt.mesh.Mesh(seg_image=seg_image)
    mesh.create_mesh(smooth_image=True, label_idx=label_idx, smooth_image_var=smooth_image_var)

    testing.assert_mesh_coordinates_same(ref_mesh, mesh.mesh, rtol=RTOL, atol=ATOL)

def test_exception_n_pixels_less_than_min_n_pixels(
    seg_image=SEG_IMAGE, 
    label_idx=5
):  
    array = sitk.GetArrayFromImage(seg_image)
    n_pixels = np.sum(array[array == label_idx])

    with pytest.raises(Exception):
        mesh = mskt.mesh.Mesh(seg_image=seg_image, label_idx=label_idx)
        mesh.create_mesh(min_n_pixels=n_pixels + 1)

def test_exception_seg_image_provided_but_no_label_idx(
    seg_image=SEG_IMAGE,
):
    with pytest.raises(Exception):
        mesh = mskt.mesh.Mesh(seg_image=seg_image)
        mesh.create_mesh()

def test_exception_no_seg_image_or_path_to_seg_image_provided_and_smooth_image_true():
    with pytest.raises(Exception):
        mesh = mskt.mesh.Mesh(smooth_image=True)
        mesh.create_mesh()

def test_exception_no_seg_image_or_path_to_seg_image_provided_and_smooth_image_false():
    with pytest.raises(Exception):
        mesh = mskt.mesh.Mesh(smooth_image=False)
        mesh.create_mesh()

def test_load_seg_image_if_not_already_loaded(
    path_seg_image=SEG_IMAGE_PATH,
    ref_mesh=MESH_NO_SMOOTHING,
    label_idx=5
):
    mesh = mskt.mesh.Mesh(path_seg_image=path_seg_image, label_idx=label_idx)
    mesh.create_mesh(smooth_image=False)

    testing.assert_mesh_coordinates_same(ref_mesh, mesh.mesh, rtol=RTOL, atol=ATOL)    

@pytest.mark.skip(reason="Test not implemented yet")
def test_specify_label_idx():
    pass  # TODO: Implement this test

@pytest.mark.skip(reason="Test not implemented yet")
def test_marching_cubes_threshold_0():
    pass  # TODO: Implement this test

@pytest.mark.skip(reason="Test not implemented yet")
def test_marching_cubes_threshold_1():
    pass  # TODO: Implement this test