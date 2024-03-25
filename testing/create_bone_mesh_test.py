from operator import sub

import pytest
import SimpleITK as sitk
from numpy.testing import assert_allclose

import pymskt as mskt
from pymskt import ATOL, RTOL

try:
    orig_femur_mesh = mskt.mesh.io.read_vtk("data/femur_mesh_orig.vtk")
    downsampled_femur_mesh = mskt.mesh.io.read_vtk("data/femur_thickness_mm_10k_pts.vtk")
except OSError:
    orig_femur_mesh = mskt.mesh.io.read_vtk("../data/femur_mesh_orig.vtk")
    downsampled_femur_mesh = mskt.mesh.io.read_vtk("../data/femur_thickness_mm_10k_pts.vtk")


def assert_images_same(image1, image2):
    """
    Helper function to assert that 2 SimpleITK images are the same.

    Parameters
    ----------
    image1 : SimpleITK.Image
        Version 1 of the image.
    image2 : SimpleITK.Image
        Version 2 of the image.
    """
    image1_array = sitk.GetArrayFromImage(image1)
    image2_array = sitk.GetArrayFromImage(image2)

    assert_allclose(image1_array, image2_array, rtol=RTOL, atol=ATOL)
    assert image1.GetOrigin() == image2.GetOrigin()
    assert image1.GetSpacing() == image2.GetSpacing()
    assert image1.GetDirection() == image2.GetDirection()


def test_creating_bone(timing=False, verbose=False):
    """
    Create femur mesh & compare its point coordinates to an already saved mesh.

    Parameters
    ----------
    timing : bool, optional
        Should the function be timed?, by default False
    verbose : bool, optional
        Should we print out results?, by default False
    """

    try:
        femur = mskt.mesh.BoneMesh(path_seg_image="data/right_knee_example.nrrd", label_idx=5)
    except OSError:
        femur = mskt.mesh.BoneMesh(path_seg_image="../data/right_knee_example.nrrd", label_idx=5)

    femur.create_mesh()
    femur_pts = femur.point_coords

    orig_femur_pts = mskt.mesh.get_mesh_physical_point_coords(orig_femur_mesh)

    assert_allclose(orig_femur_pts, femur_pts, rtol=RTOL, atol=ATOL)


@pytest.mark.skip(reason="Different results on different machines")
def test_resampling_bone(timing=False, verbose=False):
    """
    Test resampling of surface mesh using pyacvd

    Parameters
    ----------
    timing : bool, optional
        Should the function be timed?, by default False
    verbose : bool, optional
        Should we print out results?, by default False
    """
    try:
        femur = mskt.mesh.BoneMesh(path_seg_image="data/right_knee_example.nrrd", label_idx=5)
    except OSError:
        femur = mskt.mesh.BoneMesh(path_seg_image="../data/right_knee_example.nrrd", label_idx=5)

    # READ IN FEMUR MESH THAT IS ALREADY CREATED BUT NOT RESAMPLED! ONLY APPLY RESAMPLING

    femur.create_mesh()
    femur.resample_surface(subdivisions=2, clusters=10000)
    femur_pts = femur.point_coords

    downsampled_femur_pts = mskt.mesh.get_mesh_physical_point_coords(downsampled_femur_mesh)

    assert_allclose(downsampled_femur_pts, femur_pts, rtol=RTOL, atol=ATOL)


if __name__ == "__main__":
    import time

    test_creating_bone(timing=True, verbose=True)
