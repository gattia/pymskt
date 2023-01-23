
import pytest
import numpy as np
import pymskt as mskt
from pymskt.utils import testing
import SimpleITK as sitk

SEG_IMAGE_PATH = 'data/right_knee_example.nrrd'
SEG_IMAGE = sitk.ReadImage(SEG_IMAGE_PATH)
MESH_FEMUR_CROPPED = mskt.mesh.io.read_vtk('data/femur_cropped_cartilage_thick_roi_full_pts.vtk')
MESH_TIBIA_CROPPED = mskt.mesh.io.read_vtk('data/tibia_smoothed_image_cropped.vtk')

from pymskt import RTOL, ATOL

# MESH_WITH_SMOOTHING = mskt.mesh.io.read_vtk('data/femur_mesh_orig.vtk')

# Testing create mesh no smoothing
def test_create_mesh_apply_crop_femur(
    seg_image=SEG_IMAGE,
    ref_mesh=MESH_FEMUR_CROPPED,
    label_idx=5,
    crop_percent=0.7,
    bone='femur'
):
    mesh = mskt.mesh.BoneMesh(
        seg_image=seg_image, 
        label_idx=label_idx,
        bone=bone
    )
    mesh.create_mesh(
        smooth_image=True,
        crop_percent=crop_percent,
    )

    testing.assert_mesh_coordinates_same(ref_mesh, mesh.mesh, rtol=RTOL, atol=ATOL)

def test_create_mesh_apply_crop_tibia(
    seg_image=SEG_IMAGE,
    ref_mesh=MESH_TIBIA_CROPPED,
    label_idx=6,
    crop_percent=0.7,
    bone='tibia'
):
    mesh = mskt.mesh.BoneMesh(
        seg_image=seg_image, 
        label_idx=label_idx,
        bone=bone
    )
    mesh.create_mesh(
        smooth_image=True,
        crop_percent=crop_percent,
    )

    testing.assert_mesh_coordinates_same(ref_mesh, mesh.mesh, rtol=RTOL, atol=ATOL)

def test_create_mesh_apply_crop_exception_because_wrong_bone(
    seg_image=SEG_IMAGE,
    ref_mesh=MESH_TIBIA_CROPPED,
    label_idx=6,
    crop_percent=0.7,
    bone='test'
):
    mesh = mskt.mesh.BoneMesh(
        seg_image=seg_image, 
        label_idx=label_idx,
        bone=bone
    )
    
    with pytest.raises(Exception):
        mesh.create_mesh(
            smooth_image=True,
            crop_percent=crop_percent,
        )

