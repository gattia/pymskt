import numpy as np
import pytest
import SimpleITK as sitk
from numpy.testing import assert_allclose

import pymskt as mskt

FEMUR_CARTILAGE_MESH = mskt.mesh.io.read_vtk(
    "data/femur_cart_smoothed_binary_no_surface_resampling.vtk"
)
MED_TIB_CARTILAGE_MESH = mskt.mesh.io.read_vtk(
    "data/med_tib_cart_smoothed_binary_no_surface_resampling.vtk"
)
LAT_TIB_CARTILAGE_MESH = mskt.mesh.io.read_vtk(
    "data/lat_tib_cart_smoothed_binary_no_surface_resampling.vtk"
)

from pymskt import ATOL, RTOL


# LIST CARTIALGE MESHES
def test_set_list_cartilage_meshes(
    fem_cart_mesh=FEMUR_CARTILAGE_MESH,
    med_tib_cart_mesh=MED_TIB_CARTILAGE_MESH,
    lat_tib_cart_mesh=LAT_TIB_CARTILAGE_MESH,
):
    list_cart_meshes = [fem_cart_mesh, med_tib_cart_mesh, lat_tib_cart_mesh]

    list_cart_meshes = [mskt.mesh.CartilageMesh(mesh=mesh_) for mesh_ in list_cart_meshes]
    mesh = mskt.mesh.BoneMesh()
    mesh.list_cartilage_meshes = list_cart_meshes

    for mesh_idx, cart_mesh in enumerate(mesh._list_cartilage_meshes):
        rand_sample = np.random.randint(0, cart_mesh.mesh.GetNumberOfPoints(), 256)
        for rand_idx in rand_sample:
            assert_allclose(
                list_cart_meshes[mesh_idx].mesh.GetPoint(rand_idx),
                cart_mesh.mesh.GetPoint(rand_idx),
                rtol=RTOL,
                atol=ATOL,
            )


def test_set_list_cartilage_meshes_exception_items_in_list_not_CartilageMesh(
    list_cartilage_meshes=[1.0, 2.0, 3.0]
):
    mesh = mskt.mesh.BoneMesh()
    with pytest.raises(TypeError):
        mesh.list_cartilage_meshes = list_cartilage_meshes


def test_set_list_cartilage_meshes_cartilage_mesh_provided_directly_not_list(
    fem_cart_mesh=FEMUR_CARTILAGE_MESH,
):
    cart_mesh = mskt.mesh.CartilageMesh(mesh=fem_cart_mesh)
    mesh = mskt.mesh.BoneMesh()
    mesh.list_cartilage_meshes = cart_mesh

    rand_sample = np.random.randint(0, fem_cart_mesh.GetNumberOfPoints(), 256)
    for rand_idx in rand_sample:
        assert_allclose(
            fem_cart_mesh.GetPoint(rand_idx),
            mesh.list_cartilage_meshes[0].mesh.GetPoint(rand_idx),
            rtol=RTOL,
            atol=ATOL,
        )


def test_get_list_cartilage_meshes(
    fem_cart_mesh=FEMUR_CARTILAGE_MESH,
    med_tib_cart_mesh=MED_TIB_CARTILAGE_MESH,
    lat_tib_cart_mesh=LAT_TIB_CARTILAGE_MESH,
):
    list_cart_meshes = [fem_cart_mesh, med_tib_cart_mesh, lat_tib_cart_mesh]

    list_cart_meshes = [mskt.mesh.CartilageMesh(mesh=mesh_) for mesh_ in list_cart_meshes]
    mesh = mskt.mesh.BoneMesh(list_cartilage_meshes=list_cart_meshes)

    for mesh_idx, cart_mesh in enumerate(mesh.list_cartilage_meshes):
        rand_sample = np.random.randint(0, cart_mesh.mesh.GetNumberOfPoints(), 256)
        for rand_idx in rand_sample:
            assert_allclose(
                list_cart_meshes[mesh_idx].mesh.GetPoint(rand_idx),
                cart_mesh.mesh.GetPoint(rand_idx),
                rtol=RTOL,
                atol=ATOL,
            )


# LIST CARTILAGE LABELS
def test_set_list_cartilage_labels(list_cartilage_labels=[1, 2, 3, 4]):
    mesh = mskt.mesh.BoneMesh()
    mesh.list_cartilage_labels = list_cartilage_labels

    assert_allclose(mesh._list_cartilage_labels, list_cartilage_labels, rtol=RTOL, atol=ATOL)


def test_set_list_cartilage_list_fix_input_of_int_instead_of_list(list_cartilage_labels=1):
    mesh = mskt.mesh.BoneMesh()
    mesh.list_cartilage_labels = list_cartilage_labels

    assert_allclose(
        mesh._list_cartilage_labels,
        [
            list_cartilage_labels,
        ],
        rtol=RTOL,
        atol=ATOL,
    )


def test_set_list_cartilage_labels_exception_float_type_in_list(
    list_cartilage_labels=[1.0, 2.0, 3.0]
):
    mesh = mskt.mesh.BoneMesh()
    with pytest.raises(TypeError):
        mesh.list_cartilage_labels = list_cartilage_labels


def test_set_list_cartilage_list_exception_str_type_in_list(list_cartilage_labels=["1", "2", "3"]):
    mesh = mskt.mesh.BoneMesh()
    with pytest.raises(TypeError):
        mesh.list_cartilage_labels = list_cartilage_labels


def test_get_list_cartilage_labels(list_cartilage_labels=[1, 2, 3, 4]):
    mesh = mskt.mesh.BoneMesh(list_cartilage_labels=list_cartilage_labels)

    assert_allclose(mesh.list_cartilage_labels, list_cartilage_labels, rtol=RTOL, atol=ATOL)


# CROP PERCENT
def test_set_crop_percent(percent_cropped=0.1):
    mesh = mskt.mesh.BoneMesh()
    mesh.crop_percent = percent_cropped

    assert mesh._crop_percent == percent_cropped


def test_set_crop_percent_exception_str(percent_cropped="1"):
    mesh = mskt.mesh.BoneMesh()
    with pytest.raises(TypeError):
        mesh.crop_percent = percent_cropped


def test_set_crop_percent_exception_int(percent_cropped=1):
    mesh = mskt.mesh.BoneMesh()
    with pytest.raises(TypeError):
        mesh.crop_percent = percent_cropped


def test_get_crop_percent(percent_cropped=0.1):
    mesh = mskt.mesh.BoneMesh(crop_percent=percent_cropped)

    assert mesh.crop_percent == percent_cropped


# BONE
def test_get_bone(bone="test_name"):
    mesh = mskt.mesh.BoneMesh(bone=bone)

    assert mesh.bone == bone


def test_set_bone(bone="test_name"):
    mesh = mskt.mesh.BoneMesh()
    mesh.bone = bone

    assert mesh._bone == bone


def test_set_bone_type_exception_int(bone=1):
    mesh = mskt.mesh.BoneMesh()
    with pytest.raises(TypeError):
        mesh.bone = bone


def test_set_bone_type_exception_listbone(bone=["femur"]):
    mesh = mskt.mesh.BoneMesh()
    with pytest.raises(TypeError):
        mesh.bone = bone
