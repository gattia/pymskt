import pytest
import SimpleITK as sitk

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
SEG_IMAGE = sitk.ReadImage("data/right_knee_example.nrrd")

from pymskt import ATOL, RTOL


@pytest.mark.skip(
    reason="Updated create_cartilage_meshes to also 'fix' the meshes, which makes the surfaces different"
)
def test_create_cartilage_meshes_single(fem_cart_mesh=FEMUR_CARTILAGE_MESH, seg_image=SEG_IMAGE):
    mesh = mskt.mesh.BoneMesh(
        seg_image=seg_image,
        list_cartilage_labels=[
            1,
        ],
    )

    assert mesh.list_cartilage_meshes is None

    mesh.create_cartilage_meshes()

    assert len(mesh.list_cartilage_meshes) == 1

    mskt.utils.testing.assert_mesh_coordinates_same(
        mesh.list_cartilage_meshes[0], fem_cart_mesh, rtol=RTOL, atol=ATOL
    )


def test_create_cartilage_meshes_exception_no_pixels(seg_image=SEG_IMAGE):
    mesh = mskt.mesh.BoneMesh(
        seg_image=seg_image,
        list_cartilage_labels=[
            100,
        ],
    )
    with pytest.warns(UserWarning):
        mesh.create_cartilage_meshes()


@pytest.mark.skip(
    reason="Updated create_cartilage_meshes to also 'fix' the meshes, which makes the surfaces different"
)
def test_create_multiple_cartilage_meshes(
    fem_cart_mesh=FEMUR_CARTILAGE_MESH,
    med_tib_cart_mesh=MED_TIB_CARTILAGE_MESH,
    lat_tib_cart_mesh=LAT_TIB_CARTILAGE_MESH,
    seg_image=SEG_IMAGE,
):
    mesh = mskt.mesh.BoneMesh(seg_image=seg_image, list_cartilage_labels=[1, 2, 3])

    assert mesh.list_cartilage_meshes is None

    mesh.create_cartilage_meshes()

    assert len(mesh.list_cartilage_meshes) == 3

    mskt.utils.testing.assert_mesh_coordinates_same(
        mesh.list_cartilage_meshes[0], fem_cart_mesh, rtol=RTOL, atol=ATOL
    )
    mskt.utils.testing.assert_mesh_coordinates_same(
        mesh.list_cartilage_meshes[1], med_tib_cart_mesh, rtol=RTOL, atol=ATOL
    )
    mskt.utils.testing.assert_mesh_coordinates_same(
        mesh.list_cartilage_meshes[2], lat_tib_cart_mesh, rtol=RTOL, atol=ATOL
    )
