import pytest

import pymskt as mskt

REF_1_MESH = mskt.mesh.io.read_vtk(
    "data/femur_thickness_mm_regions_10k_pts_thick_smoothed_1.25_sigma.vtk"
)
REF_3_MESH = mskt.mesh.io.read_vtk(
    "data/femur_thickness_mm_regions_10k_pts_thick_smoothed_3_sigma.vtk"
)

from pymskt import ATOL, RTOL


def test_smoothing_cartilage_sigma_1(
    base_mesh_path="data/femur_thickness_mm_regions_10k_pts.vtk",
    ref_mesh=REF_1_MESH,
):
    base_mesh = mskt.mesh.io.read_vtk(base_mesh_path)
    mesh = mskt.mesh.BoneMesh(mesh=base_mesh)
    mesh.smooth_surface_scalars(scalar_sigma=1.25, scalar_array_name="thickness (mm)")
    mskt.utils.testing.assert_mesh_scalars_same(
        mesh, ref_mesh, scalarname="thickness (mm)", rtol=RTOL, atol=ATOL
    )


def test_smoothing_cartilage_sigma_3(
    base_mesh_path="data/femur_thickness_mm_regions_10k_pts.vtk", ref_mesh=REF_3_MESH
):
    base_mesh = mskt.mesh.io.read_vtk(base_mesh_path)
    mesh = mskt.mesh.BoneMesh(mesh=base_mesh)
    mesh.smooth_surface_scalars(scalar_sigma=3, scalar_array_name="thickness (mm)")
    mskt.utils.testing.assert_mesh_scalars_same(
        mesh, ref_mesh, scalarname="thickness (mm)", rtol=RTOL, atol=ATOL
    )


def test_smoothing_cartilage_sigma_test_a_mismatch(
    base_mesh_path="data/femur_thickness_mm_regions_10k_pts.vtk", ref_mesh=REF_3_MESH
):
    base_mesh = mskt.mesh.io.read_vtk(base_mesh_path)
    mesh = mskt.mesh.BoneMesh(mesh=base_mesh)
    mesh.smooth_surface_scalars(scalar_sigma=1.25, scalar_array_name="thickness (mm)")
    with pytest.raises(AssertionError):
        mskt.utils.testing.assert_mesh_scalars_same(
            mesh, ref_mesh, scalarname="thickness (mm)", rtol=RTOL, atol=ATOL
        )
