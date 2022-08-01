import pymskt as mskt
from pymskt.utils import testing

MESH = mskt.mesh.io.read_vtk('data/femur_mesh_orig.vtk')
MESH_RESAMPLED = mskt.mesh.io.read_vtk('data/femur_mesh_10k_pts.vtk')

from pymskt import RTOL, ATOL

def test_resample_surface_assert_same_as_saved_mesh(
    mesh_=MESH, 
    mesh_resampled=MESH_RESAMPLED
):
    mesh = mskt.mesh.Mesh(mesh=mesh_)
    mesh.resample_surface(subdivisions=2, clusters=10000)
    testing.assert_mesh_coordinates_same(mesh_resampled, mesh.mesh, rtol=RTOL, atol=ATOL)
