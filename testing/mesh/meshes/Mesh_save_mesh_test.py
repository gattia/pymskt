import os
import tempfile

import pymskt as mskt
from pymskt.utils import testing

MESH_WITH_SMOOTHING = mskt.mesh.io.read_vtk("data/femur_mesh_orig.vtk")

from pymskt import ATOL, RTOL


def test_saving_image(
    mesh=MESH_WITH_SMOOTHING, loc_save=tempfile.gettempdir(), filename="test.vtk"
):
    mesh = mskt.mesh.Mesh(mesh=mesh)
    mesh.save_mesh(os.path.join(loc_save, filename))

    mesh2 = mskt.mesh.io.read_vtk(os.path.join(loc_save, filename))

    testing.assert_mesh_coordinates_same(mesh.mesh, mesh2, rtol=RTOL, atol=ATOL)
