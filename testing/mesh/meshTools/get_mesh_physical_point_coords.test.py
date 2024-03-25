import numpy as np
import vtk
from numpy.testing import assert_allclose
from vtk.util.numpy_support import numpy_to_vtk

from pymskt import ATOL, RTOL
from pymskt.mesh import meshTools

NP_POINTS = np.asarray([[0, 0, 0], [1, 1, 1], [10, 10, 10]])

#
# get_mesh_physical_point_coords
#


def test_get_mesh_physical_point_coords(np_points=NP_POINTS):
    vtk_points_ = numpy_to_vtk(np_points)
    vtk_points_.SetName("test")
    vtk_points = vtk.vtkPoints()
    vtk_points.SetData(vtk_points_)

    mesh = vtk.vtkPolyData()
    mesh.SetPoints(vtk_points)

    assert_allclose(meshTools.get_mesh_physical_point_coords(mesh), np_points, rtol=RTOL, atol=ATOL)
