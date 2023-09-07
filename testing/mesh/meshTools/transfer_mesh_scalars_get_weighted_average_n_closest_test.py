from pymskt.mesh import meshTools
import vtk
import numpy as np
from vtk.util.numpy_support import numpy_to_vtk
from numpy.testing import assert_allclose
from pymskt.mesh.utils import vtk_deep_copy

from pymskt import RTOL, ATOL

#
# transfer_mesh_scalars_get_weighted_average_n_closest
#

def test_transfer_mesh_scalars_get_weighted_average_n_closest(n_points=1000):
    np_points = np.ones((n_points, 3))
    np_points[:,0] = np.arange(n_points)
    np_points[:,1] = np.arange(n_points)
    np_points[:,2] = np.arange(n_points)

    vtk_points_ = numpy_to_vtk(np_points)
    vtk_points_.SetName('test')
    vtk_points = vtk.vtkPoints()
    vtk_points.SetData(vtk_points_)

    mesh = vtk.vtkPolyData()
    mesh.SetPoints(vtk_points)

    mesh2 = vtk_deep_copy(mesh)

    np_scalars = np.random.random(n_points)
    vtk_scalars = numpy_to_vtk(np_scalars)
    vtk_scalars.SetName('test')
    # mesh.GetPointData().SetScalars(vtk_scalars)
    mesh.GetPointData().AddArray(vtk_scalars)

    transfered_scalars = meshTools.transfer_mesh_scalars_get_weighted_average_n_closest(mesh2, mesh, n=1)

    assert_allclose(np_scalars, np.squeeze(transfered_scalars['test']), rtol=RTOL, atol=ATOL)