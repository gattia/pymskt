import pytest
from pymskt.mesh import meshTools
import vtk
import numpy as np
from vtk.util.numpy_support import numpy_to_vtk
from numpy.testing import assert_allclose
from pymskt.mesh.utils import vtk_deep_copy
from scipy.stats import norm

from pymskt import RTOL, ATOL
#
# smooth_scalars_from_second_mesh_onto_base
#
@pytest.mark.skip(reason="Different results on different machines")
def test_smooth_scalars_from_second_mesh_onto_base(
    sigma=1,
    width_height=10,
    resolution=0.1,
    magnitude_impulse=100
):
    # Use the same logic as the above test. 
    # This test should be the same except that it will create identical meshes (no scalars, yet).
    # Create the mesh
    n_points = int(width_height * (1/resolution) + 1)

    x = np.linspace(0, width_height, n_points)
    xv, yv = np.meshgrid(x, x)
    np_points = np.ones((len(xv.flatten()), 3))
    np_points[:,0] = xv.flatten(order='F')
    np_points[:,1] = yv.flatten(order='F')
    np_points[:,2] = 1

    vtk_points_ = numpy_to_vtk(np_points)
    vtk_points_.SetName('test')
    vtk_points = vtk.vtkPoints()
    vtk_points.SetData(vtk_points_)

    mesh1 = vtk.vtkPolyData()
    mesh1.SetPoints(vtk_points)

    mesh2 = vtk_deep_copy(mesh1)

    # Apply impulse to mesh1.
    # Create the scalars (zeros)
    np_scalars = np.zeros(len(xv.flatten()))
    # Add impulse at the center
    np_scalars[int(len(xv.flatten('F'))/2)] = magnitude_impulse

    # Apply scalars (and impulse) to mesh. 
    vtk_scalars = numpy_to_vtk(np.copy(np_scalars))
    vtk_scalars.SetName('test')
    mesh1.GetPointData().AddArray(vtk_scalars)
    mesh1.GetPointData().SetActiveScalars('test')

    # smooth (gaussian) mesh1 scalars onto mesh2.
    # gaussian filter the points.
    smoothed_scalars = meshTools.smooth_scalars_from_second_mesh_onto_base(
        base_mesh=mesh2,
        second_mesh=mesh1,
        sigma=sigma,
        idx_coords_to_smooth_base=None,
        idx_coords_to_smooth_second=None,
        set_non_smoothed_scalars_to_zero=True
    )

    unraveled = np.reshape(smoothed_scalars, (n_points, n_points), order="F")

    # calculate the theoretical normal distribution (based on sigma etc)
    edge_sd = width_height / 2 / sigma
    x = np.linspace(-edge_sd, edge_sd, n_points)
    pdf = norm.pdf(x)
    # Normalized pdf to magnitude of the scalars:
    # This scales the whole curve based on the size of the peak (center)
    # of the curve in relation to our calcualted distribution. 
    middle_idx = int((n_points-1)/2)
    pdf = pdf / (pdf[middle_idx] / unraveled[middle_idx, middle_idx])

    # assert that the x & y axies (down the middle) follow the expected normal distribution. 
    assert_allclose(pdf, unraveled[middle_idx,:], rtol=RTOL, atol=ATOL)
    assert_allclose(pdf, unraveled[:, middle_idx], rtol=RTOL, atol=ATOL)

@pytest.mark.skip(reason="Test not implemented yet")
def test_smooth_scalars_from_second_mesh_onto_base_use_idx_coords_to_smooth():
    pass  # TODO: Implement this test

@pytest.mark.skip(reason="Test not implemented yet")
def test_smooth_scalars_from_second_mesh_onto_base_use_idx_for_second_mesh():
    pass  # TODO: Implement this test