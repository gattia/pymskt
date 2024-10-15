import numpy as np
import pytest
import vtk
from numpy.testing import assert_allclose
from scipy.stats import norm
from vtk.util.numpy_support import numpy_to_vtk, vtk_to_numpy

from pymskt import ATOL, RTOL
from pymskt.mesh import meshTools


@pytest.mark.skip(reason="Different results on different machines")
def test_gaussian_smooth_surface_scalars(
    sigma=1, width_height=10, resolution=0.1, magnitude_impulse=100
):
    # This function gaussian filters an impulse & ensures that the resulting grid follows a normal distribution.
    # Calculates the smoothed mesh/points & then compares lines through the center in the x & y axes to a normal
    # distribution.

    # Create the mesh
    n_points = int(width_height * (1 / resolution) + 1)

    x = np.linspace(0, width_height, n_points)
    xv, yv = np.meshgrid(x, x)
    np_points = np.ones((len(xv.flatten()), 3))
    np_points[:, 0] = xv.flatten(order="F")
    np_points[:, 1] = yv.flatten(order="F")
    np_points[:, 2] = 1

    vtk_points_ = numpy_to_vtk(np_points)
    vtk_points_.SetName("test")
    vtk_points = vtk.vtkPoints()
    vtk_points.SetData(vtk_points_)

    mesh = vtk.vtkPolyData()
    mesh.SetPoints(vtk_points)

    # Create the scalars (zeros)
    np_scalars = np.zeros(len(xv.flatten()))
    # Add impulse at the center
    np_scalars[int(len(xv.flatten("F")) / 2)] = magnitude_impulse

    # Apply scalars (and impulse) to mesh.
    vtk_scalars = numpy_to_vtk(np.copy(np_scalars))
    vtk_scalars.SetName("test")
    mesh.GetPointData().AddArray(vtk_scalars)
    mesh.GetPointData().SetActiveScalars("test")

    # gaussian filter the points.
    mesh2 = meshTools.gaussian_smooth_surface_scalars(
        mesh=mesh, sigma=sigma, idx_coords_to_smooth=None, array_name="test", array_idx=None
    )

    # retrieve and re-shape the filtered scalars.
    smoothed_scalars = vtk_to_numpy(mesh2.GetPointData().GetScalars())
    unraveled = np.reshape(smoothed_scalars, (n_points, n_points), order="F")

    # calculate the theoretical normal distribution (based on sigma etc)
    edge_sd = width_height / 2 / sigma
    x = np.linspace(-edge_sd, edge_sd, n_points)
    pdf = norm.pdf(x)
    # Normalized pdf to magnitude of the scalars:
    # This scales the whole curve based on the size of the peak (center)
    # of the curve in relation to our calcualted distribution.
    middle_idx = int((n_points - 1) / 2)
    pdf = pdf / (pdf[middle_idx] / unraveled[middle_idx, middle_idx])

    # assert that the x & y axies (down the middle) follow the expected normal distribution.
    assert_allclose(pdf, unraveled[middle_idx, :], rtol=RTOL, atol=ATOL)
    assert_allclose(pdf, unraveled[:, middle_idx], rtol=RTOL, atol=ATOL)


@pytest.mark.skip(reason="Test not implemented yet")
def test_gaussian_smooth_surface_scalars_use_idx_for_base_mesh():
    pass  # TODO: Implement this test
