
from lib2to3.pgen2.token import AT
from pymskt.mesh import meshTools
import vtk
import numpy as np
from vtk.util.numpy_support import numpy_to_vtk
from numpy.testing import assert_allclose

from pymskt import RTOL, ATOL
#
# get_smoothed_scalars
#

def dist(diff, epsilon=1e-7):
    return np.sqrt(np.sum(np.square(diff + epsilon)))

def test_get_smoothed_scalars(
    max_dist=1.1, # use 1.1 so only need to get single points in-line(x/y) & no diag for testing - but dont want 1.0 otherwise weighting = 0 for all other points. 
    order=2.
):
    # Create small mesh that can easily manually calculate the outcomes. 

    width_height = 1
    resolution = 1

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

    mesh = vtk.vtkPolyData()
    mesh.SetPoints(vtk_points)

    np_scalars = np.random.randint(0, 1000, len(xv.flatten()))

    np_scalars_reshaped = np.reshape(np_scalars, (width_height*2, width_height*2), order='F')
    np_scalars_smoothed_test = np.zeros_like(np_scalars_reshaped).astype(float)
    for i in range(width_height + 1):
        for j in range(width_height + 1):
            distances = [dist(0)]
            scalars = [np_scalars_reshaped[i, j]]
            if i > 0:
                scalars.append(np_scalars_reshaped[i-1, j])
                distances.append(dist(1.0))
            if j > 0:
                scalars.append(np_scalars_reshaped[i, j-1])
                distances.append(dist(1.0))
            if i < width_height:
                scalars.append(np_scalars_reshaped[i+1, j])
                distances.append(dist(1.0))
            if j < width_height:
                scalars.append(np_scalars_reshaped[i, j+1])
                distances.append(dist(1.0))
            weights = (max_dist - np.asarray(distances))**order 
            weighted_scalars = weights * np.asarray(scalars)
            normalized_point = np.sum(weighted_scalars) / np.sum(weights)
            np_scalars_smoothed_test[i, j] = normalized_point



    # Apply scalars (and impulse) to mesh. 
    vtk_scalars = numpy_to_vtk(np.copy(np_scalars))
    vtk_scalars.SetName('test')
    mesh.GetPointData().AddArray(vtk_scalars)
    mesh.GetPointData().SetActiveScalars('test')
    


    scalars_smoothed = meshTools.get_smoothed_scalars(
        mesh, 
        max_dist=max_dist, 
        order=order, 
        gaussian=False
    )

    scalars_smoothed = np.reshape(scalars_smoothed, (width_height+1, width_height+1), order='F')

    assert_allclose(np_scalars_smoothed_test, scalars_smoothed, rtol=1e-03, atol=ATOL)
