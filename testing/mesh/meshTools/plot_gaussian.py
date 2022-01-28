# import pytest
from pymskt.mesh import meshTools
import vtk
import numpy as np
from vtk.util.numpy_support import numpy_to_vtk, vtk_to_numpy
# from numpy.testing import assert_allclose
# from pymskt.mesh.utils import vtk_deep_copy

sigma = 1
width_height = 10
n_points = width_height * 10 + 1
magnitude_impulse = 100 # * 2.5061643998430685


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

np_scalars = np.zeros(len(xv.flatten()))
np_scalars[int(len(xv.flatten('F'))/2)] = magnitude_impulse

vtk_scalars = numpy_to_vtk(np.copy(np_scalars))
vtk_scalars.SetName('test')
mesh.GetPointData().AddArray(vtk_scalars)
mesh.GetPointData().SetActiveScalars('test')


mesh2 = meshTools.gaussian_smooth_surface_scalars(
    mesh=mesh, 
    sigma=sigma, 
    idx_coords_to_smooth=None, 
    array_name='test', 
    array_idx=None
)

delauney = vtk.vtkDelaunay2D()
delauney.SetInputData(mesh2)
delauney.Update()
surface = delauney.GetOutput()

data = vtk_to_numpy(mesh2.GetPointData().GetArray(0))
print('mean of data:')
print('\torig: ', np.mean(np_scalars), '\tfiltered: ', np.mean(data))
print('sum of data:')
print('\torig: ', np.sum(np_scalars), '\tfiltered: ', np.sum(data))
print('std of data:')
print('\torig: ', np.std(np_scalars), '\tfiltered: ', np.std(data))

import pyvista as pv
plotter = pv.Plotter()
plotter.add_mesh(surface, show_edges=True, edge_color='black')
plotter.show()

smoothed_scalars = vtk_to_numpy(mesh2.GetPointData().GetScalars())
unraveled = np.reshape(smoothed_scalars, (n_points, n_points), order="F")

import matplotlib.pyplot as plt
from scipy.stats import norm
edge_sd = width_height / 2 / sigma
x = np.linspace(-edge_sd, edge_sd, n_points)
pdf = norm.pdf(x)
# Normalized pdf to magnitude of the scalars: 
pdf = pdf / (pdf[50] / unraveled[50, 50])

assert np.max(np.abs(pdf - unraveled[50,:])) < 1e-4

plt.plot(pdf)
plt.plot(unraveled[50,:])
plt.show()