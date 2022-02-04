# from turtle import width
# import pytest
# from pymskt.mesh import meshTools
# import vtk
# import numpy as np
# from vtk.util.numpy_support import numpy_to_vtk, vtk_to_numpy
# from numpy.testing import assert_allclose

# NP_POINTS = np.asarray([
#         [0, 0, 0],
#         [1, 1, 1],
#         [10, 10, 10]
#     ])

# #
# # get_mesh_physical_point_coords
# #

# def test_get_mesh_physical_point_coords(np_points=NP_POINTS):
#     vtk_points_ = numpy_to_vtk(np_points)
#     vtk_points_.SetName('test')
#     vtk_points = vtk.vtkPoints()
#     vtk_points.SetData(vtk_points_)

#     mesh = vtk.vtkPolyData()
#     mesh.SetPoints(vtk_points)

#     assert_allclose(meshTools.get_mesh_physical_point_coords(mesh), np_points)

# #
# # smooth_scalars_from_second_mesh_onto_base
# #

# def test_smooth_scalars_from_second_mesh_onto_base(
#     sigma=1,
#     width_height=10,
#     resolution=0.1,
#     magnitude_impulse=100
# ):
#     # Use the same logic as the above test. 
#     # This test should be the same except that it will create identical meshes (no scalars, yet).
#     # Create the mesh
#     n_points = int(width_height * (1/resolution) + 1)

#     x = np.linspace(0, width_height, n_points)
#     xv, yv = np.meshgrid(x, x)
#     np_points = np.ones((len(xv.flatten()), 3))
#     np_points[:,0] = xv.flatten(order='F')
#     np_points[:,1] = yv.flatten(order='F')
#     np_points[:,2] = 1

#     vtk_points_ = numpy_to_vtk(np_points)
#     vtk_points_.SetName('test')
#     vtk_points = vtk.vtkPoints()
#     vtk_points.SetData(vtk_points_)

#     mesh1 = vtk.vtkPolyData()
#     mesh1.SetPoints(vtk_points)

#     mesh2 = vtk_deep_copy(mesh1)

#     # Apply impulse to mesh1.
#     # Create the scalars (zeros)
#     np_scalars = np.zeros(len(xv.flatten()))
#     # Add impulse at the center
#     np_scalars[int(len(xv.flatten('F'))/2)] = magnitude_impulse

#     # Apply scalars (and impulse) to mesh. 
#     vtk_scalars = numpy_to_vtk(np.copy(np_scalars))
#     vtk_scalars.SetName('test')
#     mesh1.GetPointData().AddArray(vtk_scalars)
#     mesh1.GetPointData().SetActiveScalars('test')

#     # smooth (gaussian) mesh1 scalars onto mesh2.
#     # gaussian filter the points.
#     smoothed_scalars = meshTools.smooth_scalars_from_second_mesh_onto_base(
#         base_mesh=mesh2,
#         second_mesh=mesh1,
#         sigma=sigma,
#         idx_coords_to_smooth_base=None,
#         idx_coords_to_smooth_second=None,
#         set_non_smoothed_scalars_to_zero=True
#     )

#     unraveled = np.reshape(smoothed_scalars, (n_points, n_points), order="F")

#     # calculate the theoretical normal distribution (based on sigma etc)
#     edge_sd = width_height / 2 / sigma
#     x = np.linspace(-edge_sd, edge_sd, n_points)
#     pdf = norm.pdf(x)
#     # Normalized pdf to magnitude of the scalars:
#     # This scales the whole curve based on the size of the peak (center)
#     # of the curve in relation to our calcualted distribution. 
#     middle_idx = int((n_points-1)/2)
#     pdf = pdf / (pdf[middle_idx] / unraveled[middle_idx, middle_idx])

#     # assert that the x & y axies (down the middle) follow the expected normal distribution. 
#     assert_allclose(pdf, unraveled[middle_idx,:], atol=1e-4)
#     assert_allclose(pdf, unraveled[:, middle_idx], atol=1e-4)

# #
# # transfer_mesh_scalars_get_weighted_average_n_closest
# #

# def test_transfer_mesh_scalars_get_weighted_average_n_closest(n_points=1000):
#     np_points = np.ones((n_points, 3))
#     np_points[:,0] = np.arange(n_points)
#     np_points[:,1] = np.arange(n_points)
#     np_points[:,2] = np.arange(n_points)

#     vtk_points_ = numpy_to_vtk(np_points)
#     vtk_points_.SetName('test')
#     vtk_points = vtk.vtkPoints()
#     vtk_points.SetData(vtk_points_)

#     mesh = vtk.vtkPolyData()
#     mesh.SetPoints(vtk_points)

#     mesh2 = vtk_deep_copy(mesh)

#     np_scalars = np.random.random(n_points)
#     vtk_scalars = numpy_to_vtk(np_scalars)
#     vtk_scalars.SetName('test')
#     # mesh.GetPointData().SetScalars(vtk_scalars)
#     mesh.GetPointData().AddArray(vtk_scalars)

#     transfered_scalars = meshTools.transfer_mesh_scalars_get_weighted_average_n_closest(mesh2, mesh, n=1)

#     assert_allclose(np_scalars, np.squeeze(transfered_scalars))

# #
# # get_smoothed_scalars
# #

# def dist(diff, epsilon=1e-7):
#     return np.sqrt(np.sum(np.square(diff + epsilon)))

# def test_get_smoothed_scalars(
#     max_dist=1.1, # use 1.1 so only need to get single points in-line(x/y) & no diag for testing - but dont want 1.0 otherwise weighting = 0 for all other points. 
#     order=2.
# ):
#     # Create small mesh that can easily manually calculate the outcomes. 

#     width_height = 1
#     resolution = 1

#     n_points = int(width_height * (1/resolution) + 1)

#     x = np.linspace(0, width_height, n_points)
#     xv, yv = np.meshgrid(x, x)
#     np_points = np.ones((len(xv.flatten()), 3))
#     np_points[:,0] = xv.flatten(order='F')
#     np_points[:,1] = yv.flatten(order='F')
#     np_points[:,2] = 1

#     vtk_points_ = numpy_to_vtk(np_points)
#     vtk_points_.SetName('test')
#     vtk_points = vtk.vtkPoints()
#     vtk_points.SetData(vtk_points_)

#     mesh = vtk.vtkPolyData()
#     mesh.SetPoints(vtk_points)

#     np_scalars = np.random.randint(0, 1000, len(xv.flatten()))

#     np_scalars_reshaped = np.reshape(np_scalars, (width_height*2, width_height*2), order='F')
#     np_scalars_smoothed_test = np.zeros_like(np_scalars_reshaped).astype(float)
#     for i in range(width_height + 1):
#         for j in range(width_height + 1):
#             distances = [dist(0)]
#             scalars = [np_scalars_reshaped[i, j]]
#             if i > 0:
#                 scalars.append(np_scalars_reshaped[i-1, j])
#                 distances.append(dist(1.0))
#             if j > 0:
#                 scalars.append(np_scalars_reshaped[i, j-1])
#                 distances.append(dist(1.0))
#             if i < width_height:
#                 scalars.append(np_scalars_reshaped[i+1, j])
#                 distances.append(dist(1.0))
#             if j < width_height:
#                 scalars.append(np_scalars_reshaped[i, j+1])
#                 distances.append(dist(1.0))
#             weights = (max_dist - np.asarray(distances))**order 
#             weighted_scalars = weights * np.asarray(scalars)
#             normalized_point = np.sum(weighted_scalars) / np.sum(weights)
#             np_scalars_smoothed_test[i, j] = normalized_point



#     # Apply scalars (and impulse) to mesh. 
#     vtk_scalars = numpy_to_vtk(np.copy(np_scalars))
#     vtk_scalars.SetName('test')
#     mesh.GetPointData().AddArray(vtk_scalars)
#     mesh.GetPointData().SetActiveScalars('test')
    


#     scalars_smoothed = meshTools.get_smoothed_scalars(
#         mesh, 
#         max_dist=max_dist, 
#         order=order, 
#         gaussian=False
#     )

#     scalars_smoothed = np.reshape(scalars_smoothed, (width_height+1, width_height+1), order='F')

#     assert_allclose(np_scalars_smoothed_test, scalars_smoothed, rtol=1e-03)

# #
# # gaussian_smooth_surface_scalars
# #

# def test_gaussian_smooth_surface_scalars(
#     sigma=1,
#     width_height=10,
#     resolution=0.1,
#     magnitude_impulse=100
# ):
#     # This function gaussian filters an impulse & ensures that the resulting grid follows a normal distribution.
#     # Calculates the smoothed mesh/points & then compares lines through the center in the x & y axes to a normal
#     # distribution. 

#     # Create the mesh
#     n_points = int(width_height * (1/resolution) + 1)

#     x = np.linspace(0, width_height, n_points)
#     xv, yv = np.meshgrid(x, x)
#     np_points = np.ones((len(xv.flatten()), 3))
#     np_points[:,0] = xv.flatten(order='F')
#     np_points[:,1] = yv.flatten(order='F')
#     np_points[:,2] = 1

#     vtk_points_ = numpy_to_vtk(np_points)
#     vtk_points_.SetName('test')
#     vtk_points = vtk.vtkPoints()
#     vtk_points.SetData(vtk_points_)

#     mesh = vtk.vtkPolyData()
#     mesh.SetPoints(vtk_points)

#     # Create the scalars (zeros)
#     np_scalars = np.zeros(len(xv.flatten()))
#     # Add impulse at the center
#     np_scalars[int(len(xv.flatten('F'))/2)] = magnitude_impulse

#     # Apply scalars (and impulse) to mesh. 
#     vtk_scalars = numpy_to_vtk(np.copy(np_scalars))
#     vtk_scalars.SetName('test')
#     mesh.GetPointData().AddArray(vtk_scalars)
#     mesh.GetPointData().SetActiveScalars('test')

#     # gaussian filter the points. 
#     mesh2 = meshTools.gaussian_smooth_surface_scalars(
#         mesh=mesh, 
#         sigma=sigma, 
#         idx_coords_to_smooth=None, 
#         array_name='test', 
#         array_idx=None
#     )

#     # retrieve and re-shape the filtered scalars. 
#     smoothed_scalars = vtk_to_numpy(mesh2.GetPointData().GetScalars())
#     unraveled = np.reshape(smoothed_scalars, (n_points, n_points), order="F")

#     # calculate the theoretical normal distribution (based on sigma etc)
#     edge_sd = width_height / 2 / sigma
#     x = np.linspace(-edge_sd, edge_sd, n_points)
#     pdf = norm.pdf(x)
#     # Normalized pdf to magnitude of the scalars:
#     # This scales the whole curve based on the size of the peak (center)
#     # of the curve in relation to our calcualted distribution. 
#     middle_idx = int((n_points-1)/2)
#     pdf = pdf / (pdf[middle_idx] / unraveled[middle_idx, middle_idx])

#     # assert that the x & y axies (down the middle) follow the expected normal distribution. 
#     assert_allclose(pdf, unraveled[middle_idx,:], atol=1e-4)
#     assert_allclose(pdf, unraveled[:, middle_idx], atol=1e-4)



# def test_gaussian_smooth_surface_scalars_use_idx_for_base_mesh():
#     raise Exception('Test not implemented')

# def test_smooth_scalars_from_second_mesh_onto_base_use_idx_coords_to_smooth():
#     raise Exception('Test not implemented')

# def test_smooth_scalars_from_second_mesh_onto_base_use_idx_for_second_mesh():
#     raise Exception('Test not implemented')

# def test_get_cartilage_properties_at_points():
#     raise Exception('Test not implemented')