import numpy as np
import pyvista as pv
import vtk

from pymskt.mesh.meshTools import resample_surface

# projected points land within float32 rounding of the surface (Embree ray tracing)
ON_SURFACE_TOL = 1e-4


def _sphere():
    return pv.Sphere(radius=10.0, theta_resolution=30, phi_resolution=30)


def _distance_to_surface(points, mesh):
    _, closest = mesh.find_closest_cell(points, return_closest_point=True)
    return np.linalg.norm(points - closest, axis=1)


def test_input_mesh_is_not_modified():
    # pyacvd's subdivide() rewrites the mesh it is given in place; resample_surface must
    # hand it a copy so the caller's mesh survives
    sphere = _sphere()
    n_points, n_cells = sphere.n_points, sphere.n_cells
    points = sphere.points.copy()

    resample_surface(sphere, subdivisions=1, clusters=500)

    assert sphere.n_points == n_points
    assert sphere.n_cells == n_cells
    assert np.array_equal(sphere.points, points)


def test_resampled_points_lie_on_the_input_surface():
    sphere = _sphere()

    out = resample_surface(sphere, subdivisions=1, clusters=500)

    assert isinstance(out, pv.PolyData)
    assert abs(out.n_points - 500) < 50  # pyacvd is approximate about the count
    assert _distance_to_surface(out.points, sphere).max() < ON_SURFACE_TOL


def test_project_to_surface_false_leaves_centroids_off_the_surface():
    sphere = _sphere()

    out = resample_surface(sphere, subdivisions=1, clusters=500, project_to_surface=False)

    # cluster centroids of a convex surface sit inside it
    assert _distance_to_surface(out.points, sphere).max() > 1e-3


def test_no_subdivision():
    sphere = _sphere()

    out = resample_surface(sphere, subdivisions=0, clusters=500)

    assert abs(out.n_points - 500) < 50
    assert _distance_to_surface(out.points, sphere).max() < ON_SURFACE_TOL


def test_accepts_vtk_polydata():
    sphere = _sphere()
    vtk_mesh = vtk.vtkPolyData()
    vtk_mesh.DeepCopy(sphere)
    n_points = vtk_mesh.GetNumberOfPoints()

    out = resample_surface(vtk_mesh, subdivisions=1, clusters=500)

    assert vtk_mesh.GetNumberOfPoints() == n_points
    assert _distance_to_surface(out.points, sphere).max() < ON_SURFACE_TOL
