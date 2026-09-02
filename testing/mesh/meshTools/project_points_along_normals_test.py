import numpy as np
import pyvista as pv
import vtk

from pymskt.mesh.meshTools import project_points_along_normals

# point-cloud-utils traces rays with Embree in single precision, so a projected point
# lands within float32 rounding of the surface rather than float64.
TOL = 1e-5


def _sphere():
    return pv.Sphere(radius=1.0, theta_resolution=40, phi_resolution=40)


def _face_centroids_and_normals(mesh):
    """Face centroids lie on the mesh, and the line through one along its face normal
    meets the face at the centroid again, so they make an exact reference."""
    m = mesh.compute_normals(cell_normals=True, point_normals=False, auto_orient_normals=True)
    return m.cell_centers().points.astype(np.float64), m.cell_normals.astype(np.float64)


def test_points_pushed_off_the_surface_come_back():
    sphere = _sphere()
    cent, norm = _face_centroids_and_normals(sphere)
    rng = np.random.default_rng(0)
    offsets = rng.uniform(-0.2, 0.2, size=len(cent))  # inward and outward
    pushed = cent + norm * offsets[:, None]
    pushed_copy, norm_copy = pushed.copy(), norm.copy()

    out = project_points_along_normals(pushed, norm, sphere)

    assert out.shape == cent.shape
    assert out.dtype == np.float64
    assert np.abs(out - cent).max() < TOL
    # inputs are not modified
    assert np.array_equal(pushed, pushed_copy)
    assert np.array_equal(norm, norm_copy)


def test_normal_length_and_sign_do_not_matter():
    sphere = _sphere()
    cent, norm = _face_centroids_and_normals(sphere)
    pushed = cent + norm * 0.1
    rng = np.random.default_rng(1)
    scale = rng.uniform(0.1, 5.0, size=len(cent))
    scale[rng.random(len(cent)) < 0.5] *= -1.0

    out = project_points_along_normals(pushed, norm * scale[:, None], sphere)

    assert np.abs(out - cent).max() < TOL


def test_zero_and_non_finite_normals_leave_points_in_place():
    sphere = _sphere()
    cent, norm = _face_centroids_and_normals(sphere)
    pushed = cent + norm * 0.1
    bad = norm.copy()
    bad[0] = 0.0
    bad[1] = np.nan
    bad[2, 0] = np.inf

    out = project_points_along_normals(pushed, bad, sphere)

    assert np.array_equal(out[:3], pushed[:3])
    assert np.abs(out[3:] - cent[3:]).max() < TOL


def test_max_distance_leaves_far_points_in_place():
    sphere = _sphere()
    cent, norm = _face_centroids_and_normals(sphere)
    near = np.arange(len(cent)) % 2 == 0
    offsets = np.where(near, 0.05, 0.5)
    pushed = cent + norm * offsets[:, None]

    out = project_points_along_normals(pushed, norm, sphere, max_distance=0.1)
    assert np.abs(out[near] - cent[near]).max() < TOL
    assert np.array_equal(out[~near], pushed[~near])

    # without the cap every point comes back
    out = project_points_along_normals(pushed, norm, sphere)
    assert np.abs(out - cent).max() < TOL


def test_nearest_intersection_in_either_direction_wins():
    sphere = _sphere()
    # a single point, for which point-cloud-utils returns scalars instead of arrays.
    # It sits above the sphere and its line crosses it twice; the direction points away
    # from the sphere, so both crossings are behind the point and the near one (z ~ +1)
    # must be chosen over the far one (z ~ -1)
    origin = np.array([[0.01, 0.02, 1.5]])
    direction = np.array([[0.0, 0.0, 1.0]])

    out = project_points_along_normals(origin, direction, sphere)

    assert out[0, 2] > 0.95
    assert np.abs(out[0, :2] - origin[0, :2]).max() < TOL  # moved along the line only


def test_accepts_vtk_polydata():
    sphere = _sphere()
    cent, norm = _face_centroids_and_normals(sphere)
    vtk_mesh = vtk.vtkPolyData()
    vtk_mesh.DeepCopy(sphere)

    out = project_points_along_normals(cent + norm * 0.1, norm, vtk_mesh)

    assert np.abs(out - cent).max() < TOL


def test_empty_input():
    sphere = _sphere()
    out = project_points_along_normals(np.empty((0, 3)), np.empty((0, 3)), sphere)
    assert out.shape == (0, 3)
