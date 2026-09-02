import numpy as np
import pyvista as pv

from pymskt.mesh import Mesh
from pymskt.mesh.meshCartilage import _triangle_edge_neighbor_counts, remove_isolated_cells


def _plane_with_flap():
    """A 4x4 triangulated plane plus one triangle hanging off a boundary edge by one edge."""
    plane = pv.Plane(i_resolution=4, j_resolution=4).triangulate()
    pts = np.vstack([plane.points, [[0.7, 0.7, 0.3]]])  # a new vertex off the plane
    faces = plane.faces.reshape(-1, 4)[:, 1:]
    # attach the flap to the first boundary edge we find (an edge used by only one triangle)
    counts = _triangle_edge_neighbor_counts(plane)
    tri = int(np.flatnonzero(counts < 3)[0])
    a, b = faces[tri, 0], faces[tri, 1]
    edge_ok = np.sum(np.all(np.isin(faces, [a, b]), axis=1)) == 1
    if not edge_ok:
        a, b = faces[tri, 1], faces[tri, 2]
    flap = [a, b, len(pts) - 1]
    all_faces = np.vstack([faces, flap])
    return pv.PolyData(pts, np.hstack([np.full((len(all_faces), 1), 3), all_faces]).ravel())


def test_neighbor_counts_match_pyvista_on_manifold_mesh():
    sphere = pv.Sphere(theta_resolution=12, phi_resolution=12)
    ours = _triangle_edge_neighbor_counts(sphere)
    ref = np.array(
        [len(sphere.cell_neighbors(i, connections="edges")) for i in range(sphere.n_cells)]
    )
    assert np.array_equal(ours, ref)


def test_closed_mesh_is_unchanged():
    sphere = pv.Sphere(theta_resolution=12, phi_resolution=12)
    out = remove_isolated_cells(sphere)
    assert isinstance(out, Mesh)
    assert out.n_cells == sphere.n_cells
    assert sphere.n_cells == pv.Sphere(theta_resolution=12, phi_resolution=12).n_cells  # untouched


def test_flap_is_removed_and_plane_kept():
    mesh = _plane_with_flap()
    n_plane = mesh.n_cells - 1
    out = remove_isolated_cells(mesh)
    assert out.n_cells == n_plane
    assert out.n_points == mesh.n_points - 1  # the flap's private vertex is cleaned away


def test_duplicate_face_is_not_treated_as_isolated():
    mesh = _plane_with_flap()
    faces = mesh.faces.reshape(-1, 4)[:, 1:]
    # duplicate one interior triangle; both copies now share all three edges with each other
    dup = np.vstack([faces, faces[[5]]])
    mesh = pv.PolyData(mesh.points, np.hstack([np.full((len(dup), 1), 3), dup]).ravel())
    counts = _triangle_edge_neighbor_counts(mesh)
    assert counts[5] >= 3 and counts[-1] >= 3
    out = remove_isolated_cells(mesh)
    assert out.n_cells == len(dup) - 1  # only the flap goes


def test_degenerate_triangle_is_dropped_and_does_not_shield_a_flap():
    mesh = _plane_with_flap()
    faces = mesh.faces.reshape(-1, 4)[:, 1:]
    flap = faces[-1]
    degenerate = np.array([[flap[0], flap[0], flap[1]]])  # shares the flap's attaching edge
    faces = np.vstack([faces, degenerate])
    mesh = pv.PolyData(mesh.points, np.hstack([np.full((len(faces), 1), 3), faces]).ravel())
    out = remove_isolated_cells(mesh)
    assert out.n_cells == len(faces) - 2  # degenerate dropped, flap removed


def test_unsigned_face_indices_are_handled():
    sphere = pv.Sphere(theta_resolution=8, phi_resolution=8)
    faces = sphere.faces.reshape(-1, 4)[:, 1:].astype(np.uint32)
    mesh = pv.PolyData(sphere.points, np.hstack([np.full((len(faces), 1), 3), faces]).ravel())
    assert np.array_equal(
        _triangle_edge_neighbor_counts(mesh), _triangle_edge_neighbor_counts(sphere)
    )
