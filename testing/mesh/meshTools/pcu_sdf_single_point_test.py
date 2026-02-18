"""
Test that pcu_sdf returns correct signed distances for a single query point.

PCU's signed_distance_to_mesh returns incorrect values when called with
exactly 1 query point (shape (1, 3)). The workaround in pcu_sdf duplicates
the point so PCU sees n >= 2, then returns only the first result.
"""

import numpy as np
import pyvista as pv

from pymskt.mesh import Mesh
from pymskt.mesh.meshTools import pcu_sdf


def test_single_point_sdf_matches_multi_point():
    """Single-point SDF should match the same point queried alongside others."""
    cyl = pv.Cylinder(
        center=[0, 0, 0], direction=(0, 0, 1), radius=5.0, height=20.0, resolution=200
    )
    mesh = Mesh(cyl.triangulate())

    surface_pt = np.array([[5.0, 0.0, 0.0]])
    interior_pt = np.array([[0.0, 0.0, 0.0]])

    sdf_single_surface = pcu_sdf(surface_pt, mesh)
    sdf_single_interior = pcu_sdf(interior_pt, mesh)
    sdf_multi = pcu_sdf(np.vstack([surface_pt, interior_pt]), mesh)

    # Single-point results must match multi-point results
    np.testing.assert_allclose(sdf_single_surface[0], sdf_multi[0], atol=1e-10)
    np.testing.assert_allclose(sdf_single_interior[0], sdf_multi[1], atol=1e-10)


def test_single_point_surface_sdf_near_zero():
    """A point on the mesh surface should have SDF close to zero."""
    cyl = pv.Cylinder(
        center=[0, 0, 0], direction=(0, 0, 1), radius=5.0, height=20.0, resolution=200
    )
    mesh = Mesh(cyl.triangulate())

    surface_pt = np.array([[5.0, 0.0, 0.0]])
    sdf = pcu_sdf(surface_pt, mesh)

    # Discretization error for res=200: r*(1-cos(pi/n)) ~ 0.0006
    assert abs(sdf[0]) < 0.01, f"Surface point SDF should be ~0, got {sdf[0]}"


def test_single_point_interior_sdf_negative():
    """A point inside the mesh should have negative SDF."""
    cyl = pv.Cylinder(
        center=[0, 0, 0], direction=(0, 0, 1), radius=5.0, height=20.0, resolution=200
    )
    mesh = Mesh(cyl.triangulate())

    interior_pt = np.array([[0.0, 0.0, 0.0]])
    sdf = pcu_sdf(interior_pt, mesh)

    assert sdf[0] < 0, f"Interior point SDF should be negative, got {sdf[0]}"


def test_single_point_exterior_sdf_positive():
    """A point outside the mesh should have positive SDF."""
    cyl = pv.Cylinder(
        center=[0, 0, 0], direction=(0, 0, 1), radius=5.0, height=20.0, resolution=200
    )
    mesh = Mesh(cyl.triangulate())

    exterior_pt = np.array([[10.0, 0.0, 0.0]])
    sdf = pcu_sdf(exterior_pt, mesh)

    assert sdf[0] > 0, f"Exterior point SDF should be positive, got {sdf[0]}"


if __name__ == "__main__":
    test_single_point_sdf_matches_multi_point()
    print("PASS: single-point matches multi-point")

    test_single_point_surface_sdf_near_zero()
    print("PASS: surface point SDF near zero")

    test_single_point_interior_sdf_negative()
    print("PASS: interior point SDF negative")

    test_single_point_exterior_sdf_positive()
    print("PASS: exterior point SDF positive")

    print("\nAll tests passed!")
