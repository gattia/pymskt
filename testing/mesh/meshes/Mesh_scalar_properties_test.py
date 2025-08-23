import numpy as np
import pytest
import pyvista as pv

import pymskt as mskt

# Test data
SPHERE = pv.Sphere()


def test_scalar_names_computed_property(sphere=SPHERE):
    """Test that scalar_names is computed dynamically."""
    mesh = mskt.mesh.Mesh(sphere)

    initial_scalar_names = mesh.scalar_names
    initial_n_scalars = mesh.n_scalars

    # Add some scalar data
    mesh.point_data["test_scalar_1"] = np.random.rand(mesh.n_points)
    mesh.point_data["test_scalar_2"] = np.random.rand(mesh.n_points)

    # Verify scalars are updated automatically
    assert mesh.n_scalars == initial_n_scalars + 2
    assert "test_scalar_1" in mesh.scalar_names
    assert "test_scalar_2" in mesh.scalar_names


def test_n_scalars_computed_property(sphere=SPHERE):
    """Test that n_scalars is computed dynamically."""
    mesh = mskt.mesh.Mesh(sphere)

    initial_n_scalars = mesh.n_scalars

    # Add scalar data
    mesh.point_data["test_scalar"] = np.random.rand(mesh.n_points)

    # Verify n_scalars is updated automatically
    assert mesh.n_scalars == initial_n_scalars + 1


def test_private_aliases_backward_compatibility(sphere=SPHERE):
    """Test that private aliases work for backward compatibility."""
    mesh = mskt.mesh.Mesh(sphere)

    # Test that private aliases return the same as public properties
    assert mesh._n_scalars == mesh.n_scalars
    assert mesh._scalar_names == mesh.scalar_names


def test_load_mesh_scalars_is_noop(sphere=SPHERE):
    """Test that load_mesh_scalars is now a no-op."""
    mesh = mskt.mesh.Mesh(sphere)

    result = mesh.load_mesh_scalars()
    assert result is None


def test_scalar_properties_consistency(sphere=SPHERE):
    """Test that scalar properties remain consistent after operations."""
    mesh = mskt.mesh.Mesh(sphere)

    # Add multiple scalars
    mesh.point_data["scalar_1"] = np.random.rand(mesh.n_points)
    mesh.point_data["scalar_2"] = np.random.rand(mesh.n_points)
    mesh.point_data["scalar_3"] = np.random.rand(mesh.n_points)

    # Verify consistency
    assert len(mesh.scalar_names) == mesh.n_scalars
    assert mesh._n_scalars == mesh.n_scalars
    assert mesh._scalar_names == mesh.scalar_names

    # Remove a scalar
    del mesh.point_data["scalar_2"]

    # Verify consistency after removal
    assert len(mesh.scalar_names) == mesh.n_scalars
    assert "scalar_2" not in mesh.scalar_names
    assert mesh._n_scalars == mesh.n_scalars
    assert mesh._scalar_names == mesh.scalar_names
