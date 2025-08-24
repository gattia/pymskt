import numpy as np
import pytest
import pyvista as pv
import vtk
from numpy.testing import assert_allclose
from vtk.util.numpy_support import numpy_to_vtk

import pymskt as mskt
from pymskt.mesh.utils import vtk_deep_copy


def test_pyvista_sphere_with_normals():
    """Test the original bug: pyvista spheres with Normals vectors + custom scalars."""

    # This exact case was failing before the fix
    source_sphere = pv.Sphere(radius=1.0)
    target_sphere = pv.Sphere(radius=1.0)

    source_mesh = mskt.mesh.Mesh(source_sphere)
    target_mesh = mskt.mesh.Mesh(target_sphere)

    # Add categorical data like the user was doing
    n_points = source_mesh.n_points
    test_labels = np.array([1, 2, 1, 2] * (n_points // 4 + 1))[:n_points]
    source_mesh.point_data["test_labels"] = test_labels

    # This was failing with "ValueError: setting an array element with a sequence"
    target_mesh.copy_scalars_from_other_mesh_to_current(
        source_mesh,
        orig_scalars_name=["test_labels"],
        new_scalars_name=["test_labels"],
        categorical=True,
        n_closest=3,
        weighted_avg=True,
    )

    # Verify it worked
    result = target_mesh.point_data["test_labels"]
    assert result.dtype in [np.int32, np.int64]
    assert np.all(np.isin(result, [1, 2]))


def test_vector_transfer():
    """Test that vector data (like Normals) transfers correctly."""

    source_sphere = pv.Sphere(radius=1.0)
    target_sphere = pv.Sphere(radius=1.0)  # Same for exact transfer

    source_mesh = mskt.mesh.Mesh(source_sphere)
    target_mesh = mskt.mesh.Mesh(target_sphere)

    # Transfer the Normals (3D vectors)
    target_mesh.copy_scalars_from_other_mesh_to_current(
        source_mesh,
        orig_scalars_name=["Normals"],
        new_scalars_name=["transferred_normals"],
        categorical=False,  # Vectors should be continuous
        n_closest=1,  # Exact transfer
        weighted_avg=True,
    )

    original_normals = source_mesh.point_data["Normals"]
    transferred_normals = target_mesh.point_data["transferred_normals"]

    # Should preserve vector structure and values
    assert transferred_normals.shape == original_normals.shape
    assert transferred_normals.ndim == 2  # Should be 2D (n_points, 3)
    assert transferred_normals.shape[1] == 3  # Should be 3D vectors
    assert np.allclose(original_normals, transferred_normals, rtol=1e-5)


def test_mixed_vectors_and_scalars():
    """Test the core fix: mixed vector and scalar arrays transfer together."""

    source_sphere = pv.Sphere(radius=1.0)
    target_sphere = pv.Sphere(radius=1.0)

    source_mesh = mskt.mesh.Mesh(source_sphere)
    target_mesh = mskt.mesh.Mesh(target_sphere)

    # Add multiple types of data
    n_points = source_mesh.n_points
    source_mesh.point_data["regions"] = np.random.randint(1, 4, n_points)
    source_mesh.point_data["thickness"] = np.random.uniform(0.5, 2.0, n_points)

    # Transfer everything: Normals (3D vectors), regions (categorical), thickness (continuous)
    target_mesh.copy_scalars_from_other_mesh_to_current(
        source_mesh,
        orig_scalars_name=["Normals", "regions", "thickness"],
        new_scalars_name=["norm_result", "reg_result", "thick_result"],
        categorical={"regions": True, "Normals": False, "thickness": False},
        n_closest=1,
        weighted_avg=True,
    )

    # Verify all data types preserved
    norm_result = target_mesh.point_data["norm_result"]
    reg_result = target_mesh.point_data["reg_result"]
    thick_result = target_mesh.point_data["thick_result"]

    assert norm_result.ndim == 2 and norm_result.shape[1] == 3  # 3D vectors
    assert reg_result.ndim == 1  # Scalar
    assert thick_result.ndim == 1  # Scalar

    assert norm_result.dtype in [np.float32, np.float64]  # Continuous
    assert reg_result.dtype in [np.int32, np.int64]  # Categorical
    assert thick_result.dtype in [np.float32, np.float64]  # Continuous


def test_vector_categorical_error():
    """Test that trying to make vectors categorical raises proper error."""

    source_sphere = pv.Sphere(radius=1.0)
    target_sphere = pv.Sphere(radius=1.0)

    source_mesh = mskt.mesh.Mesh(source_sphere)
    target_mesh = mskt.mesh.Mesh(target_sphere)

    # Should error when trying to make Normals categorical
    with pytest.raises(
        ValueError, match="Array 'Normals' is a vector.*cannot be treated as categorical"
    ):
        target_mesh.copy_scalars_from_other_mesh_to_current(
            source_mesh,
            orig_scalars_name=["Normals"],
            categorical=True,  # This should fail
            weighted_avg=True,
        )


def test_auto_detection_with_vectors():
    """Test that auto-detection correctly handles vectors as continuous."""

    source_sphere = pv.Sphere(radius=1.0)
    target_sphere = pv.Sphere(radius=1.0)

    source_mesh = mskt.mesh.Mesh(source_sphere)
    target_mesh = mskt.mesh.Mesh(target_sphere)

    # Add integer scalar data (should be detected as categorical)
    n_points = source_mesh.n_points
    source_mesh.point_data["int_regions"] = np.random.randint(1, 5, n_points).astype(np.int32)

    # Auto-detect: Normals should be continuous, int_regions should be categorical
    target_mesh.copy_scalars_from_other_mesh_to_current(
        source_mesh,
        orig_scalars_name=["Normals", "int_regions"],
        new_scalars_name=["auto_normals", "auto_regions"],
        categorical=None,  # Auto-detect
        n_closest=1,
        weighted_avg=True,
    )

    auto_normals = target_mesh.point_data["auto_normals"]
    auto_regions = target_mesh.point_data["auto_regions"]

    # Normals should be detected as continuous (float), regions as categorical (int)
    assert auto_normals.dtype in [np.float32, np.float64]
    assert auto_regions.dtype in [np.int32, np.int64]
    assert auto_normals.ndim == 2  # Vector preserved
    assert auto_regions.ndim == 1  # Scalar preserved


def test_custom_vector_data():
    """Test vector transfer with custom vector arrays."""

    # Create simple test case with known vectors
    n_points = 4
    points = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=np.float64)

    # Create source mesh
    vtk_points_ = numpy_to_vtk(points)
    vtk_points = vtk.vtkPoints()
    vtk_points.SetData(vtk_points_)
    source_mesh_vtk = vtk.vtkPolyData()
    source_mesh_vtk.SetPoints(vtk_points)

    # Create target mesh (offset points)
    target_points = points + 0.1  # Small offset
    vtk_target_points_ = numpy_to_vtk(target_points)
    vtk_target_points = vtk.vtkPoints()
    vtk_target_points.SetData(vtk_target_points_)
    target_mesh_vtk = vtk.vtkPolyData()
    target_mesh_vtk.SetPoints(vtk_target_points)

    # Add vector data
    vector_data = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 1]], dtype=np.float64)
    vtk_vectors = numpy_to_vtk(vector_data)
    vtk_vectors.SetName("test_vectors")
    source_mesh_vtk.GetPointData().AddArray(vtk_vectors)

    source_mesh = mskt.mesh.Mesh(source_mesh_vtk)
    target_mesh = mskt.mesh.Mesh(target_mesh_vtk)

    # Transfer vector data
    target_mesh.copy_scalars_from_other_mesh_to_current(
        source_mesh,
        orig_scalars_name=["test_vectors"],
        new_scalars_name=["transferred_vectors"],
        categorical=False,
        n_closest=2,
        weighted_avg=True,
    )

    transferred_vectors = target_mesh.point_data["transferred_vectors"]

    # Verify vector structure is preserved and values are reasonable
    assert transferred_vectors.shape == vector_data.shape
    assert transferred_vectors.dtype in [np.float32, np.float64]
    assert np.all(np.isfinite(transferred_vectors))  # No NaN or inf values


def test_higher_dimensional_vectors():
    """Test that the flattening approach works with higher-dimensional data."""

    # Create test mesh
    n_points = 10
    points = np.random.random((n_points, 3))

    vtk_points_ = numpy_to_vtk(points)
    vtk_points = vtk.vtkPoints()
    vtk_points.SetData(vtk_points_)
    source_mesh_vtk = vtk.vtkPolyData()
    source_mesh_vtk.SetPoints(vtk_points)
    target_mesh_vtk = vtk_deep_copy(source_mesh_vtk)

    # Add 6-component data (e.g., stress tensor: xx, yy, zz, xy, xz, yz)
    stress_data = np.random.uniform(-1, 1, (n_points, 6)).astype(np.float64)
    vtk_stress = numpy_to_vtk(stress_data)
    vtk_stress.SetName("stress_tensor")
    source_mesh_vtk.GetPointData().AddArray(vtk_stress)

    source_mesh = mskt.mesh.Mesh(source_mesh_vtk)
    target_mesh = mskt.mesh.Mesh(target_mesh_vtk)

    # Transfer 6-component data
    target_mesh.copy_scalars_from_other_mesh_to_current(
        source_mesh,
        orig_scalars_name=["stress_tensor"],
        new_scalars_name=["transferred_stress"],
        categorical=False,
        n_closest=1,  # Exact transfer
        weighted_avg=True,
    )

    transferred_stress = target_mesh.point_data["transferred_stress"]

    # Should preserve shape and values
    assert transferred_stress.shape == stress_data.shape
    assert transferred_stress.shape[1] == 6  # 6 components
    assert np.allclose(stress_data, transferred_stress, rtol=1e-10)


def test_explicit_vector_categorical_dict_error():
    """Test explicit dict specification with vector categorical error."""

    source_sphere = pv.Sphere(radius=1.0)
    target_sphere = pv.Sphere(radius=1.0)

    source_mesh = mskt.mesh.Mesh(source_sphere)
    target_mesh = mskt.mesh.Mesh(target_sphere)

    # Add scalar data too
    n_points = source_mesh.n_points
    source_mesh.point_data["scalar_data"] = np.random.randint(1, 3, n_points)

    # Should error when dict explicitly sets Normals as categorical
    with pytest.raises(
        ValueError, match="Array 'Normals' is a vector.*cannot be treated as categorical"
    ):
        target_mesh.copy_scalars_from_other_mesh_to_current(
            source_mesh,
            orig_scalars_name=["Normals", "scalar_data"],
            categorical={"Normals": True, "scalar_data": True},  # Normals=True should fail
            weighted_avg=True,
        )
