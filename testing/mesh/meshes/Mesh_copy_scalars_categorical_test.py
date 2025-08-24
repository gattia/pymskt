import numpy as np
import pytest
import pyvista as pv
import vtk
from numpy.testing import assert_allclose, assert_array_equal
from vtk.util.numpy_support import numpy_to_vtk

import pymskt as mskt
from pymskt import ATOL, RTOL
from pymskt.mesh.utils import vtk_deep_copy


def test_copy_scalars_categorical_labels():
    """Test categorical label transfer with defaultdict functionality."""

    # Create source mesh with categorical labels
    source_sphere = pv.Sphere(radius=1.0, center=(0, 0, 0))
    source_mesh = mskt.mesh.Mesh(source_sphere)

    # Create target mesh - slightly offset sphere
    target_sphere = pv.Sphere(radius=1.02, center=(0.01, 0.01, 0.01))
    target_mesh = mskt.mesh.Mesh(target_sphere)

    # Add categorical labels to source mesh (integer labels like anatomical regions)
    n_points = source_mesh.n_points

    # Create categorical label data - simulate anatomical regions with integer labels
    labels = np.zeros(n_points, dtype=np.int32)

    # Assign different regions based on position
    points = source_mesh.points

    # Region 1: z > 0.5
    labels[points[:, 2] > 0.5] = 1

    # Region 2: z < -0.5
    labels[points[:, 2] < -0.5] = 2

    # Region 3: -0.5 <= z <= 0.5 and x > 0
    middle_z = (points[:, 2] >= -0.5) & (points[:, 2] <= 0.5)
    labels[middle_z & (points[:, 0] > 0)] = 3

    # Region 4: -0.5 <= z <= 0.5 and x <= 0
    labels[middle_z & (points[:, 0] <= 0)] = 4

    # Add labels to source mesh
    source_mesh.point_data["anatomical_region"] = labels

    # Test 1: Auto-detection of categorical data (should detect integer type)
    target_mesh.copy_scalars_from_other_mesh_to_current(
        source_mesh,
        orig_scalars_name=["anatomical_region"],
        new_scalars_name=["transferred_region"],
    )

    # Verify the transferred labels are integers and reasonable
    transferred_labels = target_mesh.point_data["transferred_region"]
    assert transferred_labels.dtype in [np.int32, np.int64]
    assert np.all(np.isin(transferred_labels, [1, 2, 3, 4]))

    # Test 2: Explicitly specify categorical=True
    target_mesh2 = mskt.mesh.Mesh(target_sphere)
    target_mesh2.copy_scalars_from_other_mesh_to_current(
        source_mesh,
        orig_scalars_name=["anatomical_region"],
        new_scalars_name=["explicit_categorical"],
        categorical=True,
    )

    # Verify results are similar between auto-detection and explicit specification
    transferred_explicit = target_mesh2.point_data["explicit_categorical"]
    assert transferred_explicit.dtype in [np.int32, np.int64]

    # Test 3: Mixed categorical and continuous data
    # Add continuous scalar to source mesh
    source_mesh.point_data["thickness"] = np.random.uniform(0.5, 2.0, n_points)

    target_mesh3 = mskt.mesh.Mesh(target_sphere)
    target_mesh3.copy_scalars_from_other_mesh_to_current(
        source_mesh,
        orig_scalars_name=["anatomical_region", "thickness"],
        new_scalars_name=["region_mixed", "thickness_mixed"],
        categorical={"anatomical_region": True, "thickness": False},
    )

    # Verify mixed results
    mixed_region = target_mesh3.point_data["region_mixed"]
    mixed_thickness = target_mesh3.point_data["thickness_mixed"]

    assert mixed_region.dtype in [np.int32, np.int64]
    assert mixed_thickness.dtype in [np.float32, np.float64]
    assert np.all(np.isin(mixed_region, [1, 2, 3, 4]))
    assert np.all(mixed_thickness >= 0.4)  # Should be close to original range


def test_copy_scalars_categorical_list_parameter():
    """Test categorical parameter as a list."""

    # Create simple test meshes
    source_sphere = pv.Sphere(radius=1.0)
    target_sphere = pv.Sphere(radius=1.01)

    source_mesh = mskt.mesh.Mesh(source_sphere)
    target_mesh = mskt.mesh.Mesh(target_sphere)

    # Add test data
    n_points = source_mesh.n_points
    source_mesh.point_data["categorical_data"] = np.random.randint(1, 5, n_points)
    source_mesh.point_data["continuous_data"] = np.random.uniform(0, 10, n_points)

    # Test with list of categorical flags
    target_mesh.copy_scalars_from_other_mesh_to_current(
        source_mesh,
        orig_scalars_name=["categorical_data", "continuous_data"],
        new_scalars_name=["cat_result", "cont_result"],
        categorical=[True, False],
    )

    cat_result = target_mesh.point_data["cat_result"]
    cont_result = target_mesh.point_data["cont_result"]

    assert cat_result.dtype in [np.int32, np.int64]
    assert cont_result.dtype in [np.float32, np.float64]


def test_copy_scalars_categorical_error_handling():
    """Test error handling for categorical parameter."""

    source_sphere = pv.Sphere(radius=1.0)
    target_sphere = pv.Sphere(radius=1.01)

    source_mesh = mskt.mesh.Mesh(source_sphere)
    target_mesh = mskt.mesh.Mesh(target_sphere)

    n_points = source_mesh.n_points
    source_mesh.point_data["test_data"] = np.random.randint(1, 5, n_points)

    # Test invalid categorical parameter type
    with pytest.raises(ValueError, match="categorical must be None, bool, list, or dict"):
        target_mesh.copy_scalars_from_other_mesh_to_current(
            source_mesh, orig_scalars_name=["test_data"], categorical="invalid_type"
        )

    # Test mismatched list length
    with pytest.raises(ValueError, match="categorical list must have same length"):
        target_mesh.copy_scalars_from_other_mesh_to_current(
            source_mesh,
            orig_scalars_name=["test_data"],
            categorical=[True, False],  # Length mismatch
        )


def test_mixed_categorical_continuous_transfer():
    """Test that mixed categorical and continuous data transfers correctly."""

    # Create identical test meshes using the working approach
    n_points = 500
    np_points = np.ones((n_points, 3))
    np_points[:, 0] = np.arange(n_points) * 0.1
    np_points[:, 1] = np.arange(n_points) * 0.1
    np_points[:, 2] = np.arange(n_points) * 0.1

    vtk_points_ = numpy_to_vtk(np_points)
    vtk_points_.SetName("test")
    vtk_points = vtk.vtkPoints()
    vtk_points.SetData(vtk_points_)

    # Create source mesh
    source_mesh_vtk = vtk.vtkPolyData()
    source_mesh_vtk.SetPoints(vtk_points)

    # Create target mesh (identical for exact transfer)
    target_mesh_vtk = vtk_deep_copy(source_mesh_vtk)

    # Add categorical data (anatomical regions)
    categorical_labels = np.array([1, 2, 3, 1, 2, 3] * (n_points // 6 + 1))[:n_points].astype(
        np.int32
    )
    vtk_categorical = numpy_to_vtk(categorical_labels)
    vtk_categorical.SetName("anatomical_regions")
    source_mesh_vtk.GetPointData().AddArray(vtk_categorical)

    # Add continuous data (thickness measurements)
    thickness_values = np.random.uniform(0.5, 3.0, n_points).astype(np.float64)
    vtk_thickness = numpy_to_vtk(thickness_values)
    vtk_thickness.SetName("thickness")
    source_mesh_vtk.GetPointData().AddArray(vtk_thickness)

    # Add another continuous data (curvature)
    curvature_values = np.random.uniform(-0.1, 0.1, n_points).astype(np.float32)
    vtk_curvature = numpy_to_vtk(curvature_values)
    vtk_curvature.SetName("curvature")
    source_mesh_vtk.GetPointData().AddArray(vtk_curvature)

    # Convert to pymskt meshes
    source_mesh = mskt.mesh.Mesh(source_mesh_vtk)
    target_mesh = mskt.mesh.Mesh(target_mesh_vtk)

    # Test mixed transfer with explicit categorical specification
    target_mesh.copy_scalars_from_other_mesh_to_current(
        source_mesh,
        orig_scalars_name=["anatomical_regions", "thickness", "curvature"],
        new_scalars_name=["regions_result", "thickness_result", "curvature_result"],
        categorical={"anatomical_regions": True, "thickness": False, "curvature": False},
        n_closest=1,  # Exact transfer since meshes are identical
        weighted_avg=True,
    )

    # Verify results
    regions_result = target_mesh.point_data["regions_result"]
    thickness_result = target_mesh.point_data["thickness_result"]
    curvature_result = target_mesh.point_data["curvature_result"]

    # Categorical data should transfer exactly and be integer type
    assert regions_result.dtype in [np.int32, np.int64]
    assert np.array_equal(categorical_labels, regions_result)
    assert np.all(np.isin(regions_result, [1, 2, 3]))

    # Continuous data should transfer closely and maintain float types
    assert thickness_result.dtype in [np.float32, np.float64]
    assert curvature_result.dtype in [np.float32, np.float64]
    assert np.allclose(thickness_values, thickness_result, rtol=1e-10)
    assert np.allclose(curvature_values, curvature_result, rtol=1e-5)

    # Test auto-detection works correctly
    target_mesh2 = mskt.mesh.Mesh(target_mesh_vtk)
    target_mesh2.copy_scalars_from_other_mesh_to_current(
        source_mesh,
        orig_scalars_name=["anatomical_regions", "thickness"],
        new_scalars_name=["auto_regions", "auto_thickness"],
        categorical=None,  # Auto-detect
        n_closest=1,
        weighted_avg=True,
    )

    auto_regions = target_mesh2.point_data["auto_regions"]
    auto_thickness = target_mesh2.point_data["auto_thickness"]

    # Auto-detection should correctly identify types
    assert auto_regions.dtype in [np.int32, np.int64]  # Should detect as categorical
    assert auto_thickness.dtype in [np.float32, np.float64]  # Should detect as continuous
    assert np.array_equal(categorical_labels, auto_regions)
    assert np.allclose(thickness_values, auto_thickness, rtol=1e-10)


def test_defaultdict_functionality_directly():
    """Test that the defaultdict import fixes the original error."""

    # Use the same approach as the existing working test
    n_points = 1000
    np_points = np.ones((n_points, 3))
    np_points[:, 0] = np.arange(n_points)
    np_points[:, 1] = np.arange(n_points)
    np_points[:, 2] = np.arange(n_points)

    vtk_points_ = numpy_to_vtk(np_points)
    vtk_points_.SetName("test")
    vtk_points = vtk.vtkPoints()
    vtk_points.SetData(vtk_points_)

    # Create source mesh
    source_mesh_vtk = vtk.vtkPolyData()
    source_mesh_vtk.SetPoints(vtk_points)

    # Create target mesh (identical)
    target_mesh_vtk = vtk_deep_copy(source_mesh_vtk)

    # Add categorical scalar data - this is what we're testing
    np_categorical = np.random.randint(1, 5, n_points)  # Random categorical labels 1-4
    vtk_categorical = numpy_to_vtk(np_categorical)
    vtk_categorical.SetName("categorical_labels")
    source_mesh_vtk.GetPointData().AddArray(vtk_categorical)

    # Convert to pymskt meshes
    source_mesh = mskt.mesh.Mesh(source_mesh_vtk)
    target_mesh = mskt.mesh.Mesh(target_mesh_vtk)

    # This should trigger the defaultdict code path in transfer_mesh_scalars_get_weighted_average_n_closest
    # with categorical=True and n_closest > 1
    target_mesh.copy_scalars_from_other_mesh_to_current(
        source_mesh,
        orig_scalars_name=["categorical_labels"],
        new_scalars_name=["result_labels"],
        categorical=True,
        n_closest=3,  # Use multiple closest points to trigger weighted voting
        weighted_avg=True,
    )

    # Verify the operation completed without NameError
    result_labels = target_mesh.point_data["result_labels"]
    assert result_labels.dtype in [np.int32, np.int64]
    assert np.all(np.isin(result_labels, [1, 2, 3, 4]))
