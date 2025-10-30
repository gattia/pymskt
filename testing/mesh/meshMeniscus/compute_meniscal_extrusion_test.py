"""
Test file for meniscal extrusion computation.

TODO: Implement tests for:
- Testing extrusion calculation with synthetic data (known extrusion values)
- Testing with no extrusion (meniscus within cartilage rim)
- Edge cases: empty compartments, missing meniscus data
"""

import os

import numpy as np
import pytest

from pymskt.mesh import BoneMesh, Mesh
from pymskt.mesh.mesh_meniscus import compute_meniscal_extrusion

# ============================================================================
# Fixtures for Meniscus Shift Tests
# ============================================================================


@pytest.fixture
def tibia_with_menisci():
    """
    Fixture that provides a tibia mesh with cartilage regions and menisci.

    Returns a tuple of (tibia, medial_meniscus, lateral_meniscus, baseline_results)
    """
    test_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(test_dir, "..", "..", "..", "data")
    path_segmentation = os.path.join(data_dir, "SAG_3D_DESS_RIGHT_bones_cart_men_fib-label.nrrd")

    if not os.path.exists(path_segmentation):
        pytest.skip(f"Test data not found: {path_segmentation}")

    # Create tibia mesh with cartilage regions
    tibia = BoneMesh(path_seg_image=path_segmentation, label_idx=6, list_cartilage_labels=[2, 3])
    tibia.create_mesh()
    tibia.calc_cartilage_thickness()
    tibia.assign_cartilage_regions()

    # Create meniscus meshes
    med_meniscus = Mesh(
        path_seg_image=path_segmentation,
        label_idx=10,
    )
    med_meniscus.create_mesh()
    med_meniscus.consistent_faces()

    lat_meniscus = Mesh(
        path_seg_image=path_segmentation,
        label_idx=9,
    )
    lat_meniscus.create_mesh()
    lat_meniscus.consistent_faces()

    # Compute baseline extrusion
    baseline_results = compute_meniscal_extrusion(
        tibia_mesh=tibia,
        medial_meniscus_mesh=med_meniscus,
        lateral_meniscus_mesh=lat_meniscus,
        medial_cart_label=2,
        lateral_cart_label=3,
        scalar_array_name="labels",
        middle_percentile_range=0.1,
    )

    return tibia, med_meniscus, lat_meniscus, baseline_results


# ============================================================================
# Meniscus Shift Tests
# ============================================================================


def test_medial_meniscus_shift_medially_increases_extrusion(tibia_with_menisci):
    """
    Test that shifting medial meniscus medially increases extrusion.

    When the medial meniscus is shifted 5mm medially (away from midline),
    the extrusion value should increase (more positive or less negative).
    """
    tibia, med_meniscus, lat_meniscus, baseline_results = tibia_with_menisci
    baseline_extrusion = baseline_results["medial_extrusion_mm"]

    # Shift medial meniscus medially (5mm in +X direction for right knee)
    med_meniscus_shifted = med_meniscus.copy()
    med_meniscus_shifted.points = med_meniscus_shifted.points + np.array([5.0, 0.0, 0.0])

    # Compute extrusion with shifted meniscus
    results = compute_meniscal_extrusion(
        tibia_mesh=tibia,
        medial_meniscus_mesh=med_meniscus_shifted,
        lateral_meniscus_mesh=lat_meniscus,
        medial_cart_label=2,
        lateral_cart_label=3,
        scalar_array_name="labels",
        middle_percentile_range=0.1,
    )

    # Verify extrusion increased
    assert results["medial_extrusion_mm"] > baseline_extrusion, (
        f"Medial shift should increase extrusion. "
        f"Baseline: {baseline_extrusion:.2f}, Shifted: {results['medial_extrusion_mm']:.2f}"
    )

    print(f"\n✓ Medial meniscus medial shift test passed!")
    print(f"  Baseline: {baseline_extrusion:.2f} mm")
    print(f"  After medial shift: {results['medial_extrusion_mm']:.2f} mm")
    print(f"  Increase: {results['medial_extrusion_mm'] - baseline_extrusion:.2f} mm")


def test_medial_meniscus_shift_laterally_decreases_extrusion(tibia_with_menisci):
    """
    Test that shifting medial meniscus laterally decreases extrusion.

    When the medial meniscus is shifted 5mm laterally (toward midline),
    the extrusion value should decrease (less positive or more negative).
    """
    tibia, med_meniscus, lat_meniscus, baseline_results = tibia_with_menisci
    baseline_extrusion = baseline_results["medial_extrusion_mm"]

    # Shift medial meniscus laterally (5mm in -X direction for right knee)
    med_meniscus_shifted = med_meniscus.copy()
    med_meniscus_shifted.points = med_meniscus_shifted.points - np.array([5.0, 0.0, 0.0])

    # Compute extrusion with shifted meniscus
    results = compute_meniscal_extrusion(
        tibia_mesh=tibia,
        medial_meniscus_mesh=med_meniscus_shifted,
        lateral_meniscus_mesh=lat_meniscus,
        medial_cart_label=2,
        lateral_cart_label=3,
        scalar_array_name="labels",
        middle_percentile_range=0.1,
    )

    # Verify extrusion decreased
    assert results["medial_extrusion_mm"] < baseline_extrusion, (
        f"Lateral shift should decrease extrusion. "
        f"Baseline: {baseline_extrusion:.2f}, Shifted: {results['medial_extrusion_mm']:.2f}"
    )

    print(f"\n✓ Medial meniscus lateral shift test passed!")
    print(f"  Baseline: {baseline_extrusion:.2f} mm")
    print(f"  After lateral shift: {results['medial_extrusion_mm']:.2f} mm")
    print(f"  Decrease: {baseline_extrusion - results['medial_extrusion_mm']:.2f} mm")


def test_lateral_meniscus_shift_laterally_increases_extrusion(tibia_with_menisci):
    """
    Test that shifting lateral meniscus laterally increases extrusion.

    When the lateral meniscus is shifted 5mm laterally (away from midline),
    the extrusion value should increase (more positive or less negative).
    """
    tibia, med_meniscus, lat_meniscus, baseline_results = tibia_with_menisci
    baseline_extrusion = baseline_results["lateral_extrusion_mm"]

    # Shift lateral meniscus laterally (5mm in -X direction for right knee)
    lat_meniscus_shifted = lat_meniscus.copy()
    lat_meniscus_shifted.points = lat_meniscus_shifted.points - np.array([5.0, 0.0, 0.0])

    # Compute extrusion with shifted meniscus
    results = compute_meniscal_extrusion(
        tibia_mesh=tibia,
        medial_meniscus_mesh=med_meniscus,
        lateral_meniscus_mesh=lat_meniscus_shifted,
        medial_cart_label=2,
        lateral_cart_label=3,
        scalar_array_name="labels",
        middle_percentile_range=0.1,
    )

    # Verify extrusion increased
    assert results["lateral_extrusion_mm"] > baseline_extrusion, (
        f"Lateral shift should increase extrusion. "
        f"Baseline: {baseline_extrusion:.2f}, Shifted: {results['lateral_extrusion_mm']:.2f}"
    )

    print(f"\n✓ Lateral meniscus lateral shift test passed!")
    print(f"  Baseline: {baseline_extrusion:.2f} mm")
    print(f"  After lateral shift: {results['lateral_extrusion_mm']:.2f} mm")
    print(f"  Increase: {results['lateral_extrusion_mm'] - baseline_extrusion:.2f} mm")


def test_lateral_meniscus_shift_medially_decreases_extrusion(tibia_with_menisci):
    """
    Test that shifting lateral meniscus medially decreases extrusion.

    When the lateral meniscus is shifted 5mm medially (toward midline),
    the extrusion value should decrease (less positive or more negative).
    """
    tibia, med_meniscus, lat_meniscus, baseline_results = tibia_with_menisci
    baseline_extrusion = baseline_results["lateral_extrusion_mm"]

    # Shift lateral meniscus medially (5mm in +X direction for right knee)
    lat_meniscus_shifted = lat_meniscus.copy()
    lat_meniscus_shifted.points = lat_meniscus_shifted.points + np.array([5.0, 0.0, 0.0])

    # Compute extrusion with shifted meniscus
    results = compute_meniscal_extrusion(
        tibia_mesh=tibia,
        medial_meniscus_mesh=med_meniscus,
        lateral_meniscus_mesh=lat_meniscus_shifted,
        medial_cart_label=2,
        lateral_cart_label=3,
        scalar_array_name="labels",
        middle_percentile_range=0.1,
    )

    # Verify extrusion decreased
    assert results["lateral_extrusion_mm"] < baseline_extrusion, (
        f"Medial shift should decrease extrusion. "
        f"Baseline: {baseline_extrusion:.2f}, Shifted: {results['lateral_extrusion_mm']:.2f}"
    )

    print(f"\n✓ Lateral meniscus medial shift test passed!")
    print(f"  Baseline: {baseline_extrusion:.2f} mm")
    print(f"  After medial shift: {results['lateral_extrusion_mm']:.2f} mm")
    print(f"  Decrease: {baseline_extrusion - results['lateral_extrusion_mm']:.2f} mm")


# ============================================================================
# Convenience API Tests
# ============================================================================


def test_dict_cartilage_labels_replaces_list():
    """
    Test that dict_cartilage_labels can replace list_cartilage_labels.

    Verifies that cartilage thickness and region assignment work with only
    dict_cartilage_labels (no list_cartilage_labels needed).
    """
    # Get path to test data
    test_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(test_dir, "..", "..", "..", "data")
    path_segmentation = os.path.join(data_dir, "SAG_3D_DESS_RIGHT_bones_cart_men_fib-label.nrrd")

    if not os.path.exists(path_segmentation):
        pytest.skip(f"Test data not found: {path_segmentation}")

    # Create tibia with ONLY dict_cartilage_labels
    tibia = BoneMesh(
        path_seg_image=path_segmentation,
        label_idx=6,
        dict_cartilage_labels={"medial": 2, "lateral": 3},
    )

    # Verify list_cartilage_labels property auto-extracts from dict
    assert tibia.list_cartilage_labels == [2, 3]

    # Verify standard operations work
    tibia.create_mesh()
    tibia.calc_cartilage_thickness()  # Should work with dict values
    tibia.assign_cartilage_regions()  # Should work with dict values

    # Verify thickness and labels were assigned
    assert "thickness (mm)" in tibia.point_data
    assert "labels" in tibia.point_data

    print("\n✓ dict_cartilage_labels successfully replaces list_cartilage_labels!")


def test_set_menisci_auto_infers_labels():
    """
    Test that set_menisci() automatically infers labels from dict_cartilage_labels.

    Verifies that cartilage labels don't need to be specified explicitly
    when dict_cartilage_labels is set.
    """
    test_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(test_dir, "..", "..", "..", "data")
    path_segmentation = os.path.join(data_dir, "SAG_3D_DESS_RIGHT_bones_cart_men_fib-label.nrrd")

    if not os.path.exists(path_segmentation):
        pytest.skip(f"Test data not found: {path_segmentation}")

    # Setup tibia
    tibia = BoneMesh(
        path_seg_image=path_segmentation,
        label_idx=6,
        dict_cartilage_labels={"medial": 2, "lateral": 3},
    )
    tibia.create_mesh()
    tibia.calc_cartilage_thickness()
    tibia.assign_cartilage_regions()

    # Create menisci
    med_meniscus = Mesh(path_seg_image=path_segmentation, label_idx=10)
    med_meniscus.create_mesh()
    med_meniscus.consistent_faces()

    lat_meniscus = Mesh(path_seg_image=path_segmentation, label_idx=9)
    lat_meniscus.create_mesh()
    lat_meniscus.consistent_faces()

    # Test: set_menisci WITHOUT explicit labels (should auto-infer from dict)
    tibia.set_menisci(medial_meniscus=med_meniscus, lateral_meniscus=lat_meniscus)

    # Verify labels were cached correctly
    assert tibia._meniscal_cart_labels is not None
    assert tibia._meniscal_cart_labels["medial"] == 2
    assert tibia._meniscal_cart_labels["lateral"] == 3

    print("\n✓ set_menisci() successfully auto-infers labels from dict!")


def test_meniscal_properties_lazy_evaluation():
    """
    Test that meniscal properties auto-compute on first access (lazy evaluation).

    Verifies that calling properties like med_men_extrusion automatically
    triggers computation without explicit compute_meniscal_outcomes() call.
    """
    test_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(test_dir, "..", "..", "..", "data")
    path_segmentation = os.path.join(data_dir, "SAG_3D_DESS_RIGHT_bones_cart_men_fib-label.nrrd")

    if not os.path.exists(path_segmentation):
        pytest.skip(f"Test data not found: {path_segmentation}")

    # Setup
    tibia = BoneMesh(
        path_seg_image=path_segmentation,
        label_idx=6,
        dict_cartilage_labels={"medial": 2, "lateral": 3},
    )
    tibia.create_mesh()
    tibia.calc_cartilage_thickness()
    tibia.assign_cartilage_regions()

    med_meniscus = Mesh(path_seg_image=path_segmentation, label_idx=10)
    med_meniscus.create_mesh()
    med_meniscus.consistent_faces()

    lat_meniscus = Mesh(path_seg_image=path_segmentation, label_idx=9)
    lat_meniscus.create_mesh()
    lat_meniscus.consistent_faces()

    tibia.set_menisci(medial_meniscus=med_meniscus, lateral_meniscus=lat_meniscus)

    # Verify outcomes NOT computed yet
    assert tibia._meniscal_outcomes is None

    # Access property - should trigger auto-computation
    med_extrusion = tibia.med_men_extrusion

    # Verify outcomes NOW computed
    assert tibia._meniscal_outcomes is not None
    assert isinstance(med_extrusion, (int, float, np.number))

    # Access another property - should use cached results (no recomputation)
    lat_extrusion = tibia.lat_men_extrusion
    assert isinstance(lat_extrusion, (int, float, np.number))

    print("\n✓ Properties successfully auto-compute on first access!")
    print(f"  Medial extrusion: {med_extrusion:.2f} mm")
    print(f"  Lateral extrusion: {lat_extrusion:.2f} mm")


def test_meniscal_outcomes_caching():
    """
    Test that meniscal outcomes are properly cached and reused.

    Verifies that:
    - Results are cached after first computation
    - Property values match cached dictionary values
    - All expected metrics are present in cache
    """
    test_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(test_dir, "..", "..", "..", "data")
    path_segmentation = os.path.join(data_dir, "SAG_3D_DESS_RIGHT_bones_cart_men_fib-label.nrrd")

    if not os.path.exists(path_segmentation):
        pytest.skip(f"Test data not found: {path_segmentation}")

    # Setup
    tibia = BoneMesh(
        path_seg_image=path_segmentation,
        label_idx=6,
        dict_cartilage_labels={"medial": 2, "lateral": 3},
    )
    tibia.create_mesh()
    tibia.calc_cartilage_thickness()
    tibia.assign_cartilage_regions()

    med_meniscus = Mesh(path_seg_image=path_segmentation, label_idx=10)
    med_meniscus.create_mesh()
    med_meniscus.consistent_faces()

    lat_meniscus = Mesh(path_seg_image=path_segmentation, label_idx=9)
    lat_meniscus.create_mesh()
    lat_meniscus.consistent_faces()

    tibia.set_menisci(medial_meniscus=med_meniscus, lateral_meniscus=lat_meniscus)

    # Trigger computation via property access
    med_extrusion = tibia.med_men_extrusion
    lat_extrusion = tibia.lat_men_extrusion
    med_coverage = tibia.med_men_coverage
    lat_coverage = tibia.lat_men_coverage

    # Verify all metrics are cached
    assert "medial_extrusion_mm" in tibia._meniscal_outcomes
    assert "lateral_extrusion_mm" in tibia._meniscal_outcomes
    assert "medial_coverage_percent" in tibia._meniscal_outcomes
    assert "lateral_coverage_percent" in tibia._meniscal_outcomes

    # Verify property values match cached values
    assert med_extrusion == tibia._meniscal_outcomes["medial_extrusion_mm"]
    assert lat_extrusion == tibia._meniscal_outcomes["lateral_extrusion_mm"]
    assert med_coverage == tibia._meniscal_outcomes["medial_coverage_percent"]
    assert lat_coverage == tibia._meniscal_outcomes["lateral_coverage_percent"]

    print("\n✓ Meniscal outcomes properly cached and accessible!")


def test_meniscal_values_reasonable():
    """
    Test that computed meniscal values are reasonable.

    Verifies:
    - Extrusion values are numeric types
    - Coverage values are percentages (0-100)
    """
    test_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(test_dir, "..", "..", "..", "data")
    path_segmentation = os.path.join(data_dir, "SAG_3D_DESS_RIGHT_bones_cart_men_fib-label.nrrd")

    if not os.path.exists(path_segmentation):
        pytest.skip(f"Test data not found: {path_segmentation}")

    # Setup
    tibia = BoneMesh(
        path_seg_image=path_segmentation,
        label_idx=6,
        dict_cartilage_labels={"medial": 2, "lateral": 3},
    )
    tibia.create_mesh()
    tibia.calc_cartilage_thickness()
    tibia.assign_cartilage_regions()

    med_meniscus = Mesh(path_seg_image=path_segmentation, label_idx=10)
    med_meniscus.create_mesh()
    med_meniscus.consistent_faces()

    lat_meniscus = Mesh(path_seg_image=path_segmentation, label_idx=9)
    lat_meniscus.create_mesh()
    lat_meniscus.consistent_faces()

    tibia.set_menisci(medial_meniscus=med_meniscus, lateral_meniscus=lat_meniscus)

    # Get values
    med_extrusion = tibia.med_men_extrusion
    lat_extrusion = tibia.lat_men_extrusion
    med_coverage = tibia.med_men_coverage
    lat_coverage = tibia.lat_men_coverage

    # Verify types (accept Python and numpy numeric types)
    assert isinstance(med_extrusion, (int, float, np.number))
    assert isinstance(lat_extrusion, (int, float, np.number))
    assert isinstance(med_coverage, (int, float, np.number))
    assert isinstance(lat_coverage, (int, float, np.number))

    # Verify coverage percentages are in valid range
    assert 0 <= med_coverage <= 100, f"Medial coverage {med_coverage}% outside valid range"
    assert 0 <= lat_coverage <= 100, f"Lateral coverage {lat_coverage}% outside valid range"

    print("\n✓ All meniscal values are reasonable!")
    print(f"  Medial: {med_extrusion:.2f} mm extrusion, {med_coverage:.1f}% coverage")
    print(f"  Lateral: {lat_extrusion:.2f} mm extrusion, {lat_coverage:.1f}% coverage")


# ============================================================================
# TODO: Additional Tests
# ============================================================================


def test_extrusion_synthetic_data():
    """TODO: Test extrusion calculation with synthetic tibia and meniscus meshes."""
    pass


def test_extrusion_no_extrusion():
    """TODO: Test case where meniscus is fully within cartilage rim."""
    pass
