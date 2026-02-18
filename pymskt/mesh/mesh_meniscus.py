"""
Meniscus mesh class and analysis functions for computing meniscal outcomes, 
including extrusion and coverage.

This module provides functionality to analyze meniscal function using healthy cartilage
reference masks. Key metrics include:
- Meniscal extrusion: how far the meniscus extends beyond the cartilage rim
- Meniscal coverage: percentage of cartilage area covered by meniscus

All distances are in mm, areas in mm², and coverage in mm² and percentage.
"""

import numpy as np

from pymskt.mesh.meshes import Mesh


class MeniscusMesh(Mesh):
    """
    Class to create, store, and process meniscus meshes with specialized
    analysis functions for meniscal extrusion and coverage calculations.

    Parameters
    ----------
    mesh : vtk.vtkPolyData, optional
        vtkPolyData object that is basis of surface mesh, by default None
    seg_image : SimpleITK.Image, optional
        Segmentation image that can be used to create surface mesh, by default None
    path_seg_image : str, optional
        Path to a medical image (.nrrd) to load and create mesh from, by default None
    label_idx : int, optional
        Label of anatomy of interest, by default None
    min_n_pixels : int, optional
        All islands smaller than this size are dropped, by default 1000
    meniscus_type : str, optional
        Type of meniscus ('medial' or 'lateral'), by default None

    Attributes
    ----------
    meniscus_type : str
        Type of meniscus ('medial' or 'lateral')

    Examples
    --------
    >>> med_meniscus = MeniscusMesh(
    ...     path_seg_image='meniscus_seg.nrrd',
    ...     label_idx=1,
    ...     meniscus_type='medial'
    ... )
    """

    def __init__(
        self,
        mesh=None,
        seg_image=None,
        path_seg_image=None,
        label_idx=None,
        min_n_pixels=1000,
        meniscus_type=None,
    ):
        super().__init__(
            mesh=mesh,
            seg_image=seg_image,
            path_seg_image=path_seg_image,
            label_idx=label_idx,
            min_n_pixels=min_n_pixels,
        )
        self._meniscus_type = meniscus_type

    @property
    def meniscus_type(self):
        """Get the meniscus type."""
        return self._meniscus_type

    @meniscus_type.setter
    def meniscus_type(self, new_meniscus_type):
        """Set the meniscus type with validation."""
        if new_meniscus_type not in [None, "medial", "lateral"]:
            raise ValueError("meniscus_type must be None, 'medial', or 'lateral'")
        self._meniscus_type = new_meniscus_type


# ============================================================================
# Helper Functions
# ============================================================================


def compute_tibia_axes(
    tibia_mesh,
    medial_cart_label,
    lateral_cart_label,
    scalar_array_name="labels",
):
    """
    Compute anatomical axes (ML, IS, AP) from tibial cartilage regions.

    Uses PCA on combined cartilage points to find the tibial plateau normal (IS axis).
    The superior direction is determined by checking which side the bone is on
    relative to the cartilage. ML axis is from medial to lateral cartilage centers.
    AP axis is the cross product of ML and IS.

    Parameters
    ----------
    tibia_mesh : BoneMesh or Mesh
        Tibia mesh with scalar values indicating cartilage regions
    medial_cart_label : int or float
        Scalar value indicating medial tibial cartilage region
    lateral_cart_label : int or float
        Scalar value indicating lateral tibial cartilage region
    scalar_array_name : str, optional
        Name of scalar array containing region labels, by default 'labels'

    Returns
    -------
    dict
        Dictionary containing:
        - 'ml_axis': medial-lateral axis vector (medial to lateral, unit vector)
        - 'is_axis': inferior-superior axis vector (unit vector pointing superior)
        - 'ap_axis': anterior-posterior axis vector (unit vector)
        - 'medial_center': medial cartilage center point
        - 'lateral_center': lateral cartilage center point

    Examples
    --------
    >>> axes = compute_tibia_axes(tibia, med_cart_label=2, lat_cart_label=3)
    >>> ml_axis = axes['ml_axis']
    >>> is_axis = axes['is_axis']
    """
    # Get scalar array
    region_array = tibia_mesh[scalar_array_name]

    # Extract cartilage points
    med_tib_cart_mask = region_array == medial_cart_label
    lat_tib_cart_mask = region_array == lateral_cart_label

    med_tib_cart_points = tibia_mesh.points[med_tib_cart_mask]
    lat_tib_cart_points = tibia_mesh.points[lat_tib_cart_mask]
    tib_cart_points = np.concatenate([med_tib_cart_points, lat_tib_cart_points], axis=0)

    # Do PCA to get the three axes of the tib_cart_points and take the last
    # one as the inf/sup (normal to plateau)
    X = tib_cart_points - tib_cart_points.mean(axis=0, keepdims=True)  # (N,3)
    # PCA via SVD: X = U S Vt, rows of Vt are PCs
    U, S, Vt = np.linalg.svd(X, full_matrices=False)
    pc1, pc2, pc3 = Vt  # already orthonormal

    is_axis = pc3

    # From the PCA we can't know what is up. Check which side the bone is on
    # relative to the cartilage. The opposite direction from bone to cartilage is IS.
    mean_tib = np.mean(tibia_mesh.points, axis=0)
    mean_cart = np.mean(tib_cart_points, axis=0)

    # Update is_axis direction based on where mean_tib is relative to mean_cart
    if np.dot(mean_tib - mean_cart, is_axis) > 0:
        is_axis = -is_axis

    # Compute ML axis from cartilage centers
    med_tib_center = np.mean(med_tib_cart_points, axis=0)
    lat_tib_center = np.mean(lat_tib_cart_points, axis=0)

    ml_axis = lat_tib_center - med_tib_center
    ml_axis = ml_axis / np.linalg.norm(ml_axis)

    # Compute AP axis as cross product
    # NOTE: AP axis direction is not always same (front vs back)
    # without inputting side (right/left). So, left it just as a general axis.
    ap_axis = np.cross(ml_axis, is_axis)
    ap_axis = ap_axis / np.linalg.norm(ap_axis)

    return {
        "ml_axis": ml_axis,
        "is_axis": is_axis,
        "ap_axis": ap_axis,
        "medial_center": med_tib_center,
        "lateral_center": lat_tib_center,
    }


def _compute_extrusion_from_points(
    cart_points,
    men_points,
    ml_axis,
    side,
):
    """
    Compute extrusion by comparing ML extremes of cartilage and meniscus.

    Helper function that projects points onto ML axis and computes the
    signed extrusion distance.

    Parameters
    ----------
    cart_points : np.ndarray
        Cartilage points (N x 3)
    men_points : np.ndarray
        Meniscus points (M x 3)
    ml_axis : np.ndarray
        Medial-lateral axis vector
    side : str
        'med', 'medial', 'lat', or 'lateral'

    Returns
    -------
    float
        Extrusion distance in mm (positive = extruded beyond cartilage)
    """
    cart_points_ml = np.dot(cart_points, ml_axis)
    men_points_ml = np.dot(men_points, ml_axis)

    if side in ["med", "medial"]:
        cart_edge = np.min(cart_points_ml)
        men_edge = np.min(men_points_ml)
        extrusion = cart_edge - men_edge
    elif side in ["lat", "lateral"]:
        cart_edge = np.max(cart_points_ml)
        men_edge = np.max(men_points_ml)
        extrusion = men_edge - cart_edge
    else:
        raise ValueError(f"Invalid side: {side}, must be one of: med, medial, lat, lateral")

    return extrusion


def _compute_middle_region_extrusion(
    cart_points,
    men_points,
    ap_axis,
    ml_axis,
    side,
    middle_percentile_range,
):
    """
    Compute extrusion using middle percentile range along AP axis.

    This helper function focuses on the central portion of the AP range
    to avoid edge effects at the anterior and posterior extremes.

    Parameters
    ----------
    cart_points : np.ndarray
        Cartilage points (N x 3)
    men_points : np.ndarray
        Meniscus points (M x 3)
    ap_axis : np.ndarray
        Anterior-posterior axis vector
    ml_axis : np.ndarray
        Medial-lateral axis vector
    side : str
        'med', 'medial', 'lat', or 'lateral'
    middle_percentile_range : float
        Fraction of AP range to use (centered on middle)

    Returns
    -------
    float
        Extrusion distance in mm (positive = extruded beyond cartilage)
    """
    # Project cartilage points onto AP axis
    cart_points_ap = np.dot(cart_points, ap_axis)
    min_cart_ap = np.min(cart_points_ap)
    max_cart_ap = np.max(cart_points_ap)

    # Get the middle +/- middle_percentile_range/2 of the cartilage along AP axis
    middle_ap_cartilage = (min_cart_ap + max_cart_ap) / 2
    min_max_ap_cartilage_range = max_cart_ap - min_cart_ap
    plus_minus_ap_cartilage_range = min_max_ap_cartilage_range * middle_percentile_range / 2
    lower_ap_cartilage = middle_ap_cartilage - plus_minus_ap_cartilage_range
    upper_ap_cartilage = middle_ap_cartilage + plus_minus_ap_cartilage_range

    # Get points within the middle AP range for cartilage
    ap_cart_indices = (cart_points_ap >= lower_ap_cartilage) & (
        cart_points_ap <= upper_ap_cartilage
    )
    ml_cart_points = cart_points[ap_cart_indices]

    # Project meniscus points onto AP axis
    men_points_ap = np.dot(men_points, ap_axis)

    # Get points within the middle AP range for meniscus
    ap_men_indices = (men_points_ap >= lower_ap_cartilage) & (men_points_ap <= upper_ap_cartilage)
    ml_men_points = men_points[ap_men_indices]

    # Compute extrusion
    extrusion = _compute_extrusion_from_points(
        cart_points=ml_cart_points,
        men_points=ml_men_points,
        ml_axis=ml_axis,
        side=side,
    )

    return extrusion


def _get_single_compartment_coverage(
    tibia_mesh,
    meniscus_mesh,
    cart_label,
    is_direction,
    side_name,
    scalar_array_name,
    ray_cast_length=10.0,
):
    """
    Compute meniscal coverage for a single compartment.

    Helper function that performs ray casting from tibia to meniscus and
    computes the area of cartilage covered by meniscus.

    Parameters
    ----------
    tibia_mesh : BoneMesh or Mesh
        Tibia mesh with cartilage region labels
    meniscus_mesh : MeniscusMesh or Mesh
        Meniscus mesh for this compartment
    cart_label : int or float
        Label value for this compartment's cartilage
    is_direction : np.ndarray
        Inferior-superior direction vector for ray casting
    side_name : str
        Name for this side ('med' or 'lat') used in output keys
    scalar_array_name : str
        Name of scalar array containing region labels
    ray_cast_length : float, optional
        Length of rays to cast, by default 20.0 mm

    Returns
    -------
    dict
        Dictionary containing:
        - '{side_name}_cart_men_coverage': coverage percentage
        - '{side_name}_cart_men_area': covered area (mm²)
        - '{side_name}_cart_area': total cartilage area (mm²)
    """
    # Calculate distance from tibia to meniscus along IS direction
    tibia_mesh.calc_distance_to_other_mesh(
        list_other_meshes=[meniscus_mesh],
        ray_cast_length=ray_cast_length,
        name=f"{side_name}_men_dist_mm",
        direction=is_direction,
    )

    # Create binary masks
    binary_mask_men_above = tibia_mesh[f"{side_name}_men_dist_mm"] > 0
    binary_mask_cart = tibia_mesh[scalar_array_name] == cart_label

    tibia_mesh[f"{side_name}_men_above"] = binary_mask_men_above.astype(float)
    tibia_mesh[f"{side_name}_cart"] = binary_mask_cart.astype(float)

    # Extract cartilage submesh
    tibia_cart = tibia_mesh.copy()
    tibia_cart.remove_points(~binary_mask_cart, inplace=True)
    tibia_cart.clean(inplace=True)
    area_cart = tibia_cart.area

    # Extract covered cartilage submesh
    tibia_cart_men = tibia_cart.copy()
    tibia_cart_men.remove_points(tibia_cart_men[f"{side_name}_men_above"] == 0, inplace=True)
    tibia_cart_men.clean(inplace=True)
    area_cart_men = tibia_cart_men.area

    # Calculate coverage percentage
    if area_cart == 0:
        raise ValueError(
            f"Cartilage region is empty (area = 0) for compartment '{side_name}'. "
            "Cannot compute meniscal coverage. This likely indicates invalid input data."
        )
    percent_cart_men_coverage = (area_cart_men / area_cart) * 100

    return {
        f"{side_name}_cart_men_coverage": percent_cart_men_coverage,
        f"{side_name}_cart_men_area": area_cart_men,
        f"{side_name}_cart_area": area_cart,
    }


# ============================================================================
# Main Analysis Functions
# ============================================================================


def compute_meniscal_extrusion(
    tibia_mesh,
    medial_meniscus_mesh,
    lateral_meniscus_mesh,
    medial_cart_label,
    lateral_cart_label,
    scalar_array_name="labels",
    middle_percentile_range=0.1,
):
    """
    Compute meniscal extrusion for both medial and lateral menisci.

    Extrusion is computed by comparing the ML extremes of cartilage and meniscus
    within the middle portion of the AP range. This avoids edge effects at the
    anterior and posterior extremes.

    Parameters
    ----------
    tibia_mesh : BoneMesh or Mesh
        Tibia mesh with scalar values indicating cartilage regions from reference
    medial_meniscus_mesh : MeniscusMesh or Mesh
        Medial meniscus mesh
    lateral_meniscus_mesh : MeniscusMesh or Mesh
        Lateral meniscus mesh
    medial_cart_label : int or float
        Scalar value indicating medial cartilage region
    lateral_cart_label : int or float
        Scalar value indicating lateral cartilage region
    scalar_array_name : str, optional
        Name of scalar array containing region labels, by default 'labels'
    middle_percentile_range : float, optional
        Fraction of AP range to use for extrusion measurement (centered), by default 0.1

    Returns
    -------
    dict
        Dictionary containing extrusion metrics (all distances in mm, positive = extruded):
        - 'medial_extrusion_mm': medial extrusion distance
        - 'lateral_extrusion_mm': lateral extrusion distance
        - 'ml_axis': ML axis vector
        - 'ap_axis': AP axis vector
        - 'is_axis': IS axis vector

    Notes
    -----
    Extrusion sign convention: positive values indicate meniscus extends
    beyond the cartilage rim. Negative values indicate the meniscus is contained
    within the cartilage boundaries.

    Examples
    --------
    >>> results = compute_meniscal_extrusion(
    ...     tibia, med_meniscus, lat_meniscus,
    ...     medial_cart_label=2, lateral_cart_label=3
    ... )
    >>> print(f"Medial extrusion: {results['medial_extrusion_mm']:.2f} mm")
    """
    # Compute anatomical axes
    axes = compute_tibia_axes(tibia_mesh, medial_cart_label, lateral_cart_label, scalar_array_name)

    ml_axis = axes["ml_axis"]
    ap_axis = axes["ap_axis"]
    is_axis = axes["is_axis"]

    # Get cartilage points
    region_array = tibia_mesh[scalar_array_name]
    med_cart_indices = region_array == medial_cart_label
    lat_cart_indices = region_array == lateral_cart_label

    med_cart_points = tibia_mesh.points[med_cart_indices]
    lat_cart_points = tibia_mesh.points[lat_cart_indices]

    # Initialize results
    results = {
        "ml_axis": ml_axis,
        "ap_axis": ap_axis,
        "is_axis": is_axis,
    }

    # Compute medial extrusion (only if medial meniscus provided)
    if medial_meniscus_mesh is not None:
        med_men_points = medial_meniscus_mesh.points
        med_men_extrusion = _compute_middle_region_extrusion(
            cart_points=med_cart_points,
            men_points=med_men_points,
            ap_axis=ap_axis,
            ml_axis=ml_axis,
            side="med",
            middle_percentile_range=middle_percentile_range,
        )
        results["medial_extrusion_mm"] = med_men_extrusion

    # Compute lateral extrusion (only if lateral meniscus provided)
    if lateral_meniscus_mesh is not None:
        lat_men_points = lateral_meniscus_mesh.points
        lat_men_extrusion = _compute_middle_region_extrusion(
            cart_points=lat_cart_points,
            men_points=lat_men_points,
            ap_axis=ap_axis,
            ml_axis=ml_axis,
            side="lat",
            middle_percentile_range=middle_percentile_range,
        )
        results["lateral_extrusion_mm"] = lat_men_extrusion

    return results


def compute_meniscal_coverage(
    tibia_mesh,
    medial_meniscus_mesh,
    lateral_meniscus_mesh,
    medial_cart_label,
    lateral_cart_label,
    scalar_array_name="labels",
    ray_cast_length=10.0,
):
    """
    Compute meniscal coverage using superior-inferior ray casting.

    Coverage is computed by casting rays in the IS direction from tibial cartilage
    reference points and checking for meniscus intersections. Areas are computed
    using PyVista's mesh area calculations.

    Parameters
    ----------
    tibia_mesh : BoneMesh or Mesh
        Tibia mesh with scalar values indicating cartilage regions from reference
    medial_meniscus_mesh : MeniscusMesh or Mesh
        Medial meniscus mesh
    lateral_meniscus_mesh : MeniscusMesh or Mesh
        Lateral meniscus mesh
    medial_cart_label : int or float
        Scalar value indicating medial cartilage region
    lateral_cart_label : int or float
        Scalar value indicating lateral cartilage region
    scalar_array_name : str, optional
        Name of scalar array containing region labels, by default 'labels'
    ray_cast_length : float, optional
        Length of rays to cast in IS direction, by default 20.0 mm

    Returns
    -------
    dict
        Dictionary containing coverage metrics:
        - 'medial_coverage_percent': percentage of medial cartilage covered by meniscus
        - 'lateral_coverage_percent': percentage of lateral cartilage covered by meniscus
        - 'medial_covered_area_mm2': area of medial cartilage covered (mm²)
        - 'lateral_covered_area_mm2': area of lateral cartilage covered (mm²)
        - 'medial_total_area_mm2': total medial cartilage area (mm²)
        - 'lateral_total_area_mm2': total lateral cartilage area (mm²)

    Examples
    --------
    >>> results = compute_meniscal_coverage(
    ...     tibia, med_meniscus, lat_meniscus,
    ...     medial_cart_label=2, lateral_cart_label=3
    ... )
    >>> print(f"Medial coverage: {results['medial_coverage_percent']:.1f}%")
    """
    # Compute IS axis
    axes = compute_tibia_axes(tibia_mesh, medial_cart_label, lateral_cart_label, scalar_array_name)
    is_direction = axes["is_axis"]

    # Initialize results
    results = {}

    # Compute medial coverage (only if medial meniscus provided)
    if medial_meniscus_mesh is not None:
        med_coverage = _get_single_compartment_coverage(
            tibia_mesh=tibia_mesh,
            meniscus_mesh=medial_meniscus_mesh,
            cart_label=medial_cart_label,
            is_direction=is_direction,
            side_name="med",
            scalar_array_name=scalar_array_name,
            ray_cast_length=ray_cast_length,
        )
        results["medial_coverage_percent"] = med_coverage["med_cart_men_coverage"]
        results["medial_covered_area_mm2"] = med_coverage["med_cart_men_area"]
        results["medial_total_area_mm2"] = med_coverage["med_cart_area"]

    # Compute lateral coverage (only if lateral meniscus provided)
    if lateral_meniscus_mesh is not None:
        lat_coverage = _get_single_compartment_coverage(
            tibia_mesh=tibia_mesh,
            meniscus_mesh=lateral_meniscus_mesh,
            cart_label=lateral_cart_label,
            is_direction=is_direction,
            side_name="lat",
            scalar_array_name=scalar_array_name,
            ray_cast_length=ray_cast_length,
        )
        results["lateral_coverage_percent"] = lat_coverage["lat_cart_men_coverage"]
        results["lateral_covered_area_mm2"] = lat_coverage["lat_cart_men_area"]
        results["lateral_total_area_mm2"] = lat_coverage["lat_cart_area"]

    return results


def analyze_meniscal_metrics(
    tibia_mesh,
    medial_meniscus_mesh,
    lateral_meniscus_mesh,
    medial_cart_label,
    lateral_cart_label,
    scalar_array_name="labels",
    middle_percentile_range=0.1,
    ray_cast_length=10.0,
):
    """
    Comprehensive meniscal analysis computing both extrusion and coverage metrics.

    This is the main function for complete meniscal analysis. It computes
    meniscal extrusion using the middle AP region and meniscal coverage
    using IS-direction ray casting.

    Parameters
    ----------
    tibia_mesh : BoneMesh or Mesh
        Tibia mesh with scalar values indicating cartilage regions from reference
    medial_meniscus_mesh : MeniscusMesh or Mesh
        Medial meniscus mesh
    lateral_meniscus_mesh : MeniscusMesh or Mesh
        Lateral meniscus mesh
    medial_cart_label : int or float
        Scalar value indicating medial cartilage region
    lateral_cart_label : int or float
        Scalar value indicating lateral cartilage region
    scalar_array_name : str, optional
        Name of scalar array containing region labels, by default 'labels'
    middle_percentile_range : float, optional
        Fraction of AP range to use for extrusion measurement, by default 0.1
    ray_cast_length : float, optional
        Length of rays to cast for coverage analysis, by default 20.0 mm

    Returns
    -------
    dict
        Dictionary containing all extrusion and coverage metrics:

        Extrusion metrics (mm, positive = extruded beyond cartilage rim):
        - 'medial_extrusion_mm': medial extrusion distance
        - 'lateral_extrusion_mm': lateral extrusion distance

        Coverage metrics:
        - 'medial_coverage_percent': percentage of medial cartilage covered
        - 'lateral_coverage_percent': percentage of lateral cartilage covered
        - 'medial_covered_area_mm2': medial cartilage covered area (mm²)
        - 'lateral_covered_area_mm2': lateral cartilage covered area (mm²)
        - 'medial_total_area_mm2': total medial cartilage area (mm²)
        - 'lateral_total_area_mm2': total lateral cartilage area (mm²)

        Reference frame:
        - 'ml_axis': medial-lateral axis vector
        - 'ap_axis': anterior-posterior axis vector
        - 'is_axis': inferior-superior axis vector

    Notes
    -----
    All meshes are automatically oriented with consistent normals before analysis.

    Examples
    --------
    >>> results = analyze_meniscal_metrics(
    ...     tibia, med_meniscus, lat_meniscus,
    ...     medial_cart_label=2, lateral_cart_label=3
    ... )
    >>> print(f"Medial extrusion: {results['medial_extrusion_mm']:.2f} mm")
    >>> print(f"Medial coverage: {results['medial_coverage_percent']:.1f}%")
    """
    # Ensure tibia mesh is properly prepared
    tibia_mesh.compute_normals(auto_orient_normals=True, inplace=True)

    # Ensure meniscus meshes are properly prepared (only if not None)
    if medial_meniscus_mesh is not None:
        medial_meniscus_mesh.compute_normals(auto_orient_normals=True, inplace=True)
    if lateral_meniscus_mesh is not None:
        lateral_meniscus_mesh.compute_normals(auto_orient_normals=True, inplace=True)

    # Check that at least one meniscus is provided
    if medial_meniscus_mesh is None and lateral_meniscus_mesh is None:
        raise ValueError("At least one meniscus mesh must be provided")

    # Compute extrusion metrics (only for menisci that are present)
    extrusion_results = compute_meniscal_extrusion(
        tibia_mesh,
        medial_meniscus_mesh,
        lateral_meniscus_mesh,
        medial_cart_label,
        lateral_cart_label,
        scalar_array_name,
        middle_percentile_range,
    )

    # Compute coverage metrics (only for menisci that are present)
    coverage_results = compute_meniscal_coverage(
        tibia_mesh,
        medial_meniscus_mesh,
        lateral_meniscus_mesh,
        medial_cart_label,
        lateral_cart_label,
        scalar_array_name,
        ray_cast_length,
    )

    # Combine results
    results = {**extrusion_results, **coverage_results}

    return results
