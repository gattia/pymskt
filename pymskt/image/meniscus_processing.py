import warnings

import numpy as np
import SimpleITK as sitk
from scipy import ndimage as ndi

try:
    from scipy.optimize import minimize

    _SCIPY_AVAILABLE = True
except ImportError:
    _SCIPY_AVAILABLE = False


def _logistic(z):
    """Standard logistic sigmoid function."""
    # Clip z to avoid overflow in exp
    z = np.clip(z, -500, 500)
    return 1 / (1 + np.exp(-z))


def _cost_function(theta, X, y):
    """Cost function for logistic regression."""
    m = len(y)
    h = _logistic(X @ theta)
    # Clip predictions to avoid log(0)
    epsilon = 1e-7
    h = np.clip(h, epsilon, 1 - epsilon)
    cost = -(1 / m) * (y @ np.log(h) + (1 - y) @ np.log(1 - h))
    return cost


def _gradient(theta, X, y):
    """Gradient of the cost function for logistic regression."""
    m = len(y)
    h = _logistic(X @ theta)
    grad = (1 / m) * (X.T @ (h - y))
    return grad


def _apply_plane_split(
    seg_array, original_meniscus_mask, ml_axis, center_split, med_side_is_positive, verbose=False
):
    """Applies a splitting plane to separate original meniscus voxels."""
    if verbose:
        print(f"Applying plane split at axis {ml_axis} = {center_split}")

    cleaned_med_men_mask = np.zeros_like(seg_array, dtype=bool)
    cleaned_lat_men_mask = np.zeros_like(seg_array, dtype=bool)

    # Create masks for the two halves based on ml_axis and center_split
    med_side_mask = np.zeros_like(seg_array, dtype=bool)
    lat_side_mask = np.zeros_like(seg_array, dtype=bool)

    # Create slice objects dynamically based on ml_axis
    slices_med = [slice(None)] * seg_array.ndim
    slices_lat = [slice(None)] * seg_array.ndim

    if med_side_is_positive:
        slices_med[ml_axis] = slice(center_split, None)  # Medial is >= center
        slices_lat[ml_axis] = slice(None, center_split)  # Lateral is < center
    else:
        slices_med[ml_axis] = slice(None, center_split)  # Medial is < center
        slices_lat[ml_axis] = slice(center_split, None)  # Lateral is >= center

    med_side_mask[tuple(slices_med)] = True
    lat_side_mask[tuple(slices_lat)] = True

    # Apply these masks ONLY to the original meniscus voxels
    cleaned_med_men_mask[original_meniscus_mask & med_side_mask] = True
    cleaned_lat_men_mask[original_meniscus_mask & lat_side_mask] = True

    return cleaned_med_men_mask, cleaned_lat_men_mask


def verify_and_correct_meniscus_sides(
    seg_array,
    med_meniscus_label=4,
    lat_meniscus_label=5,
    med_tib_cart_label=2,
    lat_tib_cart_label=3,
    tib_label=7,
    ml_axis=0,
    min_cart_voxels=100,
    method="tibial_centroid",  # Options: 'tibial_centroid', 'cartilage_midpoint', 'cartilage_distance', 'logistic_meniscus'
    spacing_zyx=(1.0, 1.0, 1.0),  # Add spacing parameter (Z, Y, X order)
    verbose=False,
):
    """Helper to perform the medial/lateral cleanup logic based on tibial geometry or meniscus distribution.

    Determines the medial/lateral split point or assignment based on the chosen method.
    Creates cleaned boolean masks for medial and lateral meniscus regions.

    Parameters
    ----------
    seg_array : np.ndarray
        The input segmentation array.
    med_meniscus_label : int
        Label value for the medial meniscus.
    lat_meniscus_label : int
        Label value for the lateral meniscus.
    med_tib_cart_label : int
        Label value for the medial tibial cartilage.
    lat_tib_cart_label : int
        Label value for the lateral tibial cartilage.
    tib_label : int
        Label value for the tibia bone.
    ml_axis : int
        Index of the medial-lateral axis.
    min_cart_voxels : int
        Minimum voxels required for a tibial cartilage label to be valid for centroid calculation.
    method : str
        Method to use for splitting:
        - 'tibial_centroid': Use tibial bone centroid plane. (Default)
        - 'cartilage_midpoint': Use midpoint plane between tibial cartilage centroids.
        - 'cartilage_distance': Assign by distance to 3D tibial cartilage centroids (uses spacing).
        - 'logistic_meniscus': Use logistic regression boundary on meniscus ML coordinates.
    spacing_zyx : tuple, optional
        Physical voxel spacing in (z, y, x) order corresponding to the seg_array axes.
        Used by the 'cartilage_distance' method. Defaults to (1.0, 1.0, 1.0).
    verbose : bool
        If True, print status messages.

    Returns
    -------
    tuple: (cleaned_med_men_mask, cleaned_lat_men_mask, center_split)
        Boolean masks for the cleaned medial and lateral menisci, and the
        calculated split point coordinate along ml_axis (or None if method is 'cartilage_distance').

    Raises
    ------
    ValueError
        If required labels are missing or invalid for the chosen method.
    ImportError
        If method='logistic_meniscus' and scipy is not installed.
    """
    if verbose:
        print(f"Verifying and correcting meniscus sides using method: {method}...")

    # --- Get Locations ---
    tib_locs = np.where(seg_array == tib_label)
    med_cart_locs = np.where(seg_array == med_tib_cart_label)
    lat_cart_locs = np.where(seg_array == lat_tib_cart_label)
    original_meniscus_mask = (seg_array == med_meniscus_label) | (seg_array == lat_meniscus_label)
    meniscus_coords = np.array(np.where(original_meniscus_mask)).T  # Shape (N, 3)

    if meniscus_coords.shape[0] == 0:
        warnings.warn("No meniscus voxels found in input array.")
        return (np.zeros_like(seg_array, dtype=bool), np.zeros_like(seg_array, dtype=bool), None)

    # --- Initial Checks ---
    if tib_locs[0].size == 0:
        raise ValueError(f"Tibial bone label ({tib_label}) not found in segmentation.")

    med_cart_valid = med_cart_locs[0].size >= min_cart_voxels
    lat_cart_valid = lat_cart_locs[0].size >= min_cart_voxels

    # --- Method-Specific Logic ---
    cleaned_med_men_mask = np.zeros_like(seg_array, dtype=bool)
    cleaned_lat_men_mask = np.zeros_like(seg_array, dtype=bool)
    center_split = None
    med_side_is_positive = None  # Track determined medial side direction

    # -- Calculate Tibial Bone Centroid (needed by multiple methods) --
    middle_tib_bone = tib_locs[ml_axis].mean()

    # -- Calculate Tibial Cartilage Centroids (needed by multiple methods) --
    middle_med_cart = None
    middle_lat_cart = None
    med_cart_centroid_3d = None
    lat_cart_centroid_3d = None

    if med_cart_valid:
        middle_med_cart = med_cart_locs[ml_axis].mean()
        med_cart_centroid_3d = np.array(med_cart_locs).mean(
            axis=1
        )  # Order Z, Y, X ? Check numpy order
        if verbose:
            print(f"Medial Tibial Cartilage Centroid (ML-axis): {middle_med_cart:.2f}")
    if lat_cart_valid:
        middle_lat_cart = lat_cart_locs[ml_axis].mean()
        lat_cart_centroid_3d = np.array(lat_cart_locs).mean(axis=1)
        if verbose:
            print(f"Lateral Tibial Cartilage Centroid (ML-axis): {middle_lat_cart:.2f}")

    # -- Determine Medial Side Direction (needed for plane-based splits) --
    cartilage_contradictory = False  # Initialize flag
    if med_cart_valid and lat_cart_valid:
        med_direction = np.sign(middle_med_cart - middle_tib_bone)
        lat_direction = np.sign(middle_lat_cart - middle_tib_bone)
        if med_direction == lat_direction:
            raise ValueError(
                "Medial and lateral tibial cartilage centroids are on the same side of the tibial bone centroid. Cannot use 'cartilage_midpoint'."
            )
        cartilage_contradictory = False
        med_side_is_positive = med_direction > 0
    elif med_cart_valid or lat_cart_valid:
        cartilage_contradictory = False
        if med_cart_valid:
            med_side_is_positive = np.sign(middle_med_cart - middle_tib_bone) > 0
        else:
            med_side_is_positive = np.sign(middle_lat_cart - middle_tib_bone) < 0
        if verbose:
            print(
                "Using only one valid tibial cartilage label to determine medial/lateral direction."
            )
            print(
                "Consider using 'tibial_centroid' method instead, which does not require valid cartilage labels."
            )
    else:
        cartilage_contradictory = False  # No cartilage to contradict
        med_side_is_positive = None  # Cannot determine direction
        if verbose:
            print("No valid tibial cartilage found to determine medial/lateral direction.")

    # --- Apply Method ---
    # Determine center_split and med_side_is_positive based on method
    # Handle errors and warnings appropriately for each method first

    if method == "tibial_centroid":
        center_split = int(np.round(middle_tib_bone))
        if verbose:
            print(f"Using Tibial Bone Centroid Split Point: {center_split}")
        if med_side_is_positive is None:
            raise ValueError(
                "Cannot determine Medial direction for tibial_centroid method without valid cartilage."
            )
        cleaned_med_men_mask, cleaned_lat_men_mask = _apply_plane_split(
            seg_array, original_meniscus_mask, ml_axis, center_split, med_side_is_positive, verbose
        )

    elif method == "cartilage_midpoint":
        if not med_cart_valid or not lat_cart_valid:
            raise ValueError(
                "Method 'cartilage_midpoint' requires valid medial and lateral tibial cartilage labels."
            )
        if cartilage_contradictory:
            raise ValueError(
                "Tibial cartilage centroids are on the same side. Cannot use 'cartilage_midpoint'."
            )
        center_split = int(np.round((middle_med_cart + middle_lat_cart) / 2.0))
        if verbose:
            print(f"Using Tibial Cartilage Midpoint Split Point: {center_split}")
        med_side_is_positive = np.sign(middle_med_cart - center_split) > 0
        cleaned_med_men_mask, cleaned_lat_men_mask = _apply_plane_split(
            seg_array, original_meniscus_mask, ml_axis, center_split, med_side_is_positive, verbose
        )

    elif method == "cartilage_distance":
        if not med_cart_valid or not lat_cart_valid:
            raise ValueError(
                "Method 'cartilage_distance' requires valid medial and lateral tibial cartilage labels."
            )
        if cartilage_contradictory:
            raise ValueError(
                "Tibial cartilage centroids are on the same side. Cannot use 'cartilage_distance'."
            )

        if verbose:
            print(
                f"Using distance to 3D Tibial Cartilage Centroids: Med={med_cart_centroid_3d}, Lat={lat_cart_centroid_3d}"
            )

        # --- Incorporate Spacing --- check spacing_zyx validity
        if (
            not isinstance(spacing_zyx, (tuple, list))
            or len(spacing_zyx) != 3
            or not all(isinstance(x, (int, float)) and x > 0 for x in spacing_zyx)
        ):
            warnings.warn(
                f"Invalid spacing_zyx {spacing_zyx}; assuming isotropic voxels (1.0, 1.0, 1.0). Expected tuple/list of 3 positive numbers."
            )
            spacing_zyx = (1.0, 1.0, 1.0)
        # Add specific warning if default isotropic spacing is being used
        if tuple(spacing_zyx) == (1.0, 1.0, 1.0):
            warnings.warn(
                "Using default spacing (1.0, 1.0, 1.0) for 'cartilage_distance'. Results may be inaccurate for anisotropic voxels. Provide correct 'spacing_zyx'."
            )

        spacing_array = np.array(spacing_zyx)
        if verbose:
            print(f"Using spacing (ZYX): {spacing_array}")
        # --------------------------

        # Calculate scaled coordinate differences
        # meniscus_coords is (N, 3) [z,y,x], centroids are (3,) [z,y,x]
        diff_med = meniscus_coords - med_cart_centroid_3d
        diff_lat = meniscus_coords - lat_cart_centroid_3d

        scaled_diff_med = diff_med * spacing_array  # Element-wise multiplication
        scaled_diff_lat = diff_lat * spacing_array

        # Calculate Euclidean distance on scaled differences
        dist_med = np.linalg.norm(scaled_diff_med, axis=1)
        dist_lat = np.linalg.norm(scaled_diff_lat, axis=1)
        # --------------------------

        # Assign based on minimum distance
        assign_med = dist_med < dist_lat
        assign_lat = ~assign_med  # Assign ties to lateral

        # Get original indices as tuples for assignment
        meniscus_indices_tuple = tuple(meniscus_coords.T)

        # Assign boolean values to masks
        cleaned_med_men_mask[meniscus_indices_tuple] = assign_med
        cleaned_lat_men_mask[meniscus_indices_tuple] = assign_lat
        center_split = None  # Not applicable

    elif method == "logistic_meniscus":
        if not _SCIPY_AVAILABLE:
            raise ImportError("Method 'logistic_meniscus' requires scipy.optimize.")
        if verbose:
            print("Using Logistic Regression on Meniscus ML coordinates.")
        meniscus_ml_coords = meniscus_coords[:, ml_axis]
        original_med_mask_bool = seg_array[original_meniscus_mask] == med_meniscus_label
        meniscus_labels_binary = np.where(original_med_mask_bool, 0, 1)
        X = np.vstack([np.ones(len(meniscus_ml_coords)), meniscus_ml_coords]).T
        theta = np.zeros(X.shape[1])
        result = minimize(
            _cost_function,
            theta,
            args=(X, meniscus_labels_binary),
            method="BFGS",
            jac=_gradient,
            options={"maxiter": 400, "disp": verbose},
        )
        if not result.success:
            raise RuntimeError(f"Logistic regression failed to converge: {result.message}")
        theta_optimized = result.x

        if np.abs(theta_optimized[1]) < 1e-6:  # Coefficient is near zero -> Fallback
            warnings.warn(
                "Logistic regression ML coefficient is near zero. Falling back to tibial centroid split."
            )
            center_split = int(np.round(middle_tib_bone))
            if verbose:
                print(f"Using Fallback Tibial Bone Centroid Split Point: {center_split}")
            if med_side_is_positive is None:
                raise ValueError(
                    "Cannot determine Medial direction for logistic fallback without valid cartilage."
                )
            cleaned_med_men_mask, cleaned_lat_men_mask = _apply_plane_split(
                seg_array,
                original_meniscus_mask,
                ml_axis,
                center_split,
                med_side_is_positive,
                verbose,
            )
        else:  # Logistic regression succeeded
            center_split = -theta_optimized[0] / theta_optimized[1]
            if verbose:
                print(f"Logistic Regression Boundary Split Point: {center_split:.2f}")
            center_split = int(np.round(center_split))
            logistic_med_side_is_positive = theta_optimized[1] < 0
            if (
                med_side_is_positive is not None
                and med_side_is_positive != logistic_med_side_is_positive
            ):
                warnings.warn(
                    "Logistic regression direction disagrees with cartilage-derived direction. Using logistic result."
                )
            med_side_is_positive = (
                logistic_med_side_is_positive  # Use logistic result for direction
            )
            cleaned_med_men_mask, cleaned_lat_men_mask = _apply_plane_split(
                seg_array,
                original_meniscus_mask,
                ml_axis,
                center_split,
                med_side_is_positive,
                verbose,
            )

    else:
        raise ValueError(f"Unknown method: {method}.")

    # --- Get Indices from Cleaned Masks ---
    med_men_indices = np.array(np.where(cleaned_med_men_mask)).T
    lat_men_indices = np.array(np.where(cleaned_lat_men_mask)).T
    n_med = med_men_indices.shape[0]
    n_lat = lat_men_indices.shape[0]

    return cleaned_med_men_mask, cleaned_lat_men_mask, center_split


def verify_and_correct_meniscus_sides_sitk(
    seg_image: sitk.Image,
    med_meniscus_label=4,
    lat_meniscus_label=5,
    med_tib_cart_label=2,
    lat_tib_cart_label=3,
    tib_label=7,
    ml_axis=0,
    min_cart_voxels=100,
    method="tibial_centroid",
    verbose=False,
):
    """
    SimpleITK wrapper for verify_and_correct_meniscus_sides.

    Extracts the numpy array and spacing from the sitk.Image and calls the
    main verification function.

    Parameters
    ----------
    seg_image : sitk.Image
        The input segmentation image.
    med_meniscus_label : int, optional
        Label value for medial meniscus. Defaults to 4.
    lat_meniscus_label : int, optional
        Label value for lateral meniscus. Defaults to 5.
    med_tib_cart_label : int, optional
        Label value for medial tibial cartilage. Defaults to 2.
    lat_tib_cart_label : int, optional
        Label value for lateral tibial cartilage. Defaults to 3.
    tib_label : int, optional
        Label value for tibia bone. Defaults to 7.
    ml_axis : int, optional
        Index of the medial-lateral axis. Defaults to 0.
    min_cart_voxels : int, optional
        Minimum voxels for valid tibial cartilage centroid. Defaults to 100.
    method : str, optional
        Method for splitting ('tibial_centroid', 'cartilage_midpoint',
        'cartilage_distance', 'logistic_meniscus'). Defaults to 'tibial_centroid'.
    verbose : bool, optional
        If True, print status messages. Defaults to False.

    Returns
    -------
    tuple: (cleaned_med_men_mask, cleaned_lat_men_mask, center_split)
        Boolean numpy masks for the cleaned medial and lateral menisci, and the
        calculated split point coordinate along ml_axis (or None if method is 'cartilage_distance').

    Raises
    ------
    ValueError
        If required labels are missing or invalid for the chosen method.
    ImportError
        If method='logistic_meniscus' and scipy is not installed.
    """
    seg_array = sitk.GetArrayFromImage(seg_image)
    spacing_xyz = seg_image.GetSpacing()
    # Reverse spacing for numpy array indexing (z, y, x)
    spacing_zyx = tuple(reversed(spacing_xyz))

    # warning that ml axis is for np array indexing, not the original
    # sitk image axis

    return verify_and_correct_meniscus_sides(
        seg_array=seg_array,
        med_meniscus_label=med_meniscus_label,
        lat_meniscus_label=lat_meniscus_label,
        med_tib_cart_label=med_tib_cart_label,
        lat_tib_cart_label=lat_tib_cart_label,
        tib_label=tib_label,
        ml_axis=ml_axis,
        min_cart_voxels=min_cart_voxels,
        method=method,
        spacing_zyx=spacing_zyx,  # Pass extracted and ordered spacing
        verbose=verbose,
    )


# === Helper Function for Angle-Based Region Classification ===
def _classify_voxel_regions_by_angle(
    voxel_indices,
    center_ml,
    center_ap,
    anterior_reference_angle_rad,
    ml_axis,
    ap_axis,
    spacing_ml,  # Physical spacing along the ML axis
    spacing_ap,  # Physical spacing along the AP axis
):
    """Calculates polar angles in physical space and classifies voxels.

    Args:
        voxel_indices (np.ndarray): Shape (N, 3) array of voxel ZYX indices.
        center_ml (float): ML coordinate of the polar center (in voxels).
        center_ap (float): AP coordinate of the polar center (in voxels).
        anterior_reference_angle_rad (float): Angle (radians) of the Anterior direction.
        ml_axis (int): Index of the ML axis in voxel_indices.
        ap_axis (int): Index of the AP axis in voxel_indices.
        spacing_ml (float): Physical distance per voxel along ml_axis.
        spacing_ap (float): Physical distance per voxel along ap_axis.

    Returns:
        tuple: (is_anterior, is_middle, is_posterior) - Boolean arrays (N,).
    """
    if voxel_indices.shape[0] == 0:
        # Return empty boolean arrays if no voxels
        return np.array([], dtype=bool), np.array([], dtype=bool), np.array([], dtype=bool)

    # Calculate voxel differences
    delta_ap = voxel_indices[:, ap_axis] - center_ap
    delta_ml = voxel_indices[:, ml_axis] - center_ml

    # Scale differences to physical space
    scaled_delta_ap = delta_ap * spacing_ap
    scaled_delta_ml = delta_ml * spacing_ml

    # Calculate raw angles in physical space
    raw_angles_rad = np.arctan2(scaled_delta_ap, scaled_delta_ml)

    # Rotate angles so Anterior (Patella direction) is 0 degrees
    shifted_angles_rad = raw_angles_rad - anterior_reference_angle_rad
    voxel_angles_deg = np.degrees(shifted_angles_rad) % 360

    # Define anatomical regions based on angle relative to Anterior (0 degrees)
    is_anterior = (voxel_angles_deg >= 300) | (voxel_angles_deg < 60)  # 0 +/- 60 deg
    is_posterior = (voxel_angles_deg >= 120) & (voxel_angles_deg < 240)  # 180 +/- 60 deg
    is_middle = ~is_anterior & ~is_posterior

    return is_anterior, is_middle, is_posterior


# ============================================================


def subdivide_meniscus_regions(
    seg_image,
    med_meniscus_label=4,  # Example label
    lat_meniscus_label=5,  # Example label
    med_tib_cart_label=2,
    lat_tib_cart_label=3,
    tib_label=6,  # Default tibial bone label
    pat_label=None,  # Optional: Label for Patella bone
    pat_cart_label=None,  # Optional: Label for Patellar Cartilage
    ml_axis=0,
    ap_axis=2,  # Assuming AP is axis 2 for angle calculation
    is_axis=1,  # Assuming IS is axis 1
    min_cart_voxels=100,  # Min voxels for tibial cartilage to be considered valid
    cleanup_method="tibial_centroid",  # Method passed to verify_and_correct_meniscus_sides
    center_method="tibia",  # Center for polar coords: 'tibia' or 'cartilage'
    label_map=None,  # Optional dict to override default subregion labels
    verbose=False,
    cartilage_center_offset_fraction: float = 0.25,  # Fraction to shift cartilage centers inward
):
    """
    Subdivides medial and lateral meniscus segmentations into anterior, middle,
    and posterior regions using a polar coordinate system.

    Includes a step to clean up medial/lateral meniscus labels based on the specified method.

    The center of the polar coordinate system can be either the centroid of the
    tibial bone (`center_method='tibia'`) or the centroids of the respective
    medial/lateral tibial cartilage plates (`center_method='cartilage'`). Using
    'cartilage' may provide better results for more C-shaped menisci like the lateral
    one, but requires valid cartilage segmentations.

    The Anterior direction (0 degrees) for the polar system is defined by the vector
    pointing from the tibial centroid to the patellar centroid. Therefore, either
    `pat_label` or `pat_cart_label` must be provided and present in the segmentation.

    Parameters
    ----------
    seg_image : sitk.Image
        Input segmentation image.
    med_meniscus_label : int
        Label value for medial meniscus.
    lat_meniscus_label : int
        Label value for lateral meniscus.
    med_tib_cart_label : int
        Label value for medial tibial cartilage.
    lat_tib_cart_label : int
        Label value for lateral tibial cartilage.
    tib_label : int
        Label value for tibia bone.
    pat_label : int, optional
        Label value for the patella bone. Required if `pat_cart_label` is not provided or found.
    pat_cart_label : int, optional
        Label value for the patellar cartilage. Used if `pat_label` is not provided or found.
    ml_axis : int
        Index of the medial-lateral axis.
    ap_axis : int
        Index of the anterior-posterior axis.
    is_axis : int
        Index of the inferior-superior axis.
    min_cart_voxels : int
        Minimum voxels for valid tibial cartilage centroid.
    cleanup_method : str
        Method for medial/lateral cleanup ('tibial_centroid', 'cartilage_midpoint',
        'cartilage_distance', 'logistic_meniscus').
    center_method : {'tibia', 'cartilage'}, optional
        Defines the center for the polar coordinate system used for angle calculations.
        - 'tibia': Uses the centroid of the `tib_label` globally. (Default)
        - 'cartilage': Uses the centroid of `med_tib_cart_label` for the medial
          meniscus and `lat_tib_cart_label` for the lateral meniscus. Requires
          both cartilage labels to be present and meet `min_cart_voxels`.
    label_map : dict, optional
        Dictionary mapping subregion names (e.g., "Medial_Anterior") to desired
        integer label values. If None, uses default labels starting from 101.
    cartilage_center_offset_fraction : float, optional
        When `center_method='cartilage'`, this fraction (0.0 to < 0.5) determines how far
        to shift the polar centers inward along the line connecting the medial and
        lateral cartilage centroids. 0.0 means use the exact cartilage centroids.
        A value like 0.25 shifts each center 25% of the distance towards the other.
        Defaults to 0.0.
    verbose : bool
        If True, print status messages.

    Returns
    -------
    sitk.Image
        New segmentation image with 6 meniscus sub-region labels.

    Raises
    ------
    ValueError
        If required labels (tibia, patella, cartilage if `center_method='cartilage'`)
        are missing, or if cleanup method fails, or if `center_method` is invalid.
    ImportError
        If cleanup_method='logistic_meniscus' and scipy is not installed.
    """

    if verbose:
        print("Starting meniscus sub-region division...")

    seg_array = sitk.GetArrayFromImage(seg_image)
    output_array = np.zeros_like(seg_array)  # Initialize output array

    # Keep non-meniscus labels
    meniscus_labels = [med_meniscus_label, lat_meniscus_label]
    output_array[~np.isin(seg_array, meniscus_labels)] = seg_array[
        ~np.isin(seg_array, meniscus_labels)
    ]

    # --- Medial/Lateral Cleanup ---
    try:
        # Get spacing and reverse for numpy indexing (z, y, x)
        spacing_xyz = seg_image.GetSpacing()
        spacing_zyx = tuple(reversed(spacing_xyz))

        cleaned_med_men_mask, cleaned_lat_men_mask, center_split = (
            verify_and_correct_meniscus_sides(
                seg_array=seg_array.copy(),
                med_meniscus_label=med_meniscus_label,
                lat_meniscus_label=lat_meniscus_label,
                med_tib_cart_label=med_tib_cart_label,
                lat_tib_cart_label=lat_tib_cart_label,
                tib_label=tib_label,
                ml_axis=ml_axis,
                min_cart_voxels=min_cart_voxels,
                method=cleanup_method,
                spacing_zyx=spacing_zyx,  # Pass correctly ordered spacing
                verbose=verbose,
            )
        )
    except (ValueError, RuntimeError, ImportError) as e:
        # Re-raise errors from the helper function
        raise e

    # --- Get Indices from Cleaned Masks ---
    med_men_indices = np.array(np.where(cleaned_med_men_mask)).T
    lat_men_indices = np.array(np.where(cleaned_lat_men_mask)).T
    n_med = med_men_indices.shape[0]
    n_lat = lat_men_indices.shape[0]

    # --- Get Physical Spacing for Relevant Axes ---
    # Assumes seg_array is ZYX, sitk image spacing is XYZ
    spacing_xyz = seg_image.GetSpacing()
    numpy_to_sitk_axis_map = {0: 2, 1: 1, 2: 0}  # Numpy ZYX -> SITK XYZ
    try:
        ml_axis_sitk = numpy_to_sitk_axis_map[ml_axis]
        ap_axis_sitk = numpy_to_sitk_axis_map[ap_axis]
    except KeyError:
        raise ValueError(f"Invalid ml_axis ({ml_axis}) or ap_axis ({ap_axis}). Must be 0, 1, or 2.")
    spacing_ml = spacing_xyz[ml_axis_sitk]
    spacing_ap = spacing_xyz[ap_axis_sitk]
    if verbose:
        print(
            f"Using spacing: ML={spacing_ml:.4f} (axis {ml_axis}), AP={spacing_ap:.4f} (axis {ap_axis})"
        )

    # --- Define Polar Coordinate System Center(s) ---
    if verbose:
        print(f"Defining polar coordinate system center using method: {center_method}...")

    # Tibial centroid is always needed for the Patella->Tibia vector (anterior reference)
    tib_locs = np.where(seg_array == tib_label)
    if tib_locs[0].size == 0:
        raise ValueError(
            f"Tibial bone label ({tib_label}) not found after cleanup check."
        )  # Should not happen if helper ran

    tib_center_ml = tib_locs[ml_axis].mean()
    tib_center_ap = tib_locs[ap_axis].mean()
    if verbose:
        print(f"Tibial Centroid (ML, AP): ({tib_center_ml:.2f}, {tib_center_ap:.2f})")

    med_center_ml, med_center_ap = None, None
    lat_center_ml, lat_center_ap = None, None

    if center_method == "tibia":
        med_center_ml, med_center_ap = tib_center_ml, tib_center_ap
        lat_center_ml, lat_center_ap = tib_center_ml, tib_center_ap
        if verbose:
            print("Using global tibial centroid as polar center.")
    elif center_method == "cartilage":
        if verbose:
            print("Using individual tibial cartilage centroids as polar centers.")
        med_cart_locs = np.where(seg_array == med_tib_cart_label)
        lat_cart_locs = np.where(seg_array == lat_tib_cart_label)

        if med_cart_locs[0].size < min_cart_voxels:
            raise ValueError(
                f"Medial tibial cartilage label {med_tib_cart_label} has insufficient voxels ({med_cart_locs[0].size} < {min_cart_voxels}) for center_method='cartilage'."
            )
        if lat_cart_locs[0].size < min_cart_voxels:
            raise ValueError(
                f"Lateral tibial cartilage label {lat_tib_cart_label} has insufficient voxels ({lat_cart_locs[0].size} < {min_cart_voxels}) for center_method='cartilage'."
            )

        med_center_ml = med_cart_locs[ml_axis].mean()
        med_center_ap = med_cart_locs[ap_axis].mean()
        lat_center_ml = lat_cart_locs[ml_axis].mean()
        lat_center_ap = lat_cart_locs[ap_axis].mean()
        if verbose:
            print(f"Medial Cartilage Center (ML, AP): ({med_center_ml:.2f}, {med_center_ap:.2f})")
        if verbose:
            print(f"Lateral Cartilage Center (ML, AP): ({lat_center_ml:.2f}, {lat_center_ap:.2f})")

        # Optionally adjust centers inward along the line connecting them
        if cartilage_center_offset_fraction > 0:
            if verbose:
                print(
                    f"Adjusting cartilage centers inward by fraction: {cartilage_center_offset_fraction}"
                )
            vec_ml = lat_center_ml - med_center_ml
            vec_ap = lat_center_ap - med_center_ap

            adj_med_center_ml = med_center_ml + cartilage_center_offset_fraction * vec_ml
            adj_med_center_ap = med_center_ap + cartilage_center_offset_fraction * vec_ap
            adj_lat_center_ml = lat_center_ml - cartilage_center_offset_fraction * vec_ml
            adj_lat_center_ap = lat_center_ap - cartilage_center_offset_fraction * vec_ap

            # Update the centers to be used
            med_center_ml, med_center_ap = adj_med_center_ml, adj_med_center_ap
            lat_center_ml, lat_center_ap = adj_lat_center_ml, adj_lat_center_ap

            if verbose:
                print(
                    f"Adjusted Medial Center (ML, AP): ({med_center_ml:.2f}, {med_center_ap:.2f})"
                )
                print(
                    f"Adjusted Lateral Center (ML, AP): ({lat_center_ml:.2f}, {lat_center_ap:.2f})"
                )

    else:
        raise ValueError(
            f"Invalid center_method: '{center_method}'. Choose 'tibia' or 'cartilage'."
        )

    # --- Determine Anterior Reference Angle (if using patella) ---
    anterior_reference_angle_rad = None
    if verbose:
        print("Using patella centroid to define Anterior direction.")
    pat_anchor_label = None
    # Prioritize patella bone, then cartilage
    if pat_label is not None and np.any(seg_array == pat_label):
        pat_anchor_label = pat_label
        if verbose:
            print(f"Using patella bone label {pat_label} for anchor.")
    elif pat_cart_label is not None and np.any(seg_array == pat_cart_label):
        pat_anchor_label = pat_cart_label
        if verbose:
            print(f"Using patellar cartilage label {pat_cart_label} for anchor.")
    else:
        raise ValueError(
            "This function requires either pat_label or pat_cart_label to be provided and present in the segmentation to define the Anterior direction."
        )

    pat_locs = np.where(seg_array == pat_anchor_label)
    pat_center_ml = pat_locs[ml_axis].mean()
    pat_center_ap = pat_locs[ap_axis].mean()
    if verbose:
        print(f"Patellar Anchor Centroid (ML, AP): ({pat_center_ml:.2f}, {pat_center_ap:.2f})")

    # Calculate vector from TIBIAL center to patellar center (defines Anterior direction)
    vec_ml = pat_center_ml - tib_center_ml
    vec_ap = pat_center_ap - tib_center_ap

    # Calculate angle of this vector (this is the Anterior direction)
    anterior_reference_angle_rad = np.arctan2(vec_ap, vec_ml)
    if verbose:
        print(
            f"Anterior Reference Angle (Patella direction = 0 degrees): {np.degrees(anterior_reference_angle_rad):.2f} degrees relative to ML axis."
        )

    # --- Classify Voxels into Regions using Helper Function ---
    if verbose:
        print("Classifying meniscus voxels into Anterior/Middle/Posterior regions...")

    # Initialize empty arrays
    med_is_anterior, med_is_middle, med_is_posterior = (np.array([], dtype=bool),) * 3
    lat_is_anterior, lat_is_middle, lat_is_posterior = (np.array([], dtype=bool),) * 3

    if center_method == "cartilage":
        if n_med > 0:
            med_is_anterior, med_is_middle, med_is_posterior = _classify_voxel_regions_by_angle(
                med_men_indices,
                med_center_ml,
                med_center_ap,
                anterior_reference_angle_rad,
                ml_axis,
                ap_axis,
                spacing_ml,
                spacing_ap,
            )
        if n_lat > 0:
            lat_is_anterior, lat_is_middle, lat_is_posterior = _classify_voxel_regions_by_angle(
                lat_men_indices,
                lat_center_ml,
                lat_center_ap,
                anterior_reference_angle_rad,
                ml_axis,
                ap_axis,
                spacing_ml,
                spacing_ap,
            )
        if verbose:
            print("Region classification complete using cartilage centers.")

    else:  # center_method == 'tibia'
        all_men_indices = np.vstack((med_men_indices, lat_men_indices))
        if all_men_indices.shape[0] > 0:
            is_anterior, is_middle, is_posterior = _classify_voxel_regions_by_angle(
                all_men_indices,
                tib_center_ml,
                tib_center_ap,
                anterior_reference_angle_rad,
                ml_axis,
                ap_axis,
                spacing_ml,
                spacing_ap,
            )

            # Split the results
            med_is_anterior = is_anterior[:n_med]
            med_is_middle = is_middle[:n_med]
            med_is_posterior = is_posterior[:n_med]
            lat_is_anterior = is_anterior[n_med:]
            lat_is_middle = is_middle[n_med:]
            lat_is_posterior = is_posterior[n_med:]
        if verbose:
            print("Region classification complete using tibial center.")

    # --- Assign Sub-regions ---
    if verbose:
        print("Assigning voxels to sub-regions...")

    # Define default or use provided label map
    if label_map is None:
        label_map = {
            "Medial_Anterior": 101,
            "Medial_Middle": 102,
            "Medial_Posterior": 103,
            "Lateral_Anterior": 104,
            "Lateral_Middle": 105,
            "Lateral_Posterior": 106,
        }

    # --- Split masks and assign labels ---
    med_subregion_labels = np.zeros(n_med, dtype=output_array.dtype)
    lat_subregion_labels = np.zeros(n_lat, dtype=output_array.dtype)

    if n_med > 0:
        med_subregion_labels[med_is_anterior] = label_map["Medial_Anterior"]
        med_subregion_labels[med_is_middle] = label_map["Medial_Middle"]
        med_subregion_labels[med_is_posterior] = label_map["Medial_Posterior"]

        # Check for unassigned medial voxels (shouldn't happen with ~is_anterior & ~is_posterior)
        unassigned_med = med_subregion_labels == 0
        if np.any(unassigned_med):
            warnings.warn(
                f"{np.sum(unassigned_med)} medial meniscus voxels were not assigned a sub-region label. Check angle range definitions."
            )

    if n_lat > 0:
        lat_subregion_labels[lat_is_anterior] = label_map["Lateral_Anterior"]
        lat_subregion_labels[lat_is_middle] = label_map["Lateral_Middle"]
        lat_subregion_labels[lat_is_posterior] = label_map["Lateral_Posterior"]

        # Check for unassigned lateral voxels
        unassigned_lat = lat_subregion_labels == 0
        if np.any(unassigned_lat):
            warnings.warn(
                f"{np.sum(unassigned_lat)} lateral meniscus voxels were not assigned a sub-region label. Check angle range definitions."
            )

    # Assign labels back to the output array
    if n_med > 0:
        med_indices_tuple = tuple(med_men_indices.T)
        output_array[med_indices_tuple] = med_subregion_labels
    if n_lat > 0:
        lat_indices_tuple = tuple(lat_men_indices.T)
        output_array[lat_indices_tuple] = lat_subregion_labels

    # --- Create Output Image ---
    if verbose:
        print("Creating final output image...")
    final_output_image = sitk.GetImageFromArray(output_array.astype(seg_array.dtype))
    final_output_image.CopyInformation(seg_image)

    if verbose:
        print("Meniscus sub-region division complete.")
    return final_output_image
