"""
Module for calculating T2 relaxation time maps from multi-echo MRI data.
"""

import copy
import os
import tempfile
import warnings

import nibabel as nib
import numpy as np
import SimpleITK as sitk
from scipy.optimize import curve_fit

# # Need pydicom to read metadata easily before loading the whole series
# try:
#     import pydicom
# except ImportError:
#     warnings.warn("pydicom is not installed. Cannot read DICOM metadata directly. Will rely solely on SimpleITK.")
#     pydicom = None


def calculate_t2_linear_fit(data_4d, tes, t2_cutoff=100, r2_cutoff=0.7):
    """
    Calculate T2, PD, and R2 maps using log-linear least squares.

    Parameters
    ----------
    data_4d : np.ndarray
        4D NumPy array with shape [slices, rows, cols, echoes].
    tes : list or np.ndarray
        List or array of echo times (ms) corresponding to the 4th dimension
        of `data_4d`.
    t2_cutoff : float, optional
        Upper threshold for valid T2 values (ms). Values above this or below 0
        will be set to 0. Default is 100.
    r2_cutoff : float, optional
        Lower threshold for R-squared (goodness-of-fit). Pixels with R2 below
        this value will have their T2 set to 0. Default is 0.7.

    Returns
    -------
    dict
        A dictionary containing the calculated maps as 3D NumPy arrays:
        {'t2': t2_map_np, 'pd': pd_map_np, 'r2': r2_map_np}
    """
    if not isinstance(data_4d, np.ndarray) or data_4d.ndim != 4:
        raise ValueError("data_4d must be a 4D NumPy array.")
    if not isinstance(tes, (list, np.ndarray)):
        raise ValueError("tes must be a list or NumPy array.")
    if len(tes) != data_4d.shape[3]:
        raise ValueError("Length of tes must match the size of the 4th dimension of data_4d.")

    tes = np.asarray(tes)  # Ensure tes is a numpy array for calculations

    four_d_shape = data_4d.shape
    # Log transform the T2 data for linear least squares estimate.
    # Handle potential log(0) or log(negative) issues.
    with np.errstate(divide="ignore", invalid="ignore"):
        # Add a small epsilon to avoid log(0)
        epsilon = np.finfo(float).eps
        log_data = np.log(np.maximum(data_4d, epsilon))
        # Mark voxels where input data was non-positive
        invalid_input_mask = data_4d <= 0

    num_pixels = np.prod(four_d_shape[:3])

    # Prepare regressors (independent variable - TEs)
    num_echoes = len(tes)
    regress_te = np.vstack([np.ones(num_echoes), tes]).T

    # Pre-calculate the pseudo-inverse part: (X.T * X)^-1 * X.T
    try:
        # Using pinv for potential stability issues with matrix inversion
        regress_part_one = np.linalg.pinv(regress_te.T @ regress_te) @ regress_te.T
    except np.linalg.LinAlgError:
        raise RuntimeError("Could not compute pseudo-inverse of TE matrix. Check TE values.")

    # Prepare dependent variable (log signal)
    # Reshape log_data to [num_echoes, num_pixels]
    dependent_variable = log_data.reshape(-1, num_echoes).T  # Now shape is [num_echoes, num_pixels]

    # Perform the regression: beta = regress_part_one * Y
    regression_result = regress_part_one @ dependent_variable  # Result shape: [2, num_pixels]

    # Extract T2 (decay) and PD (intercept)
    # Slope = -1 / T2 --> T2 = -1 / Slope
    with np.errstate(divide="ignore", invalid="ignore"):
        t2_values = -1.0 / regression_result[1, :]  # Shape: [num_pixels]
    pd_values_log = regression_result[0, :]  # Shape: [num_pixels]
    pd_values = np.exp(pd_values_log)  # Convert log PD back

    # --- Filtering and Masking ---
    # Initialize maps
    t2_map_flat = np.zeros(num_pixels, dtype=float)
    pd_map_flat = np.zeros(num_pixels, dtype=float)
    r2_map_flat = np.zeros(num_pixels, dtype=float)

    # Identify valid pixels for fitting (finite results, positive T2 before cutoff)
    valid_fit_mask = np.isfinite(t2_values) & np.isfinite(pd_values) & (t2_values > 0)

    # Apply T2 cutoff
    t2_cutoff_mask = t2_values <= t2_cutoff
    valid_mask = valid_fit_mask & t2_cutoff_mask

    # Apply filtering based on initial invalid inputs (where signal <= 0 for any echo)
    # Flatten the invalid input mask and check if *any* echo was invalid for a pixel
    invalid_input_flat = invalid_input_mask.reshape(-1, num_echoes).any(
        axis=1
    )  # Shape: [num_pixels]
    valid_mask = valid_mask & (~invalid_input_flat)

    # --- Calculate R-squared only for valid pixels ---
    if np.any(valid_mask):
        # Subset data for R2 calculation
        log_data_valid = dependent_variable[:, valid_mask]  # Shape [num_echoes, num_valid_pixels]
        coeffs_valid = regression_result[:, valid_mask]  # Shape [2, num_valid_pixels]

        # Calculate predicted values
        predicted_log_signal = regress_te @ coeffs_valid  # Shape [num_echoes, num_valid_pixels]

        # Calculate residuals
        residuals = log_data_valid - predicted_log_signal

        # Calculate Sum of Squares Total (SST)
        mean_log_signal_valid = np.mean(log_data_valid, axis=0)  # Shape [num_valid_pixels]
        sst = np.sum(
            (log_data_valid - mean_log_signal_valid) ** 2, axis=0
        )  # Shape [num_valid_pixels]

        # Calculate Sum of Squares Residual (SSR)
        ssr = np.sum(residuals**2, axis=0)  # Shape [num_valid_pixels]

        # Calculate R-squared
        # Avoid division by zero if SST is zero (happens if signal is constant)
        r_squared = np.zeros_like(ssr)
        valid_sst_mask = sst > np.finfo(float).eps
        r_squared[valid_sst_mask] = 1.0 - (ssr[valid_sst_mask] / sst[valid_sst_mask])
        r_squared[r_squared < 0] = 0  # Clamp R2 to be non-negative

        # Apply R2 cutoff
        r2_cutoff_mask = r_squared >= r2_cutoff
        final_valid_mask_indices = np.where(valid_mask)[0][r2_cutoff_mask]

        # Populate maps only for pixels passing all criteria
        t2_map_flat[final_valid_mask_indices] = t2_values[final_valid_mask_indices]
        pd_map_flat[final_valid_mask_indices] = pd_values[final_valid_mask_indices]
        r2_map_flat[final_valid_mask_indices] = r_squared[
            r2_cutoff_mask
        ]  # Assign the calculated r2 for valid pixels

    # Reshape flat maps back to 3D
    t2_map_np = t2_map_flat.reshape(four_d_shape[:3])
    pd_map_np = pd_map_flat.reshape(four_d_shape[:3])
    r2_map_np = r2_map_flat.reshape(four_d_shape[:3])

    return {"t2": t2_map_np, "pd": pd_map_np, "r2": r2_map_np}


# --- Helper function for non-linear fit ---
def _t2Func(TE, PD, T2):
    """Mono-exponential decay function for T2 fitting."""
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        # Add epsilon to T2 to prevent division by zero if T2 is somehow zero
        t2_safe = np.maximum(T2, np.finfo(float).eps)
        return PD * np.exp(-TE / t2_safe)


# --- T2DataLoader Class (Implementation Step 4) ---
class T2DataLoader:
    """
    Loads multi-echo T2 data from various sources and prepares it for mapping.

    Handles data loading, TE extraction/validation, and provides access to
    the 4D data array, TEs, and reference image geometry.

    Parameters
    ----------
    source : str or SimpleITK.Image or np.ndarray
        Data source. Interpretation depends on `input_type`:
        - 'dicom': Path to the folder containing DICOM series.
        - 'parrec': Path to the `.PAR` file (expects `.REC` in same directory).
        - 'nifti_series', 'nrrd_series': Path to the folder containing echo images.
        - 'sitk_image': A 4D SimpleITK Image object.
        - 'numpy_array': A 4D NumPy array ([slices, rows, cols, echoes]).
    input_type : {'dicom', 'parrec', 'nifti_series', 'nrrd_series', 'sitk_image', 'numpy_array'}
        Specifies the type of data source.
    tes : list or np.ndarray, optional
        Echo times (ms). Required if `input_type` is 'nifti_series',
        'nrrd_series', 'sitk_image', or 'numpy_array'. For 'dicom' and
        'parrec', if not provided, the loader attempts to extract them from
        metadata. If provided for 'dicom' or 'parrec', it overrides extracted TEs.
    **kwargs : dict
        Additional arguments specific to loading methods (e.g., `drop_echo_1`).
        Supported kwargs:
        - `drop_echo_1` (bool): If True, removes the first echo data and TE.
        - `round_decimals` (int): Number of decimals to round TEs to (for DICOM/PARREC).

    Attributes
    ----------
    _data_4d : np.ndarray
        The loaded 4D data array ([slices, rows, cols, echoes]).
    _tes : np.ndarray
        The validated echo times array (ms).
    _ref_image : SimpleITK.Image
        A SimpleITK image containing the geometry (origin, spacing, direction)
        of the loaded data, suitable for creating output maps.
    """

    SUPPORTED_INPUT_TYPES = {
        "dicom",
        "parrec",
        "nifti_series",
        "nrrd_series",
        "sitk_image",
        "numpy_array",
    }

    def __init__(self, source, input_type, tes=None, **kwargs):
        if input_type not in self.SUPPORTED_INPUT_TYPES:
            raise ValueError(
                f"Unsupported input_type: {input_type}. "
                f"Must be one of {self.SUPPORTED_INPUT_TYPES}"
            )

        self._data_4d = None
        self._tes = None
        self._ref_image = None  # Stores geometry

        # Process kwargs relevant to loading
        self.drop_echo_1 = kwargs.get("drop_echo_1", False)
        self.round_decimals = kwargs.get(
            "round_decimals", 3
        )  # Relevant for DICOM/PARREC TE extraction

        # Dispatch to appropriate loading method
        if input_type == "dicom":
            if not isinstance(source, str) or not os.path.isdir(source):
                raise ValueError("For 'dicom' input_type, 'source' must be a valid directory path.")
            self._load_dicom(source, user_tes=tes)
        elif input_type == "parrec":
            if not isinstance(source, str) or not source.lower().endswith(".par"):
                raise ValueError(
                    "For 'parrec' input_type, 'source' must be a valid path to a .PAR file."
                )
            self._load_parrec(source, user_tes=tes)
        elif input_type == "nifti_series" or input_type == "nrrd_series":
            if not isinstance(source, list) or not all(isinstance(p, str) for p in source):
                raise ValueError(
                    f"For '{input_type}', 'source' must be a list of file paths (strings)."
                )
            if not source:
                raise ValueError(f"For '{input_type}', the 'source' list cannot be empty.")
            if tes is None:
                raise ValueError(f"For '{input_type}', 'tes' must be provided.")
            # Pass the list of filepaths directly
            self._load_file_series(filepaths=source, tes=tes)
        elif input_type == "sitk_image":
            # Expect a list of SimpleITK Images
            if not isinstance(source, list) or not all(
                isinstance(img, sitk.Image) for img in source
            ):
                raise ValueError(
                    "For 'sitk_image' input_type, 'source' must be a list of SimpleITK.Image objects."
                )
            if not source:
                raise ValueError("For 'sitk_image' input_type, the 'source' list cannot be empty.")
            if tes is None:
                raise ValueError("For 'sitk_image' input_type, 'tes' must be provided.")
            # Pass the list of images directly
            self._process_image_list(image_list=source, tes=tes)
        elif input_type == "numpy_array":
            if not isinstance(source, np.ndarray):
                raise ValueError("For 'numpy_array' input_type, 'source' must be a NumPy array.")
            if tes is None:
                raise ValueError("For 'numpy_array' input_type, 'tes' must be provided.")
            self._process_numpy_array(source, tes)

        # --- Post-loading processing (common steps) ---
        # 1. Validate shapes match
        if self._data_4d is not None and self._tes is not None:
            if self._data_4d.ndim != 4:
                raise RuntimeError(
                    f"Loaded data array has {self._data_4d.ndim} dimensions, expected 4."
                )
            if self._data_4d.shape[3] != len(self._tes):
                raise RuntimeError(
                    f"Number of echoes in data ({self._data_4d.shape[3]}) "
                    f"does not match number of TEs ({len(self._tes)})."
                )
        else:
            # This should ideally not happen if loading methods are correct
            raise RuntimeError("Data loading failed: _data_4d or _tes not set.")

        # 3. Ensure reference image has 3D geometry
        if self._ref_image is not None and self._ref_image.GetDimension() == 4:
            # Extract 3D geometry from the 4D reference if needed
            size_4d = list(self._ref_image.GetSize())
            index_4d = [0] * 4
            size_3d = size_4d[:3]
            extractor = sitk.ExtractImageFilter()
            extractor.SetSize(size_3d + [0])  # Size 0 in 4th dim extracts the 3D volume
            extractor.SetIndex(index_4d)
            ref_3d = extractor.Execute(self._ref_image)

            # Create a new image with only 3D info
            dummy_array = np.zeros(ref_3d.GetSize()[::-1])  # Z,Y,X order for numpy
            new_ref_image = sitk.GetImageFromArray(dummy_array)
            new_ref_image.CopyInformation(ref_3d)
            self._ref_image = new_ref_image

        elif self._ref_image is None and input_type == "numpy_array":
            warnings.warn(
                "Input was NumPy array without geometry info. Creating default reference image."
            )
            # Create a dummy reference image with default spacing/origin/direction
            # Using Z,Y,X order for GetImageFromArray
            dummy_array = np.zeros(self._data_4d.shape[:3][::-1])
            self._ref_image = sitk.GetImageFromArray(dummy_array)
            # Set default information
            self._ref_image.SetSpacing([1.0, 1.0, 1.0])
            self._ref_image.SetOrigin([0.0, 0.0, 0.0])
            self._ref_image.SetDirection(np.eye(3).flatten())

    # --- Internal Loading Methods ---
    def _load_dicom(self, folder_path, user_tes):
        """
        Loads a DICOM series by reading metadata from each file, determining the
        structure (slices, echoes), and constructing a 4D array.
        Mimics the logic from T2Reader.load_images_and_info_dicom_series and
        T2Reader.construct_4d_array.
        """
        if not os.path.isdir(folder_path):
            raise FileNotFoundError(f"DICOM source directory not found: {folder_path}")

        # 1. Get list of files
        try:
            list_filenames = [
                os.path.join(folder_path, f)
                for f in os.listdir(folder_path)
                if f.lower().endswith(".dcm")
            ]
            if not list_filenames:
                raise FileNotFoundError(f"No .dcm files found in {folder_path}")
        except OSError as e:
            raise IOError(f"Could not list files in directory {folder_path}: {e}") from e

        # 2. Read metadata from all files
        image_info_list = []
        unique_tes = set()
        unique_origins = (
            set()
        )  # Using origin as a slice identifier might be more robust than SliceLocation
        ref_spacing = None
        ref_direction = None
        slice_shape_rows_cols = None
        te_tag = "0018|0081"
        ipp_tag = "0020|0032"  # Image Position Patient

        meta_reader = sitk.ImageFileReader()
        meta_reader.LoadPrivateTagsOn()

        for filename in list_filenames:
            try:
                meta_reader.SetFileName(filename)
                meta_reader.ReadImageInformation()

                # Essential tags
                if not meta_reader.HasMetaDataKey(te_tag) or not meta_reader.HasMetaDataKey(
                    ipp_tag
                ):
                    warnings.warn(
                        f"Skipping file {filename}: Missing required DICOM tag (EchoTime or ImagePositionPatient)."
                    )
                    continue

                te_str = meta_reader.GetMetaData(te_tag)
                ipp_str = meta_reader.GetMetaData(ipp_tag)

                # Basic geometry info (check consistency)
                current_spacing = meta_reader.GetSpacing()
                current_direction = meta_reader.GetDirection()
                current_size = meta_reader.GetSize()  # Should be 2D or 3D with size 1 in one dim

                # Check for consistent geometry across slices
                if ref_spacing is None:
                    ref_spacing = current_spacing
                    ref_direction = current_direction
                    # Determine slice shape (rows, cols) from the size (x, y[, z=1])
                    if len(current_size) == 2:  # 2D image
                        slice_shape_rows_cols = (current_size[1], current_size[0])  # height, width
                    elif len(current_size) == 3 and current_size[2] == 1:  # 3D image, single slice
                        slice_shape_rows_cols = (current_size[1], current_size[0])  # height, width
                    else:
                        warnings.warn(
                            f"Skipping file {filename}: Unexpected image dimensions {current_size}."
                        )
                        continue
                elif current_spacing != ref_spacing or current_direction != ref_direction:
                    warnings.warn(
                        f"Skipping file {filename}: Inconsistent spacing or direction detected."
                    )
                    continue
                # Check slice dimensions
                current_slice_shape = None
                if len(current_size) == 2:
                    current_slice_shape = (current_size[1], current_size[0])
                elif len(current_size) == 3 and current_size[2] == 1:
                    current_slice_shape = (current_size[1], current_size[0])
                if current_slice_shape != slice_shape_rows_cols:
                    warnings.warn(
                        f"Skipping file {filename}: Inconsistent slice dimensions {current_slice_shape} vs {slice_shape_rows_cols}."
                    )
                    continue

                # Parse TE and Image Position Patient (Origin)
                try:
                    te = round(float(te_str), self.round_decimals)
                    # Origin is read directly as tuple by SITK
                    origin = meta_reader.GetOrigin()  # This is a tuple
                    unique_origins.add(origin)
                    unique_tes.add(te)

                    image_info_list.append(
                        {
                            "filename": filename,
                            "te": te,
                            "origin": origin,  # Use origin tuple as slice identifier
                        }
                    )

                except ValueError as e:
                    warnings.warn(
                        f"Skipping file {filename}: Could not parse TE '{te_str}' or origin '{ipp_str}'. Error: {e}"
                    )
                    continue

            except Exception as e:
                warnings.warn(f"Could not read metadata from {filename}: {e}")

        if not image_info_list:
            raise IOError(
                f"Could not successfully read metadata from any DICOM files in {folder_path}."
            )
        if not unique_tes:
            raise ValueError("Could not extract any EchoTime values from DICOM metadata.")
        if not unique_origins:
            raise ValueError(
                "Could not extract any ImagePositionPatient (origin) values from DICOM metadata."
            )

        # Sort unique TEs and Origins
        sorted_unique_tes = np.sort(list(unique_tes))

        # Sort origins - crucial for correct slice order.
        # SimpleITK origins are (x,y,z). Sorting usually happens along the axis with largest variation.
        # Often, this is the Z axis. Let's sort primarily by Z, then Y, then X.
        sorted_unique_origins = sorted(list(unique_origins), key=lambda o: (o[2], o[1], o[0]))

        # 3. Determine 4D shape and pre-allocate array
        num_slices = len(sorted_unique_origins)
        num_echoes = len(sorted_unique_tes)
        rows, cols = slice_shape_rows_cols
        final_shape = (num_slices, rows, cols, num_echoes)
        self._data_4d = np.zeros(final_shape, dtype=np.float32)

        # Create mapping from TE/Origin to index
        te_to_index = {te: i for i, te in enumerate(sorted_unique_tes)}
        origin_to_index = {origin: i for i, origin in enumerate(sorted_unique_origins)}

        # 4. Load pixel data and populate 4D array
        pixel_reader = sitk.ImageFileReader()
        pixel_reader.SetOutputPixelType(sitk.sitkFloat32)  # Ensure float output

        loaded_count = 0
        for info in image_info_list:
            try:
                slice_idx = origin_to_index[info["origin"]]
                te_idx = te_to_index[info["te"]]

                # Read pixel data for this file
                pixel_reader.SetFileName(info["filename"])
                slice_image = pixel_reader.Execute()
                slice_array = sitk.GetArrayFromImage(
                    slice_image
                )  # Shape [1, rows, cols] or [rows, cols]

                # Ensure array is 2D [rows, cols]
                if slice_array.ndim == 3 and slice_array.shape[0] == 1:
                    slice_array = slice_array.squeeze(axis=0)
                elif slice_array.ndim != 2:
                    warnings.warn(
                        f"Unexpected array dimension {slice_array.ndim} for file {info['filename']}. Skipping."
                    )
                    continue

                # Check shape consistency again after loading pixels
                if slice_array.shape != (rows, cols):
                    warnings.warn(
                        f"Pixel data shape mismatch {slice_array.shape} vs expected {(rows, cols)} for file {info['filename']}. Skipping."
                    )
                    continue

                # Place in 4D array
                self._data_4d[slice_idx, :, :, te_idx] = slice_array
                loaded_count += 1

            except KeyError:
                warnings.warn(
                    f"Could not find index for origin {info['origin']} or TE {info['te']} for file {info['filename']}. This shouldn't happen."
                )
            except Exception as e:
                warnings.warn(f"Error processing pixel data for file {info['filename']}: {e}")

        expected_count = num_slices * num_echoes
        if loaded_count < expected_count:
            warnings.warn(
                f"Successfully loaded pixel data for {loaded_count} slice/echo combinations, but expected {expected_count}. Array may have missing data."
            )
            # Should we raise an error here? Maybe depends on how critical completeness is.
            # For now, just warn.
        elif loaded_count > expected_count:
            warnings.warn(
                f"Loaded data for {loaded_count} slice/echo combinations, but expected only {expected_count}. Possible duplicate data?"
            )

        # 5. Set attributes
        self._tes = sorted_unique_tes
        # Create reference image from geometry of the first slice/echo
        ref_origin = sorted_unique_origins[0]
        # Construct a 3D reference image representing the volume dimensions and geometry
        # Size needs to be (cols, rows, num_slices) for SITK (x,y,z)
        ref_image_3d_size = (cols, rows, num_slices)
        ref_image_3d = sitk.Image(ref_image_3d_size, sitk.sitkFloat32)
        ref_image_3d.SetSpacing(
            ref_spacing
        )  # Assuming ref_spacing captured (x,y,z) or (x,y) correctly
        ref_image_3d.SetDirection(ref_direction)  # Assuming ref_direction captured 3x3 matrix
        # The origin of the 3D volume is the origin of the first slice in the sorted list
        ref_image_3d.SetOrigin(ref_origin)
        self._ref_image = ref_image_3d

        # Optional: Override TEs if user provided them
        if user_tes is not None:
            user_tes_arr = np.asarray(user_tes)
            if len(user_tes_arr) == num_echoes:
                warnings.warn("User-provided TEs are overriding extracted TEs.")
                self._tes = user_tes_arr
            else:
                warnings.warn(
                    f"User-provided TEs length ({len(user_tes_arr)}) does not match number of echoes ({num_echoes}). Ignoring user TEs."
                )

    def _load_parrec(self, par_path, user_tes):
        """Loads PAR/REC data using nibabel and SimpleITK for orientation consistency."""
        # 1. Load Nibabel Image
        try:
            # Load without explicit scaling, rely on get_fdata dtype
            nib_image = nib.load(par_path)
        except FileNotFoundError:
            raise FileNotFoundError(f"PAR file not found: {par_path}")
        except Exception as e:
            raise IOError(f"Failed to load PAR/REC file {par_path}: {e}") from e

        # 2. Validate Dimensions
        if nib_image.ndim != 4:
            raise ValueError(
                f"PAR/REC file must be 4-dimensional. Found {nib_image.ndim} dimensions in {par_path}"
            )
        # Load data without forcing dtype - let nibabel handle native type initially
        nib_array = nib_image.get_fdata()

        # 3. Extract TEs from Header
        try:
            # Mimic T2Reader: get echo times and iterate to find unique values
            list_all_echo_times = nib_image.header.get_def("echo_time")
            if list_all_echo_times is None:
                raise KeyError("Could not find 'echo_time' in PAR header definitions.")

            unique_tes_list = []
            for echo_time in list_all_echo_times:
                rounded_te = round(float(echo_time), self.round_decimals)
                if rounded_te not in unique_tes_list:
                    unique_tes_list.append(rounded_te)

            # Sort the unique TEs
            extracted_tes = np.sort(np.array(unique_tes_list, dtype=float))

            # Basic check: do we have at least one TE?
            if len(extracted_tes) == 0:
                raise ValueError("No valid echo times extracted from PAR header.")

            # Note: Length comparison with data dimension happens later in __init__
            # after potentially removing TE=0 and applying drop_echo_1

        except KeyError as e:
            raise ValueError(
                f"Could not extract echo times from PAR header in {par_path}: {e}"
            ) from e
        except Exception as e:
            raise ValueError(
                f"Error processing echo times from PAR header in {par_path}: {e}"
            ) from e

        # Check if user provided TEs override extracted ones
        if user_tes is not None:
            user_tes_arr = np.asarray(user_tes)
            if len(user_tes_arr) == len(extracted_tes):
                warnings.warn("User-provided TEs are overriding TEs extracted from PAR header.")
                final_tes = user_tes_arr
            else:
                warnings.warn(
                    f"User-provided TEs length ({len(user_tes_arr)}) does not match number of echoes ({len(extracted_tes)}) extracted from PAR header. Using extracted TEs."
                )
                final_tes = extracted_tes
        else:
            final_tes = extracted_tes

        # Identify and potentially handle TE=0 (vendor map)
        # Logic adapted from T2Reader
        zero_te_indices = np.where(final_tes == 0)[0]
        vendor_map_image = None  # Placeholder for potential vendor map sitk image
        vendor_map_idx = None

        if len(zero_te_indices) > 1:
            warnings.warn(
                f"Found multiple ({len(zero_te_indices)}) zero echo times in {par_path}. This is unexpected. Using the first encountered."
            )
            vendor_map_idx = zero_te_indices[0]
        elif len(zero_te_indices) == 1:
            vendor_map_idx = zero_te_indices[0]

        # Create list of valid echo indices (excluding vendor map index)
        valid_echo_indices = np.arange(len(final_tes))
        valid_tes_list = list(final_tes)  # Work with lists for easy deletion

        if vendor_map_idx is not None:
            # Remove the zero TE and its index
            del valid_tes_list[vendor_map_idx]
            valid_echo_indices = np.delete(valid_echo_indices, vendor_map_idx)

        if len(valid_tes_list) == 0:
            raise ValueError(
                f"No non-zero echo times found after processing PAR/REC file {par_path}"
            )

        # Convert back to numpy array after potential modification
        current_tes = np.asarray(valid_tes_list)

        # Find the index of the first (minimum) TE among the valid echoes *in the original nib_array*
        min_te_val = np.min(current_tes)
        # Find where this min TE occurred in the original final_tes array (which includes TE=0)
        original_indices_of_min_te = np.where(final_tes == min_te_val)[0]
        # Filter these indices to only include those that are still valid (not the TE=0 index)
        valid_original_indices_of_min_te = [
            idx for idx in original_indices_of_min_te if idx in valid_echo_indices
        ]

        if not valid_original_indices_of_min_te:
            raise ValueError(
                f"Could not locate the minimum TE ({min_te_val}) among valid echo indices."
            )
        # Use the first occurrence if multiple echoes have the same minimum TE
        min_te_original_idx = valid_original_indices_of_min_te[0]

        # 4. Reconstruct into SimpleITK space for consistent geometry
        #    Use a temporary directory for intermediate NIfTI files
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_nii_filename = os.path.join(tmpdir, "temp_echo.nii.gz")

            # Save the first valid echo (minimum TE) to get reference geometry
            first_echo_array = nib_array[:, :, :, min_te_original_idx]
            # Create a nibabel Nifti1Image for this echo
            nib_first_echo = nib.Nifti1Image(
                first_echo_array, affine=nib_image.affine, header=nib_image.header
            )
            nib.save(nib_first_echo, tmp_nii_filename)

            # Read back with SimpleITK to get the reference geometry
            try:
                ref_sitk_image = sitk.ReadImage(tmp_nii_filename)
                ref_sitk_image.SetMetaData(
                    "intent_code", ""
                )  # Clear potential intent code if needed
                self._ref_image = sitk.Cast(ref_sitk_image, sitk.sitkFloat32)  # Store ref image
                ref_array = sitk.GetArrayFromImage(self._ref_image)  # Shape [slices, rows, cols]
                ref_shape_3d = ref_array.shape
            except Exception as e:
                raise IOError(
                    f"Failed to read temporary NIfTI file {tmp_nii_filename} with SimpleITK: {e}"
                ) from e

            # Pre-allocate the final 4D array ([slices, rows, cols, echoes])
            # ref_shape_3d is (Z, Y, X) from SimpleITK
            num_valid_echoes = len(current_tes)
            self._data_4d = np.zeros(ref_shape_3d + (num_valid_echoes,), dtype=np.float32)

            # Also handle the vendor map NIfTI conversion (only if it exists)
            if vendor_map_idx is not None:
                vendor_map_array = nib_array[:, :, :, vendor_map_idx]  # Shape (X, Y, Z)
                nib_vendor_map = nib.Nifti1Image(
                    vendor_map_array, affine=nib_image.affine, header=nib_image.header
                )
                tmp_vendor_nii = os.path.join(tmpdir, "temp_vendor_map.nii.gz")
                nib.save(nib_vendor_map, tmp_vendor_nii)
                try:
                    # Store the vendor map separately if needed later? For now, just load it.
                    vendor_map_image = sitk.ReadImage(tmp_vendor_nii)
                    # Optional: store vendor_map_image as an attribute if needed downstream
                    # self.vendor_map_image = vendor_map_image
                except Exception as e:
                    warnings.warn(
                        f"Failed to read temporary vendor map NIfTI file {tmp_vendor_nii}: {e}"
                    )

            # --- Optimized Echo Loading ---
            # Directly extract, transpose, and place echoes from nib_array into self._data_4d
            # Assumes nib_array is (X, Y, Z, T) and self._data_4d needs (Z, Y, X, T)
            if nib_array.ndim != 4:
                # This check should be redundant due to earlier validation, but for safety:
                raise RuntimeError("Internal error: nib_array is not 4D before echo extraction.")

            nib_x_dim, nib_y_dim, nib_z_dim, _ = nib_array.shape
            sitk_z_dim, sitk_y_dim, sitk_x_dim = ref_shape_3d

            # Double check dimension consistency after potential SITK interpretation
            if not (
                nib_x_dim == sitk_x_dim and nib_y_dim == sitk_y_dim and nib_z_dim == sitk_z_dim
            ):
                warnings.warn(
                    "Dimension mismatch between nibabel array ({nib_x_dim},{nib_y_dim},{nib_z_dim}) "
                    f"and SimpleITK reference ({sitk_x_dim},{sitk_y_dim},{sitk_z_dim}). "
                    "Transposing based on common conventions, but verify results."
                )

            for i, original_idx in enumerate(valid_echo_indices):
                # Extract 3D volume for the echo (X, Y, Z)
                echo_nib_xyz = nib_array[:, :, :, original_idx]

                # Transpose to SITK order (Z, Y, X)
                echo_sitk_zyx = np.transpose(echo_nib_xyz, (2, 1, 0))

                # Verify shape before assignment (should match ref_shape_3d)
                if echo_sitk_zyx.shape != ref_shape_3d:
                    raise ValueError(
                        f"Shape mismatch after transpose for echo index {original_idx}. "
                        f"Expected {ref_shape_3d}, got {echo_sitk_zyx.shape}."
                    )

                # Place transposed array into the final 4D array
                self._data_4d[:, :, :, i] = echo_sitk_zyx

        # 5. Apply drop_echo_1 if requested (operates on the final self._data_4d and self._tes)
        self._tes = current_tes  # Assign the valid TEs before potential drop

        if self.drop_echo_1:
            if len(self._tes) <= 1:
                warnings.warn(
                    "drop_echo_1=True but only one echo remains after processing. Cannot drop."
                )
            else:
                min_te_index_in_valid = np.argmin(self._tes)
                self._data_4d = np.delete(self._data_4d, min_te_index_in_valid, axis=3)
                self._tes = np.delete(self._tes, min_te_index_in_valid)
                warnings.warn(
                    "Dropped the first echo (lowest TE) as requested by drop_echo_1=True."
                )

    def _load_file_series(self, filepaths, tes):
        """Loads a series of 3D NIfTI or NRRD files from a list of file paths."""
        if not isinstance(filepaths, list) or not all(isinstance(p, str) for p in filepaths):
            raise ValueError("'filepaths' must be a list of file paths (strings).")
        if not filepaths:
            raise ValueError("The 'filepaths' list cannot be empty.")
        if not isinstance(tes, (list, np.ndarray)):
            raise ValueError(
                "'tes' must be provided as a list or NumPy array for file series loading."
            )
        tes_array = np.asarray(tes)
        if len(tes_array) != len(filepaths):
            raise ValueError(
                f"Number of provided TEs ({len(tes_array)}) does not match number "
                f"of files ({len(filepaths)}) in the 'filepaths' list."
            )

        # Load images from paths into a list
        image_list = []
        reader = sitk.ImageFileReader()
        # Keep OutputPixelType as default to avoid casting yet, let _process_image_list handle it
        # reader.SetOutputPixelType(sitk.sitkFloat32)

        for i, filepath in enumerate(filepaths):
            try:
                # Check file existence before attempting read
                if not os.path.exists(filepath):
                    raise FileNotFoundError(f"File not found at index {i}: {filepath}")
                reader.SetFileName(filepath)
                image = reader.Execute()
                image_list.append(image)
            except Exception as e:
                raise IOError(f"Failed to read file at index {i} ({filepath}): {e}") from e

        # Call the method responsible for processing the list of images
        # This method will handle geometry checks, array stacking, and setting attributes
        self._process_image_list(image_list=image_list, tes=tes_array)

    def _process_image_list(self, image_list, tes):
        """Internal method to process a list of 3D SimpleITK images."""
        # 1. Input validation (already done in __init__, but ensure list not empty)
        if not image_list:
            raise ValueError("The 'image_list' cannot be empty.")

        # 2. Sort TEs and images
        tes_array = np.asarray(tes)
        if len(tes_array) != len(image_list):
            raise ValueError(
                f"Mismatch between number of TEs ({len(tes_array)}) and images ({len(image_list)})."
            )

        sort_indices = np.argsort(tes_array)
        sorted_tes = tes_array[sort_indices]
        sorted_image_list = [image_list[i] for i in sort_indices]

        # 3. Geometry check and collect arrays
        ref_size = None
        ref_spacing = None
        ref_origin = None
        ref_direction = None
        echo_arrays = []

        for i, image in enumerate(sorted_image_list):
            try:
                if image.GetDimension() != 3:
                    raise ValueError(
                        f"Image at index {i} (sorted) is not 3D (Dimension: {image.GetDimension()})."
                    )

                current_size = image.GetSize()
                current_spacing = image.GetSpacing()
                current_origin = image.GetOrigin()
                current_direction = image.GetDirection()

                if i == 0:
                    # First image sets the reference geometry
                    ref_size = current_size
                    ref_spacing = current_spacing
                    ref_origin = current_origin
                    ref_direction = current_direction

                    # Create the 3D reference image (use float32 for consistency)
                    self._ref_image = sitk.Image(ref_size, sitk.sitkFloat32)
                    self._ref_image.SetSpacing(ref_spacing)
                    self._ref_image.SetOrigin(ref_origin)
                    self._ref_image.SetDirection(ref_direction)

                    # Get the first array (ensure float32)
                    first_array = sitk.GetArrayFromImage(sitk.Cast(image, sitk.sitkFloat32))
                    echo_arrays.append(first_array)
                else:
                    # Check consistency with reference geometry
                    geom_mismatch = False
                    mismatch_reason = []
                    if current_size != ref_size:
                        geom_mismatch = True
                        mismatch_reason.append(f"size ({current_size} vs {ref_size})")
                    if not np.allclose(current_spacing, ref_spacing):
                        geom_mismatch = True
                        mismatch_reason.append(f"spacing ({current_spacing} vs {ref_spacing})")
                    if not np.allclose(current_origin, ref_origin):
                        geom_mismatch = True
                        mismatch_reason.append(f"origin ({current_origin} vs {ref_origin})")
                    if not np.allclose(current_direction, ref_direction):
                        geom_mismatch = True
                        mismatch_reason.append("direction")

                    if geom_mismatch:
                        raise ValueError(
                            f"Geometric mismatch in image at index {i} (sorted). Differs from first image in: {', '.join(mismatch_reason)}."
                        )

                    # Geometry matches, get array (ensure float32)
                    current_array = sitk.GetArrayFromImage(sitk.Cast(image, sitk.sitkFloat32))
                    echo_arrays.append(current_array)

            except Exception as e:
                # Add context to potential errors during processing
                raise RuntimeError(f"Error processing image at index {i} (sorted): {e}") from e

        # 4. Stack arrays into 4D
        try:
            # sitk.GetArrayFromImage returns ZYX, stack along a new last axis (T)
            # Resulting shape: (Z, Y, X, T)
            stacked_data = np.stack(echo_arrays, axis=-1)
        except ValueError as e:
            raise RuntimeError(
                f"Failed to stack image arrays into 4D: {e}. Ensure all images have identical dimensions."
            ) from e

        # 5. Set attributes and handle drop_echo_1
        self._tes = sorted_tes  # Already sorted
        self._data_4d = stacked_data

        if self.drop_echo_1:
            if len(self._tes) <= 1:
                warnings.warn("drop_echo_1=True but only one echo exists. Cannot drop.")
            else:
                # Find the index of the minimum TE (which should be the first one after sorting)
                min_te_index = 0  # Since TEs are sorted
                self._data_4d = np.delete(self._data_4d, min_te_index, axis=3)
                self._tes = np.delete(self._tes, min_te_index)
                warnings.warn(
                    "Dropped the first echo (lowest TE) as requested by drop_echo_1=True."
                )

    def _process_numpy_array(self, numpy_array, tes):
        # TODO: Process pre-loaded 4D NumPy array
        #       Validate TEs against the 4th dimension size.
        #       Assume [slices, rows, cols, echoes] format.
        #       Create a default reference image with basic geometry.
        #       Set self._data_4d, self._tes, self._ref_image
        warnings.warn("NumPy array processing not yet implemented.")
        self._data_4d = None  # Placeholder
        self._tes = np.asarray(tes)  # Placeholder
        self._ref_image = None  # Placeholder
        raise NotImplementedError("NumPy array processing is not yet implemented.")

    # --- Public Getter Methods ---
    def get_data_array(self):
        """Returns the loaded 4D data as a NumPy array ([slices, rows, cols, echoes])."""
        if self._data_4d is None:
            raise RuntimeError("Data has not been loaded successfully.")
        return self._data_4d

    def get_tes(self):
        """Returns the validated echo times as a NumPy array (ms)."""
        if self._tes is None:
            raise RuntimeError("TEs have not been loaded successfully.")
        return self._tes

    def get_reference_image(self):
        """
        Returns a 3D SimpleITK image with the geometry of the loaded data.

        This image can be used to correctly store calculated maps (T2, PD, R2)
        by copying its spacing, origin, and direction information.
        """
        if self._ref_image is None:
            raise RuntimeError("Reference image geometry has not been loaded successfully.")
        return self._ref_image


def calculate_t2_nonlinear_fit(data_4d, tes, p0_maps=None, t2_cutoff=100, r2_cutoff=0.8, mask=None):
    """
    Calculate T2, PD, and R2 maps using non-linear least squares fitting.

    Uses scipy.optimize.curve_fit with a mono-exponential model.
    Requires initial parameter estimates, typically from a linear fit.

    Parameters
    ----------
    data_4d : np.ndarray
        4D NumPy array with shape [slices, rows, cols, echoes].
    tes : list or np.ndarray
        List or array of echo times (ms) corresponding to the 4th dimension
        of `data_4d`.
    p0_maps : dict, optional
        Dictionary containing initial guess maps (from linear fit) as 3D
        NumPy arrays: {'t2': t2_map_linear, 'pd': pd_map_linear}.
        If None, the function cannot proceed.
    t2_cutoff : float, optional
        Upper threshold for valid T2 values (ms). Fitted T2 values above this
        will be set to 0. Default is 100.
    r2_cutoff : float, optional
        Lower threshold for R-squared (goodness-of-fit). Pixels with R2 below
        this value will have their T2 set to 0. Default is 0.8.
    mask : np.ndarray, optional
        3D boolean NumPy array with the same shape as `data_4d[:3]`.
        If provided, fitting is only performed where mask is True.
        If None, fitting is attempted for all voxels where input data is finite
        and p0 estimates are valid.

    Returns
    -------
    dict
        A dictionary containing the calculated maps as 3D NumPy arrays:
        {'t2': t2_map_nl, 'pd': pd_map_nl, 'r2': r2_map_nl}

    Raises
    ------
    ValueError
        If input shapes are incorrect, TEs don't match data, or `p0_maps` is missing/invalid.
    """
    # --- Input Validation ---
    if not isinstance(data_4d, np.ndarray) or data_4d.ndim != 4:
        raise ValueError("data_4d must be a 4D NumPy array.")
    if not isinstance(tes, (list, np.ndarray)):
        raise ValueError("tes must be a list or NumPy array.")
    tes = np.asarray(tes)
    if len(tes) != data_4d.shape[3]:
        raise ValueError("Length of tes must match the size of the 4th dimension of data_4d.")

    shape_3d = data_4d.shape[:3]

    if (
        p0_maps is None
        or not isinstance(p0_maps, dict)
        or "t2" not in p0_maps
        or "pd" not in p0_maps
    ):
        raise ValueError(
            "p0_maps dictionary with 't2' and 'pd' keys (from linear fit) is required for non-linear fit."
        )

    p0_t2 = p0_maps["t2"]
    p0_pd = p0_maps["pd"]

    if not isinstance(p0_t2, np.ndarray) or p0_t2.shape != shape_3d:
        raise ValueError(f"p0_maps['t2'] must be a NumPy array with shape {shape_3d}")
    if not isinstance(p0_pd, np.ndarray) or p0_pd.shape != shape_3d:
        raise ValueError(f"p0_maps['pd'] must be a NumPy array with shape {shape_3d}")

    if mask is not None:
        if not isinstance(mask, np.ndarray) or mask.dtype != bool or mask.shape != shape_3d:
            raise ValueError(f"mask must be a boolean NumPy array with shape {shape_3d}")
        # Ensure we only fit voxels where mask is True AND p0 is valid
        valid_p0_mask = np.isfinite(p0_t2) & (p0_t2 > 0) & np.isfinite(p0_pd) & (p0_pd > 0)
        pixels_to_fit_mask = mask & valid_p0_mask
    else:
        # If no mask, fit where p0 estimates are valid
        valid_p0_mask = np.isfinite(p0_t2) & (p0_t2 > 0) & np.isfinite(p0_pd) & (p0_pd > 0)
        pixels_to_fit_mask = valid_p0_mask

    # Also exclude pixels where any input data point is non-finite or non-positive
    invalid_input_mask_3d = np.any(~np.isfinite(data_4d) | (data_4d <= 0), axis=3)
    pixels_to_fit_mask = pixels_to_fit_mask & (~invalid_input_mask_3d)

    # Get indices of pixels to fit
    pixels_to_fit_indices = np.where(pixels_to_fit_mask)

    # Initialize output maps
    t2_map_nl = np.zeros(shape_3d, dtype=float)
    pd_map_nl = np.zeros(shape_3d, dtype=float)
    r2_map_nl = np.zeros(shape_3d, dtype=float)

    # --- Voxel-wise Fitting ---
    num_pixels_to_fit = len(pixels_to_fit_indices[0])
    if num_pixels_to_fit == 0:
        warnings.warn(
            "No valid pixels found to perform non-linear fitting based on p0 maps and mask."
        )
        return {"t2": t2_map_nl, "pd": pd_map_nl, "r2": r2_map_nl}

    # Extract data for fitting voxels only for efficiency
    fit_data = data_4d[pixels_to_fit_indices]
    fit_p0_pd = p0_pd[pixels_to_fit_indices]
    fit_p0_t2 = p0_t2[pixels_to_fit_indices]

    fitted_pd = np.zeros(num_pixels_to_fit)
    fitted_t2 = np.zeros(num_pixels_to_fit)
    fitted_r2 = np.zeros(num_pixels_to_fit)

    # Consider adding progress bar if fitting many voxels (e.g., using tqdm if available)
    for i in range(num_pixels_to_fit):
        ydata = fit_data[i, :]  # Signal decay for this voxel
        p0 = (fit_p0_pd[i], fit_p0_t2[i])  # Initial guess from linear fit

        try:
            # Add bounds: PD > 0, T2 > 0
            # Bounds might help convergence, especially if p0 is poor.
            # Lower bounds: [0, 0]; Upper bounds: [Inf, Inf] (or a reasonable large value)
            bounds = ([0, 0], [np.inf, 100])
            popt, pcov = curve_fit(
                f=_t2Func,
                xdata=tes,
                ydata=ydata,
                p0=p0,
                method="lm",  # Trust Region Reflective often good with bounds
                #    bounds=bounds,
                #    max_nfev=1000 # Increase max function evaluations if needed
            )
            pd_fit, t2_fit = popt

            # Calculate R-squared for the fit
            residuals = ydata - _t2Func(tes, *popt)
            ss_res = np.sum(residuals**2)
            ss_tot = np.sum((ydata - np.mean(ydata)) ** 2)
            if ss_tot < np.finfo(float).eps:  # Avoid division by zero if data is constant
                r_squared = 1.0 if ss_res < np.finfo(float).eps else 0.0
            else:
                r_squared = 1.0 - (ss_res / ss_tot)
                if r_squared < 0:
                    r_squared = 0  # Clamp R2 >= 0

            # Apply cutoffs
            if t2_fit > t2_cutoff or t2_fit <= 0:
                t2_fit = 0.0
                pd_fit = 0.0  # Also zero PD if T2 is invalid
                r_squared = 0.0  # Zero R2 if T2 is invalid
            elif r_squared < r2_cutoff:
                t2_fit = 0.0  # Zero T2 if fit is poor
                pd_fit = 0.0  # Zero PD if fit is poor
                r_squared = 0.0  # Zero R2 if fit is poor

        except (RuntimeError, ValueError, TypeError) as err:
            # Common issues: Couldn't converge, p0 invalid, input data issues
            # warnings.warn(f"Curve fit failed for voxel index {i}: {err}")
            pd_fit, t2_fit, r_squared = 0.0, 0.0, 0.0

        fitted_pd[i] = pd_fit
        fitted_t2[i] = t2_fit
        fitted_r2[i] = r_squared

    # --- Place fitted values back into the full maps ---
    t2_map_nl[pixels_to_fit_indices] = fitted_t2
    pd_map_nl[pixels_to_fit_indices] = fitted_pd
    r2_map_nl[pixels_to_fit_indices] = fitted_r2

    return {"t2": t2_map_nl, "pd": pd_map_nl, "r2": r2_map_nl}


# --- High-level function (Implementation Step 5) ---
def calculate_t2_map(source, input_type, method="nonlinear", tes=None, **kwargs):
    """
    Calculates T2, PD, and R2 maps from multi-echo data.

    This function orchestrates the data loading and fitting process.

    Parameters
    ----------
    source : str or SimpleITK.Image or np.ndarray
        Data source. See `T2DataLoader` for details.
    input_type : {'dicom', 'parrec', 'nifti_series', 'nrrd_series', 'sitk_image', 'numpy_array'}
        Specifies the type of data source.
    method : {'linear', 'nonlinear'}, optional
        Specifies the fitting method to use for the final maps. 'nonlinear'
        first performs a linear fit to get initial estimates. Default is 'nonlinear'.
    tes : list or np.ndarray, optional
        Echo times (ms). Required for some input types (see `T2DataLoader`).
    **kwargs :
        Additional keyword arguments passed to both `T2DataLoader` and the
        fitting functions (`calculate_t2_linear_fit`, `calculate_t2_nonlinear_fit`).
        Common arguments include:
        - `drop_echo_1` (bool): Passed to `T2DataLoader`.
        - `t2_cutoff` (float): Used by fitting functions (default: 100).
        - `r2_cutoff` (float): Used by fitting functions (default: 0.7 for linear, 0.8 for non-linear).
        - `mask` (np.ndarray): 3D boolean mask passed to `calculate_t2_nonlinear_fit`.

    Returns
    -------
    dict
        A dictionary containing the calculated maps as 3D SimpleITK.Image objects:
        {'t2_map': sitk.Image, 'pd_map': sitk.Image, 'r2_map': sitk.Image}

    Raises
    ------
    ValueError
        If `method` is invalid or required inputs are missing.
    RuntimeError
        If data loading or fitting fails.
    """
    if method not in ["linear", "nonlinear"]:
        raise ValueError(f"Invalid method: '{method}'. Must be 'linear' or 'nonlinear'.")

    # Extract fitting-specific kwargs, using defaults if not provided
    # Defaults match those in the fitting functions for consistency
    linear_t2_cutoff = kwargs.get("t2_cutoff", 100)
    linear_r2_cutoff = kwargs.get("r2_cutoff", 0.7)
    nonlinear_t2_cutoff = kwargs.get("t2_cutoff", 100)
    nonlinear_r2_cutoff = kwargs.get("r2_cutoff", 0.8)
    mask = kwargs.get("mask", None)

    # Instantiate loader - pass all kwargs; loader will pick relevant ones
    try:
        loader = T2DataLoader(source, input_type, tes=tes, **kwargs)
        data_4d = loader.get_data_array()
        loaded_tes = loader.get_tes()
        ref_image = loader.get_reference_image()
    except Exception as e:
        raise RuntimeError(f"Data loading failed: {e}") from e

    # --- Perform Fitting ---
    # Linear fit is always needed, either as final result or for p0
    try:
        linear_results = calculate_t2_linear_fit(
            data_4d, loaded_tes, t2_cutoff=linear_t2_cutoff, r2_cutoff=linear_r2_cutoff
        )
    except Exception as e:
        raise RuntimeError(f"Linear T2 fitting failed: {e}") from e

    if method == "linear":
        final_results_np = linear_results
    elif method == "nonlinear":
        try:
            # Use linear results as initial guess (p0)
            nonlinear_results = calculate_t2_nonlinear_fit(
                data_4d,
                loaded_tes,
                p0_maps=linear_results,  # Pass linear t2/pd
                t2_cutoff=nonlinear_t2_cutoff,
                r2_cutoff=nonlinear_r2_cutoff,
                mask=mask,
            )  # Pass optional mask
            final_results_np = nonlinear_results
        except Exception as e:
            raise RuntimeError(f"Non-linear T2 fitting failed: {e}") from e

    # --- Create Output SimpleITK Images ---
    output_maps_sitk = {}
    for map_name, map_array_np in final_results_np.items():
        try:
            # Get array (Z, Y, X) and convert to SITK image
            map_image_sitk = sitk.GetImageFromArray(map_array_np)
            # Copy geometry from the reference image loaded by T2DataLoader
            map_image_sitk.CopyInformation(ref_image)
            output_maps_sitk[f"{map_name}_map"] = map_image_sitk  # e.g., 't2_map', 'pd_map'
        except Exception as e:
            warnings.warn(f"Failed to create SimpleITK image for map '{map_name}': {e}")
            output_maps_sitk[f"{map_name}_map"] = None  # Indicate failure for this map

    return output_maps_sitk
