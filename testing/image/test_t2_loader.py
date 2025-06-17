"""
Tests for the T2DataLoader class in pymskt.image.t2.
"""

import os
import tempfile
import warnings
from pathlib import Path

import numpy as np
import pytest
import SimpleITK as sitk
from numpy.testing import assert_allclose

from pymskt.image import t2  # Import the module itself for helper fn

# Assuming the main code is importable like this
from pymskt.image.t2 import T2DataLoader

# Define path to test data (relative to workspace root)
PARREC_TEST_DATA_PATH = "data/Par_Rec_T2_Test/T2.PAR"
MESE_DICOM_PATH = "data/MESE_T2_dicom_data_anon"


# Helper function to prepare test data
def prepare_sitk_test_data_from_dicom(dicom_folder, output_folder, file_format="nii.gz"):
    """Loads DICOM series and saves echoes."""
    print(f"Preparing {file_format} series from {dicom_folder} to {output_folder}")
    if not os.path.isdir(dicom_folder):
        raise FileNotFoundError(f"DICOM source directory not found: {dicom_folder}")

    try:
        # Use T2DataLoader to load the DICOM series
        # Assuming T2DataLoader's _load_dicom works correctly
        # Use drop_echo_1=False to get all echoes for this preparation step
        loader = t2.T2DataLoader(source=dicom_folder, input_type="dicom", drop_echo_1=False)
        data_4d = loader.get_data_array()  # Shape (Z, Y, X, T)
        tes = loader.get_tes()  # Sorted TEs
        ref_image = loader.get_reference_image()  # 3D reference geometry
        print(f"Loaded {data_4d.shape[3]} echoes with TEs: {tes}")
    except Exception as e:
        print(f"Error loading DICOM series using T2DataLoader: {e}")
        raise

    # Ensure output directory exists
    os.makedirs(output_folder, exist_ok=True)
    print(f"Saving individual echo images to: {output_folder}")

    num_echoes = data_4d.shape[3]
    if len(tes) != num_echoes:
        raise RuntimeError(
            f"Mismatch between number of TEs ({len(tes)}) and data dimension ({num_echoes})."
        )

    saved_filepaths = []
    for i in range(num_echoes):
        te_value = tes[i]
        echo_array_3d = data_4d[..., i]  # Extract 3D array (Z, Y, X)

        # Create 3D SimpleITK image from the NumPy array
        sitk_echo_image = sitk.GetImageFromArray(echo_array_3d)

        # --- Crucially, copy geometry from the reference image ---
        sitk_echo_image.CopyInformation(ref_image)

        # Construct filename
        te_str = f"{te_value:.2f}".replace(".", "p")
        filename = f"echo_TE_{te_str}.{file_format}"
        output_path = os.path.join(output_folder, filename)

        try:
            print(f"  Saving echo {i+1}/{num_echoes}: {filename} (TE={te_value})")
            sitk.WriteImage(sitk_echo_image, output_path)
            saved_filepaths.append(output_path)
        except Exception as e:
            print(f"Error saving file {output_path}: {e}")
            # Optional: decide whether to raise immediately or collect errors
            # raise

    if len(saved_filepaths) != num_echoes:
        warnings.warn("Number of saved files does not match the number of echoes.")

    print(f"Finished preparing test data. Saved {len(saved_filepaths)} files.")
    return saved_filepaths, tes


# Pytest fixture to prepare MESE echo files
@pytest.fixture(scope="module")
def mese_sitk_series_data():
    """Pytest fixture to prepare MESE echo files in a temporary directory."""
    if not os.path.exists(MESE_DICOM_PATH):
        pytest.skip(f"MESE DICOM test data not found at {MESE_DICOM_PATH}")

    # Create a temporary directory for the output files
    with tempfile.TemporaryDirectory() as tmpdir:
        print(f"Created temporary directory for test data: {tmpdir}")
        try:
            filepaths, tes = prepare_sitk_test_data_from_dicom(MESE_DICOM_PATH, tmpdir)
            # Load the images back now for the fixture to return them
            images = [sitk.ReadImage(f) for f in filepaths]
            yield images, tes  # Yield the list of images and TEs
        except Exception as e:
            print(f"Error during test data preparation: {e}")
            pytest.fail("Failed to prepare test data fixture.")
    # Temporary directory is automatically cleaned up here
    print("Cleaned up temporary test data directory.")


# --- Tests for __init__ Validation ---


def test_t2dataloader_init_invalid_input_type():
    """Test that T2DataLoader raises ValueError for unsupported input types."""
    with pytest.raises(ValueError, match="Unsupported input_type"):
        T2DataLoader(source=None, input_type="invalid_type")


def test_t2dataloader_init_missing_tes_nifti(tmp_path):
    """Test ValueError if tes are missing for nifti_series."""
    # Create a dummy directory and dummy files
    d = tmp_path / "nifti_dir"
    d.mkdir()
    (d / "file1.nii").touch()
    (d / "file2.nii").touch()
    dummy_files = [str(d / "file1.nii"), str(d / "file2.nii")]
    with pytest.raises(ValueError, match="'tes' must be provided"):
        T2DataLoader(source=dummy_files, input_type="nifti_series", tes=None)


def test_t2dataloader_init_missing_tes_nrrd(tmp_path):
    """Test ValueError if tes are missing for nrrd_series."""
    d = tmp_path / "nrrd_dir"
    d.mkdir()
    (d / "file1.nrrd").touch()
    (d / "file2.nrrd").touch()
    dummy_files = [str(d / "file1.nrrd"), str(d / "file2.nrrd")]
    with pytest.raises(ValueError, match="'tes' must be provided"):
        T2DataLoader(source=dummy_files, input_type="nrrd_series", tes=None)


def test_t2dataloader_init_missing_tes_sitk():
    """Test ValueError if tes are missing for sitk_image."""
    dummy_image_1 = sitk.Image([10, 10, 5], sitk.sitkFloat32)  # 3D image
    dummy_image_2 = sitk.Image([10, 10, 5], sitk.sitkFloat32)  # 3D image
    dummy_list = [dummy_image_1, dummy_image_2]
    with pytest.raises(ValueError, match="'tes' must be provided"):
        T2DataLoader(source=dummy_list, input_type="sitk_image", tes=None)


def test_t2dataloader_init_missing_tes_numpy():
    """Test ValueError if tes are missing for numpy_array."""
    dummy_array = np.zeros((5, 10, 10, 3))  # 4D array
    with pytest.raises(ValueError, match="'tes' must be provided"):
        T2DataLoader(source=dummy_array, input_type="numpy_array", tes=None)


def test_t2dataloader_init_invalid_source_dicom():
    """Test ValueError if source is not a directory for dicom."""
    with pytest.raises(ValueError, match="must be a valid directory path"):
        T2DataLoader(source="not_a_real_dir/file.dcm", input_type="dicom")


@pytest.mark.skip(reason="PAR/REC test data not available")
def test_t2dataloader_init_invalid_source_parrec():
    """Test ValueError if source is not a .par file for parrec."""
    with pytest.raises(ValueError, match="must be a valid path to a .PAR file"):
        T2DataLoader(source="image.rec", input_type="parrec")
    with pytest.raises(ValueError, match="must be a valid path to a .PAR file"):
        T2DataLoader(source="not_a_real_dir/", input_type="parrec")


def test_t2dataloader_init_invalid_source_nifti_series():
    """Test ValueError if source list is empty for nifti_series."""
    with pytest.raises(ValueError, match="the 'source' list cannot be empty"):
        T2DataLoader(source=[], input_type="nifti_series", tes=[10, 20])


def test_t2dataloader_init_invalid_source_nrrd_series():
    """Test ValueError if source list is empty for nrrd_series."""
    with pytest.raises(ValueError, match="the 'source' list cannot be empty"):
        T2DataLoader(source=[], input_type="nrrd_series", tes=[10, 20])


def test_t2dataloader_init_invalid_source_sitk():
    """Test ValueError if source is not a list of sitk.Image for sitk_image."""
    dummy_array = np.zeros((5, 10, 10))  # Use a NumPy array in the list
    with pytest.raises(ValueError, match="must be a list of SimpleITK.Image objects"):
        T2DataLoader(source=[dummy_array], input_type="sitk_image", tes=[10, 20, 30])


def test_t2dataloader_init_invalid_source_numpy():
    """Test ValueError if source is not a np.ndarray for numpy_array."""
    dummy_image = sitk.Image([10, 10, 5, 3], sitk.sitkFloat32)
    with pytest.raises(ValueError, match="must be a NumPy array"):
        T2DataLoader(source=dummy_image, input_type="numpy_array", tes=[10, 20, 30])


# --- Placeholder Tests for Loading (Requires Data/Mocks) ---
# Example structure - these will fail until loaders are implemented


def test_t2dataloader_load_dicom():
    """Test loading a DICOM series from the anonymized test data."""
    # Define path relative to project root (assuming tests run from root)
    dicom_data_path = Path("data/MESE_T2_dicom_data_anon")
    if not dicom_data_path.is_dir():
        pytest.skip(f"Test data directory not found: {dicom_data_path}")

    expected_tes = np.array([6.312, 12.624, 18.936, 25.248, 31.56, 37.872, 44.184, 50.496])
    expected_shape = (24, 256, 256, 8)
    expected_ref_size = (256, 256, 24)
    expected_ref_spacing = (0.625, 0.625, 4.0)
    expected_ref_origin = (-29.0364, -103.429, 56.2757)
    expected_ref_direction = (
        -0.02183430508768329,
        -0.14477205880584315,
        -0.9892240970126311,
        0.9995105915952667,
        0.019010007721790664,
        -0.02484344773371071,
        0.022401794799036173,
        -0.9892824018426362,
        0.14428613583517622,
    )

    # Load the data
    loader = T2DataLoader(source=str(dicom_data_path), input_type="dicom")

    # Assertions
    assert isinstance(loader.get_data_array(), np.ndarray)
    assert loader.get_data_array().shape == expected_shape
    assert loader.get_data_array().dtype == np.float32

    assert isinstance(loader.get_tes(), np.ndarray)
    np.testing.assert_allclose(loader.get_tes(), expected_tes, rtol=1e-5, atol=1e-5)

    ref_img = loader.get_reference_image()
    assert isinstance(ref_img, sitk.Image)
    assert ref_img.GetDimension() == 3
    assert ref_img.GetSize() == expected_ref_size
    np.testing.assert_allclose(ref_img.GetSpacing(), expected_ref_spacing, rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(ref_img.GetOrigin(), expected_ref_origin, rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(ref_img.GetDirection(), expected_ref_direction, rtol=1e-5, atol=1e-5)


# @pytest.mark.skip(reason="Requires PAR/REC test data or mocking")
# def test_t2dataloader_load_parrec():
#     # Setup: path_to_par_file = ...
#     # loader = T2DataLoader(source=path_to_par_file, input_type='parrec')
#     # assert loader.get_tes() == expected_tes
#     # assert loader.get_data_array().shape == expected_shape
#     # assert isinstance(loader.get_reference_image(), sitk.Image)
#     pytest.fail("PAR/REC test not implemented")


def test_t2dataloader_load_numpy():
    """Test loading from a numpy array (should work with current placeholders if they are updated)."""
    shape = (3, 10, 12, 4)  # slices, rows, cols, echoes
    tes = [10.0, 20.0, 30.0, 40.0]
    data = np.random.rand(*shape)

    # This will currently fail because _process_numpy_array raises NotImplementedError
    # We expect a RuntimeError from the __init__ post-validation check
    # OR NotImplementedError if the check doesn't run first.
    with pytest.raises((RuntimeError, NotImplementedError)):
        T2DataLoader(source=data, input_type="numpy_array", tes=tes)

    # Once implemented, the test would look like:
    # loader = T2DataLoader(source=data, input_type='numpy_array', tes=tes)
    # assert np.array_equal(loader.get_tes(), np.array(tes))
    # assert np.array_equal(loader.get_data_array(), data)
    # assert loader.get_data_array().shape == shape
    # ref_img = loader.get_reference_image()
    # assert isinstance(ref_img, sitk.Image)
    # assert ref_img.GetSize() == (shape[2], shape[1], shape[0]) # SITK order: x, y, z
    # assert ref_img.GetSpacing() == (1.0, 1.0, 1.0) # Default
    # assert ref_img.GetOrigin() == (0.0, 0.0, 0.0) # Default


# ====================================
# Tests for T2DataLoader (Data Loading)
# ====================================


@pytest.mark.skip(reason="Requires PAR/REC test data")
def test_t2dataloader_parrec():
    """Test T2DataLoader with PAR/REC input type."""
    # Check if the test data exists
    if not os.path.exists(PARREC_TEST_DATA_PATH):
        pytest.skip(f"Test data not found at {PARREC_TEST_DATA_PATH}")

    # Instantiate the loader
    try:
        loader = t2.T2DataLoader(source=PARREC_TEST_DATA_PATH, input_type="parrec")
    except Exception as e:
        pytest.fail(f"T2DataLoader initialization failed for PAR/REC: {e}")

    # Check loaded data array
    data_array = loader.get_data_array()
    assert isinstance(data_array, np.ndarray), "Data array should be a NumPy array"
    assert data_array.ndim == 4, f"Data array should be 4D, but got {data_array.ndim}D"
    # Specific shape check based on provided info (Z, Y, X, T)
    expected_shape_zyxt = (46, 448, 448, 12)
    assert (
        data_array.shape == expected_shape_zyxt
    ), f"Data array shape {data_array.shape} does not match expected {expected_shape_zyxt}"

    # Check loaded TEs
    tes = loader.get_tes()
    assert isinstance(tes, np.ndarray), "TEs should be a NumPy array"
    assert tes.ndim == 1, "TEs should be a 1D array"
    assert (
        len(tes) == data_array.shape[3]
    ), "Number of TEs should match the 4th dimension of the data array"
    # Specific TE value check
    expected_tes = np.array(
        [6.36, 12.72, 19.09, 25.45, 31.81, 38.17, 44.53, 50.90, 57.26, 63.62, 69.98, 76.34]
    )
    assert np.allclose(
        tes, expected_tes, rtol=1e-3
    ), f"Loaded TEs {tes} do not match expected {expected_tes}"
    # Ensure TE=0 was handled correctly (assuming it might have been present originally)
    assert 0.0 not in tes, "TE=0 should not be present in the final TEs array"

    # Check reference image
    ref_image = loader.get_reference_image()
    assert isinstance(ref_image, sitk.Image), "Reference image should be a SimpleITK image"
    assert ref_image.GetDimension() == 3, "Reference image should be 3D"
    # SimpleITK GetSize is (X, Y, Z), data_array shape is (Z, Y, X, T)
    expected_size_xyz = (expected_shape_zyxt[2], expected_shape_zyxt[1], expected_shape_zyxt[0])
    assert (
        ref_image.GetSize() == expected_size_xyz
    ), f"Reference image size {ref_image.GetSize()} does not match expected {expected_size_xyz}"
    # Specific spacing check (X, Y, Z)
    expected_spacing_xyz = (0.313, 0.313, 2.0)
    assert np.allclose(
        ref_image.GetSpacing(), expected_spacing_xyz, rtol=1e-3
    ), f"Reference image spacing {ref_image.GetSpacing()} does not match expected {expected_spacing_xyz}"

    # Test with drop_echo_1=True
    try:
        loader_drop1 = t2.T2DataLoader(
            source=PARREC_TEST_DATA_PATH, input_type="parrec", drop_echo_1=True
        )
    except Exception as e:
        pytest.fail(f"T2DataLoader initialization failed for PAR/REC with drop_echo_1=True: {e}")

    data_array_drop1 = loader_drop1.get_data_array()
    tes_drop1 = loader_drop1.get_tes()

    assert (
        data_array_drop1.shape[3] == data_array.shape[3] - 1
    ), "Data array should have one less echo after drop_echo_1"
    assert len(tes_drop1) == len(tes) - 1, "Should have one less TE after drop_echo_1"
    assert np.min(tes_drop1) > np.min(tes), "The minimum TE should be higher after drop_echo_1"


def test_t2dataloader_sitk_list(mese_sitk_series_data):
    """Test T2DataLoader with input_type='sitk_image' using a list of images."""
    images, tes = mese_sitk_series_data  # Get data from fixture

    # Check if fixture failed to yield data (e.g., DICOM loading issue)
    if images is None or tes is None:
        pytest.fail("Test data fixture did not provide valid images/TEs.")

    # Instantiate the loader
    try:
        # Use drop_echo_1=False initially to check all echoes load correctly
        loader = t2.T2DataLoader(source=images, input_type="sitk_image", tes=tes, drop_echo_1=False)
    except Exception as e:
        pytest.fail(f"T2DataLoader(sitk_image list) initialization failed: {e}")

    # Get loaded data
    data_array = loader.get_data_array()
    loaded_tes = loader.get_tes()
    ref_image = loader.get_reference_image()

    # --- Assertions (similar to PARREC test) ---
    assert isinstance(data_array, np.ndarray), "Data array should be a NumPy array"
    assert data_array.ndim == 4, f"Data array should be 4D, but got {data_array.ndim}D"
    assert data_array.shape[3] == len(
        images
    ), "Number of echoes in array must match number of input images"

    assert isinstance(loaded_tes, np.ndarray), "Loaded TEs should be a NumPy array"
    assert loaded_tes.ndim == 1, "Loaded TEs should be a 1D array"
    assert (
        len(loaded_tes) == data_array.shape[3]
    ), "Number of loaded TEs should match data array echo dimension"
    # Check if TEs match the input TEs (they should be sorted)
    assert np.allclose(loaded_tes, np.sort(tes)), "Loaded TEs do not match sorted input TEs"

    assert isinstance(ref_image, sitk.Image), "Reference image should be a SimpleITK image"
    assert ref_image.GetDimension() == 3, "Reference image should be 3D"
    # Check geometry consistency with the first input image
    first_image = images[0]
    assert ref_image.GetSize() == first_image.GetSize(), "Reference image size mismatch"
    assert np.allclose(
        ref_image.GetSpacing(), first_image.GetSpacing()
    ), "Reference image spacing mismatch"
    assert np.allclose(
        ref_image.GetOrigin(), first_image.GetOrigin()
    ), "Reference image origin mismatch"
    assert np.allclose(
        ref_image.GetDirection(), first_image.GetDirection()
    ), "Reference image direction mismatch"

    # Test with drop_echo_1=True
    try:
        # Use drop_echo_1=True
        loader_drop1 = t2.T2DataLoader(
            source=images, input_type="sitk_image", tes=tes, drop_echo_1=True
        )
    except Exception as e:
        pytest.fail(
            f"T2DataLoader(sitk_image list) initialization with drop_echo_1=True failed: {e}"
        )

    data_array_drop1 = loader_drop1.get_data_array()
    tes_drop1 = loader_drop1.get_tes()

    assert (
        data_array_drop1.shape[3] == data_array.shape[3] - 1
    ), "Data array should have one less echo after drop_echo_1 (sitk_list)"
    assert (
        len(tes_drop1) == len(loaded_tes) - 1
    ), "Should have one less TE after drop_echo_1 (sitk_list)"
    assert np.min(tes_drop1) > np.min(
        loaded_tes
    ), "The minimum TE should be higher after drop_echo_1 (sitk_list)"


# --- Tests for Getters ---

# @pytest.mark.skip(reason="Requires successful loading first")
# def test_t2dataloader_getters_fail_before_load():
#      """Test that getters raise RuntimeError if loading failed."""
# Need a way to instantiate the class such that loading *fails*
# e.g., by mocking an internal loader to raise an exception or return None

# Assuming loader is instantiated but failed internally:
# with pytest.raises(RuntimeError):
#     loader.get_data_array()
# with pytest.raises(RuntimeError):
#     loader.get_tes()
# with pytest.raises(RuntimeError):
#     loader.get_reference_image()
#      pytest.fail("Getter failure test not implemented")

# --- Tests for Options ---

# @pytest.mark.skip(reason="Requires successful loading first")
# def test_t2dataloader_drop_echo_1():
#      """Test the drop_echo_1 functionality."""
# Setup loader with successful load (e.g., numpy array)
# shape = (3, 10, 12, 4)
# tes = [10., 20., 30., 40.]
# data = np.random.rand(*shape)
# loader = T2DataLoader(source=data, input_type='numpy_array', tes=tes, drop_echo_1=True)
# assert np.array_equal(loader.get_tes(), np.array(tes[1:]))
# assert loader.get_data_array().shape == (shape[0], shape[1], shape[2], shape[3] - 1)
#      pytest.fail("drop_echo_1 test not implemented")

# @pytest.mark.skip(reason="Requires successful loading first")
# def test_t2dataloader_drop_echo_1_single_echo():
#      """Test that drop_echo_1 warns and does nothing if only one echo exists."""
# shape = (3, 10, 12, 1)
# tes = [10.]
# data = np.random.rand(*shape)
# with pytest.warns(UserWarning, match="Cannot drop first echo"):
#      loader = T2DataLoader(source=data, input_type='numpy_array', tes=tes, drop_echo_1=True)
# assert np.array_equal(loader.get_tes(), np.array(tes))
# assert loader.get_data_array().shape == shape
#      pytest.fail("drop_echo_1 single echo test not implemented")
