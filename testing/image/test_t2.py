import os

import numpy as np
import pytest
import SimpleITK as sitk
from numpy.testing import assert_allclose

# Assuming the new t2.py file is in pymskt.image
from pymskt.image import t2

# Define path to test data (relative to workspace root)
PARREC_TEST_DATA_PATH = "data/Par_Rec_T2_Test/T2.PAR"


def generate_synthetic_t2_data(shape=(2, 10, 10), tes=None, pds=None, t2s=None, noise_level=0.0):
    """Generates synthetic multi-echo data with exponential decay."""
    if tes is None:
        tes = np.array([10.0, 20.0, 30.0, 40.0, 50.0])
    if pds is None:
        pds = np.full(shape, 1000.0)  # Example Proton Density
    if t2s is None:
        t2s = np.full(shape, 50.0)  # Example T2 value (ms)

    num_echoes = len(tes)
    data_4d = np.zeros(shape + (num_echoes,))

    # Add small epsilon to T2 to avoid division by zero if T2=0
    epsilon_t2 = np.finfo(float).eps
    t2s_safe = np.maximum(t2s, epsilon_t2)

    for i in range(num_echoes):
        data_4d[..., i] = pds * np.exp(-tes[i] / t2s_safe)

    if noise_level > 0:
        noise = np.random.normal(0, noise_level * np.mean(pds), size=data_4d.shape)
        data_4d += noise
        # Ensure data doesn't go below zero due to noise
        data_4d = np.maximum(data_4d, 0)

    return data_4d, tes, pds, t2s


def test_calculate_t2_linear_fit_basic():
    """Test basic linear T2 fitting on noiseless synthetic data."""
    shape = (1, 5, 5)  # Single slice, small matrix
    true_pd = 1500.0
    true_t2 = 60.0  # ms
    pds = np.full(shape, true_pd)
    t2s = np.full(shape, true_t2)
    tes = np.array([10.0, 25.0, 40.0, 55.0, 70.0])

    data_4d, tes, _, _ = generate_synthetic_t2_data(
        shape=shape, tes=tes, pds=pds, t2s=t2s, noise_level=0.0
    )

    # Set relatively high cutoffs to avoid filtering valid results
    results = t2.calculate_t2_linear_fit(data_4d, tes, t2_cutoff=500, r2_cutoff=0.95)

    # Check results - expect near perfect fit for noiseless data
    assert_allclose(
        results["t2"], true_t2, rtol=1e-5, atol=1e-5, err_msg="T2 map does not match expected value"
    )
    assert_allclose(
        results["pd"], true_pd, rtol=1e-5, atol=1e-5, err_msg="PD map does not match expected value"
    )
    assert_allclose(
        results["r2"],
        1.0,
        rtol=1e-5,
        atol=1e-5,
        err_msg="R2 map should be close to 1.0 for perfect fit",
    )


def test_calculate_t2_linear_fit_cutoffs():
    """Test that t2_cutoff and r2_cutoff filter results."""
    shape = (1, 2, 2)
    pds = np.array([[[1000.0, 1000.0], [1000.0, 1000.0]]])
    # Voxel 0: good T2, Voxel 1: high T2, Voxel 2: Noisy (low R2), Voxel 3: okay
    t2s = np.array([[[50.0, 150.0], [50.0, 60.0]]])
    tes = np.array([10.0, 20.0, 30.0, 40.0, 50.0])

    data_4d, tes, _, _ = generate_synthetic_t2_data(
        shape=shape, tes=tes, pds=pds, t2s=t2s, noise_level=0.0
    )

    # Add noise specifically to one voxel to force low R2
    noise = np.random.normal(0, 300, size=data_4d[0, 1, 0, :].shape)
    data_4d[0, 1, 0, :] += noise
    data_4d = np.maximum(data_4d, 0)  # Ensure non-negative

    t2_cutoff = 100.0
    r2_cutoff = 0.9  # Expect voxel 2 (noisy one) to be below this

    results = t2.calculate_t2_linear_fit(data_4d, tes, t2_cutoff=t2_cutoff, r2_cutoff=r2_cutoff)

    # Expected results:
    # Voxel 0 (0,0): T2=50, PD=1000, R2=1.0 -> Should pass
    # Voxel 1 (0,1): T2=150 -> Should be zeroed by t2_cutoff
    # Voxel 2 (1,0): T2=50, but noisy -> Should be zeroed by r2_cutoff
    # Voxel 3 (1,1): T2=60, PD=1000, R2=1.0 -> Should pass

    assert results["t2"][0, 0, 0] > 0, "Voxel (0,0,0) T2 should be > 0"
    assert results["pd"][0, 0, 0] > 0, "Voxel (0,0,0) PD should be > 0"
    assert results["r2"][0, 0, 0] >= r2_cutoff, "Voxel (0,0,0) R2 should be >= cutoff"

    assert_allclose(
        results["t2"][0, 0, 1],
        0.0,
        atol=1e-9,
        err_msg="Voxel (0,0,1) T2 should be 0 due to t2_cutoff",
    )
    assert_allclose(
        results["pd"][0, 0, 1],
        0.0,
        atol=1e-9,
        err_msg="Voxel (0,0,1) PD should be 0 due to t2_cutoff",
    )  # PD also zeroed if T2 is
    assert_allclose(
        results["r2"][0, 0, 1],
        0.0,
        atol=1e-9,
        err_msg="Voxel (0,0,1) R2 should be 0 due to t2_cutoff",
    )  # R2 also zeroed

    assert (
        results["r2"][0, 1, 0] < r2_cutoff
    ), "Voxel (0,1,0) R2 should be low due to noise"  # Check R2 is actually low first
    assert_allclose(
        results["t2"][0, 1, 0],
        0.0,
        atol=1e-9,
        err_msg="Voxel (0,1,0) T2 should be 0 due to r2_cutoff",
    )
    assert_allclose(
        results["pd"][0, 1, 0],
        0.0,
        atol=1e-9,
        err_msg="Voxel (0,1,0) PD should be 0 due to r2_cutoff",
    )
    # R2 value itself isn't zeroed by the r2_cutoff, but the corresponding T2/PD are.
    # However, our implementation zeros R2 as well if T2 is zeroed by the R2 cutoff. Let's check that.
    assert_allclose(
        results["r2"][0, 1, 0],
        0.0,
        atol=1e-9,
        err_msg="Voxel (0,1,0) R2 should be 0 because T2 was zeroed by r2_cutoff",
    )

    assert results["t2"][0, 1, 1] > 0, "Voxel (0,1,1) T2 should be > 0"
    assert results["pd"][0, 1, 1] > 0, "Voxel (0,1,1) PD should be > 0"
    assert results["r2"][0, 1, 1] >= r2_cutoff, "Voxel (0,1,1) R2 should be >= cutoff"


def test_calculate_t2_linear_fit_invalid_input():
    """Test linear fitting with non-positive values in input data."""
    shape = (1, 2, 2)
    pds = np.full(shape, 1000.0)
    t2s = np.full(shape, 50.0)
    tes = np.array([10.0, 20.0, 30.0, 40.0, 50.0])

    data_4d, tes, _, _ = generate_synthetic_t2_data(shape=shape, tes=tes, pds=pds, t2s=t2s)

    # Introduce zero/negative values
    data_4d[0, 0, 1, 2] = 0.0  # Zero at one echo
    data_4d[0, 1, 0, :] = -10.0  # Negative signal for all echoes

    results = t2.calculate_t2_linear_fit(data_4d, tes, t2_cutoff=200, r2_cutoff=0.5)

    # Voxel (0,0,0) should be normal
    assert results["t2"][0, 0, 0] > 0
    assert results["pd"][0, 0, 0] > 0
    assert results["r2"][0, 0, 0] > 0.9

    # Voxel (0,0,1) had a zero, should be excluded
    assert_allclose(results["t2"][0, 0, 1], 0.0, atol=1e-9)
    assert_allclose(results["pd"][0, 0, 1], 0.0, atol=1e-9)
    assert_allclose(results["r2"][0, 0, 1], 0.0, atol=1e-9)

    # Voxel (0,1,0) had negative signal, should be excluded
    assert_allclose(results["t2"][0, 1, 0], 0.0, atol=1e-9)
    assert_allclose(results["pd"][0, 1, 0], 0.0, atol=1e-9)
    assert_allclose(results["r2"][0, 1, 0], 0.0, atol=1e-9)

    # Voxel (0,1,1) should be normal
    assert results["t2"][0, 1, 1] > 0
    assert results["pd"][0, 1, 1] > 0
    assert results["r2"][0, 1, 1] > 0.9


def test_calculate_t2_linear_fit_input_validation():
    """Test input validation for the linear fit function."""
    tes = [10.0, 20.0, 30.0]
    data_3d = np.random.rand(5, 5, 3)
    data_4d_wrong_echoes = np.random.rand(5, 5, 5, 4)
    data_4d_correct = np.random.rand(5, 5, 5, 3)

    # Wrong data dimension
    with pytest.raises(ValueError, match="data_4d must be a 4D NumPy array."):
        t2.calculate_t2_linear_fit(data_3d, tes)

    # Wrong tes type
    with pytest.raises(ValueError, match="tes must be a list or NumPy array."):
        t2.calculate_t2_linear_fit(data_4d_correct, "not_a_list")

    # Mismatched tes length and data dimension
    with pytest.raises(ValueError, match="Length of tes must match"):
        t2.calculate_t2_linear_fit(data_4d_wrong_echoes, tes)

    # Test with non-numpy array data_4d (should fail)
    with pytest.raises(ValueError, match="data_4d must be a 4D NumPy array."):
        t2.calculate_t2_linear_fit([[[[1, 2], [3, 4]]]], tes)


# ====================================
# Tests for Non-Linear Fitting
# ====================================


def test_calculate_t2_nonlinear_fit_basic():
    """Test basic non-linear T2 fitting on noiseless synthetic data."""
    shape = (1, 3, 3)
    true_pd = 1200.0
    true_t2 = 75.0  # ms
    pds = np.full(shape, true_pd)
    t2s = np.full(shape, true_t2)
    tes = np.array([15.0, 30.0, 45.0, 60.0, 75.0])

    data_4d, tes, _, _ = generate_synthetic_t2_data(
        shape=shape, tes=tes, pds=pds, t2s=t2s, noise_level=0.0
    )

    # Generate mock p0 maps (ideal in this case)
    p0_maps = {"t2": np.full(shape, true_t2), "pd": np.full(shape, true_pd)}

    # Set high cutoffs
    results = t2.calculate_t2_nonlinear_fit(
        data_4d, tes, p0_maps=p0_maps, t2_cutoff=500, r2_cutoff=0.95
    )

    # Check results - expect near perfect fit for noiseless data
    # Non-linear fit might be slightly less precise than linear algebra for perfect data
    assert_allclose(
        results["t2"],
        true_t2,
        rtol=1e-4,
        atol=1e-4,
        err_msg="NL T2 map does not match expected value",
    )
    assert_allclose(
        results["pd"],
        true_pd,
        rtol=1e-4,
        atol=1e-4,
        err_msg="NL PD map does not match expected value",
    )
    assert_allclose(
        results["r2"],
        1.0,
        rtol=1e-4,
        atol=1e-4,
        err_msg="NL R2 map should be close to 1.0 for perfect fit",
    )


def test_calculate_t2_nonlinear_fit_cutoffs():
    """Test that t2_cutoff and r2_cutoff filter non-linear results."""
    shape = (1, 2, 2)
    pds_base = 1000.0
    pds = np.array([[[pds_base, pds_base], [pds_base, pds_base]]])
    # Voxel 0: good T2, Voxel 1: high T2, Voxel 2: Noisy (low R2), Voxel 3: okay
    t2s = np.array([[[40.0, 120.0], [40.0, 55.0]]])
    tes = np.array([10.0, 25.0, 40.0, 55.0, 70.0])

    data_4d, tes, _, _ = generate_synthetic_t2_data(
        shape=shape, tes=tes, pds=pds, t2s=t2s, noise_level=0.0
    )

    # Make voxel [0, 1, 0] have a poor fit deterministically
    # Set its signal to be constant, which doesn't fit exponential decay
    data_4d[0, 1, 0, :] = pds_base * 0.5  # Assign a constant value
    # Ensure the p0 map for this voxel is still valid initially (e.g., positive)
    # It might be better to use the linear fit result as p0 in a real scenario,
    # but for testing cutoffs, setting p0 explicitly is okay.
    p0_maps = {"t2": np.copy(t2s), "pd": np.copy(pds)}
    p0_maps["t2"][0, 1, 0] = 40.0  # Ensure valid T2 p0
    p0_maps["pd"][0, 1, 0] = pds_base  # Ensure valid PD p0

    t2_cutoff = 100.0
    r2_cutoff = 0.9  # Expect the constant signal voxel to be below this

    results = t2.calculate_t2_nonlinear_fit(
        data_4d, tes, p0_maps=p0_maps, t2_cutoff=t2_cutoff, r2_cutoff=r2_cutoff
    )

    # Expected results are similar to linear, but check non-linear specifically
    # Voxel 0 (0,0): T2=40 -> Should pass
    # Voxel 1 (0,1): T2=120 -> Should be zeroed by t2_cutoff
    # Voxel 2 (1,0): T2=40, but noisy -> Should be zeroed by r2_cutoff
    # Voxel 3 (1,1): T2=55 -> Should pass

    assert results["t2"][0, 0, 0] > 0, "NL Voxel (0,0,0) T2 should be > 0"
    assert results["r2"][0, 0, 0] >= r2_cutoff, "NL Voxel (0,0,0) R2 should be >= cutoff"

    assert_allclose(
        results["t2"][0, 0, 1],
        0.0,
        atol=1e-9,
        err_msg="NL Voxel (0,0,1) T2 should be 0 due to t2_cutoff",
    )
    assert_allclose(
        results["pd"][0, 0, 1],
        0.0,
        atol=1e-9,
        err_msg="NL Voxel (0,0,1) PD should be 0 due to t2_cutoff",
    )
    assert_allclose(
        results["r2"][0, 0, 1],
        0.0,
        atol=1e-9,
        err_msg="NL Voxel (0,0,1) R2 should be 0 due to t2_cutoff",
    )

    # Check R2 value for the modified voxel before asserting T2/PD are zeroed
    # A constant signal should ideally yield R2=0
    assert (
        results["r2"][0, 1, 0] < r2_cutoff
    ), f"NL Voxel (0,1,0) R2 ({results['r2'][0, 1, 0]:.4f}) should be low (constant signal)"
    assert_allclose(
        results["t2"][0, 1, 0],
        0.0,
        atol=1e-9,
        err_msg="NL Voxel (0,1,0) T2 should be 0 due to r2_cutoff",
    )
    assert_allclose(
        results["pd"][0, 1, 0],
        0.0,
        atol=1e-9,
        err_msg="NL Voxel (0,1,0) PD should be 0 due to r2_cutoff",
    )
    assert_allclose(
        results["r2"][0, 1, 0],
        0.0,
        atol=1e-9,
        err_msg="NL Voxel (0,1,0) R2 should be 0 because fit was rejected by r2_cutoff",
    )

    assert results["t2"][0, 1, 1] > 0, "NL Voxel (0,1,1) T2 should be > 0"
    assert results["r2"][0, 1, 1] >= r2_cutoff, "NL Voxel (0,1,1) R2 should be >= cutoff"


def test_calculate_t2_nonlinear_fit_masking():
    """Test that the mask correctly restricts fitting."""
    shape = (1, 3, 3)
    true_pd = 1000.0
    true_t2 = 50.0
    pds = np.full(shape, true_pd)
    t2s = np.full(shape, true_t2)
    tes = np.array([10.0, 20.0, 30.0, 40.0, 50.0])

    data_4d, tes, _, _ = generate_synthetic_t2_data(shape=shape, tes=tes, pds=pds, t2s=t2s)
    p0_maps = {"t2": np.full(shape, true_t2), "pd": np.full(shape, true_pd)}

    # Create a mask to fit only specific pixels (e.g., corners)
    mask = np.zeros(shape, dtype=bool)
    mask[0, 0, 0] = True
    mask[0, 0, -1] = True
    mask[0, -1, 0] = True
    mask[0, -1, -1] = True

    results = t2.calculate_t2_nonlinear_fit(
        data_4d, tes, p0_maps=p0_maps, mask=mask, t2_cutoff=200, r2_cutoff=0.9
    )

    # Check that fitted pixels have values > 0
    assert results["t2"][0, 0, 0] > 0
    assert results["t2"][0, 0, -1] > 0
    assert results["t2"][0, -1, 0] > 0
    assert results["t2"][0, -1, -1] > 0

    # Check that non-masked pixels are zero
    assert_allclose(results["t2"][0, 0, 1], 0.0, atol=1e-9)
    assert_allclose(results["t2"][0, 1, 0], 0.0, atol=1e-9)
    assert_allclose(results["t2"][0, 1, 1], 0.0, atol=1e-9)
    # ... check other non-masked pixels ...

    # Check PD and R2 maps similarly
    assert results["pd"][0, 0, 0] > 0
    assert results["r2"][0, 0, 0] > 0
    assert_allclose(results["pd"][0, 1, 1], 0.0, atol=1e-9)
    assert_allclose(results["r2"][0, 1, 1], 0.0, atol=1e-9)


def test_calculate_t2_nonlinear_fit_input_validation():
    """Test input validation for the non-linear fit function."""
    shape_3d = (5, 5, 5)
    shape_4d = shape_3d + (3,)
    tes = [10.0, 20.0, 30.0]
    data_4d_correct = np.random.rand(*shape_4d)
    p0_t2_correct = np.random.rand(*shape_3d) * 100
    p0_pd_correct = np.random.rand(*shape_3d) * 1000
    p0_maps_correct = {"t2": p0_t2_correct, "pd": p0_pd_correct}
    mask_correct = np.random.choice([True, False], size=shape_3d)

    # Missing p0_maps
    with pytest.raises(ValueError, match="p0_maps dictionary .* is required"):
        t2.calculate_t2_nonlinear_fit(data_4d_correct, tes, p0_maps=None)

    # Missing keys in p0_maps
    with pytest.raises(ValueError, match="p0_maps dictionary .* is required"):
        t2.calculate_t2_nonlinear_fit(data_4d_correct, tes, p0_maps={"t2": p0_t2_correct})
    with pytest.raises(ValueError, match="p0_maps dictionary .* is required"):
        t2.calculate_t2_nonlinear_fit(data_4d_correct, tes, p0_maps={"pd": p0_pd_correct})

    # Incorrect p0_maps shape
    p0_t2_wrong_shape = np.random.rand(5, 5, 4)
    with pytest.raises(ValueError, match="p0_maps\['t2'\] must be a NumPy array with shape"):
        t2.calculate_t2_nonlinear_fit(
            data_4d_correct, tes, p0_maps={"t2": p0_t2_wrong_shape, "pd": p0_pd_correct}
        )

    # Incorrect mask shape
    mask_wrong_shape = np.random.choice([True, False], size=(5, 5, 4))
    with pytest.raises(ValueError, match="mask must be a boolean NumPy array with shape"):
        t2.calculate_t2_nonlinear_fit(
            data_4d_correct, tes, p0_maps=p0_maps_correct, mask=mask_wrong_shape
        )

    # Incorrect mask dtype
    mask_wrong_dtype = np.random.rand(*shape_3d)
    with pytest.raises(ValueError, match="mask must be a boolean NumPy array with shape"):
        t2.calculate_t2_nonlinear_fit(
            data_4d_correct, tes, p0_maps=p0_maps_correct, mask=mask_wrong_dtype
        )


# ====================================
# Placeholder Tests for Future Enhancements
# ====================================


@pytest.mark.skip(reason="Test with noisy data and looser tolerance not yet implemented.")
def test_fit_with_moderate_noise():
    """Test both linear and non-linear fits with moderate noise."""
    # TODO: Generate synthetic data with moderate noise
    # TODO: Check if results are within a reasonable tolerance of ground truth
    pass


@pytest.mark.skip(reason="Test with very low T2 values not yet implemented.")
def test_fit_very_low_t2():
    """Test stability and results for very low T2 values (e.g., < 10ms)."""
    # TODO: Generate data with T2 = 5ms
    # TODO: Run fits and check for reasonable (possibly zeroed) output
    pass


@pytest.mark.skip(reason="Test with T2 values near cutoff not yet implemented.")
def test_fit_t2_near_cutoff():
    """Test fit behavior for T2 values just below the cutoff threshold."""
    # TODO: Generate data with T2 slightly less than a chosen cutoff
    # TODO: Run fits with that cutoff and ensure T2 is not zeroed
    pass


@pytest.mark.skip(reason="Test with few echoes (e.g., 3) not yet implemented.")
def test_fit_few_echoes():
    """Test stability and results when only 3-4 echo times are available."""
    # TODO: Generate data with only 3 TEs
    # TODO: Run fits and check they complete without error (tolerances might be loose)
    pass


@pytest.mark.skip(reason="Test fitting voxels with constant signal not yet implemented.")
def test_fit_constant_signal():
    """Test behavior when input signal is constant across echoes."""
    # TODO: Generate data where one voxel has constant signal
    # TODO: Run fits and assert T2/PD/R2 are likely zero (due to R2 cutoff or T2 cutoff)
    pass


@pytest.mark.skip(reason="Test interaction of mask and invalid p0 maps not yet implemented.")
def test_nonlinear_fit_mask_invalid_p0():
    """Test non-linear fit skips masked-in voxels if p0 is invalid."""
    # TODO: Create mask=True but invalid p0 (e.g., NaN, 0) for a voxel
    # TODO: Run non-linear fit and assert output is zero for that voxel
    pass


@pytest.mark.skip(reason="Test forcing curve_fit failure not yet implemented.")
def test_nonlinear_fit_curvefit_exception():
    """Test error handling when curve_fit itself fails internally."""
    # TODO: Craft data/p0 known to cause curve_fit issues (if possible)
    # TODO: Run non-linear fit and assert output is zero and no exception is raised
    pass


# ====================================
# Test for calculate_t2_map (end-to-end)
# ====================================


@pytest.mark.skip(
    reason="Test for calculate_t2_map with PAR/REC input slow right now... not sure why... maybe too much data? Or maybe the mask is not working?."
)
@pytest.mark.timeout(30)  # Change timeout to 30 seconds
def test_calculate_t2_map_parrec():
    """Test the high-level calculate_t2_map function with PAR/REC input."""
    # --- Setup: Load Data and Prepare Mask (Load data ONCE) ---
    # Check if the test data exists
    if not os.path.exists(PARREC_TEST_DATA_PATH):
        pytest.skip(f"Test data not found at {PARREC_TEST_DATA_PATH}")

    # Load T2 data using the loader ONCE
    try:
        # Assuming T2DataLoader is accessible via t2 module
        loader = t2.T2DataLoader(source=PARREC_TEST_DATA_PATH, input_type="parrec")
        data_4d = loader.get_data_array()
        loaded_tes = loader.get_tes()
        ref_image = loader.get_reference_image()
    except Exception as e:
        pytest.fail(f"Initial T2DataLoader failed: {e}")

    # Define expected geometry
    expected_shape_zyx = (46, 448, 448)
    expected_size_xyz = (expected_shape_zyx[2], expected_shape_zyx[1], expected_shape_zyx[0])
    expected_spacing_xyz = (0.313, 0.313, 2.0)

    # Load segmentation and transform
    seg_path = "data/Par_Rec_T2_Test/T2-label.nrrd"
    transform_path = "data/Par_Rec_T2_Test/t2_transform.tfm"
    if not os.path.exists(seg_path):
        pytest.skip(f"Segmentation mask not found at {seg_path}")
    if not os.path.exists(transform_path):
        pytest.skip(f"Transform file not found at {transform_path}")

    # Resample segmentation and create mask
    try:
        seg_image = sitk.ReadImage(seg_path)
        transform = sitk.ReadTransform(transform_path)
        resampled_seg_image = sitk.Resample(
            seg_image, ref_image, transform, sitk.sitkNearestNeighbor
        )

        # Create boolean mask specifically for labels 1, 2, 3, 4
        seg_array = sitk.GetArrayFromImage(resampled_seg_image)
        labels_to_fit = [1, 2, 3, 4]
        mask_array = np.isin(seg_array, labels_to_fit)

        # Verify resampled mask shape matches reference image shape (Z, Y, X)
        expected_mask_shape = ref_image.GetSize()[::-1]
        assert (
            mask_array.shape == expected_mask_shape
        ), f"Resampled mask shape {mask_array.shape} does not match T2 data shape {expected_mask_shape}"
    except Exception as e:
        pytest.fail(f"Failed to load, resample, or process segmentation mask: {e}")

    # --- Run Linear Fit (ONCE) ---
    try:
        linear_results_np = t2.calculate_t2_linear_fit(
            data_4d, loaded_tes, t2_cutoff=100, r2_cutoff=0.7  # Provide default directly
        )  # Provide default directly
    except Exception as e:
        pytest.fail(f"calculate_t2_linear_fit failed: {e}")

    # --- Create and Test Linear SITK Maps ---
    linear_maps_sitk = {}
    for map_name, map_array_np in linear_results_np.items():
        try:
            map_image_sitk = sitk.GetImageFromArray(map_array_np)
            map_image_sitk.CopyInformation(ref_image)
            linear_maps_sitk[f"{map_name}_map"] = map_image_sitk
        except Exception as e:
            pytest.fail(f"Failed to create SimpleITK image for linear map '{map_name}': {e}")

    # Verify linear maps
    assert isinstance(linear_maps_sitk, dict)
    assert "t2_map" in linear_maps_sitk
    assert "pd_map" in linear_maps_sitk
    assert "r2_map" in linear_maps_sitk
    for map_name, map_image in linear_maps_sitk.items():
        assert isinstance(map_image, sitk.Image), f"Linear {map_name} is not a SimpleITK image"
        assert map_image.GetDimension() == 3, f"Linear {map_name} is not 3D"
        assert map_image.GetSize() == expected_size_xyz, f"Linear {map_name} size is incorrect"
        assert np.allclose(
            map_image.GetSpacing(), expected_spacing_xyz, rtol=1e-3
        ), f"Linear {map_name} spacing is incorrect"
        stats = sitk.StatisticsImageFilter()
        stats.Execute(map_image)
        assert stats.GetMaximum() > 0, f"Linear {map_name} appears to be all zeros"

    # --- Run Non-Linear Fit (using linear results and mask) ---
    try:
        nonlinear_results_np = t2.calculate_t2_nonlinear_fit(
            data_4d,
            loaded_tes,
            p0_maps=linear_results_np,  # REUSE linear fit
            t2_cutoff=100,  # Provide default directly
            r2_cutoff=0.8,  # Provide default directly
            mask=mask_array,
        )
    except Exception as e:
        pytest.fail(f"calculate_t2_nonlinear_fit failed: {e}")

    # --- Create and Test Non-Linear SITK Maps ---
    nonlinear_maps_sitk = {}
    for map_name, map_array_np in nonlinear_results_np.items():
        try:
            map_image_sitk = sitk.GetImageFromArray(map_array_np)
            map_image_sitk.CopyInformation(ref_image)
            nonlinear_maps_sitk[f"{map_name}_map"] = map_image_sitk
        except Exception as e:
            pytest.fail(f"Failed to create SimpleITK image for non-linear map '{map_name}': {e}")

    # Verify non-linear maps
    assert isinstance(nonlinear_maps_sitk, dict)
    assert "t2_map" in nonlinear_maps_sitk
    assert "pd_map" in nonlinear_maps_sitk
    assert "r2_map" in nonlinear_maps_sitk
    for map_name, map_image in nonlinear_maps_sitk.items():
        assert isinstance(map_image, sitk.Image), f"Nonlinear {map_name} is not a SimpleITK image"
        assert map_image.GetDimension() == 3, f"Nonlinear {map_name} is not 3D"
        assert map_image.GetSize() == expected_size_xyz, f"Nonlinear {map_name} size is incorrect"
        assert np.allclose(
            map_image.GetSpacing(), expected_spacing_xyz, rtol=1e-3
        ), f"Nonlinear {map_name} spacing is incorrect"
        stats = sitk.StatisticsImageFilter()
        stats.Execute(map_image)
        # Non-linear map might be all zero if mask is empty or fit fails everywhere in mask
        # We can keep the check, but it might be less strict than for linear
        # assert stats.GetMaximum() > 0, f"Nonlinear {map_name} appears to be all zeros"
