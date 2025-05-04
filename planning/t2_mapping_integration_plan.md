# Planning Document: T2 Mapping Integration into pymskt

This document outlines the plan for integrating T2 mapping functionality into the `pymskt` library, based on existing code and discussion.

**Target Location:** Initially, the core logic will reside in `pymskt/image/t2.py`. A potential move to a `pymskt/image/quantitative/` submodule can be considered later.

**Core Goals:**

1.  Provide robust T2 mapping capabilities (linear and non-linear fitting).
2.  Support common input data formats:
    *   DICOM series folder.
    *   Philips PAR/REC file pair.
    *   Series of 3D NIfTI (.nii/.nii.gz) or NRRD (.nrrd) files.
    *   Pre-loaded 4D SimpleITK Image or NumPy array.
3.  Offer a simple, high-level API for common use cases.
4.  Provide access to lower-level functions for flexibility.
5.  Integrate cleanly with `pymskt` conventions.

**Proposed Structure & API:**

1.  **High-Level API Function:**
    *   `pymskt.image.calculate_t2_map(source, input_type, method='nonlinear', tes=None, **kwargs)`
        *   `source`: Path to data (folder for DICOM/NIfTI series, path to `.PAR` file for PAR/REC), or a 4D SimpleITK Image/NumPy array.
        *   `input_type`: String ('dicom', 'parrec', 'nifti_series', 'nrrd_series', 'sitk_image', 'numpy_array'). Guides the loading process.
        *   `method`: String ('linear', 'nonlinear'). Specifies the fitting method. Non-linear implies linear is run first for initialization.
        *   `tes`: List or NumPy array of echo times (ms). Required if `input_type` is 'nifti_series', 'nrrd_series', 'sitk_image', or 'numpy_array'. Optional for 'dicom' and 'parrec' (attempt to auto-detect from metadata, but can be overridden if provided).
        *   `**kwargs`: Pass additional parameters to loading and calculation functions (e.g., `t2_cutoff`, `r2_cutoff`, `drop_echo_1`, `seg_image`, `seg_labels_for_t2`, etc.). Loading specific kwargs (e.g., `drop_echo_1`) will be handled by the loader, calculation kwargs by the calculation functions.
        *   **Returns:** A dictionary containing SimpleITK images for 't2_map', 'pd_map', 'r2_map'.

2.  **Public Calculation Functions (in `pymskt/image/t2.py`):**
    *   `calculate_t2_linear_fit(data_4d, tes, t2_cutoff=100, r2_cutoff=0.7)`
        *   Takes 4D NumPy array (`[slices, rows, cols, echoes]`) and TEs.
        *   Returns dictionary of 3D NumPy arrays: {'t2': t2_map, 'pd': pd_map, 'r2': r2_map}.
        *   Contains the logic currently in the standalone `t2map_linear` function.
    *   `calculate_t2_nonlinear_fit(data_4d, tes, p0_maps=None, t2_cutoff=100, r2_cutoff=0.8, mask=None)`
        *   Takes 4D NumPy array, TEs.
        *   `p0_maps`: Optional dictionary containing 'pd' and 't2' NumPy arrays from linear fit for initialization. If `None`, linear fit might be run internally or raise an error.
        *   `mask`: Optional 3D NumPy boolean array specifying voxels to fit.
        *   Returns dictionary of 3D NumPy arrays: {'t2': t2_map, 'pd': pd_map, 'r2': r2_map}.
        *   Contains the non-linear fitting logic using `scipy.optimize.curve_fit`.

3.  **Data Loading Class (in `pymskt/image/t2.py`):**
    *   `T2DataLoader`
        *   `__init__(self, source, input_type, tes=None, **kwargs)`:
            *   Takes data source, type, optional TEs, and loading-specific kwargs (e.g., `drop_echo_1`, `round_decimals`).
            *   Based on `input_type`, calls internal methods (`_load_dicom`, `_load_parrec`, etc.).
            *   Performs loading, TE extraction/validation, data consolidation into 4D array, geometry extraction.
            *   Uses Python's `tempfile` module for temporary files (e.g., PAR/REC).
            *   Stores loaded data internally.
        *   Internal Methods: `_load_dicom`, `_load_parrec`, `_load_file_series`, `_process_sitk_image`, `_process_numpy_array`. These encapsulate the logic from the old `T2Reader`.
        *   Public Methods/Properties:
            *   `get_data_array()`: Returns the 4D NumPy array.
            *   `get_tes()`: Returns the validated list/array of TEs.
            *   `get_reference_image()`: Returns a SimpleITK image holding the correct geometry (origin, spacing, direction) for reconstructing output maps.

**Implementation Steps:**

1.  **Setup:** Create `pymskt/image/t2.py` if it doesn't exist (or clear the existing playground content). Add necessary imports (`numpy`, `scipy`, `SimpleITK`, `nibabel`, `os`, `warnings`, `copy`, `tempfile`).
2.  **Refactor `t2map_linear`:** Move the logic from the current standalone `t2map_linear` into the new `calculate_t2_linear_fit` function, adjusting inputs/outputs as planned.
3.  **Refactor Non-linear Fit:** Extract the non-linear fitting logic from `T2Reader.calc_t2_map_nonlinear` into `calculate_t2_nonlinear_fit`, generalizing inputs (take 4D array, TEs, optional initial guess maps, optional mask) and returning NumPy arrays. Define the static `t2Func` (or make it internal `_t2Func`).
4.  **Implement `T2DataLoader` Class:**
    *   Define the class structure with `__init__`.
    *   Implement `_load_dicom` by adapting logic from `T2Reader` (`load_images_and_info_dicom_series`, `construct_4d_array`).
    *   Implement `_load_parrec` by adapting logic from `T2Reader` (`load_par_rec_data`, `get_par_rec_tes_echos`, `get_te_1_image_and_4D_array_from_par_rec`). Replace `/tmp` usage with `tempfile`.
    *   Implement `_load_file_series` for NIfTI/NRRD (requires reading files, stacking, handling TEs).
    *   Implement `_process_sitk_image` and `_process_numpy_array` to handle direct input of 4D data (extracting geometry, validating TEs).
    *   Implement the public getter methods (`get_data_array`, `get_tes`, `get_reference_image`).
    *   Add logic in `__init__` to dispatch to the correct internal loading method based on `input_type`. Handle common logic like `drop_echo_1`.
5.  **Implement High-Level Function (`calculate_t2_map`):**
    *   Instantiate `loader = T2DataLoader(source, input_type, tes, **kwargs)`.
    *   Retrieve `data_4d = loader.get_data_array()`, `actual_tes = loader.get_tes()`, `ref_image = loader.get_reference_image()`.
    *   Call `calculate_t2_linear_fit`, passing relevant `kwargs`.
    *   If `method=='nonlinear'`, call `calculate_t2_nonlinear_fit`, passing linear results and relevant `kwargs`.
    *   Package the final NumPy result maps into SimpleITK images using `ref_image.CopyInformation()` or setting origin/spacing/direction manually.
    *   Return the dictionary of SimpleITK images.
6.  **Cleanup:** Remove the old `T2Reader` class entirely from `pymskt/image/t2.py`.
7.  **Dependencies:** Ensure `SimpleITK`, `nibabel`, `numpy`, `scipy` are listed in `pymskt`'s dependencies (`setup.py` or `requirements.txt`).
8.  **Error Handling & Warnings:** Add robust checks (e.g., mismatched TEs, incorrect input types, fitting errors, file not found) and informative messages/warnings throughout the loader and calculation functions.
9.  **Docstrings:** Write clear docstrings for all public functions/classes (`calculate_t2_map`, `calculate_t2_linear_fit`, `calculate_t2_nonlinear_fit`, `T2DataLoader`) and their methods, explaining parameters, returns, and usage. Add module-level docstring to `t2.py`.
10. **Testing:** Create unit tests (e.g., in `pymskt/tests/test_image_t2.py`) covering:
    *   `T2DataLoader` with each `input_type`.
    *   Linear and non-linear calculations.
    *   The high-level `calculate_t2_map` function.
    *   Edge cases (e.g., single slice data, drop echo 1, cutoffs).

**Timeline:** Phased approach:
1.  Phase 1: Refactor linear/non-linear calculations, implement high-level function and basic `T2DataLoader` assuming input is 4D array/image + TEs. Add basic tests.
2.  Phase 2: Implement DICOM and PAR/REC loading within `T2DataLoader`. Add tests.
3.  Phase 3: Implement NIfTI/NRRD series loading within `T2DataLoader`. Add tests.
4.  Phase 4: Comprehensive documentation, error handling, and refinement.

## Current Status (as of 2025-04-27)

**Progress:**
*   **(Step 2 & 3)** Core calculation functions (`calculate_t2_linear_fit`, `calculate_t2_nonlinear_fit`) are implemented and refactored in `pymskt/image/t2.py`.
*   **(Step 4)** A data loading class (`T2DataLoader`) skeleton has been added to `pymskt/image/t2.py` with the basic structure, `__init__` logic, getters, and dispatching for various input types.
*   **(Step 10)** Initial validation tests for `T2DataLoader.__init__` have been added to `testing/image/test_t2_loader.py`.

**Next Steps / Considerations:**
*   **(Step 4 cont.)** Implement internal loading methods within `T2DataLoader` (_load_dicom, _load_parrec, _load_file_series, _process_sitk_image, _process_numpy_array).
    *   This involves adapting logic from the old `T2Reader` in `pymskt/image/t2_example.py` and handling TE extraction/validation, data assembly, and geometry extraction.
*   **(Step 10 cont.)** Finalize the strategy for test data (DICOM, PAR/REC) - either using real sample data or mocking file I/O, and implement tests for loading methods.
*   **(Step 5)** Implement the high-level API function `pymskt.image.calculate_t2_map`.
*   **(Step 10 cont.)** Add comprehensive tests for the high-level function and edge cases.
*   **(Step 8 & 9)** Add robust error handling, warnings, and docstrings throughout.
*   Refine handling of 4D image geometry (especially direction vectors).
*   **(Step 6)** Remove old code from `pymskt/image/t2_example.py` once fully replaced.