import os

import numpy as np
import pytest
import SimpleITK as sitk

from pymskt.image.cartilage_processing import (
    get_aligned_cartilage_subregions,
    get_cartilage_subregions,
    getAnteriorOfWeightBearing,
)

SEG_IMAGE = sitk.ReadImage("data/right_knee_example.nrrd")


def run_cartilage_subregion_test(
    seg_image, fem_cart_label_idx, wb_region_percent_dist, truth_wb_factor
):
    """
    Runs the cartilage subregion alignment pipeline test on the provided segmentation image.

    It takes a base segmentation and splits it into subregions.
    It then applied a known affine transform to simulate misalignment on the original segmentation.
    It then runs the alignment subregion division pipeline and compares the recovered segmentation
    with the ground truth. The pipeline varifies that the subregions created from the misaligned
    segmentation are roughly the same as the subregions created from the original segmentation.


    Parameters:
      seg_image: sitk.Image, the input segmentation image
      fem_cart_label_idx: int, label index for femoral cartilage
      wb_region_percent_dist: float, percentage distance used by the pipeline
      truth_wb_factor: float, factor to compute the ground truth posterior WB slice
    """
    # Extract image array and view
    img_array = sitk.GetArrayFromImage(seg_image)
    img_view = sitk.GetArrayViewFromImage(seg_image)

    # Compute parameters for ground truth segmentation
    troch_notch_y, troch_notch_x = getAnteriorOfWeightBearing(
        img_view, femurIndex=fem_cart_label_idx
    )
    loc_fem_z, loc_fem_y, loc_fem_x = np.where(img_view == fem_cart_label_idx)
    post_femur_slice = np.max(loc_fem_x)
    posterior_wb_slice = np.round(
        (post_femur_slice - troch_notch_x) * truth_wb_factor + troch_notch_x
    ).astype(int)
    fem_y_midpoint = np.round(np.mean(loc_fem_y)).astype(int)

    # Compute ground truth segmentation
    gt_array = get_cartilage_subregions(
        img_array,
        anteriorWBslice=troch_notch_x,
        posteriorWBslice=posterior_wb_slice,
        trochY=troch_notch_y,
        femurLabel=fem_cart_label_idx,
        medTibiaLabel=2,
        latTibiaLabel=3,
        antFemurMask=11,
        medWbFemurMask=12,
        latWbFemurMask=13,
        medPostFemurMask=14,
        latPostFemurMask=15,
        mid_fem_y=fem_y_midpoint,
    )
    gt_image = sitk.GetImageFromArray(gt_array)
    gt_image.CopyInformation(seg_image)

    # Define a known affine transform (5 deg rotation about z-axis and translation)
    angle = np.deg2rad(5)
    transform = sitk.AffineTransform(3)
    cos_angle, sin_angle = np.cos(angle), np.sin(angle)
    matrix = (cos_angle, -sin_angle, 0, sin_angle, cos_angle, 0, 0, 0, 1)
    transform.SetMatrix(matrix)
    transform.SetTranslation((1.0, -1.0, 0.5))

    # Apply the transform to simulate misalignment
    resampler = sitk.ResampleImageFilter()
    resampler.SetReferenceImage(seg_image)
    resampler.SetInterpolator(sitk.sitkNearestNeighbor)
    resampler.SetTransform(transform)
    transformed_image = resampler.Execute(seg_image)

    # Run the alignment/recovery pipeline
    recovered_image = get_aligned_cartilage_subregions(
        transformed_image,
        wb_region_percent_dist=wb_region_percent_dist,
        femurLabel=fem_cart_label_idx,
        femurBoneLabel=5,
        medTibiaLabel=2,
        latTibiaLabel=3,
        antFemurMask=11,
        medWbFemurMask=12,
        latWbFemurMask=13,
        medPostFemurMask=14,
        latPostFemurMask=15,
        reference_image_input=seg_image,  # using the original image as reference
        ref_image_fem_bone_label=5,
    )

    # Apply the inverse transform to recovered_image so it aligns with seg_image
    inverse_transform = transform.GetInverse()
    resampler_inv = sitk.ResampleImageFilter()
    resampler_inv.SetReferenceImage(seg_image)
    resampler_inv.SetInterpolator(sitk.sitkNearestNeighbor)
    resampler_inv.SetTransform(inverse_transform)
    final_recovered_image = resampler_inv.Execute(recovered_image)

    # Compare the recovered segmentation with the ground truth
    recovered_array = sitk.GetArrayFromImage(final_recovered_image)
    diff = np.abs(recovered_array.astype(np.int32) - gt_array.astype(np.int32))
    n_voxels = np.prod(diff.shape)
    percent_wrong = 1 - ((n_voxels - np.sum(diff != 0)) / n_voxels)
    print(f"Percent wrong: {percent_wrong:.2%}")
    assert (
        percent_wrong < 0.1
    ), "More than 0.1% of the voxels are wrong; pipeline did not recover the segmentation correctly."

    # Verify label consistency between recovered segmentation and the transformed segmentation
    recovered_aligned_array = sitk.GetArrayFromImage(recovered_image)
    transformed_array = sitk.GetArrayFromImage(transformed_image)
    expected_femur_label = fem_cart_label_idx
    fem_cart_subregion_labels = {11, 12, 13, 14, 15}

    invalid_voxels = np.logical_and(
        np.isin(recovered_aligned_array, list(fem_cart_subregion_labels)),
        transformed_array != expected_femur_label,
    )
    num_invalid_voxels = np.sum(invalid_voxels)
    print(
        "Number of voxels with recovered femur subregion labels outside the original femur region:",
        num_invalid_voxels,
    )
    assert (
        num_invalid_voxels == 0
    ), "Some recovered femur cartilage subregion labels are found outside of the original femur region in transformed_image."

    invalid_voxels2 = np.logical_and(
        transformed_array == expected_femur_label,
        ~np.isin(recovered_aligned_array, list(fem_cart_subregion_labels)),
    )
    num_invalid_voxels2 = np.sum(invalid_voxels2)
    print(
        "Number of voxels in the original femur region not recovered as a femur subregion:",
        num_invalid_voxels2,
    )
    assert (
        num_invalid_voxels2 == 0
    ), "Some voxels from the original femur region in transformed_image do not have a valid femur cartilage subregion label in recovered_image."

    print(
        "Femur cartilage subregion labels are correctly aligned with the original femur cartilage region."
    )


def test_right_knee_pipeline_with_real_data():
    # For the right knee, use the original segmentation and matching wb factor
    run_cartilage_subregion_test(
        SEG_IMAGE, fem_cart_label_idx=1, wb_region_percent_dist=0.6, truth_wb_factor=0.6
    )


def test_left_knee_pipeline():
    # For the left knee, flip the segmentation image to simulate a left knee
    left_seg_array = np.flip(sitk.GetArrayFromImage(SEG_IMAGE), axis=0)
    seg_image = sitk.GetImageFromArray(left_seg_array)
    seg_image.CopyInformation(SEG_IMAGE)
    # Note: here we use a different truth_wb_factor (0.5) while still passing 0.6 to the alignment function
    run_cartilage_subregion_test(
        seg_image, fem_cart_label_idx=1, wb_region_percent_dist=0.6, truth_wb_factor=0.5
    )
