import os
import warnings
from tempfile import gettempdir

import numpy as np
import requests
import SimpleITK as sitk
from scipy import ndimage as ndi


def CofM(array):
    """
    Get center of mass for a row of a binary 2D image.
    Parameters
    ----------
    array : 1D array
        Individual row of a 2D image.
    Returns
    -------
    centerPixels :
        Average location of 1s in the row
    Notes
    -----
    Calculates the average location of cartilage for the row of image being analyzed.
    Returns 0 if there are no pixels

    """
    pixels = np.where(array == 1)
    centerPixels = np.mean(pixels)
    nans = np.isnan(centerPixels)
    if nans == True:
        centerPixels = 0
    return centerPixels


def get_y_CofM(flattenedSeg):
    """
    Get CofM of femoral cartilage for each row of the flattened segmentation.
    Parameters
    ----------
    flattenedSeg : 2D array
        Axial flattened, and filled in femoral cartilage segmentation.
    Returns
    -------
    yCofM :
        Find the CofM for each row of the image.
    Notes
    -----
    Get the x/y coordinates for the CofM for each row of the flattened segmentation.

    """
    locationFemur = np.where(flattenedSeg == 1)
    yCofM = np.zeros((flattenedSeg.shape[0], 2), dtype=int)

    # only calculate for rows with cartilage.
    minRow = np.min(locationFemur[0])
    maxRow = np.max(locationFemur[0])

    # iterate over rows of image, get CofM, store CofM for row.
    for x in range(minRow, maxRow):
        yCofM[x, 0] = x  # store the x-coordinate (row) we calcualted CofM for.
        yCofM[x, 1] = int(
            CofM(flattenedSeg[x, :])
        )  # store the CofM value (make it an integer for indexing)

    # remove 0.2 * (maxRow - minRow) of pixels from the most medial and most lateral side of the femur
    offset = int(0.2 * (maxRow - minRow))
    if minRow + offset < maxRow - offset:
        yCofM = yCofM[
            minRow + offset : maxRow - offset, :
        ]  # remove 10 most medial and most lateral pixels of femoral cartilage.
    else:
        # fallback to original range if removing pixels is not possible.
        warnings.warn(
            "Not enough pixels to remove most medial and most lateral pixels of femoral cartilage."
        )
    return yCofM


def absolute_CofM(flattenedSeg):
    """
    Get absolute CofM of all the femoral cartilage pixels
    Parameters
    ----------
    flattenedSeg : 2D array
        Axial flattened, and filled in femoral cartilage segmentation.
    Returns
    -------
    centerX :
        The CofM in the X direction for the segmentation
    centerY :
        The CofM in the Y direction for the segmentation
    Notes
    -----
    Get the x/y coordinates for the CofM for the whole flattened segmentation

    """
    femurPoints = np.where(flattenedSeg == 1)
    centerX = np.mean(femurPoints[0])
    centerY = np.mean(femurPoints[1])
    return (centerX, centerY)


def findNotch(flattenedSeg, trochleaPositionX=1000):
    """
    Get the X Y position of the trochlear notch - where medial/lateral sides of the femur meet.
    Parameters
    ----------
    flattenedSeg : 2D array
        Axial flattened, and filled in femoral cartilage segmentation.
    Returns
    -------
    trochleaPositionY :
        Y position of trochlear notch
    trochleaPositionX :
        X position of trochlear notch
    Notes
    -----
    Get the x/y coordinates for the trochlear notch. This is an iterative method that assumes things about the shape the
    femoral cartilage.

    """
    # Goal is to find the most anterior point that is between the medial/lateral condyles

    # First guess at the troch notch in the 1st axis (med/lat axis) is the location with the smallest value for
    # the 2nd axis CofM. This is because in axis 1, negative is anterior and we expect the most anterior CofM should
    # roughly align with the trochlear notch.
    y_CofM = get_y_CofM(flattenedSeg)
    first_guess = y_CofM[np.argmin(y_CofM[:, 1]), 0]
    # the second guess is just the CofM of the whole cartilage.
    centerX, centerY = absolute_CofM(flattenedSeg)
    second_guess = centerX

    # We use the 2 guesses to help define a search space for the trochlear notch.
    offset = int(0.25 * (flattenedSeg.shape[0]))
    min_search = int(np.min((first_guess, second_guess)) - offset)
    max_search = int(np.max((first_guess, second_guess)) + offset)
    # check if search space is valid
    if min_search > max_search or min_search < 0 or max_search > flattenedSeg.shape[0]:
        warnings.warn(
            "Avoiding invalid search space for trochlear notch,\
                       the search space will be set to the full range of the flattened segmentation."
        )
        min_search = 0
        max_search = flattenedSeg.shape[0]

    # now, we iterate over all of the rows (axis 1) of the search space (moving in the medial/lateral direction)
    # we are looking for the row where the most posterior point (back of femur) is furthest anterior (notch).
    for y in range(min_search, max_search):
        # At each row, we find most posterior pixel labeled as cartilage.
        try:
            trochleaPosition_test = np.max(np.where(flattenedSeg[y, :] == 1))
        except ValueError:
            # if there is no cartilage we'll get a ValueError exception.
            # in that case, set this value to be the max it can be (the size of the first axis)
            trochleaPosition_test = flattenedSeg.shape[1]
        # if the most posterior point for this row is more anterior than the current trochleaPositionX,
        # then update this to be the new trochlear notch.
        if trochleaPosition_test < trochleaPositionX:
            trochleaPositionX = trochleaPosition_test
            trochleaPositionY = y

    return (trochleaPositionY, trochleaPositionX + 1)


def getAnteriorOfWeightBearing(segArray, femurIndex=1):
    """
    Prepare full segmentation and extract the trochlear notch location.
    Parameters
    ----------
    flattenedSeg : 2D array
        Axial flattened, and filled in femoral cartilage segmentation.
    femurIndex : int
        Index of the label used to localize the femur in the array.
    Returns
    -------
    trochleaPositionY :
        Y position of trochlear notch
    trochleaPositionX :
        X position of trochlear notch
    Notes
    -----
    Get the x/y coordinates for the trochlear notch. This is an iterative method that assumes things about the shape the femoral cartilage.
    First flatten and fill any holes in the segmentation.

    """

    femurSegmentation = np.zeros_like(segArray)
    femurSegmentation[segArray == femurIndex] = 1
    flattenedSegmentation = np.amax(femurSegmentation, axis=1)
    flattened_seg_filled = ndi.binary_fill_holes(flattenedSegmentation)
    trochY, trochX = findNotch(flattened_seg_filled)
    return (trochY, trochX)


def get_superior_fem_cart_region(seg_array, fem_cart_idx=2, ratio=0.33):
    # get the average location of the fem cartilage
    # in the IS direction for each slice.
    mean_fem_ap = np.zeros(seg_array.shape[2])
    # iteratve over slices
    for ap_idx in range(seg_array.shape[2]):
        # get locations where fem cartilage is
        slice_ = np.where(seg_array[:, :, ap_idx] == fem_cart_idx)
        # get average of fem cart locations
        mean_ = np.mean(slice_[1])
        # store result
        mean_fem_ap[ap_idx] = mean_

    # get the top of the femoral cartilage globally (in IS)
    top_fem_cart = np.min(np.where(seg_array == fem_cart_idx)[1])

    # create a superior decision boundary between the
    # top of the fem cartilag and the average of the fem cartilage
    # at that slice. This position is a ratio between the two. If
    # the ratio = 0.5 its the middle, if 1.0 its the top
    # if 0.0 its the mean of the fem cart at that slice.
    # default 0.33 is above the middle of the femur for that slice,
    # but below the midway point.
    wb_IS_cutoff = mean_fem_ap * (1 - ratio) + top_fem_cart * ratio

    # create a mask for all cartilage above this wb_IS_cutoff
    superior_mask = np.zeros_like(seg_array)

    # set the parts anterior to the fem cart as all having
    # the same mask that would happen at the most anterior border
    # and the ones behind the fem cart to be the same as the last
    # point.
    # get all fem cart points.
    loc_fem_cart = np.where(seg_array == fem_cart_idx)
    # Get the anterior point of fem cart
    ap_start_idx = np.min(loc_fem_cart[-1])
    # get posterior point.
    ap_end_idx = np.max(loc_fem_cart[-1])

    # get the anterior/posterior parts of the mask.
    superior_mask[:, : int(wb_IS_cutoff[ap_start_idx]), :ap_start_idx] = 1
    superior_mask[:, : int(wb_IS_cutoff[ap_end_idx]), ap_end_idx:] = 1

    # iterate over all other slices between the ant/post and
    # fill mask accordingly
    for ap_idx in range(ap_start_idx, ap_end_idx):
        # get the border for the superior/inferior regions
        border_ = wb_IS_cutoff[ap_idx]
        if np.isfinite(border_):
            # If its finite, then make it the official border used
            # for masking.
            border = int(border_)
            # If it wasnt finite, then this isnt updated,
            # which means that the last time it was finite will
            # be used. This handles the casess where cartilage holes
            # could potentially cause there to be no values.
            # Thought - its unlikely there wuold be complete holes
            # along the entire medial/lateral condyles.
        superior_mask[:, :border, ap_idx] = 1

    return superior_mask


def get_cartilage_subregions(
    segArray,
    anteriorWBslice,
    posteriorWBslice,
    trochY,
    femurLabel=1,
    medTibiaLabel=2,
    latTibiaLabel=3,
    antFemurMask=5,
    medWbFemurMask=6,
    latWbFemurMask=7,
    medPostFemurMask=8,
    latPostFemurMask=9,
    mid_fem_y=None,
):
    """
    Take cartilage segmentation, and decompose femoral cartilage into subregions of interest.
    Parameters
    ----------
    segArray : array
        3D array with segmentation for the cartialge regions.
    anteriorWBslice : int
        Slice that seperates the anterior and weight bearing femoral cartilage.
    posteriorWBslice : int
        Slice that seperates the weight bearing and posterior femoral cartilage.
    trochY : int
        Slice that differentiates medial / lateral femur - trochlear notch Y component.
    femurLabel : int
        Label that femur is in the segArray
    medTibiaLabel : int
        Label that medial tibia is in the segArray
    latTibiaLabel : int
        Label that lateral tibia is in the segArray
    antFemurMask : int
        Label anterior femur should be labeled in final segmentation.
    medWbFemurMask : int
        Label medial weight bearing femur should be labeled in final segmentation.
    latWbFemurMask : int
        Label lateral weight bearing femur should be labeled in final segmentation.
    medPostFemurMask : int
        Label medial posterior femur should be labeled in final segmentation.
    latPostFemurMask : int
        Label lateral posterior femur should be labeled in final segmentation.
    mid_fem_y : int
        Deprecated - not used.
    Returns
    -------
    final_segmentation : array
        3D array with the updated segmentations - including weightbearing, medial/latera, anterior, and posterior.
    Notes
    -----

    """

    if mid_fem_y is not None:
        # warning that this is deprecated and not used.
        warnings.warn("mid_fem_y is deprecated and not used.")

    # array to store final segmentation
    final_segmentation = np.zeros_like(segArray)

    # create masks for ant/wb/posterior femur
    anterior_femur_mask = np.zeros_like(segArray)
    anterior_femur_mask[:, :, :anteriorWBslice] = 1

    wb_femur_mask = np.zeros_like(segArray)
    wb_femur_mask[:, :, anteriorWBslice:posteriorWBslice] = 1

    superior_mask = get_superior_fem_cart_region(segArray, fem_cart_idx=femurLabel, ratio=0.33)

    # only get weight bearing from the lower part of the femur - to make
    # sure don't accidentall get the top of the posterior condyle when it wraps back around.

    # If its the wb_femur region, but above the superior mask cut line
    # then assign to be wb_reror and actually assign to be posteror_femur_mask
    wb_femur_mask_error = wb_femur_mask * superior_mask
    wb_femur_mask[wb_femur_mask_error == 1] = 0  # fix the wb_femur_mask

    posterior_femur_mask = np.zeros_like(segArray)
    posterior_femur_mask[:, :, posteriorWBslice:] = 1

    posterior_femur_mask = np.max((wb_femur_mask_error, posterior_femur_mask), axis=0)

    # create seg of just femur - and then break it into the sub-regions
    femurSegArray = np.zeros_like(segArray)
    femurSegArray[segArray == femurLabel] = 1

    # find the center of the medial/lateral tibia - use to distinguish M/L femur ROIs
    locationMedialTibia = np.asarray(np.where(segArray == medTibiaLabel))
    locationLateralTibia = np.asarray(np.where(segArray == latTibiaLabel))

    centerMedialTibia = locationMedialTibia.mean(axis=1)
    centerLateralTibia = locationLateralTibia.mean(axis=1)

    med_femur_mask = np.zeros_like(segArray)
    lat_femur_mask = np.zeros_like(segArray)
    if centerMedialTibia[0] > trochY:
        med_femur_mask[trochY:, :, :] = 1
        lat_femur_mask[:trochY, :, :] = 1
    else:
        med_femur_mask[:trochY, :, :] = 1
        lat_femur_mask[trochY:, :, :] = 1

    final_segmentation[segArray != femurLabel] = segArray[segArray != femurLabel]
    final_segmentation += (femurSegArray * anterior_femur_mask) * antFemurMask
    final_segmentation += (femurSegArray * wb_femur_mask * med_femur_mask) * medWbFemurMask
    final_segmentation += (femurSegArray * wb_femur_mask * lat_femur_mask) * latWbFemurMask
    final_segmentation += (femurSegArray * posterior_femur_mask * med_femur_mask) * medPostFemurMask
    final_segmentation += (femurSegArray * posterior_femur_mask * lat_femur_mask) * latPostFemurMask

    return final_segmentation


def verify_and_correct_med_lat_tib_cart(
    seg_array,  # sitk.GetArrayViewFromImage(seg)
    tib_label=6,
    med_tib_cart_label=2,
    lat_tib_cart_label=3,
    ml_axis=0,
    split_method="geometric_tibia",
):
    """
    Verify that the medial and lateral tibial cartilage are correctly labeled.
    Parameters
    ----------
    seg_array : array
        3D array with segmentation for the cartilage/bone regions.
    tib_label : int
        Label that tibial cartilage is in the seg_array
    med_tib_cart_label : int
        Label that medial tibial cartilage is in the seg_array
    lat_tib_cart_label : int
        Label that lateral tibial cartilage is in the seg_array
    ml_axis : int
        Medial/lateral axis of the acquired knee MRI.

    Returns
    -------
    seg_array : array
        3D array with segmentation for the cartilage/bone regions.
        The tibial cartilage regions will have been updated to ensure
        all tib cart on med/lat sides are correctly classified.

    """
    # get binary array for tibia
    array_tib = np.zeros_like(seg_array)
    array_tib[seg_array == tib_label] = 1
    # get binary array for tib cart
    array_tib_cart = np.zeros_like(seg_array)
    array_tib_cart[(seg_array == lat_tib_cart_label) + (seg_array == med_tib_cart_label)] = 1

    # get the locatons of med/lat cartilage & get their centroids
    med_cart_locs_ = np.where(seg_array == med_tib_cart_label)
    med_cart_locs = np.asarray(med_cart_locs_)
    lat_cart_locs_ = np.where(seg_array == lat_tib_cart_label)
    lat_cart_locs = np.asarray(lat_cart_locs_)
    middle_med_cart = med_cart_locs[ml_axis, :].mean()
    middle_lat_cart = lat_cart_locs[ml_axis, :].mean()

    # get location of tibia to get centroid of tibial plateau
    tib_locs = np.asarray(np.where(seg_array == tib_label))
    middle_tib = tib_locs[ml_axis, :].mean()
    center_tibia_slice = int(middle_tib)

    # infer the direction(s) for medial/lateral
    med_direction = np.sign(middle_med_cart - middle_tib)
    lat_direction = np.sign(middle_lat_cart - middle_tib)
    if med_direction == lat_direction:
        raise Exception("Middle of med and lat tibial cartilage on same side of centerline!")

    # create med/lat cartilage masks - binary for updating seg masks
    med_tib_cart_mask = np.zeros_like(seg_array)
    lat_tib_cart_mask = np.zeros_like(seg_array)

    if split_method == "geometric_tibia":
        center_ = center_tibia_slice
    elif split_method == "geometric_cartilage":
        raise Exception("Not implemented yet!")
    elif split_method == "logistic_cartilage":
        from scipy.optimize import minimize

        def logistic(z):
            return 1 / (1 + np.exp(-z))

        def cost_function(theta, X, y):
            predictions = logistic(X @ theta)
            errors = y * np.log(predictions) + (1 - y) * np.log(1 - predictions)
            return -np.mean(errors)

        def gradient(theta, X, y):
            predictions = logistic(X @ theta)
            return X.T @ (predictions - y) / len(y)

        med_lat_axis_med_cart_locs = med_cart_locs_[ml_axis]
        med_lat_axis_lat_cart_locs = lat_cart_locs_[ml_axis]

        # med = np.asarray(flat_med_cart_locs).T
        # lat = np.asarray(flat_lat_cart_locs).T
        # pre-allocate data for logistic regression
        locs = np.zeros(
            (med_lat_axis_med_cart_locs.shape[0] + med_lat_axis_lat_cart_locs.shape[0], 2)
        )
        # add intercept term
        locs[:, 0] = 1
        # add locations (along ML axis)
        locs[:, 1] = np.concatenate(
            (med_lat_axis_med_cart_locs, med_lat_axis_lat_cart_locs), axis=0
        )
        # create labels array (dependent variable)
        labels = np.zeros(locs.shape[0])
        labels[med_lat_axis_med_cart_locs.shape[0] :] = 1

        # pre-allocate coefficients
        m, n = locs.shape
        theta = np.zeros(n)
        # run logistic regression
        result = minimize(
            cost_function, theta, args=(locs, labels), jac=gradient, options={"maxiter": 400}
        )
        theta = result.x

        X = np.zeros((seg_array.shape[ml_axis], 2))
        X[:, 0] = 1
        X[:, 1] = np.arange(seg_array.shape[ml_axis])
        predictions = logistic(X @ theta)
        center_ = int(np.argmin(abs(predictions - 0.5)))

    if med_direction > 0:
        med_tib_cart_mask[center_:, ...] = 1
        lat_tib_cart_mask[:center_, ...] = 1
    elif med_direction < 0:
        med_tib_cart_mask[:center_, ...] = 1
        lat_tib_cart_mask[center_:, ...] = 1

    # create new med/lat cartilage arrays
    new_med_cart_array = array_tib_cart * med_tib_cart_mask
    new_lat_cart_array = array_tib_cart * lat_tib_cart_mask

    # make copy of original segmentation array & update
    # med/lat tibial cartilage labels
    new_seg_array = seg_array.copy()
    new_seg_array[new_med_cart_array == 1] = med_tib_cart_label
    new_seg_array[new_lat_cart_array == 1] = lat_tib_cart_label

    return new_seg_array


def get_knee_segmentation_with_femur_subregions(
    seg_image,
    fem_cart_label_idx=1,
    wb_region_percent_dist=0.6,
    femur_label=5,
    med_tibia_label=2,
    lat_tibia_label=3,
    ant_femur_mask=11,
    med_wb_femur_mask=12,
    lat_wb_femur_mask=13,
    med_post_femur_mask=14,
    lat_post_femur_mask=15,
    verify_med_lat_tib_cart=True,
    tibia_label=6,
    ml_axis=0,
    ref_image_reg_percent_sampling=0.1,
):
    """
    Give seg image of knee. Return seg image with all sub-regions of femur included.

    Parameters
    ----------
    seg_image : SimpleITK.Image
        SimpleITK image of the segmentation to be processed.
    fem_cart_label_idx : int, optional
        Label of femoral cartilage, by default 1
    wb_region_percent_dist : float, optional
        How large weightbearing region is (from not to posterior of condyles), by default 0.6
    femur_label : int, optional
        Seg label for the femur cartilage, by default 1
    med_tibia_label : int, optional
        Seg label for the medial tibia cartilage, by default 2
    lat_tibia_label : int, optional
        Seg label for the lateral tibia cartilage, by default 3
    ant_femur_mask : int, optional
        Seg label for the anterior femur region, by default 11
    med_wb_femur_mask : int, optional
        Seg label for medial weight-bearing femur, by default 12
    lat_wb_femur_mask : int, optional
        Seg label for lateral weight-bearing femur, by default 13
    med_post_femur_mask : int, optional
        Seg label for medial posterior femur, by default 14
    lat_post_femur_mask : int, optional
        Seg label for lateral posterior femur, by default 15
    verify_med_lat_tib_cart : bool, optional
        Whether to verify that medial and lateral tibial cartilage is on same side of centerline, by default True
    tibia_label : int, optional
        Seg label for the tibia, by default 6
    ml_axis : int, optional
        Medial/lateral axis of the acquired knee MRI, by default 0

    Returns
    -------
    SimpleITK.Image
        Image of the new/updated segmentation
    """

    new_seg_image = get_aligned_cartilage_subregions(
        seg_image,
        wb_region_percent_dist=wb_region_percent_dist,
        femurLabel=fem_cart_label_idx,
        femurBoneLabel=femur_label,
        medTibiaLabel=med_tibia_label,
        latTibiaLabel=lat_tibia_label,
        antFemurMask=ant_femur_mask,
        medWbFemurMask=med_wb_femur_mask,
        latWbFemurMask=lat_wb_femur_mask,
        medPostFemurMask=med_post_femur_mask,
        latPostFemurMask=lat_post_femur_mask,
        reg_percent_sampling=ref_image_reg_percent_sampling,
    )

    new_seg_array = sitk.GetArrayFromImage(new_seg_image)

    if verify_med_lat_tib_cart:
        new_seg_array = verify_and_correct_med_lat_tib_cart(
            new_seg_array,
            tib_label=tibia_label,
            med_tib_cart_label=med_tibia_label,
            lat_tib_cart_label=lat_tibia_label,
            ml_axis=ml_axis,
        )
    seg_label_image = sitk.GetImageFromArray(new_seg_array)
    seg_label_image.CopyInformation(seg_image)
    return seg_label_image


def combine_depth_region_segs(orig_seg, depth_segs):
    if isinstance(orig_seg, sitk.Image):
        type_orig_seg = "sitk"
        orig_seg_array = sitk.GetArrayFromImage(orig_seg)
    elif isinstance(orig_seg, np.ndarray):
        type_orig_seg = "np"
        orig_seg_array = orig_seg

    # combine all of the depth segs that exist into one:
    if isinstance(depth_segs, np.ndarray):
        depth_segs = [depth_segs]
    elif isinstance(depth_segs, sitk.Image):
        depth_segs = [sitk.GetArrayFromImage(depth_segs)]
    elif isinstance(depth_segs, (list, tuple)):
        if all(isinstance(i, np.ndarray) for i in depth_segs):
            pass
        elif all(isinstance(i, sitk.Image) for i in depth_segs):
            depth_segs = [sitk.GetArrayFromImage(i) for i in depth_segs]
        else:
            raise ValueError("depth_segs must be a list of numpy arrays or SimpleITK images")

    # assert that all depth segs are the same size, and that they match the orig_seg size
    assert all(
        i.shape == orig_seg_array.shape for i in depth_segs
    ), "all depth segs must be the same size as orig_seg"

    # finally, assert that depth_segs only has 3 unique values (0, 100, 200)
    # if it happens to have 0, 1, 2 then convert to 0, 100, 200
    if np.unique(depth_segs).tolist() == [0, 1, 2]:
        depth_segs = [i * 100 for i in depth_segs]

    # combine the depth segs into a single mask.
    new_seg_combined = np.zeros_like(orig_seg_array, dtype=np.uint16)
    for i, depth_seg in enumerate(depth_segs):
        # if happens to be 0,1,2 then convert to 0,100,200
        if np.unique(depth_seg).tolist() == [0, 1, 2]:
            depth_seg = depth_seg * 100
        # assert that depth_seg only has allowed values (0, 100, 200)
        unique_labels = np.unique(depth_seg)
        allowed_labels = {0, 100, 200}
        assert set(unique_labels).issubset(
            allowed_labels
        ), f"depth_segs must only contain values within {allowed_labels}. Found: {unique_labels}"

        # could do += but this might end up with higher values in a voxel if
        # the same two masks are accidentally added twice.
        new_seg_combined[depth_seg == 100] = 100
        new_seg_combined[depth_seg == 200] = 200

    # finally, add the orig_seg back in.
    new_seg_combined += orig_seg_array.astype(
        np.uint16
    )  # this way label 1 will be 101 and 201 for the deep and superficial regions.

    if type_orig_seg == "sitk":
        new_seg_combined = sitk.GetImageFromArray(new_seg_combined)
        new_seg_combined.CopyInformation(orig_seg)

    return new_seg_combined


def get_aligned_cartilage_subregions(
    seg_image_sitk,
    wb_region_percent_dist=0.6,
    femurLabel=1,
    femurBoneLabel=5,
    medTibiaLabel=2,
    latTibiaLabel=3,
    antFemurMask=11,
    medWbFemurMask=12,
    latWbFemurMask=13,
    medPostFemurMask=14,
    latPostFemurMask=15,
    reference_image_input=os.path.join(
        os.path.dirname(os.path.dirname(__file__)), "data", "right_knee_example.nrrd"
    ),
    ref_image_fem_bone_label=5,
    reg_percent_sampling=0.1,
    print_transform=False,
):
    """

    Aligns the segmentation image to the reference image, splits the femur cartilage into subregions,
    and then resamples the subregions back to the original image space.

    Parameters:
    ----------
    seg_image_sitk: sitk.Image
        The segmentation image to be processed.
    wb_region_percent_dist: float, optional
        The percentage distance used by the pipeline to determine the weight-bearing region, by default 0.6
    femurLabel: int, optional
        The label of the femur cartilage, by default 1
    femurBoneLabel: int, optional
        The label of the femur bone, by default 5
    medTibiaLabel: int, optional
        The label of the medial tibia cartilage, by default 2
    latTibiaLabel: int, optional
        The label of the lateral tibia cartilage, by default 3
    antFemurMask: int, optional
        The label of the anterior femur region, by default 11
    medWbFemurMask: int, optional
        The label of the medial weight-bearing femur region, by default 12
    latWbFemurMask: int, optional
        The label of the lateral weight-bearing femur region, by default 13
    medPostFemurMask: int, optional
        The label of the medial posterior femur region, by default 14
    latPostFemurMask: int, optional
        The label of the lateral posterior femur region, by default 15
    reference_image_input: str or sitk.Image, optional
        The reference image to align the segmentation to, by default the right knee example image
    ref_image_fem_bone_label: int, optional
        The label of the femur bone in the reference image, by default 5
    reg_percent_sampling: float, optional
        The percentage of voxels to sample for registration, by default 0.1
    print_transform: bool, optional
        Whether to print the transform parameters, by default False

    Returns:
    --------
    sitk.Image
        The processed segmentation image with subregions aligned to the reference image.

    Notes:
    This function wraps get_cartilage_subregions by:
        1. Reading in the reference image from a file if a filepath is provided, and then binarizing it to include only the femur bone (label=femurBoneLabel).
        2. Determining knee laterality on seg_image_sitk using the medial and lateral tibial cartilage labels.
            If the new knee is left, it flips the binarized reference image along the medial/lateral axis.
        3. Performing a similarity registration using only the femur bone as the moving image.
            The moving image is created by thresholding seg_image_sitk to only keep the femur bone.
        4. Resampling the full segmentation (seg_image_sitk) into the aligned (reference) space using the computed transform.
        5. Running the existing get_cartilage_subregions on the aligned segmentation array. This is the main point, we want to
        break cartilage into subregions when the 3D array is aligned along the knee anatomical axes.
        6. Inversely resampling the processed subregions back to the original image space.
        7. For any voxel originally identified as femoral cartilage (using femurLabel) that does not directly carry a valid subregion label upon inverse resampling,
            assigning it the nearest valid label via a distance transform that uses the image spacing.
    """
    if isinstance(reference_image_input, str):
        if not os.path.exists(reference_image_input):
            # Use temp directory for downloaded assets
            data_dir = os.path.join(gettempdir(), "pymskt_data")
            os.makedirs(data_dir, exist_ok=True)

            dest_path = os.path.join(data_dir, "right_knee_example.nrrd")

            if not os.path.exists(dest_path):
                try:
                    url = "https://github.com/gattia/pymskt/raw/main/data/right_knee_example.nrrd"
                    response = requests.get(url, timeout=10)
                    response.raise_for_status()

                    with open(dest_path, "wb") as f:
                        f.write(response.content)
                except requests.exceptions.RequestException as e:
                    raise RuntimeError(
                        f"Failed to download reference image: {str(e)}\n"
                        "You can manually download it from:\n"
                        f"{url}\n"
                        f"and place it at: {dest_path}"
                    ) from e

            reference_image_input = dest_path

        try:
            reference_image = sitk.ReadImage(reference_image_input)
        except RuntimeError as e:
            raise RuntimeError(
                f"Failed to load reference image at {reference_image_input}\n"
                "The file may be corrupted. Try deleting it to trigger a re-download."
            ) from e
    else:
        reference_image = reference_image_input

    # Binarize the reference image to keep only the femur bone (label=femurBoneLabel)
    binary_reference = sitk.BinaryThreshold(
        reference_image,
        lowerThreshold=ref_image_fem_bone_label,
        upperThreshold=ref_image_fem_bone_label,
        insideValue=1,
        outsideValue=0,
    )
    # Cast the binary reference to Float32 for registration.
    binary_reference = sitk.Cast(binary_reference, sitk.sitkFloat32)

    # Step 1: Determine knee laterality using the tibial cartilage labels in seg_image_sitk.
    label_stats = sitk.LabelShapeStatisticsImageFilter()
    label_stats.Execute(seg_image_sitk)
    try:
        med_centroid = np.array(label_stats.GetCentroid(medTibiaLabel))
        lat_centroid = np.array(label_stats.GetCentroid(latTibiaLabel))
    except Exception as e:
        print(e)
        raise ValueError(
            "Could not compute centroids for tibial labels. Ensure they are present in segmentation."
        ) from e

    # For a right knee, we expect the lateral tibia to have a higher x-coordinate than the medial tibia.
    is_right_knee = lat_centroid[0] < med_centroid[0]

    # If the new knee is left (i.e. not right), flip the binary reference image along the medial/lateral axis.
    if not is_right_knee:
        binary_ref_array = sitk.GetArrayFromImage(binary_reference)
        # flip array along the med/lat axis
        binary_ref_array = np.flip(binary_ref_array, axis=0)
        binary_reference_flipped = sitk.GetImageFromArray(binary_ref_array)
        binary_reference_flipped.CopyInformation(binary_reference)
        binary_reference = binary_reference_flipped

    # Step 2: Create a moving image from seg_image_sitk that includes only the femur bone.
    moving_femur = sitk.BinaryThreshold(
        seg_image_sitk,
        lowerThreshold=femurBoneLabel,
        upperThreshold=femurBoneLabel,
        insideValue=1,
        outsideValue=0,
    )
    # Cast the moving image to Float32 for registration as well.
    moving_femur = sitk.Cast(moving_femur, sitk.sitkFloat32)

    # Step 3: Register the moving femur to the binary reference image using a similarity transform.
    registration_method = sitk.ImageRegistrationMethod()
    initial_transform = sitk.CenteredTransformInitializer(
        binary_reference,
        moving_femur,
        sitk.Similarity3DTransform(),
        sitk.CenteredTransformInitializerFilter.MOMENTS,
    )
    registration_method.SetInitialTransform(initial_transform, inPlace=False)
    registration_method.SetMetricAsMeanSquares()
    registration_method.SetInterpolator(sitk.sitkNearestNeighbor)
    registration_method.SetMetricSamplingPercentage(reg_percent_sampling)
    # registration_method.SetOptimizerAsRegularStepGradientDescent(learningRate=1.0,
    #                                                              minStep=1e-4,
    #                                                              numberOfIterations=200,
    #                                                              gradientMagnitudeTolerance=1e-6)
    registration_method.SetOptimizerAsGradientDescent(
        learningRate=1.0,
        numberOfIterations=200,
        convergenceMinimumValue=1e-6,
        convergenceWindowSize=10,
    )
    final_transform = registration_method.Execute(binary_reference, moving_femur)

    if print_transform:
        # get the transform parameters
        print(np.array(final_transform.GetParameters()).reshape(-1, 3).T)
    # Step 4: Resample the full segmentation into the aligned (reference) space using the computed transform.
    resampler = sitk.ResampleImageFilter()
    resampler.SetReferenceImage(binary_reference)
    resampler.SetInterpolator(sitk.sitkNearestNeighbor)
    resampler.SetTransform(final_transform)
    aligned_seg_image = resampler.Execute(seg_image_sitk)
    aligned_seg_array = sitk.GetArrayFromImage(aligned_seg_image)

    troch_notch_y, troch_notch_x = getAnteriorOfWeightBearing(
        sitk.GetArrayViewFromImage(aligned_seg_image), femurIndex=femurLabel
    )
    loc_fem_z, loc_fem_y, loc_fem_x = np.where(
        sitk.GetArrayViewFromImage(aligned_seg_image) == femurLabel
    )
    post_femur_slice = np.max(loc_fem_x)
    posterior_wb_slice = np.round(
        (post_femur_slice - troch_notch_x) * wb_region_percent_dist + troch_notch_x
    ).astype(int)

    # Get midpoint of femoral cartilage in the inferior/superior direction
    fem_y_midpoint = np.round(np.mean(loc_fem_y)).astype(int)

    # Step 5: Run the existing get_cartilage_subregions on the aligned segmentation array.
    subregion_array = get_cartilage_subregions(
        aligned_seg_array,
        anteriorWBslice=troch_notch_x,
        posteriorWBslice=posterior_wb_slice,
        trochY=troch_notch_y,
        femurLabel=femurLabel,
        medTibiaLabel=medTibiaLabel,
        latTibiaLabel=latTibiaLabel,
        antFemurMask=antFemurMask,
        medWbFemurMask=medWbFemurMask,
        latWbFemurMask=latWbFemurMask,
        medPostFemurMask=medPostFemurMask,
        latPostFemurMask=latPostFemurMask,
    )

    # Create a SimpleITK image from subregion_array in the aligned space.
    subregion_image_aligned = sitk.GetImageFromArray(subregion_array)
    subregion_image_aligned.CopyInformation(binary_reference)

    # Step 6: Inversely resample the processed subregion segmentation back to the original image space.
    inverse_transform = final_transform.GetInverse()
    resampler_back = sitk.ResampleImageFilter()
    resampler_back.SetReferenceImage(seg_image_sitk)
    resampler_back.SetInterpolator(sitk.sitkNearestNeighbor)
    resampler_back.SetTransform(inverse_transform)
    subregion_image_original = resampler_back.Execute(subregion_image_aligned)

    # Step 7: For every voxel in the original femoral cartilage region that did not receive a valid subregion label,
    # assign it the nearest valid label via a distance transform that uses the image spacing.
    orig_seg_array = sitk.GetArrayFromImage(seg_image_sitk)
    processed_seg_array = sitk.GetArrayFromImage(subregion_image_original)

    # everywhere that was not femur cartilage in original, set to the
    # values from the original seg_image_sitk (orig_seg_array)
    processed_seg_array[orig_seg_array != femurLabel] = orig_seg_array[orig_seg_array != femurLabel]

    # Define the valid femoral cartilage subregion labels.
    valid_labels = {
        antFemurMask,
        medWbFemurMask,
        latWbFemurMask,
        medPostFemurMask,
        latPostFemurMask,
    }
    fem_cart_mask = orig_seg_array == femurLabel
    valid_mask = np.isin(processed_seg_array, list(valid_labels))

    # Create the missing mask and find its indices.
    missing_mask = fem_cart_mask & (~valid_mask)
    missing_voxel_coords = np.nonzero(missing_mask)
    fixed_array = processed_seg_array.copy()

    # Only compute the distance transform if there are missing voxels.
    if missing_voxel_coords[0].size > 0:
        # Get the physical spacing (SimpleITK spacing is in x,y,z order)
        spacing = seg_image_sitk.GetSpacing()
        # Convert spacing to numpy array ordering (z, y, x)
        sampling = (spacing[2], spacing[1], spacing[0])
        distance, indices = ndi.distance_transform_edt(
            ~valid_mask, sampling=sampling, return_indices=True
        )
        nearest_z = indices[0][missing_voxel_coords]
        nearest_y = indices[1][missing_voxel_coords]
        nearest_x = indices[2][missing_voxel_coords]
        fixed_array[missing_voxel_coords] = processed_seg_array[nearest_z, nearest_y, nearest_x]

    final_subregion_image = sitk.GetImageFromArray(fixed_array)
    final_subregion_image.CopyInformation(seg_image_sitk)

    return final_subregion_image
