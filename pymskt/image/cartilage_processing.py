import SimpleITK as sitk
import numpy as np
from scipy import ndimage as ndi


def CofM(array):
    '''
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
    
    '''
    pixels = np.where(array==1)
    centerPixels = np.mean(pixels)
    nans = np.isnan(centerPixels)
    if nans == True:
        centerPixels = 0
    return(centerPixels)

def get_y_CofM(flattenedSeg):
    '''
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
    
    '''
    locationFemur = np.where(flattenedSeg==1)
    yCofM = np.zeros((flattenedSeg.shape[0], 2), dtype=int)
    
    # only calculate for rows with cartilage. 
    minRow = np.min(locationFemur[0])
    maxRow = np.max(locationFemur[0])
    
    # iterate over rows of image, get CofM, store CofM for row. 
    for x in range(minRow, maxRow):
        yCofM[x, 0] = x #store the x-coordinate (row) we calcualted CofM for. 
        yCofM[x, 1] = int(CofM(flattenedSeg[x, :])) # store the CofM value (make it an integer for indexing)
    yCofM = yCofM[minRow+10:maxRow-10,:] # remove 10 most medial and most lateral pixels of femoral cartilage. 
    return(yCofM) 

def absolute_CofM(flattenedSeg):
    '''
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
    
    '''
    femurPoints = np.where(flattenedSeg==1)
    centerX = np.mean(femurPoints[0])
    centerY = np.mean(femurPoints[1])
    return(centerX, centerY)

def findNotch(flattenedSeg, trochleaPositionX=1000):
    '''
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
    
    '''
    # Goal is to find the most anterior point that is between the medial/lateral condyles

    # First guess at the troch notch in the 1st axis (med/lat axis) is the location with the smallest value for
    # the 2nd axis CofM. This is because in axis 1, negative is anterior and we expect the most anterior CofM should 
    # roughly align with the trochlear notch.
    y_CofM = get_y_CofM(flattenedSeg)
    first_guess = y_CofM[np.argmin(y_CofM[:,1]), 0]
    # the second guess is just the CofM of the whole cartilage. 
    centerX, centerY = absolute_CofM(flattenedSeg)
    second_guess = centerX

    # We use the 2 guesses to help define a search space for the trochlear notch.
    min_search = int(np.min((first_guess,second_guess))-20)
    max_search = int(np.max((first_guess,second_guess))+20)

    # now, we iterate over all of the rows (axis 1) of the search space (moving in the medial/lateral direction)
    # we are looking for the row where the most posterior point (back of femur) is furthest anterior (notch). 
    for y in range(min_search, max_search):
        # At each row, we find most posterior pixel labeled as cartilage. 
        try:
            trochleaPosition_test = np.max(np.where(flattenedSeg[y,:]==1))
        except ValueError:
            # if there is no cartilage we'll get a ValueError exception. 
            # in that case, set this value to be the max it can be (the size of the first axis)
            trochleaPosition_test = flattenedSeg.shape[1]
        # if the most posterior point for this row is more anterior than the current trochleaPositionX,
        # then update this to be the new trochlear notch.
        if trochleaPosition_test < trochleaPositionX:
            trochleaPositionX = trochleaPosition_test
            trochleaPositionY = y

    return(trochleaPositionY, trochleaPositionX+1)

def getAnteriorOfWeightBearing(segArray, femurIndex=1):
    '''
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
    
    '''

    femurSegmentation = np.zeros_like(segArray)
    femurSegmentation[segArray == femurIndex] = 1
    flattenedSegmentation = np.amax(femurSegmentation, axis=1)
    flattened_seg_filled = ndi.binary_fill_holes(flattenedSegmentation)
    trochY, trochX = findNotch(flattened_seg_filled)
    return(trochY, trochX)
    
def getCartilageSubRegions(segArray, anteriorWBslice, posteriorWBslice, trochY,
                           femurLabel=1, medTibiaLabel=2, latTibiaLabel=3, antFemurMask=5, 
                           medWbFemurMask=6, latWbFemurMask=7, medPostFemurMask=8, latPostFemurMask=9):
    '''
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
    Returns
    -------
    final_segmentation : array
        3D array with the updated segmentations - including weightbearing, medial/latera, anterior, and posterior. 
    Notes
    -----
    
    '''
    
    #array to store final segmentation
    final_segmentation = np.zeros_like(segArray)
    
    #create masks for ant/wb/posterior femur
    anterior_femur_mask = np.zeros_like(segArray)
    anterior_femur_mask[:,:,:anteriorWBslice] = 1

    wb_femur_mask = np.zeros_like(segArray)
    wb_femur_mask[:,:,anteriorWBslice:posteriorWBslice] = 1

    posterior_femur_mask = np.zeros_like(segArray)
    posterior_femur_mask[:,:,posteriorWBslice:] = 1
    
    #create seg of just femur - and then break it into the sub-regions
    femurSegArray = np.zeros_like(segArray)
    femurSegArray[segArray==femurLabel] = 1
    
    #find the center of the medial/lateral tibia - use to distinguish M/L femur ROIs
    locationMedialTibia = np.asarray(np.where(segArray==medTibiaLabel))
    locationLateralTibia = np.asarray(np.where(segArray==latTibiaLabel))
    
    centerMedialTibia = locationMedialTibia.mean(axis=1)
    centerLateralTibia = locationLateralTibia.mean(axis=1)

    med_femur_mask = np.zeros_like(segArray)
    lat_femur_mask = np.zeros_like(segArray)
    if centerMedialTibia[0] > trochY:
        med_femur_mask[trochY:,:,:] = 1
        lat_femur_mask[:trochY,:,:] = 1
    else:
        med_femur_mask[:trochY,:,:] = 1
        lat_femur_mask[trochY:,:,:] = 1

    final_segmentation[segArray!=femurLabel] = segArray[segArray!=femurLabel] 
    final_segmentation += (femurSegArray * anterior_femur_mask) * antFemurMask
    final_segmentation += (femurSegArray * wb_femur_mask * med_femur_mask) * medWbFemurMask
    final_segmentation += (femurSegArray * wb_femur_mask * lat_femur_mask) * latWbFemurMask
    final_segmentation += (femurSegArray * posterior_femur_mask * med_femur_mask) * medPostFemurMask
    final_segmentation += (femurSegArray * posterior_femur_mask * lat_femur_mask) * latPostFemurMask
    
    return(final_segmentation)


def get_knee_segmentation_with_femur_subregions(seg_image,
                                                fem_cart_label_idx=1,
                                                wb_region_percent_dist=0.6,
                                                femur_label=1,
                                                med_tibia_label=2,
                                                lat_tibia_label=3,
                                                ant_femur_mask=11,
                                                med_wb_femur_mask=12,
                                                lat_wb_femur_mask=13,
                                                med_post_femur_mask=14,
                                                lat_post_femur_mask=15
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

    Returns
    -------
    SimpleITK.Image
        Image of the new/updated segmentation
    """
    troch_notch_y, troch_notch_x = getAnteriorOfWeightBearing(sitk.GetArrayViewFromImage(seg_image),
                                                              femurIndex=fem_cart_label_idx)
    loc_fem_z, loc_fem_y, loc_fem_x = np.where(sitk.GetArrayViewFromImage(seg_image) == fem_cart_label_idx)
    post_femur_slice = np.max(loc_fem_x)
    posterior_wb_slice = np.round((post_femur_slice - troch_notch_x) * wb_region_percent_dist + troch_notch_x).astype(int)
    new_seg_array = getCartilageSubRegions(sitk.GetArrayViewFromImage(seg_image),
                                           anteriorWBslice=troch_notch_x,
                                           posteriorWBslice=posterior_wb_slice,
                                           trochY=troch_notch_y,
                                           femurLabel=femur_label,
                                           medTibiaLabel=med_tibia_label,
                                           latTibiaLabel=lat_tibia_label,
                                           antFemurMask=ant_femur_mask,
                                           medWbFemurMask=med_wb_femur_mask,
                                           latWbFemurMask=lat_wb_femur_mask,
                                           medPostFemurMask=med_post_femur_mask,
                                           latPostFemurMask=lat_post_femur_mask
                                           )
    seg_label_image = sitk.GetImageFromArray(new_seg_array)
    seg_label_image.CopyInformation(seg_image)
    return seg_label_image
