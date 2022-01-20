from numpy.testing import assert_allclose
import SimpleITK as sitk


def assert_images_same(image1, image2):
    """
    Helper function to assert that 2 SimpleITK images are the same. 

    Parameters
    ----------
    image1 : SimpleITK.Image
        Version 1 of the image. 
    image2 : SimpleITK.Image
        Version 2 of the image. 
    """    
    image1_array = sitk.GetArrayFromImage(image1)
    image2_array = sitk.GetArrayFromImage(image2)

    assert_allclose(image1_array, image2_array)
    assert image1.GetOrigin() == image2.GetOrigin()
    assert image1.GetSpacing() == image2.GetSpacing()
    assert image1.GetDirection() == image2.GetDirection()