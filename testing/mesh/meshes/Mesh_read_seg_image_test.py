import pytest
import SimpleITK as sitk

import pymskt as mskt
from pymskt.utils import testing

PATH_SEG_IMAGE = "data/right_knee_example.nrrd"
SEG_IMAGE = sitk.ReadImage(PATH_SEG_IMAGE)

from pymskt import ATOL, RTOL


def test_image_path_provided_to_init(path_seg_image=PATH_SEG_IMAGE, seg_image=SEG_IMAGE):
    mesh = mskt.mesh.Mesh(path_seg_image=path_seg_image)
    mesh.read_seg_image()

    testing.assert_images_same(seg_image, mesh.seg_image, rtol=RTOL, atol=ATOL)


def test_image_path_provided_to_read_seg_image_directly(
    path_seg_image=PATH_SEG_IMAGE, seg_image=SEG_IMAGE
):
    mesh = mskt.mesh.Mesh()
    mesh.read_seg_image(path_seg_image=path_seg_image)

    testing.assert_images_same(seg_image, mesh.seg_image, rtol=RTOL, atol=ATOL)


def test_exception_when_no_path_provided():
    with pytest.raises(Exception):
        mesh = mskt.mesh.Mesh()
        mesh.read_seg_image()


if __name__ == "__main__":
    import time

    # BELOW CAN BE USED TO RUN TESTS DIRECTLY
    # CAN BE ACHIEVED BY CALLING EACH TEST INDIVIDUALLY, like:
    test_image_path_provided_to_init(path_seg_image=PATH_SEG_IMAGE)
