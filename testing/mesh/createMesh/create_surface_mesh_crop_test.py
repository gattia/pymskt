"""
`create_surface_mesh(crop_to_label=True)` must give the same mesh as meshing the whole
volume. The reference-mesh tests in `testing/mesh/meshes` guard this against saved
full-volume meshes; here the two paths are compared directly, including the clipping of
the crop box at the image border.
"""

import numpy as np
import pytest
import pyvista as pv
import SimpleITK as sitk

from pymskt import ATOL, RTOL
from pymskt.mesh.createMesh import _crop_to_label, create_surface_mesh
from pymskt.utils import testing

SEG_IMAGE = sitk.ReadImage("data/right_knee_example.nrrd")


def _max_surface_distance(mesh_a, mesh_b):
    """Largest closest-point distance between the two surfaces, in both directions."""
    a, b = pv.wrap(mesh_a), pv.wrap(mesh_b)
    _, on_b = b.find_closest_cell(a.points, return_closest_point=True)
    _, on_a = a.find_closest_cell(b.points, return_closest_point=True)
    return max(
        np.linalg.norm(a.points - on_b, axis=1).max(),
        np.linalg.norm(b.points - on_a, axis=1).max(),
    )


def _mesh_both_ways(image, label_idx, image_smooth_var, **kwargs):
    meshes = []
    for crop in (False, True):
        meshes.append(
            create_surface_mesh(
                image,
                label_idx,
                image_smooth_var,
                tmp_filename=f"crop_test_{label_idx}_{int(crop)}.nrrd",
                crop_to_label=crop,
                **kwargs,
            )
        )
    return meshes


def _assert_same_mesh(full, cropped):
    assert cropped.GetNumberOfPoints() == full.GetNumberOfPoints()
    assert cropped.GetNumberOfCells() == full.GetNumberOfCells()
    assert _max_surface_distance(full, cropped) < 1e-4
    # marching cubes visits cells in the same order in both runs, so vertices line up too
    testing.assert_mesh_coordinates_same(pv.wrap(full), pv.wrap(cropped), rtol=RTOL, atol=ATOL)


@pytest.mark.parametrize(
    "label_idx, image_smooth_var",
    [
        pytest.param(5, 1.0, id="femur_bone"),
        pytest.param(1, 0.3125 / 2, id="femur_cartilage"),
        pytest.param(7, 1.0, id="patella_bone"),
    ],
)
def test_crop_matches_full_volume(label_idx, image_smooth_var):
    full, cropped = _mesh_both_ways(SEG_IMAGE, label_idx, image_smooth_var)
    _assert_same_mesh(full, cropped)


@pytest.mark.parametrize(
    "kwargs",
    [
        pytest.param(dict(filter_binary_image=False), id="no_smoothing"),
        pytest.param(
            dict(surface_extraction_method="discrete_marching_cubes"), id="discrete_marching_cubes"
        ),
        pytest.param(dict(surface_extraction_method="flying_edges"), id="flying_edges"),
        pytest.param(dict(mc_threshold=0.1), id="low_threshold"),
    ],
)
def test_crop_matches_full_volume_other_extraction_settings(kwargs):
    full, cropped = _mesh_both_ways(SEG_IMAGE, 7, 1.0, **kwargs)
    _assert_same_mesh(full, cropped)


def _image_with_label_on_border():
    """
    Small image (z, y, x) whose label 3 touches the x=0 and z=0 faces, with a second label
    away from the edges, anisotropic spacing, an offset origin and a flipped direction.
    """
    array = np.zeros((24, 30, 40), dtype=np.int8)
    array[0:10, 8:20, 0:15] = 3
    array[12:22, 8:25, 20:38] = 2
    image = sitk.GetImageFromArray(array)
    image.SetSpacing((0.5, 0.5, 1.0))
    image.SetOrigin((-10.0, 5.0, 20.0))
    image.SetDirection((-1, 0, 0, 0, -1, 0, 0, 0, 1))
    return image


def test_crop_box_is_clipped_to_the_image():
    image = _image_with_label_on_border()
    cropped = _crop_to_label(image, 3, image_smooth_var=0.25)

    # sigma = 0.5 mm -> 1 voxel in x/y, 0.5 voxel in z -> margins of 6, 6, 4 voxels.
    # x and z are clipped at the image edge (label starts at 0); y keeps both margins.
    assert cropped.GetSize() == (15 + 6, 12 + 6 + 6, 10 + 4)  # (x, y, z)
    assert cropped.GetOrigin() == image.TransformIndexToPhysicalPoint((0, 8 - 6, 0))
    assert cropped.GetDirection() == image.GetDirection()
    assert cropped.GetSpacing() == image.GetSpacing()

    # a voxel of the crop sits at the same physical position as in the full image
    assert cropped.TransformIndexToPhysicalPoint((3, 4, 5)) == pytest.approx(
        image.TransformIndexToPhysicalPoint((3, 4 + 2, 5))
    )


@pytest.mark.parametrize("set_seg_border_to_zeros", [True, False])
@pytest.mark.parametrize("filter_binary_image", [True, False])
def test_crop_matches_full_volume_label_on_border(set_seg_border_to_zeros, filter_binary_image):
    image = _image_with_label_on_border()
    full, cropped = _mesh_both_ways(
        image,
        3,
        0.25,
        filter_binary_image=filter_binary_image,
        set_seg_border_to_zeros=set_seg_border_to_zeros,
    )
    assert full.GetNumberOfPoints() > 0
    _assert_same_mesh(full, cropped)


def test_crop_matches_full_volume_tiny_label():
    # one voxel, no smoothing: margin 2 gives a 5-voxel box, widened to 10 along x so that
    # vtkNrrdReader does not read the crop as a 2D image with 5 components per pixel
    array = np.zeros((20, 20, 20), dtype=np.int8)
    array[9, 10, 11] = 1
    image = sitk.GetImageFromArray(array)
    image.SetSpacing((0.4, 0.4, 0.7))

    assert _crop_to_label(image, 1).GetSize() == (10, 5, 5)
    full, cropped = _mesh_both_ways(image, 1, 0.1, filter_binary_image=False)
    assert full.GetNumberOfPoints() > 0
    _assert_same_mesh(full, cropped)


def test_crop_missing_label_raises():
    with pytest.raises(ValueError, match="not present"):
        _crop_to_label(_image_with_label_on_border(), 99, image_smooth_var=0.25)
