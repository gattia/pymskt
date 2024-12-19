import numpy as np
from numpy.testing import assert_allclose

import pymskt as mskt


def test_get_data_along_line(
    line_resolution=9, rand_int=np.random.randint(0, 10000), categorical=True
):
    uniform_image = mskt.image.create_vtk_image(
        origin=[0, 0, 0], dimensions=[20, 20, 20], spacing=[1, 1, 1], scalar=rand_int
    )

    probe = mskt.mesh.meshTools.ProbeVtkImageDataAlongLine(
        line_resolution=line_resolution,
        vtk_image=uniform_image,
        data_categorical=categorical,
    )

    scalars = probe.get_data_along_line(start_pt=(0, 0, 0), end_pt=(19, 19, 19))

    assert_allclose(np.ones(line_resolution + 1) * rand_int, scalars)


def test_get_data_along_line_that_is_outside_volume_assert_no_data_returned(
    line_resolution=9, rand_int=np.random.randint(0, 10000), categorical=True
):
    uniform_image = mskt.image.create_vtk_image(
        origin=[0, 0, 0], dimensions=[20, 20, 20], spacing=[1, 1, 1], scalar=rand_int
    )

    probe = mskt.mesh.meshTools.ProbeVtkImageDataAlongLine(
        line_resolution=line_resolution,
        vtk_image=uniform_image,
        data_categorical=categorical,
    )

    # Create a line that is outside of the volume - should still be nearest neighbours
    # so end up with the same data/result.
    scalars = probe.get_data_along_line(start_pt=(100, 100, 100), end_pt=(200, 200, 200))

    assert len(scalars) == 0


def test_scalars_retrieved_appropriately_line_in_middle_of_cells_categorical_false(
    line_resolution=9, categorical=False
):
    data = np.zeros((2, 2, 10))
    for i in range(line_resolution + 1):
        data[:, 0, i] = i
        data[:, 1, i] = i + 1

    vtk_image = mskt.image.create_vtk_image(origin=[0, 0, 0], spacing=[1, 1, 1], data=data)

    probe = mskt.mesh.meshTools.ProbeVtkImageDataAlongLine(
        line_resolution=line_resolution, vtk_image=vtk_image, data_categorical=categorical
    )

    scalars = probe.get_data_along_line(start_pt=(0.5, 0.5, 0), end_pt=(0.5, 0.5, line_resolution))

    assert_allclose(scalars, [i + 0.5 for i in range(line_resolution + 1)])


def test_scalars_retrieved_appropriately_line_just_off_center_categorical_true(
    line_resolution=9, categorical=True
):
    data = np.zeros((2, 2, 10))
    for i in range(line_resolution + 1):
        data[:, 0, i] = i
        data[:, 1, i] = i + 1

    vtk_image = mskt.image.create_vtk_image(origin=[0, 0, 0], spacing=[1, 1, 1], data=data)

    probe = mskt.mesh.meshTools.ProbeVtkImageDataAlongLine(
        line_resolution=line_resolution, vtk_image=vtk_image, data_categorical=categorical
    )

    scalars = probe.get_data_along_line(start_pt=(0.5, 0.6, 0), end_pt=(0.5, 0.6, line_resolution))

    assert_allclose(scalars, [i + 1 for i in range(line_resolution + 1)])
