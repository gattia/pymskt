import numpy as np
import pymskt as mskt
from numpy.testing import assert_allclose

def test_save_mean_data_along_line_that_is_outside_volume_assert_filler_is_used_properly(
    line_resolution = 10,
    rand_int = np.random.randint(0, 10000),
    categorical=True,
    filler=np.random.randint(-10000, 10000),
    save_mean=True
):  
    uniform_image = mskt.image.create_vtk_image(
        origin=[0,0,0],
        dimensions=[20,20,20],
        spacing=[1,1,1],
        scalar=rand_int
    )

    probe = mskt.mesh.meshTools.ProbeVtkImageDataAlongLine(
        line_resolution=line_resolution,
        vtk_image=uniform_image,
        data_categorical=categorical,
        filler=filler,
        save_mean=save_mean
    )

    # Create a line that is outside of the volume - should still be nearest neighbours
    # so end up with the same data/result. 
    probe.save_data_along_line(
        start_pt=(100,100,100),
        end_pt=(200,200,200)
    )

    assert_allclose(probe.mean_data, [filler])

def test_save_std_data_along_line_that_is_outside_volume_assert_filler_is_used_properly(
    line_resolution = 10,
    rand_int = np.random.randint(0, 10000),
    categorical=True,
    filler=np.random.randint(-10000, 10000),
    save_std=True
):  
    uniform_image = mskt.image.create_vtk_image(
        origin=[0,0,0],
        dimensions=[20,20,20],
        spacing=[1,1,1],
        scalar=rand_int
    )

    probe = mskt.mesh.meshTools.ProbeVtkImageDataAlongLine(
        line_resolution=line_resolution,
        vtk_image=uniform_image,
        data_categorical=categorical,
        filler=filler,
        save_std=save_std
    )

    # Create a line that is outside of the volume - should still be nearest neighbours
    # so end up with the same data/result. 
    probe.save_data_along_line(
        start_pt=(100,100,100),
        end_pt=(200,200,200)
    )

    assert_allclose(probe.std_data, [filler])

def test_mean_is_correct_when_uniform(
    line_resolution = 10,
    rand_int = np.random.randint(0, 10000),
    categorical=False,
    save_mean=True
):  
    uniform_image = mskt.image.create_vtk_image(
        origin=[0,0,0],
        dimensions=[20,20,20],
        spacing=[1,1,1],
        scalar=rand_int
    )

    probe = mskt.mesh.meshTools.ProbeVtkImageDataAlongLine(
        line_resolution=line_resolution,
        vtk_image=uniform_image,
        data_categorical=categorical,
        save_mean=save_mean
    )

    probe.save_data_along_line(
        start_pt=(0,0,0),
        end_pt=(19,19,19)
    )

    assert len(probe.mean_data) == 1
    assert probe.mean_data[0] == rand_int

def test_std_is_correct_when_uniform(
    line_resolution = 10,
    rand_int = np.random.randint(0, 10000),
    categorical=False,
    save_std=True
):  
    uniform_image = mskt.image.create_vtk_image(
        origin=[0,0,0],
        dimensions=[20,20,20],
        spacing=[1,1,1],
        scalar=rand_int
    )

    probe = mskt.mesh.meshTools.ProbeVtkImageDataAlongLine(
        line_resolution=line_resolution,
        vtk_image=uniform_image,
        data_categorical=categorical,
        save_std=save_std
    )

    probe.save_data_along_line(
        start_pt=(0,0,0),
        end_pt=(19,19,19)
    )

    assert np.abs(probe.std_data[0] - 0) < 1e-7
