import pymskt as mskt
import numpy as np
from numpy.testing import assert_allclose

# TEST GET EACH VALUE
def test_get_mean(
    mean_list=list(np.random.random(1000))
):  
    vtk_image = mskt.image.create_vtk_image()
    probe = mskt.mesh.meshTools.ProbeVtkImageDataAlongLine(
        save_mean=True,
        vtk_image=vtk_image,
        line_resolution=10
    )
    probe._mean_data.append(mean_list)

    assert_allclose(mean_list, np.squeeze(probe.mean_data))

def test_get_std(
    std_list=list(np.random.random(1000))
):
    vtk_image = mskt.image.create_vtk_image()
    probe = mskt.mesh.meshTools.ProbeVtkImageDataAlongLine(
        save_std=True,
        vtk_image=vtk_image,
        line_resolution=10
    )
    probe._std_data.append(std_list)

    assert_allclose(std_list, np.squeeze(probe.std_data))

def test_get_mode(
    mode_list=list(np.random.randint(0, 1000, 1000))
):
    vtk_image = mskt.image.create_vtk_image()
    probe = mskt.mesh.meshTools.ProbeVtkImageDataAlongLine(
        save_most_common=True,
        vtk_image=vtk_image,
        line_resolution=10
    )
    probe._most_common_data.append(mode_list)

    assert_allclose(mode_list, np.squeeze(probe.most_common_data))

# TEST GET EACH VALUE WHEN ITS NOT CALCULATED
def test_get_mean_not_specified():  
    vtk_image = mskt.image.create_vtk_image()
    probe = mskt.mesh.meshTools.ProbeVtkImageDataAlongLine(
        save_mean=False,
        vtk_image=vtk_image,
        line_resolution=10
    )
    assert probe.mean_data is None

def test_get_std_not_specified():
    vtk_image = mskt.image.create_vtk_image()
    probe = mskt.mesh.meshTools.ProbeVtkImageDataAlongLine(
        save_std=False,
        vtk_image=vtk_image,
        line_resolution=10
    )
    
    assert probe.std_data is None

def test_get_mode_not_specified():
    vtk_image = mskt.image.create_vtk_image()
    probe = mskt.mesh.meshTools.ProbeVtkImageDataAlongLine(
        save_most_common=False,
        vtk_image=vtk_image,
        line_resolution=10
    )

    assert probe.most_common_data is None