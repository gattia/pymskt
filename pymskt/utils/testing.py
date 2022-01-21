from numpy.testing import assert_allclose
import SimpleITK as sitk
import pymskt as mskt


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

def assert_mesh_coordinates_same(mesh1, mesh2, rtol=1e-3):
    """
    Helper function to assert that 2 vtkPolyData meshes points are the same. 

    Parameters
    ----------
    mesh1 : vtk.vtkPolyData
        Version 1 of the mesh
    mesh2 : vtk.vtkPolyData
        Version 2 of the mesh
    """

    mesh1_pts = mskt.mesh.get_mesh_physical_point_coords(mesh1)
    mesh2_pts = mskt.mesh.get_mesh_physical_point_coords(mesh2)

    assert_allclose(mesh1_pts, mesh2_pts, rtol=rtol)

