from typing import Optional

import SimpleITK as sitk
import vtk
from numpy.testing import assert_allclose
from vtk.util.numpy_support import vtk_to_numpy

import pymskt as mskt


def assert_images_same(image1, image2, rtol=1e-4, atol=1e-5):
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

    assert_allclose(image1_array, image2_array, rtol=rtol, atol=atol)
    assert image1.GetOrigin() == image2.GetOrigin()
    assert image1.GetSpacing() == image2.GetSpacing()
    assert image1.GetDirection() == image2.GetDirection()


def assert_mesh_coordinates_same(mesh1, mesh2, rtol=1e-4, atol=1e-5):
    """
    Helper function to assert that 2 vtkPolyData meshes points are the same.

    Parameters
    ----------
    mesh1 : vtk.vtkPolyData
        Version 1 of the mesh
    mesh2 : vtk.vtkPolyData
        Version 2 of the mesh
    """

    assert_allclose(mesh1.points, mesh2.points, rtol=rtol, atol=atol)


def assert_mesh_scalars_same(mesh1, mesh2, scalarname, scalarname2=None, rtol=1e-4, atol=1e-5):
    """
    Helper function to assert that the scalars on 2 vtkPolyData meshes as the same.

    Parameters
    ----------
    mesh1 : vtk.vtkPolyData
        Version 1 of the mesh
    mesh2 : vtk.vtkPolyData
        Version 2 of the mesh
    scalarname : str
        String of the name associated with the scalars we are comparing
    scalarname2 : str, optional
        String of the name associated with the scalars on the second mesh.
        This is only needed if the second mesh scarals have a different name than the first.
    """
    if scalarname2 is None:
        scalarname2 = scalarname
    scalars1 = mesh1.point_data[scalarname]
    scalars2 = mesh2.point_data[scalarname2]

    assert_allclose(scalars1, scalars2, rtol=rtol, atol=atol)
