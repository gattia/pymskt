import numpy as np
import vtk
from numpy.testing import assert_allclose
from vtk.util.numpy_support import vtk_to_numpy

import pymskt as mskt
from pymskt import ATOL, RTOL

MESH = mskt.mesh.io.read_vtk("data/femur_mesh_10k_pts.vtk")


def test_translation_with_transform(mesh=MESH):
    orig_points = vtk_to_numpy(mesh.GetPoints().GetData())

    translation = [10, 20, 30]
    fourXfour = np.identity(4)
    fourXfour[:3, 3] = translation

    transform = vtk.vtkTransform()
    transform.SetMatrix(fourXfour.flatten())

    transformed_mesh = mskt.mesh.meshTransform.apply_transform(mesh, transform)
    transformed_points = vtk_to_numpy(transformed_mesh.GetPoints().GetData())

    assert_allclose(transformed_points - translation, orig_points, rtol=RTOL, atol=ATOL)
