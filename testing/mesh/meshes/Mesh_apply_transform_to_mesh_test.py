import pytest
import pymskt as mskt
import vtk
import numpy as np
from vtk.util.numpy_support import vtk_to_numpy
from numpy.testing import assert_allclose


MESH = mskt.mesh.io.read_vtk('data/femur_mesh_10k_pts.vtk')

def test_translation_with_transform(mesh_=MESH):
    mesh = mskt.mesh.Mesh(mesh=mesh_)
    orig_points = vtk_to_numpy(mesh.mesh.GetPoints().GetData())

    translation = [10, 20, 30]
    fourXfour = np.identity(4)
    fourXfour[:3, 3] = translation

    transform = vtk.vtkTransform()
    transform.SetMatrix(fourXfour.flatten())

    mesh.apply_transform_to_mesh(transform=transform)
    
    transformed_points = vtk_to_numpy(mesh.mesh.GetPoints().GetData())

    assert_allclose(transformed_points - translation, orig_points)

def test_translation_with_transform_dont_save_transform(mesh_=MESH, save_transform=False):
    mesh = mskt.mesh.Mesh(mesh=mesh_)

    assert len(mesh.list_applied_transforms) == 0

    translation = [10, 20, 30]
    fourXfour = np.identity(4)
    fourXfour[:3, 3] = translation

    transform = vtk.vtkTransform()
    transform.SetMatrix(fourXfour.flatten())

    mesh.apply_transform_to_mesh(transform=transform, save_transform=save_transform)

    assert len(mesh.list_applied_transforms) == 0

def test_translation_with_transform_save_mesh(mesh_=MESH, save_transform=True):
    mesh = mskt.mesh.Mesh(mesh=mesh_)

    assert len(mesh.list_applied_transforms) == 0
    
    translation = [10, 20, 30]
    fourXfour = np.identity(4)
    fourXfour[:3, 3] = translation

    transform = vtk.vtkTransform()
    transform.SetMatrix(fourXfour.flatten())

    mesh.apply_transform_to_mesh(transform=transform, save_transform=save_transform)

    assert len(mesh.list_applied_transforms) == 1
    

def test_translation_with_transformer(mesh_=MESH):
    mesh = mskt.mesh.Mesh(mesh=mesh_)
    orig_points = vtk_to_numpy(mesh.mesh.GetPoints().GetData())

    translation = [10, 20, 30]
    fourXfour = np.identity(4)
    fourXfour[:3, 3] = translation

    transform = vtk.vtkTransform()
    transform.SetMatrix(fourXfour.flatten())

    transformer = vtk.vtkTransformPolyDataFilter()
    transformer.SetTransform(transform)

    mesh.apply_transform_to_mesh(transformer=transformer)

    transformed_points = vtk_to_numpy(mesh.mesh.GetPoints().GetData())

    assert_allclose(transformed_points - translation, orig_points)

def test_dont_provide_transform_or_transformer_raise_exception(mesh_=MESH):
    mesh = mskt.mesh.Mesh(mesh=mesh_)
    with pytest.raises(Exception):
        mesh.apply_transform_to_mesh()

def test_rotation_with_transform():
    raise Exception('Test not implemented')

def test_rotation_with_transformer():
    raise Exception('Test not implemented')

def test_reverse_most_recent_transform_with_translation():
    raise Exception('Test not implemented')

def test_reverse_all_transforms_with_translation():
    raise Exception('Test not implemented')