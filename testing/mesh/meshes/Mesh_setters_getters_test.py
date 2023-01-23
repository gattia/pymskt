from importlib.resources import path
import pymskt as mskt
import SimpleITK as sitk
import numpy as np
from numpy.testing import assert_allclose
import vtk

from pymskt import RTOL, ATOL

SEG_IMAGE = sitk.ReadImage('data/right_knee_example.nrrd')
MESH = mskt.mesh.io.read_vtk('data/femur_mesh_10k_pts.vtk')

# SEG IMAGE
def test_set_seg_image(seg_image=SEG_IMAGE):
    mesh = mskt.mesh.Mesh()
    mesh.seg_image = seg_image

    assert mesh._seg_image == seg_image

def test_get_seg_image(seg_image=SEG_IMAGE):
    mesh = mskt.mesh.Mesh()
    mesh._seg_image = seg_image

    assert mesh.seg_image == seg_image

# MESH
def test_set_mesh(vtk_mesh=MESH):
    mesh = mskt.mesh.Mesh()
    mesh.mesh = vtk_mesh

    assert mesh.mesh == vtk_mesh

def test_get_mesh(vtk_mesh=MESH):
    mesh = mskt.mesh.Mesh()
    mesh._mesh = vtk_mesh

    assert mesh.mesh == vtk_mesh

# POINT COORDINATES
def test_get_point_coords(vtk_mesh=MESH):
    mesh = mskt.mesh.Mesh(mesh=vtk_mesh)
    pt_coords = mesh.point_coords

    assert type(pt_coords) == np.ndarray
    assert pt_coords.shape[0] == mesh.mesh.GetNumberOfPoints()
    assert pt_coords.shape[1] == 3

def test_set_point_coords(vtk_mesh=MESH):
    mesh = mskt.mesh.Mesh(mesh=vtk_mesh)
    pt_coords = mesh.point_coords

    translation = [10, 20, 30]
    pt_coords = pt_coords + translation

    mesh.point_coords = pt_coords

    rand_sample = np.random.randint(0, mesh.mesh.GetNumberOfPoints(), 256)
    for rand_idx in rand_sample:
        assert_allclose(pt_coords[rand_idx,:], mesh.mesh.GetPoint(rand_idx), rtol=RTOL, atol=ATOL)

# PATH TO SEG IMAGE
def test_get_path_to_seg_image(path_seg_image='test/location/fake/data.csv'):
    mesh = mskt.mesh.Mesh(path_seg_image=path_seg_image)
    assert mesh.path_seg_image == path_seg_image

def test_set_path_to_seg_image(path_seg_image='test/location/fake/data.csv'):
    mesh = mskt.mesh.Mesh()
    mesh.path_seg_image=path_seg_image
    assert mesh._path_seg_image == path_seg_image

# LABEL INDEX
def test_get_label_idx(label_idx=3):
    mesh = mskt.mesh.Mesh(label_idx=label_idx)
    assert mesh.label_idx == label_idx

def test_set_label_idx(label_idx=3):
    mesh = mskt.mesh.Mesh()
    mesh.label_idx = label_idx
    assert mesh._label_idx == label_idx

# MINIMUM NUMBER OF PIXELS
def test_get_min_n_pixels(min_n_pixels=100):
    mesh = mskt.mesh.Mesh(min_n_pixels=min_n_pixels)
    assert mesh.min_n_pixels == min_n_pixels

def test_set_min_n_pixels(min_n_pixels=100):
    mesh = mskt.mesh.Mesh()
    mesh.min_n_pixels = min_n_pixels
    assert mesh._min_n_pixels == min_n_pixels

# LIST OF APPLIED TRANSFORMS
def test_get_list_applied_transforms(n_rotations=10, vtk_mesh=MESH):
    mesh = mskt.mesh.Mesh(mesh=vtk_mesh)

    rand_rotations = np.random.random((4, 4, n_rotations))
    for rot_idx in range(n_rotations):
        transform = vtk.vtkTransform()
        transform.SetMatrix(rand_rotations[:, :, rot_idx].flatten())
        mesh.apply_transform_to_mesh(transform=transform)
    
    list_transforms = mesh.list_applied_transforms
    
    for transform_idx, transform in enumerate(list_transforms):
        translation = transform.GetPosition()
        assert_allclose(translation, rand_rotations[:3, 3, transform_idx], rtol=RTOL, atol=ATOL)
