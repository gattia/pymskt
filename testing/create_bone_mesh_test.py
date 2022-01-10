from operator import sub
import pytest
import pymskt as mskt
from numpy.testing import assert_allclose

try:
    orig_femur_mesh = mskt.mesh.io.read_vtk('data/femur_mesh_orig.vtk')
    downsampled_femur_mesh = mskt.mesh.io.read_vtk('data/femur_thickness_mm_10k_pts.vtk')
except OSError:
    orig_femur_mesh = mskt.mesh.io.read_vtk('../data/femur_mesh_orig.vtk')
    downsampled_femur_mesh = mskt.mesh.io.read_vtk('../data/femur_thickness_mm_10k_pts.vtk')

def test_orig_femur_bone(timing=False,
                         verbose=False,
                         rtol=1e-3):
    try:
        femur = mskt.mesh.BoneMesh(path_seg_image='data/right_knee_example.nrrd', label_idx=5)
    except OSError:
        femur = mskt.mesh.BoneMesh(path_seg_image='../data/right_knee_example.nrrd', label_idx=5)
    
    femur.create_mesh()
    femur_pts = femur.point_coords

    orig_femur_pts = mskt.mesh.get_mesh_physical_point_coords(orig_femur_mesh)

    assert_allclose(orig_femur_pts, femur_pts, rtol=rtol)

def test_downsampled_femur_bone(timing=False,
                         verbose=False,
                         rtol=1e-3):
    try:
        femur = mskt.mesh.BoneMesh(path_seg_image='data/right_knee_example.nrrd', label_idx=5)
    except OSError:
        femur = mskt.mesh.BoneMesh(path_seg_image='../data/right_knee_example.nrrd', label_idx=5)
    
    femur.create_mesh()
    femur.resample_surface(subdivisions=2, clusters=10000)
    femur_pts = femur.point_coords

    downsampled_femur_pts = mskt.mesh.get_mesh_physical_point_coords(downsampled_femur_mesh)

    assert_allclose(downsampled_femur_pts, femur_pts, rtol=rtol)



if __name__ == "__main__":
    import time
    test_orig_femur_bone(timing=True, verbose=True)
