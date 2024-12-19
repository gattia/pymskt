import numpy as np

import pymskt as mskt
from pymskt.utils import testing

MESH = mskt.mesh.io.read_vtk("data/femur_mesh_orig.vtk")
MESH_RESAMPLED = mskt.mesh.Mesh(mskt.mesh.io.read_vtk("data/femur_mesh_10k_pts.vtk"))

from pymskt import ATOL, RTOL


def test_resample_surface_assert_same_as_saved_mesh(mesh_=MESH, mesh_resampled=MESH_RESAMPLED):
    mesh = mskt.mesh.Mesh(mesh=mesh_)
    mesh.resample_surface(subdivisions=2, clusters=10000)
    # compute ASSD as a percentage of the mean edge length
    # assert it is < 3% of average edge length
    mean_edge_1 = np.mean(mesh.edge_lengths)
    mean_edge_2 = np.mean(mesh_resampled.edge_lengths)
    mean_edges = np.mean([mean_edge_1, mean_edge_2])
    assd = mesh.get_assd(mesh_resampled)
    assd_percentage = assd / mean_edges * 100
    assert assd_percentage < 3, f"ASSD is {assd_percentage:.2f}% of the mean edge length"
