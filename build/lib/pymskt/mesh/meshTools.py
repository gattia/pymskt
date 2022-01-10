import os

import vtk
from vtk.util.numpy_support import vtk_to_numpy

import SimpleITK as sitk

import numpy as np

from pymskt.utils import n2l, l2n, safely_delete_tmp_file
from pymskt.mesh.utils import is_hit, get_intersect, get_surface_normals, get_obb_surface
import pymskt.image as pybtimage
import pymskt.mesh.createMesh as createMesh 
import pymskt.mesh.meshTransform as meshTransform



class ProbeVtkImageDataAlongLine:
    def __init__(self,
                 line_resolution,
                 vtk_image,
                 save_data_in_class=True,
                 save_mean=False,
                 save_std=False,
                 save_most_common=False,
                 filler=0,
                 non_zero_only=True
                 ):
        self.save_mean = save_mean
        self.save_std = save_std
        self.save_most_common = save_most_common
        self.filler = filler
        self.non_zero_only = non_zero_only

        self.line = vtk.vtkLineSource()
        self.line.SetResolution(line_resolution)

        self.probe_filter = vtk.vtkProbeFilter()
        self.probe_filter.SetSourceData(vtk_image)

        if save_data_in_class is True:
            if self.save_mean is True:
                self._mean_data = []
            if self.save_std is True:
                self._std_data = []
            if self.save_most_common is True:
                self._most_common_data = []

    def get_data_along_line(self,
                            start_pt,
                            end_pt):
        self.line.SetPoint1(start_pt)
        self.line.SetPoint2(end_pt)

        self.probe_filter.SetInputConnection(self.line.GetOutputPort())
        self.probe_filter.Update()
        scalars = vtk_to_numpy(self.probe_filter.GetOutput().GetPointData().GetScalars())

        if self.non_zero_only is True:
            scalars = scalars[scalars != 0]

        return scalars

    def save_data_along_line(self,
                             start_pt,
                             end_pt):
        scalars = self.get_data_along_line(start_pt, end_pt)
        if len(scalars) > 0:
            if self.save_mean is True:
                self._mean_data.append(np.mean(scalars))
            if self.save_std is True:
                self._std_data.append(np.std(scalars, ddof=1))
            if self.save_most_common is True:
                # most_common is for getting segmentations and trying to assign a bone region
                # to be a cartilage ROI. This is becuase there might be a normal vector that
                # cross > 1 cartilage region (e.g., weight-bearing vs anterior fem cartilage)
                self._most_common_data.append(np.bincount(scalars).argmax())
        else:
            self.append_filler()

    def append_filler(self):
        if self.save_mean is True:
            self._mean_data.append(self.filler)
        if self.save_std is True:
            self._std_data.append(self.filler)
        if self.save_most_common is True:
            self._most_common_data.append(self.filler)

    @property
    def mean_data(self):
        if self.save_mean is True:
            return self._mean_data
        else:
            return None

    @property
    def std_data(self):
        if self.save_std is True:
            return self._std_data
        else:
            return None

    @property
    def most_common_data(self):
        if self.save_most_common is True:
            return self._most_common_data
        else:
            return None


def get_cartilage_properties_at_points(surface_bone,
                                       surface_cartilage,
                                       t2_vtk_image=None,
                                       seg_vtk_image=None,
                                       ray_cast_length=20.,
                                       percent_ray_length_opposite_direction=0.25,
                                       no_thickness_filler=0.,
                                       no_t2_filler=0.,
                                       no_seg_filler=0,
                                       line_resolution=100):  # Could be nan??
    """
    :param points:
    :param point_normals:
    :param obb:
    :param t2_vtk_image:
    :param seg_vtk_image:
    :param ray_length:
    :param percent_ray_length_opposite_direction:
    :param no_thickness_filler:
    :param no_t2_filler:
    :param no_seg_filler:
    :param line_resolution:
    :return:


    Will return list of data for:
        Cartilage thickness
        Mean T2 at each point on bone
        Most common cartilage label at each point on bone (normal to surface).
    """

    normals = get_surface_normals(surface_bone)
    points = surface_bone.GetPoints()
    obb_cartilage = get_obb_surface(surface_cartilage)
    point_normals = normals.GetOutput().GetPointData().GetNormals()

    thickness_data = []
    if (t2_vtk_image is not None) or (seg_vtk_image is not None):
        # if T2 data, or a segmentation image is provided, then setup Probe tool to
        # get T2 values and/or cartilage ROI for each bone vertex. 
        line = vtk.vtkLineSource()
        line.SetResolution(line_resolution)

        if t2_vtk_image is not None:
            t2_data_probe = ProbeVtkImageDataAlongLine(line_resolution,
                                                  t2_vtk_image,
                                                  save_mean=True,
                                                  filler=no_t2_filler)
        if seg_vtk_image is not None:
            seg_data_probe = ProbeVtkImageDataAlongLine(line_resolution,
                                                   seg_vtk_image,
                                                   save_most_common=True,
                                                   filler=no_seg_filler)
    # Loop through all points
    for idx in range(points.GetNumberOfPoints()):
        point = points.GetPoint(idx)
        normal = point_normals.GetTuple(idx)

        end_point_ray = n2l(l2n(point) + ray_cast_length*l2n(normal))
        start_point_ray = n2l(l2n(point) + ray_cast_length*percent_ray_length_opposite_direction*(-l2n(normal)))

        # Check if there are any intersections for the given ray
        if is_hit(obb_cartilage, start_point_ray, end_point_ray):  # intersections were found
            # Retrieve coordinates of intersection points and intersected cell ids
            points_intersect, cell_ids_intersect = get_intersect(obb_cartilage, start_point_ray, end_point_ray)
    #         points
            if len(points_intersect) == 2:
                thickness_data.append(np.sqrt(np.sum(np.square(l2n(points_intersect[0]) - l2n(points_intersect[1])))))
                if t2_vtk_image is not None:
                    t2_data_probe.save_data_along_line(start_pt=points_intersect[0],
                                                       end_pt=points_intersect[1])
                if seg_vtk_image is not None:
                    seg_data_probe.save_data_along_line(start_pt=points_intersect[0],
                                                        end_pt=points_intersect[1])

            else:
                thickness_data.append(no_thickness_filler)
                if t2_vtk_image is not None:
                    t2_data_probe.append_filler()
                if seg_vtk_image is not None:
                    seg_data_probe.append_filler()
        else:
            thickness_data.append(no_thickness_filler)
            if t2_vtk_image is not None:
                t2_data_probe.append_filler()
            if seg_vtk_image is not None:
                seg_data_probe.append_filler()

    if (t2_vtk_image is None) & (seg_vtk_image is None):
        return np.asarray(thickness_data, dtype=np.float)
    elif (t2_vtk_image is not None) & (seg_vtk_image is not None):
        return (np.asarray(thickness_data, dtype=np.float),
                np.asarray(t2_data_probe.mean_data, dtype=np.float),
                np.asarray(seg_data_probe.most_common_data, dtype=np.int)
                )
    elif t2_vtk_image is not None:
        return (np.asarray(thickness_data, dtype=np.float),
                np.asarray(t2_data_probe.mean_data, dtype=np.float)
                )
    elif seg_vtk_image is not None:
        return (np.asarray(thickness_data, dtype=np.float),
                np.asarray(seg_data_probe.most_common_data, dtype=np.int)
                )




def get_mesh_physical_point_coords(mesh):
    point_coordinates = np.zeros((mesh.GetNumberOfPoints(), 3))
    for pt_idx in range(mesh.GetNumberOfPoints()):
        point_coordinates[pt_idx, :] = mesh.GetPoint(pt_idx)
    return point_coordinates




