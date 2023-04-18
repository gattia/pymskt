import os
import time
from turtle import distance

import vtk
from vtk.util.numpy_support import vtk_to_numpy, numpy_to_vtk

import SimpleITK as sitk
import pyacvd
import pyvista as pv

import numpy as np

from pymskt.utils import n2l, l2n, safely_delete_tmp_file
from pymskt.mesh.utils import is_hit, get_intersect, get_surface_normals, get_obb_surface, vtk_deep_copy
import pymskt.image as pybtimage
import pymskt.mesh.createMesh as createMesh
import pymskt.mesh.meshTransform as meshTransform
from pymskt.cython_functions import gaussian_kernel
import pymeshfix as mf

epsilon = 1e-7

class ProbeVtkImageDataAlongLine:
    """
    Class to find values along a line. This is used to get things like the mean T2 value normal
    to a bones surface & within the cartialge region. This is done by defining a line in a
    particualar location. 

    Parameters
    ----------
    line_resolution : float
        How many points to create along the line. 
    vtk_image : vtk.vtkImageData
        Image read into vtk so that we can apply the probe to it. 
    save_data_in_class : bool, optional
        Whether or not to save data along the line(s) to the class, by default True
    save_mean : bool, optional
        Whether the mean value should be saved along the line, by default False
    save_std : bool, optional
        Whether the standard deviation of the data along the line should be
        saved, by default False
    save_most_common : bool, optional
        Whether the mode (most common) value should be saved used for identifying cartilage
        regions on the bone surface, by default False
    filler : int, optional
        What value should be placed at locations where we don't have a value
        (e.g., where we don't have T2 values), by default 0
    non_zero_only : bool, optional
        Only save non-zero values along the line, by default True
        This is done becuase zeros are normally regions of error (e.g.
        poor T2 relaxation fit) and thus would artifically reduce the outcome
        along the line. 
    
    
    Attributes
    ----------
    save_mean : bool
        Whether the mean value should be saved along the line, by default False
    save_std : bool
        Whether the standard deviation of the data along the line should be
        saved, by default False
    save_most_common : bool 
        Whether the mode (most common) value should be saved used for identifying cartilage
        regions on the bone surface, by default False
    filler : float
        What value should be placed at locations where we don't have a value
        (e.g., where we don't have T2 values), by default 0
    non_zero_only : bool 
        Only save non-zero values along the line, by default True
        This is done becuase zeros are normally regions of error (e.g.
        poor T2 relaxation fit) and thus would artifically reduce the outcome
        along the line. 
    line : vtk.vtkLineSource
        Line to put into `probe_filter` and to determine mean/std/common values for. 
    probe_filter : vtk.vtkProbeFilter
        Filter to use to get the image data along the line. 
    _mean_data : list
        List of the mean values for each vertex / line projected
    _std_data : list
        List of standard deviation of each vertex / line projected
    _most_common_data : list
        List of most common data of each vertex / line projected
    
    Methods
    -------


    """    
    def __init__(self,
                 line_resolution,
                 vtk_image,
                 save_data_in_class=True,
                 save_mean=False,
                 save_std=False,
                 save_most_common=False,
                 save_max=False,
                 filler=0,
                 non_zero_only=True,
                 data_categorical=False
                 ):
        """[summary]

        Parameters
        ----------
        line_resolution : float
            How many points to create along the line. 
        vtk_image : vtk.vtkImageData
            Image read into vtk so that we can apply the probe to it. 
        save_data_in_class : bool, optional
            Whether or not to save data along the line(s) to the class, by default True
        save_mean : bool, optional
            Whether the mean value should be saved along the line, by default False
        save_std : bool, optional
            Whether the standard deviation of the data along the line should be
            saved, by default False
        save_most_common : bool, optional
            Whether the mode (most common) value should be saved used for identifying cartilage
            regions on the bone surface, by default False
        save_max : bool, optional
            Whether the max value should be saved along the line, be default False
        filler : int, optional
            What value should be placed at locations where we don't have a value
            (e.g., where we don't have T2 values), by default 0
        non_zero_only : bool, optional
            Only save non-zero values along the line, by default True
            This is done becuase zeros are normally regions of error (e.g.
            poor T2 relaxation fit) and thus would artifically reduce the outcome
            along the line.
        data_categorical : bool, optional
            Specify whether or not the data is categorical to determine the interpolation
            method that should be used. 
        """        
        self.save_mean = save_mean
        self.save_std = save_std
        self.save_most_common = save_most_common
        self.save_max = save_max
        self.filler = filler
        self.non_zero_only = non_zero_only

        self.line = vtk.vtkLineSource()
        self.line.SetResolution(line_resolution)

        self.probe_filter = vtk.vtkProbeFilter()
        self.probe_filter.SetSourceData(vtk_image)
        if data_categorical is True:
            self.probe_filter.CategoricalDataOn()

        if save_data_in_class is True:
            if self.save_mean is True:
                self._mean_data = []
            if self.save_std is True:
                self._std_data = []
            if self.save_most_common is True:
                self._most_common_data = []
            if self.save_max is True:
                self._max_data = []

    def get_data_along_line(self,
                            start_pt,
                            end_pt):
        """
        Function to get scalar values along a line between `start_pt` and `end_pt`. 

        Parameters
        ----------
        start_pt : list
            List of the x,y,z position of the starting point in the line. 
        end_pt : list
            List of the x,y,z position of the ending point in the line. 

        Returns
        -------
        numpy.ndarray
            numpy array of scalar values obtained along the line.
        """        
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
        """
        Save the appropriate outcomes to a growing list. 

        Parameters
        ----------
        start_pt : list
            List of the x,y,z position of the starting point in the line. 
        end_pt : list
            List of the x,y,z position of the ending point in the line. 
        """        
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
            if self.save_max is True:
                self._max_data.append(np.max(scalars))
        else:
            self.append_filler()

    def append_filler(self):
        """
        Add filler value to the requisite lists (_mean_data, _std_data, etc.) as 
        appropriate. 
        """        
        if self.save_mean is True:
            self._mean_data.append(self.filler)
        if self.save_std is True:
            self._std_data.append(self.filler)
        if self.save_most_common is True:
            self._most_common_data.append(self.filler)
        if self.save_max is True:
            self._max_data.append(self.filler)

    @property
    def mean_data(self):
        """
        Return the `_mean_data`

        Returns
        -------
        list
            List of mean values along each line tested. 
        """        
        if self.save_mean is True:
            return self._mean_data
        else:
            return None

    @property
    def std_data(self):
        """
        Return the `_std_data`

        Returns
        -------
        list
            List of the std values along each line tested. 
        """        
        if self.save_std is True:
            return self._std_data
        else:
            return None

    @property
    def most_common_data(self):
        """
        Return the `_most_common_data`

        Returns
        -------
        list
            List of the most common value for each line tested. 
        """        
        if self.save_most_common is True:
            return self._most_common_data
        else:
            return None
    
    @property
    def max_data(self):
        """
        Return the `_max_data`

        Returns
        -------
        list
            List of the most common value for each line tested. 
        """        
        if self.save_max is True:
            return self._max_data
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
    Extract cartilage outcomes (T2 & thickness) at all points on bone surface. 

    Parameters
    ----------
    surface_bone : BoneMesh
        Bone mesh containing vtk.vtkPolyData - get outcomes for nodes (vertices) on
        this mesh
    surface_cartilage : CartilageMesh
        Cartilage mesh containing vtk.vtkPolyData - for obtaining cartilage outcomes.
    t2_vtk_image : vtk.vtkImageData, optional
        vtk object that contains our Cartilage T2 data, by default None
    seg_vtk_image : vtk.vtkImageData, optional
        vtk object that contains the segmentation mask(s) to help assign
        labels to bone surface (e.g., most common), by default None
    ray_cast_length : float, optional
        Length (mm) of ray to cast from bone surface when trying to find cartilage (inner &
        outter shell), by default 20.0
    percent_ray_length_opposite_direction : float, optional
        How far to project ray inside of the bone. This is done just in case the cartilage
        surface ends up slightly inside of (or coincident with) the bone surface, by default 0.25
    no_thickness_filler : float, optional
        Value to use instead of thickness (if no cartilage), by default 0.
    no_t2_filler : float, optional
        Value to use instead of T2 (if no cartilage), by default 0.
    no_seg_filler : int, optional
        Value to use if no segmentation label available (because no cartilage?), by default 0
    line_resolution : int, optional
        Number of points to have along line, by default 100

    Returns
    -------
    list
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
                                                   filler=no_seg_filler,
                                                   data_categorical=True)
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
        return np.asarray(thickness_data, dtype=float)
    elif (t2_vtk_image is not None) & (seg_vtk_image is not None):
        return (np.asarray(thickness_data, dtype=float),
                np.asarray(t2_data_probe.mean_data, dtype=float),
                np.asarray(seg_data_probe.most_common_data, dtype=int)
                )
    elif t2_vtk_image is not None:
        return (np.asarray(thickness_data, dtype=float),
                np.asarray(t2_data_probe.mean_data, dtype=float)
                )
    elif seg_vtk_image is not None:
        return (np.asarray(thickness_data, dtype=float),
                np.asarray(seg_data_probe.most_common_data, dtype=int)
                )

def get_distance_other_surface_at_points(surface,
                                         other_surface,
                                         ray_cast_length=20.,
                                         percent_ray_length_opposite_direction=0.25,
                                         no_distance_filler=0.
                                         ):  # Could be nan??
    """
    Extract cartilage outcomes (T2 & thickness) at all points on bone surface. 

    Parameters
    ----------
    surface_bone : BoneMesh
        Bone mesh containing vtk.vtkPolyData - get outcomes for nodes (vertices) on
        this mesh
    surface_cartilage : CartilageMesh
        Cartilage mesh containing vtk.vtkPolyData - for obtaining cartilage outcomes.
    t2_vtk_image : vtk.vtkImageData, optional
        vtk object that contains our Cartilage T2 data, by default None
    seg_vtk_image : vtk.vtkImageData, optional
        vtk object that contains the segmentation mask(s) to help assign
        labels to bone surface (e.g., most common), by default None
    ray_cast_length : float, optional
        Length (mm) of ray to cast from bone surface when trying to find cartilage (inner &
        outter shell), by default 20.0
    percent_ray_length_opposite_direction : float, optional
        How far to project ray inside of the bone. This is done just in case the cartilage
        surface ends up slightly inside of (or coincident with) the bone surface, by default 0.25
    no_thickness_filler : float, optional
        Value to use instead of thickness (if no cartilage), by default 0.
    no_t2_filler : float, optional
        Value to use instead of T2 (if no cartilage), by default 0.
    no_seg_filler : int, optional
        Value to use if no segmentation label available (because no cartilage?), by default 0
    line_resolution : int, optional
        Number of points to have along line, by default 100

    Returns
    -------
    list
        Will return list of data for:
            Cartilage thickness
            Mean T2 at each point on bone
            Most common cartilage label at each point on bone (normal to surface).
    """    

    normals = get_surface_normals(surface)
    points = surface.GetPoints()
    obb_other_surface = get_obb_surface(other_surface)
    point_normals = normals.GetOutput().GetPointData().GetNormals()

    distance_data = []
    # Loop through all points
    for idx in range(points.GetNumberOfPoints()):
        point = points.GetPoint(idx)
        normal = point_normals.GetTuple(idx)

        end_point_ray = n2l(l2n(point) + ray_cast_length*l2n(normal))
        start_point_ray = n2l(l2n(point) + ray_cast_length*percent_ray_length_opposite_direction*(-l2n(normal)))

        # Check if there are any intersections for the given ray
        if is_hit(obb_other_surface, start_point_ray, end_point_ray):  # intersections were found
            # Retrieve coordinates of intersection points and intersected cell ids
            points_intersect, cell_ids_intersect = get_intersect(obb_other_surface, start_point_ray, end_point_ray)
    #         points
            # if len(points_intersect) == 1:
            distance_data.append(np.sqrt(np.sum(np.square(l2n(point) - l2n(points_intersect[0])))))
            # else:
                # distance_data.append(no_distance_filler)

        else:
            distance_data.append(no_distance_filler)

    return np.asarray(distance_data, dtype=float)

def set_mesh_physical_point_coords(mesh, new_points):
    """
    Convenience function to update the x/y/z point coords of a mesh

    Nothing is returned becuase the mesh object is updated in-place. 

    Parameters
    ----------
    mesh : vtk.vtkPolyData
        Mesh object we want to update the point coordinates for
    new_points : np.ndarray
        Numpy array shaped n_points x 3. These are the new point coordinates that
        we want to update the mesh to have. 

    """
    orig_point_coords = get_mesh_physical_point_coords(mesh)
    if new_points.shape == orig_point_coords.shape:
        mesh.GetPoints().SetData(numpy_to_vtk(new_points))


def get_mesh_physical_point_coords(mesh):
    """
    Get a numpy array of the x/y/z location of each point (vertex) on the `mesh`.

    Parameters
    ----------
    mesh : 
        [description]

    Returns
    -------
    numpy.ndarray
        n_points x 3 array describing the x/y/z position of each point. 
    
    Notes
    -----
    Below is the original method used to retrieve the point coordinates. 
    
    point_coordinates = np.zeros((mesh.GetNumberOfPoints(), 3))
    for pt_idx in range(mesh.GetNumberOfPoints()):
        point_coordinates[pt_idx, :] = mesh.GetPoint(pt_idx)
    """    
    
    point_coordinates = vtk_to_numpy(mesh.GetPoints().GetData())
    return point_coordinates

def get_mesh_point_features(mesh, features):
    """
    Get a numpy array of the x/y/z location of each point (vertex) on the `mesh`.

    Parameters
    ----------
    mesh : 
        vtkPolyData object
    
    features : list
        List of strings associated with features to retrieve.

    Returns
    -------
    numpy.ndarray
        n_points x len(features) array of the featurs at each point/vertex. 
    
    Notes
    -----

    """
    # ensure this is a list
    if isinstance(features, str):
        features = [features]

    vertex_features = np.zeros((mesh.GetNumberOfPoints(), len(features)))
    for i, feature in enumerate(features):
        feature_vec = vtk_to_numpy(mesh.GetPointData().GetArray(feature))
        vertex_features[:, i] = feature_vec.copy()
    return vertex_features

def set_mesh_point_features(mesh, features, feature_names=None):
    if len(features.shape) == 1:
        if feature_names is None:
            feature_name = 'feature_1'
        elif type(feature_name) in (list, tuple):
            feature_name = feature_names[0]
        else:
            feature_name = feature_names
        scalars = numpy_to_vtk(features)
        scalars.SetName(feature_name)
        mesh.GetPointData().AddArray(scalars)
    elif len(features.shape) == 2:
        n_pts = mesh.GetNumberOfPoints()
        if features.shape[1] == n_pts:
            features = features.T
        assert features.shape[0] == n_pts, "Features must be n_points x n_features"
        if feature_names is None:
            feature_names = [f'feature_{i}' for i in range(features.shape[1])]
        elif type(feature_names) in (list, tuple):
            assert len(feature_names) == features.shape[1], "Must provide a feature name for each feature"
        else:
            feature_names = [feature_names]
        for feature_idx in range(features.shape[1]):
            scalars = numpy_to_vtk(features[:, feature_idx])
            scalars.SetName(feature_names[feature_idx])
            mesh.GetPointData().AddArray(scalars)
    return mesh

def smooth_scalars_from_second_mesh_onto_base(base_mesh,
                                              second_mesh,
                                              sigma=1.,
                                              idx_coords_to_smooth_base=None,
                                              idx_coords_to_smooth_second=None,
                                              set_non_smoothed_scalars_to_zero=True
                                              ):  # sigma is equal to fwhm=2 (1mm in each direction)
    """
    Function to copy surface scalars from one mesh to another. This is done in a "smoothing" fashioon
    to get a weighted-average of the closest point - this is because the points on the 2 meshes won't
    be coincident with one another. The weighted average is done using a gaussian smoothing.

    Parameters
    ----------
    base_mesh : vtk.vtkPolyData
        The base mesh to smooth the scalars from `second_mesh` onto. 
    second_mesh : vtk.vtkPolyData
        The mesh with the scalar values that we want to pass onto the `base_mesh`.
    sigma : float, optional
        Sigma (standard deviation) of gaussian filter to apply to scalars, by default 1.
    idx_coords_to_smooth_base : list, optional
        List of the indices of nodes that are of interest for transferring (typically cartilage), 
        by default None
    idx_coords_to_smooth_second : list, optional
        List of the indices of the nodes that are of interest on the second mesh, by default None
    set_non_smoothed_scalars_to_zero : bool, optional
        Whether or not to set all notes that are not smoothed to zero, by default True

    Returns
    -------
    numpy.ndarray
        An array of the scalar values for each node on the base mesh that includes the scalar values
        transfered (smoothed) from the secondary mesh. 
    """    
    base_mesh_pts = get_mesh_physical_point_coords(base_mesh)
    if idx_coords_to_smooth_base is not None:
        base_mesh_pts = base_mesh_pts[idx_coords_to_smooth_base, :]
    second_mesh_pts = get_mesh_physical_point_coords(second_mesh)
    if idx_coords_to_smooth_second is not None:
        second_mesh_pts = second_mesh_pts[idx_coords_to_smooth_second, :]
    gauss_kernel = gaussian_kernel(base_mesh_pts, second_mesh_pts, sigma=sigma)
    second_mesh_scalars = np.copy(vtk_to_numpy(second_mesh.GetPointData().GetScalars()))
    if idx_coords_to_smooth_second is not None:
        # If sub-sampled second mesh - then only give the scalars from those sub-sampled points on mesh.
        second_mesh_scalars = second_mesh_scalars[idx_coords_to_smooth_second]

    smoothed_scalars_on_base = np.sum(gauss_kernel * second_mesh_scalars, axis=1)

    if idx_coords_to_smooth_base is not None:
        # if sub-sampled baseline mesh (only want to set cartilage to certain points/vertices), then
        # set the calculated smoothed scalars to only those nodes (and leave all other nodes the same as they were
        # originally.
        if set_non_smoothed_scalars_to_zero is True:
            base_mesh_scalars = np.zeros(base_mesh.GetNumberOfPoints())
        else:
            base_mesh_scalars = np.copy(vtk_to_numpy(base_mesh.GetPointData().GetScalars()))
        base_mesh_scalars[idx_coords_to_smooth_base] = smoothed_scalars_on_base
        return base_mesh_scalars

    else:
        return smoothed_scalars_on_base


def transfer_mesh_scalars_get_weighted_average_n_closest(new_mesh, 
                                                         old_mesh, 
                                                         n=3, 
                                                         return_mesh=False, 
                                                         create_new_mesh=False
                                                        ):
    """
    Transfer scalars from old_mesh to new_mesh using the weighted-average of the `n` closest
    nodes/points/vertices. Similar but not exactly the same as `smooth_scalars_from_second_mesh_onto_base`
    
    This function is ideally used for things like transferring cartilage thickness values from one mesh to another 
    after they have been registered together. This is necessary for things like performing statistical analyses or
    getting aggregate statistics. 

    Parameters
    ----------
    new_mesh : vtk.vtkPolyData
        The new mesh that we want to transfer scalar values onto. Also `base_mesh` from
        `smooth_scalars_from_second_mesh_onto_base` 
    old_mesh : vtk.vtkPolyData
        The mesh that we want to transfer scalars from. Also called `second_mesh` from 
        `smooth_scalars_from_second_mesh_onto_base`
    n : int, optional
        The number of closest nodes that we want to get weighed average of, by default 3

    Returns
    -------
    numpy.ndarray
        An array of the scalar values for each node on the `new_mesh` that includes the scalar values
        transfered (smoothed) from the `old_mesh`. 
    """    

    kDTree = vtk.vtkKdTreePointLocator()
    kDTree.SetDataSet(old_mesh)
    kDTree.BuildLocator()

    n_arrays = old_mesh.GetPointData().GetNumberOfArrays()
    array_names = [old_mesh.GetPointData().GetArray(array_idx).GetName() for array_idx in range(n_arrays)]
    new_scalars = np.zeros((new_mesh.GetNumberOfPoints(), n_arrays))
    scalars_old_mesh = [np.copy(vtk_to_numpy(old_mesh.GetPointData().GetArray(array_name))) for array_name in array_names]
    # print('len scalars_old_mesh', len(scalars_old_mesh))
    # scalars_old_mesh = np.copy(vtk_to_numpy(old_mesh.GetPointData().GetScalars()))
    for new_mesh_pt_idx in range(new_mesh.GetNumberOfPoints()):
        point = new_mesh.GetPoint(new_mesh_pt_idx)
        closest_ids = vtk.vtkIdList()
        kDTree.FindClosestNPoints(n, point, closest_ids)

        list_scalars = []
        distance_weighting = []
        for closest_pts_idx in range(closest_ids.GetNumberOfIds()):
            pt_idx = closest_ids.GetId(closest_pts_idx)
            _point = old_mesh.GetPoint(pt_idx)
            list_scalars.append([scalars[pt_idx] for scalars in scalars_old_mesh])
            distance_weighting.append(1 / np.sqrt(np.sum(np.square(np.asarray(point) - np.asarray(_point) + epsilon))))
    
        total_distance = np.sum(distance_weighting)
        # print('list_scalars', list_scalars)
        # print('distance_weighting', distance_weighting)
        # print('total_distance', total_distance)
        normalized_value = np.sum(np.asarray(list_scalars) * np.expand_dims(np.asarray(distance_weighting), axis=1),
                                  axis=0) / total_distance
        # print('new_mesh_pt_idx', new_mesh_pt_idx)
        # print('normalized_value', normalized_value)
        # print('new_scalars shape', new_scalars.shape)
        new_scalars[new_mesh_pt_idx, :] = normalized_value
    if return_mesh is False:
        return new_scalars
    else:
        if create_new_mesh is True:
            # create a new memory/vtk object. 
            new_mesh_ = vtk_deep_copy(new_mesh)
        else:
            # this just makes a reference to the existing new_mesh in memory. 
            new_mesh_ = new_mesh
        for idx, array_name in enumerate(array_names):
            new_array = numpy_to_vtk(new_scalars[:, idx])
            new_array.SetName(array_name)
            new_mesh_.GetPointData().AddArray(new_array)
        return new_mesh_

def get_smoothed_scalars(mesh, max_dist=2.0, order=2, gaussian=False):
    """
    perform smoothing of scalars on the nodes of a surface mesh. 
    return the smoothed values of the nodes so they can be used as necessary. 
    (e.g. to replace originals or something else)
    Smoothing is done for all data within `max_dist` and uses a simple weighted average based on
    the distance to the power of `order`. Default is squared distance (`order=2`)

    Parameters
    ----------
    mesh : vtk.vtkPolyData
        Surface mesh that we want to smooth scalars of. 
    max_dist : float, optional
        Maximum distance of nodes that we want to smooth (mm), by default 2.0
    order : int, optional
        Order of the polynomial used for weighting other nodes within `max_dist`, by default 2
    gaussian : bool, optional
        Should this use a gaussian smoothing, or weighted average, by default False

    Returns
    -------
    numpy.ndarray
        An array of the scalar values for each node on the `mesh` after they have been smoothed. 
    """    

    kDTree = vtk.vtkKdTreePointLocator()
    kDTree.SetDataSet(mesh)
    kDTree.BuildLocator()

    thickness_smoothed = np.zeros(mesh.GetNumberOfPoints())
    scalars = l2n(mesh.GetPointData().GetScalars())
    for idx in range(mesh.GetNumberOfPoints()):
        if scalars[idx] >0:  # don't smooth nodes with thickness == 0 (or negative? if that were to happen)
            point = mesh.GetPoint(idx)
            closest_ids = vtk.vtkIdList()
            kDTree.FindPointsWithinRadius(max_dist, point, closest_ids) # This will return a value ( 0 or 1). Can use that for debudding.

            list_scalars = []
            list_distances = []
            for closest_pt_idx in range(closest_ids.GetNumberOfIds()):
                pt_idx = closest_ids.GetId(closest_pt_idx)
                _point = mesh.GetPoint(pt_idx)
                list_scalars.append(scalars[pt_idx])
                list_distances.append(np.sqrt(np.sum(np.square(np.asarray(point) - np.asarray(_point) + epsilon))))

            distances_weighted = (max_dist - np.asarray(list_distances))**order
            scalars_weights = distances_weighted * np.asarray(list_scalars)
            normalized_value = np.sum(scalars_weights) / np.sum(distances_weighted)
            thickness_smoothed[idx] = normalized_value
    return thickness_smoothed

def gaussian_smooth_surface_scalars(mesh, sigma=1., idx_coords_to_smooth=None, array_name='thickness (mm)', array_idx=None):
    """
    The following is another function to smooth the scalar values on the surface of a mesh. 
    This one performs a gaussian smoothing using the supplied sigma and only smooths based on 
    the input `idx_coords_to_smooth`. If no `idx_coords_to_smooth` is provided, then all of the
    points are smoothed. `idx_coords_to_smooth` should be a list of indices of points to include. 

    e.g., coords_to_smooth = np.where(vtk_to_numpy(mesh.GetPointData().GetScalars())>0.01)[0]
    This would give only coordinates where the scalar values of the mesh are >0.01. This example is
    useful for cartilage where we might only want to smooth in locations that we have cartilage and
    ignore the other areas. 

    Parameters
    ----------
    mesh : vtk.vtkPolyData
        This is a surface mesh of that we want to smooth the scalars of. 
    sigma : float, optional
        The standard deviation of the gaussian filter to apply, by default 1.
    idx_coords_to_smooth : list, optional
        List of the indices of the vertices (points) that we want to include in the
        smoothing. For example, we can only smooth values that are cartialge and ignore
        all non-cartilage points, by default None
    array_name : str, optional
        Name of the scalar array that we want to smooth/filter, by default 'thickness (mm)'
    array_idx : int, optional
        The index of the scalar array that we want to smooth/filter - this is an alternative
        option to `array_name`, by default None

    Returns
    -------
    vtk.vtkPolyData
        Return the original mesh for which the scalars have been smoothed. However, this is not
        necessary becuase if the original mesh still exists then it should have been updated
        during the course of the pipeline. 
    """    

    point_coordinates = get_mesh_physical_point_coords(mesh)
    if idx_coords_to_smooth is not None:
        point_coordinates = point_coordinates[idx_coords_to_smooth, :]
    kernel = gaussian_kernel(point_coordinates, point_coordinates, sigma=sigma)

    original_array = mesh.GetPointData().GetArray(array_idx if array_idx is not None else array_name)
    original_scalars = np.copy(vtk_to_numpy(original_array))

    if idx_coords_to_smooth is not None:
        smoothed_scalars = np.sum(kernel * original_scalars[idx_coords_to_smooth],
                                    axis=1)
        original_scalars[idx_coords_to_smooth] = smoothed_scalars
        smoothed_scalars = original_scalars
    else:
        smoothed_scalars = np.sum(kernel * original_scalars, axis=1)

    smoothed_scalars = numpy_to_vtk(smoothed_scalars)
    smoothed_scalars.SetName(original_array.GetName())
    original_array.DeepCopy(smoothed_scalars) # Assign the scalars back to the original mesh

    # return the mesh object - however, if the original is not deleted, it should be smoothed
    # appropriately. 
    return mesh

def resample_surface(mesh, subdivisions=2, clusters=10000):
    """
    Resample a surface mesh using the ACVD algorithm: 
    Version used: 
    - https://github.com/pyvista/pyacvd
    Original version w/ more references: 
    - https://github.com/valette/ACVD

    Parameters
    ----------
    mesh : vtk.vtkPolyData
        Polydata mesh to be re-sampled. 
    subdivisions : int, optional
        Subdivide the mesh to have more points before clustering, by default 2
        Probably not necessary for very dense meshes.
    clusters : int, optional
        The number of clusters (points/vertices) to create during resampling 
        surafce, by default 10000
        - This is not exact, might have slight differences.
    
        Returns
    -------
    vtk.vtkPolyData :
        Return the resampled mesh. This will be a pyvista version of the vtk mesh
        but this is usable in all vtk function so it is not an issue. 
        

    """        
    pv_smooth_mesh = pv.wrap(mesh)
    clus = pyacvd.Clustering(pv_smooth_mesh)
    clus.subdivide(subdivisions)
    clus.cluster(clusters)
    mesh = clus.create_mesh()

    return mesh

def check_mesh_types(mesh, return_type='pyvista'):
    """
    Check the type of the input mesh and return the appropriate mesh type.

    Parameters
    ----------
    mesh : vtk.vtkPolyData or pyvista.PolyData or Mesh or BoneMesh or CartilageMesh
        The mesh to be checked.
    return_type : str, optional
        The type of mesh to return. Options are 'pyvista' or 'vtk', by default 'pyvista'
    
    Returns
    -------
    pyvista.PolyData or vtk.vtkPolyData
        The mesh in the appropriate type.
    """
    from pymskt.mesh import Mesh, BoneMesh, CartilageMesh

    if isinstance(mesh, (Mesh, BoneMesh, CartilageMesh)):
        mesh_ = mesh.mesh
    elif isinstance(mesh, vtk.vtkPolyData):
        mesh_ = pv.wrap(mesh)
    elif isinstance(mesh, pv.PolyData):
        mesh_ = mesh
    else:
        raise Exception(f"Mesh type not recognized: {type(mesh)}")
    
    return mesh_

def fix_mesh(mesh, verbose=True):
    """

    Parameters
    ----------
    mesh : vtk.vtkPolyData or pyvista.PolyData or Mesh or BoneMesh or CartilageMesh
        The mesh to be fixed. 
    verbose : bool, optional
        Print out the status of the mesh repair, by default True
    
    Returns
    -------
    pyvista.PolyData or Mesh or BoneMesh or CartilageMesh 
        The fixed mesh. If the input mesh is a Mesh, BoneMesh, or CartilageMesh, then
        the mesh attribute of the input mesh will be updated and returned. The updated
        mesh will be of type pyvista.PolyData. Otherwise, the fixed mesh will be 
        returned as type pyvista.PolyData.

    Raises
    ------
    Exception
        If the mesh type is not recognized, raise an exception.

    Notes
    -----
    This function is a wrapper for the meshfix package. 
             
    """
    from pymskt.mesh import Mesh, BoneMesh, CartilageMesh
    
    mesh_ = check_mesh_types(mesh)

    connected = mesh_.connectivity(largest=False)

    cell_ids = np.unique(connected['RegionId'])
    fixed_objects = []
    for idx in cell_ids:
        obj = connected.threshold([idx-0.5, idx+0.5])
        obj = obj.extract_surface()

        meshfix = mf.MeshFix(obj)
        meshfix.repair(verbose=verbose)
        if idx == 0:
            #if first iteration create new object
            new_object = meshfix.mesh
        else:
            #if not first iteration append to new object
            new_object += meshfix.mesh

    if isinstance(mesh, (Mesh, BoneMesh, CartilageMesh)):
        mesh.mesh = new_object
        return mesh
    else:
        return new_object



def get_mesh_edge_lengths(mesh):
    """
    Get the edge lengths of a mesh.

    Parameters
    ----------
    mesh : vtk.vtkPolyData or pyvista.PolyData or Mesh or BoneMesh or CartilageMesh
        The mesh to extract edge lengths from.
    
    Returns
    -------
    numpy.ndarray
        The edge lengths of the mesh.
    """
    
    mesh_ = check_mesh_types(mesh)
    
    faces = mesh_.faces.reshape(-1, 4)

    # extract all edges (triangle is ABC)
    edges0 = faces[:, 1:3]  # edges AB
    edges1 = faces[:, (1, 3)]  # edges AC
    edges2 = faces[:, 2:]  # edges BC

    edges = np.vstack((edges0, edges1, edges2))

    points = mesh_.points
    
    edge_lengths = np.sqrt(np.sum(np.square(points[edges[:,0],:] - points[edges[:,1], :]), axis=1))

    return edge_lengths
    



### THE FOLLOWING IS AN OLD/ORIGINAL VERSION OF THIS THAT SMOOTHED ALL ARRAYS ATTACHED TO MESH
# def gaussian_smooth_surface_scalars(mesh, sigma=(1,), idx_coords_to_smooth=None):
#     """
#     The following is another function to smooth the scalar values on the surface of a mesh. 
#     This one performs a gaussian smoothing using the supplied sigma and only smooths based on 
#     the input `idx_coords_to_smooth`. If no `idx_coords_to_smooth` is provided, then all of the
#     points are smoothed. `idx_coords_to_smooth` should be a list of indices of points to include. 

#     e.g., coords_to_smooth = np.where(vtk_to_numpy(mesh.GetPointData().GetScalars())>0.01)[0]
#     This would give only coordinates where the scalar values of the mesh are >0.01. This example is
#     useful for cartilage where we might only want to smooth in locations that we have cartilage and
#     ignore the other areas. 

#     """
#     point_coordinates = get_mesh_physical_point_coords(mesh)
#     if idx_coords_to_smooth is not None:
#         point_coordinates = point_coordinates[idx_coords_to_smooth, :]
#     kernels = []
#     if isinstance(sigma, (list, tuple)):
#         for sig in sigma:
#             kernels.append(gaussian_kernel(point_coordinates, point_coordinates, sigma=sig))
#     elif isinstance(sigma, (float, int)):
#         kernels.append(gaussian_kernel(point_coordinates, point_coordinates, sigma=sigma))

#     n_arrays = mesh.GetPointData().GetNumberOfArrays()
#     if n_arrays > len(kernels):
#         if len(kernels) == 1:
#             kernels = [kernels[0] for x in range(n_arrays)]
#     for array_idx in range(n_arrays):
#         original_array = mesh.GetPointData().GetArray(array_idx)
#         original_scalars = np.copy(vtk_to_numpy(original_array))

#         if idx_coords_to_smooth is not None:
#             smoothed_scalars = np.sum(kernels[array_idx] * original_scalars[idx_coords_to_smooth],
#                                       axis=1)
#             original_scalars[idx_coords_to_smooth] = smoothed_scalars
#             smoothed_scalars = original_scalars
#         else:
#             smoothed_scalars = np.sum(kernels[array_idx] * original_scalars, axis=1)

#         smoothed_scalars = numpy_to_vtk(smoothed_scalars)
#         smoothed_scalars.SetName(original_array.GetName())
#         original_array.DeepCopy(smoothed_scalars)

#     return mesh

# def get_smoothed_cartilage_thickness_values(loc_nrrd_images,
#                                             seg_image_name,
#                                             bone_label,
#                                             list_cart_labels,
#                                             image_smooth_var=1.0,
#                                             smooth_cart=False,
#                                             image_smooth_var_cart=1.0,
#                                             ray_cast_length=10.,
#                                             percent_ray_len_opposite_dir=0.2,
#                                             smooth_surface_scalars=True,
#                                             smooth_only_cartilage_values=True,
#                                             scalar_gauss_sigma=1.6986436005760381,  # This is a FWHM = 4
#                                             bone_pyacvd_subdivisions=2,
#                                             bone_pyacvd_clusters=20000,
#                                             crop_bones=False,
#                                             crop_percent=0.7,
#                                             bone=None,
#                                             loc_t2_map_nrrd=None,
#                                             t2_map_filename=None,
#                                             t2_smooth_sigma_multiple_of_thick=3,
#                                             assign_seg_label_to_bone=False,
#                                             mc_threshold=0.5,
#                                             bone_label_threshold=5000,
#                                             path_to_seg_transform=None,
#                                             reverse_seg_transform=True,
#                                             verbose=False):
#     """

#     :param loc_nrrd_images:
#     :param seg_image_name:
#     :param bone_label:
#     :param list_cart_labels:
#     :param image_smooth_var:
#     :param loc_tmp_save:
#     :param tmp_bone_filename:
#     :param smooth_cart:
#     :param image_smooth_var_cart:
#     :param tmp_cart_filename:
#     :param ray_cast_length:
#     :param percent_ray_len_opposite_dir:
#     :param smooth_surface_scalars:
#     :param smooth_surface_scalars_gauss:
#     :param smooth_only_cartilage_values:
#     :param scalar_gauss_sigma:
#     :param scalar_smooth_max_dist:
#     :param scalar_smooth_order:
#     :param bone_pyacvd_subdivisions:
#     :param bone_pyacvd_clusters:
#     :param crop_bones:
#     :param crop_percent:
#     :param bone:
#     :param tmp_cropped_image_filename:
#     :param loc_t2_map_nrrd:.
#     :param t2_map_filename:
#     :param t2_smooth_sigma_multiple_of_thick:
#     :param assign_seg_label_to_bone:
#     :param multiple_cart_labels_separate:
#     :param mc_threshold:
#     :return:

#     Notes:
#     multiple_cart_labels_separate REMOVED from the function.
#     """
#     # Read segmentation image
#     seg_image = sitk.ReadImage(os.path.join(loc_nrrd_images, seg_image_name))
#     seg_image = set_seg_border_to_zeros(seg_image, border_size=1)

#     seg_view = sitk.GetArrayViewFromImage(seg_image)
#     n_pixels_labelled = sum(seg_view[seg_view == bone_label])

#     if n_pixels_labelled < bone_label_threshold:
#         raise Exception('The bone does not exist in this segmentation!, only {} pixels detected, threshold # is {}'.format(n_pixels_labelled, 
#                                                                                                                            bone_label_threshold))
    
#     # Read segmentation in vtk format if going to assign labels to surface.
#     # Also, if femur break it up into its parts.
#     if assign_seg_label_to_bone is True:
#         tmp_filename = ''.join(random.choice(string.ascii_lowercase) for i in range(10)) + '.nrrd'
#         if bone == 'femur':
#             new_seg_image = qc.get_knee_segmentation_with_femur_subregions(seg_image,
#                                                                            fem_cart_label_idx=1)
#             sitk.WriteImage(new_seg_image, os.path.join('/tmp', tmp_filename))
#         else:
#             sitk.WriteImage(seg_image, os.path.join('/tmp', tmp_filename))
#         vtk_seg_reader = read_nrrd('/tmp',
#                                    tmp_filename,
#                                    set_origin_zero=True
#                                    )
#         vtk_seg = vtk_seg_reader.GetOutput()

#         seg_transformer = SitkVtkTransformer(seg_image)

#         # Delete tmp files
#         safely_delete_tmp_file('/tmp',
#                                tmp_filename)

#     # Crop the bones if that's an option/thing.
#     if crop_bones is True:
#         if 'femur' in bone:
#             bone_crop_distal = True
#         elif 'tibia' in bone:
#             bone_crop_distal = False
#         else:
#             raise Exception('var bone should be "femur" or "tiba" got: {} instead'.format(bone))

#         seg_image = crop_bone_based_on_width(seg_image,
#                                              bone_label,
#                                              percent_width_to_crop_height=crop_percent,
#                                              bone_crop_distal=bone_crop_distal)

#     if verbose is True:
#         tic = time.time()

#     # Create bone mesh/smooth/resample surface points.
#     ns_bone_mesh = BoneMesh(seg_image=seg_image,
#                             label_idx=bone_label)
#     if verbose is True:
#         print('Loaded mesh')
#     ns_bone_mesh.create_mesh(smooth_image=True,
#                              smooth_image_var=image_smooth_var,
#                              marching_cubes_threshold=mc_threshold
#                              )
#     if verbose is True:
#        print('Smoothed bone surface')
#     ns_bone_mesh.resample_surface(subdivisions=bone_pyacvd_subdivisions,
#                                   clusters=bone_pyacvd_clusters)
#     if verbose is True:
#        print('Resampled surface')
#     n_bone_points = ns_bone_mesh._mesh.GetNumberOfPoints()

#     if verbose is True:
#         toc = time.time()
#         print('Creating bone mesh took: {}'.format(toc - tic))
#         tic = time.time()

#     # Pre-allocate empty arrays for t2/labels if they are being placed on surface.
#     if assign_seg_label_to_bone is True:
#         # Apply inverse transform to get it into the space of the image.
#         # This is easier than the reverse function.
#         if assign_seg_label_to_bone is True:
#             ns_bone_mesh.apply_transform_to_mesh(transform=seg_transformer.get_inverse_transform())

#             labels = np.zeros(n_bone_points, dtype=int)

#     thicknesses = np.zeros(n_bone_points, dtype=float)
#     if verbose is True:
#        print('Number bone mesh points: {}'.format(n_bone_points))

#     # Iterate over cartilage labels
#     # Create mesh & store thickness + cartilage label + t2 in arrays
#     for cart_label_idx in list_cart_labels:
#         # Test to see if this particular cartilage label even exists in the label :P
#         # This is important for people that may have no cartilage (of a particular type)
#         seg_array_view = sitk.GetArrayViewFromImage(seg_image)
#         n_pixels_with_cart = np.sum(seg_array_view == cart_label_idx)
#         if n_pixels_with_cart == 0:
#             print("Not analyzing cartilage for label {} because it doesnt have any pixels!".format(cart_label_idx))
#             continue

#         ns_cart_mesh = CartilageMesh(seg_image=seg_image,
#                                      label_idx=cart_label_idx)
#         ns_cart_mesh.create_mesh(smooth_image=smooth_cart,
#                                  smooth_image_var=image_smooth_var_cart,
#                                  marching_cubes_threshold=mc_threshold)

#         # Perform Thickness & label simultaneously. 

#         if assign_seg_label_to_bone is True:
#             ns_cart_mesh.apply_transform_to_mesh(transform=seg_transformer.get_inverse_transform())

#         node_data = get_cartilage_properties_at_points(ns_bone_mesh._mesh,
#                                                        ns_cart_mesh._mesh,
#                                                        t2_vtk_image=None,
#                                                        seg_vtk_image=vtk_seg if assign_seg_label_to_bone is True else None,
#                                                        ray_cast_length=ray_cast_length,
#                                                        percent_ray_length_opposite_direction=percent_ray_len_opposite_dir
#                                                        )
#         if assign_seg_label_to_bone is False:
#             thicknesses += node_data
#         else:
#             thicknesses += node_data[0]
#             labels += node_data[1]

#         if verbose is True:
#             print('Cartilage label: {}'.format(cart_label_idx))
#             print('Mean thicknesses (all): {}'.format(np.mean(thicknesses)))

#     if verbose is True:
#         toc = time.time()
#         print('Calculating all thicknesses: {}'.format(toc - tic))
#         tic = time.time()

#     # Assign thickness & T2 data (if it exists) to bone surface.
#     thickness_scalars = numpy_to_vtk(thicknesses)
#     thickness_scalars.SetName('thickness (mm)')
#     ns_bone_mesh._mesh.GetPointData().SetScalars(thickness_scalars)
    
#     # Smooth surface scalars
#     if smooth_surface_scalars is True:
#         if smooth_only_cartilage_values is True:
#             loc_cartilage = np.where(vtk_to_numpy(ns_bone_mesh._mesh.GetPointData().GetScalars())>0.01)[0]
#             ns_bone_mesh.mesh = gaussian_smooth_surface_scalars(ns_bone_mesh.mesh,
#                                                                 sigma=scalar_gauss_sigma,
#                                                                 idx_coords_to_smooth=loc_cartilage)
#         else:
#             ns_bone_mesh.mesh = gaussian_smooth_surface_scalars(ns_bone_mesh.mesh, sigma=scalar_gauss_sigma)

#         if verbose is True:
#             toc = time.time()
#             print('Smoothing scalars took: {}'.format(toc - tic))

#     # Add the label values to the bone after smoothing is finished.
#     if assign_seg_label_to_bone is True:
#         label_scalars = numpy_to_vtk(labels)
#         label_scalars.SetName('Cartilage Region')
#         ns_bone_mesh._mesh.GetPointData().AddArray(label_scalars)

#     if assign_seg_label_to_bone is True:
#         # Transform bone back to the position it was in before rotating it (for the t2 analysis)
#         ns_bone_mesh.reverse_all_transforms()

#     return ns_bone_mesh.mesh