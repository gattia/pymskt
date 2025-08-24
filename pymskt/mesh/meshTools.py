from collections import defaultdict

import numpy as np
import point_cloud_utils as pcu
import pyacvd
import pymeshfix as mf
import pyvista as pv
import vtk
from scipy.spatial import cKDTree
from vtk.util.numpy_support import numpy_to_vtk, vtk_to_numpy

from pymskt.mesh.utils import (
    get_intersect,
    get_obb_surface,
    get_surface_normals,
    is_hit,
    vtk_deep_copy,
)
from pymskt.utils import gaussian_kernel, l2n, n2l, safely_delete_tmp_file

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

    def __init__(
        self,
        line_resolution,
        vtk_image,
        save_data_in_class=True,
        save_mean=False,
        save_std=False,
        save_most_common=False,
        save_max=False,
        filler=0,
        non_zero_only=True,
        data_categorical=False,
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

    def get_data_along_line(self, start_pt, end_pt):
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

    def save_data_along_line(self, start_pt, end_pt):
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


def get_cartilage_properties_at_points(
    surface_bone,
    surface_cartilage,
    t2_vtk_image=None,
    seg_vtk_image=None,
    ray_cast_length=20.0,
    percent_ray_length_opposite_direction=0.25,
    no_thickness_filler=0.0,
    no_t2_filler=0.0,
    no_seg_filler=0,
    line_resolution=100,
    n_intersections=2,
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
    n_intersections : int, optional
        Number of intersections to expect when casting ray from bone surface to cartilage

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
            t2_data_probe = ProbeVtkImageDataAlongLine(
                line_resolution, t2_vtk_image, save_mean=True, filler=no_t2_filler
            )
        if seg_vtk_image is not None:
            seg_data_probe = ProbeVtkImageDataAlongLine(
                line_resolution,
                seg_vtk_image,
                save_most_common=True,
                filler=no_seg_filler,
                data_categorical=True,
            )

    print("INTERSECTION IS: ", n_intersections)

    # Loop through all points
    for idx in range(points.GetNumberOfPoints()):
        point = points.GetPoint(idx)
        normal = point_normals.GetTuple(idx)

        end_point_ray = n2l(l2n(point) + ray_cast_length * l2n(normal))
        start_point_ray = n2l(
            l2n(point) + ray_cast_length * percent_ray_length_opposite_direction * (-l2n(normal))
        )

        # Check if there are any intersections for the given ray
        if is_hit(obb_cartilage, start_point_ray, end_point_ray):  # intersections were found
            # Retrieve coordinates of intersection points and intersected cell ids
            points_intersect, cell_ids_intersect = get_intersect(
                obb_cartilage, start_point_ray, end_point_ray
            )
            #         points
            if len(points_intersect) == n_intersections:
                if n_intersections == 2:
                    thickness_data.append(
                        np.sqrt(
                            np.sum(np.square(l2n(points_intersect[0]) - l2n(points_intersect[1])))
                        )
                    )
                elif n_intersections == 1:
                    thickness_data.append(no_thickness_filler)
                    points_intersect = [start_point_ray, end_point_ray]

                if t2_vtk_image is not None:
                    t2_data_probe.save_data_along_line(
                        start_pt=points_intersect[0], end_pt=points_intersect[1]
                    )
                if seg_vtk_image is not None:
                    seg_data_probe.save_data_along_line(
                        start_pt=points_intersect[0], end_pt=points_intersect[1]
                    )

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
        return (
            np.asarray(thickness_data, dtype=float),
            np.asarray(t2_data_probe.mean_data, dtype=float),
            np.asarray(seg_data_probe.most_common_data, dtype=int),
        )
    elif t2_vtk_image is not None:
        return (
            np.asarray(thickness_data, dtype=float),
            np.asarray(t2_data_probe.mean_data, dtype=float),
        )
    elif seg_vtk_image is not None:
        return (
            np.asarray(thickness_data, dtype=float),
            np.asarray(seg_data_probe.most_common_data, dtype=int),
        )


def get_distance_other_surface_at_points(
    surface,
    other_surface,
    ray_cast_length=20.0,
    percent_ray_length_opposite_direction=0.25,
    no_distance_filler=0.0,
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

        end_point_ray = n2l(l2n(point) + ray_cast_length * l2n(normal))
        start_point_ray = n2l(
            l2n(point) + ray_cast_length * percent_ray_length_opposite_direction * (-l2n(normal))
        )

        # Check if there are any intersections for the given ray
        if is_hit(obb_other_surface, start_point_ray, end_point_ray):  # intersections were found
            # Retrieve coordinates of intersection points and intersected cell ids
            points_intersect, cell_ids_intersect = get_intersect(
                obb_other_surface, start_point_ray, end_point_ray
            )
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
            feature_name = "feature_1"
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
            feature_names = [f"feature_{i}" for i in range(features.shape[1])]
        elif type(feature_names) in (list, tuple):
            assert (
                len(feature_names) == features.shape[1]
            ), "Must provide a feature name for each feature"
        else:
            feature_names = [feature_names]
        for feature_idx in range(features.shape[1]):
            scalars = numpy_to_vtk(features[:, feature_idx])
            scalars.SetName(feature_names[feature_idx])
            mesh.GetPointData().AddArray(scalars)
    return mesh


def smooth_scalars_from_second_mesh_onto_base(
    base_mesh,
    second_mesh,
    sigma=1.0,
    idx_coords_to_smooth_base=None,
    idx_coords_to_smooth_second=None,
    set_non_smoothed_scalars_to_zero=True,
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


def transfer_mesh_scalars_get_weighted_average_n_closest(
    new_mesh,
    old_mesh,
    n=3,
    return_mesh=False,
    create_new_mesh=False,
    max_dist=None,
    categorical=None,
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
    categorical : bool, dict, or None, optional
        Specify whether scalars should be treated as categorical. If None (default),
        auto-detects based on data type per array. If bool, applies to all arrays.
        If dict, keys should be array names with bool values.

    Returns
    -------
    dict
        A dict of the scalar values with keys = scalar names and the scalar value for
        each node that includes the scalar values transfered (smoothed) from the `old_mesh`.
    """

    kDTree = vtk.vtkKdTreePointLocator()
    kDTree.SetDataSet(old_mesh)
    kDTree.BuildLocator()

    n_arrays = old_mesh.GetPointData().GetNumberOfArrays()
    array_names = [
        old_mesh.GetPointData().GetArray(array_idx).GetName() for array_idx in range(n_arrays)
    ]

    # Flatten all arrays and track their original shapes
    original_arrays = [
        np.copy(vtk_to_numpy(old_mesh.GetPointData().GetArray(array_name)))
        for array_name in array_names
    ]
    array_shapes = {}  # Track original shapes for reshaping later
    flattened_scalars = []  # All arrays flattened to 1D per point
    flat_to_original = []  # Map flat index back to (array_idx, component_idx)

    for array_idx, (array_name, arr) in enumerate(zip(array_names, original_arrays)):
        array_shapes[array_name] = arr.shape[1:] if arr.ndim > 1 else ()

        if arr.ndim == 1:
            # Scalar array - just add as-is
            flattened_scalars.append(arr)
            flat_to_original.append((array_idx, None))
        else:
            # Vector/tensor array - flatten components
            n_components = arr.shape[1] if arr.ndim == 2 else np.prod(arr.shape[1:])
            arr_reshaped = arr.reshape(arr.shape[0], n_components)
            for comp_idx in range(n_components):
                flattened_scalars.append(arr_reshaped[:, comp_idx])
                flat_to_original.append((array_idx, comp_idx))

    # Initialize output arrays with correct shapes
    new_scalars = {}
    for array_name in array_names:
        if array_shapes[array_name]:
            # Vector/tensor - initialize with full shape
            new_scalars[array_name] = np.zeros(
                (new_mesh.GetNumberOfPoints(),) + array_shapes[array_name]
            )
        else:
            # Scalar - initialize as 1D
            new_scalars[array_name] = np.zeros(new_mesh.GetNumberOfPoints())

    # Use flattened arrays for processing (this makes the rest of the code work unchanged)
    scalars_old_mesh = flattened_scalars
    flat_array_names = [f"flat_{i}" for i in range(len(flattened_scalars))]

    # Handle categorical parameter for flattened arrays
    if categorical is None:
        # Auto-detect categorical for each original array, then expand to flattened components
        categorical_flags = {}
        for array_name, arr in zip(array_names, original_arrays):
            # Only scalar arrays can be categorical - vectors are automatically continuous
            if arr.ndim > 1:
                categorical_flags[array_name] = False
            else:
                categorical_flags[array_name] = np.issubdtype(arr.dtype, np.integer)
    elif isinstance(categorical, bool):
        # Apply same categorical flag to all arrays, but validate vectors
        categorical_flags = {}
        for array_name, arr in zip(array_names, original_arrays):
            if categorical and arr.ndim > 1:
                raise ValueError(
                    f"Array '{array_name}' is a vector ({arr.shape}) and cannot be treated as categorical"
                )
            categorical_flags[array_name] = categorical
    elif isinstance(categorical, dict):
        # Use provided per-array flags, but validate vectors
        categorical_flags = categorical.copy()
        # Fill in missing arrays with auto-detection and validate
        for array_name, arr in zip(array_names, original_arrays):
            if array_name in categorical_flags and categorical_flags[array_name] and arr.ndim > 1:
                raise ValueError(
                    f"Array '{array_name}' is a vector ({arr.shape}) and cannot be treated as categorical"
                )
            if array_name not in categorical_flags:
                if arr.ndim > 1:
                    categorical_flags[array_name] = False
                else:
                    categorical_flags[array_name] = np.issubdtype(arr.dtype, np.integer)
    else:
        raise ValueError("categorical must be None, bool, or dict")

    # Create flattened categorical flags - each flattened component inherits from its parent array
    flat_categorical_flags = {}
    for flat_idx, (orig_array_idx, comp_idx) in enumerate(flat_to_original):
        orig_array_name = array_names[orig_array_idx]
        flat_categorical_flags[flat_array_names[flat_idx]] = categorical_flags[orig_array_name]

    # Create temporary flat output arrays for processing
    flat_new_scalars = {}
    for flat_name in flat_array_names:
        flat_new_scalars[flat_name] = np.zeros(new_mesh.GetNumberOfPoints())

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
            dist_ = np.sqrt(np.sum(np.square(np.asarray(point) - np.asarray(_point) + epsilon)))
            if max_dist is not None:
                if dist_ > max_dist:
                    continue
            list_scalars.append([scalars[pt_idx] for scalars in scalars_old_mesh])
            distance_weighting.append(1 / dist_)

        if len(list_scalars) == 0:
            # no points within max_dist, skip (which leaves the scalar(s) at zero)
            continue
        # compute the total distance
        total_distance = np.sum(distance_weighting)

        # Process each flattened array - this is the original logic that now works!
        arr = np.asarray(list_scalars)
        for flat_idx, flat_name in enumerate(flat_array_names):
            if flat_categorical_flags[flat_name]:
                # Distance-weighted voting for categorical data
                array_values = arr[:, flat_idx]
                if len(array_values) > 0:
                    label_weights = defaultdict(float)
                    for label, weight in zip(array_values, distance_weighting):
                        label_weights[int(label)] += weight
                    chosen_label = max(label_weights, key=label_weights.get)
                    flat_new_scalars[flat_name][new_mesh_pt_idx] = chosen_label
            else:
                # Weighted average for continuous data
                normalized_value = (
                    np.sum(arr[:, flat_idx] * np.asarray(distance_weighting)) / total_distance
                )
                flat_new_scalars[flat_name][new_mesh_pt_idx] = normalized_value

    # Reshape flattened results back to original array structures
    for flat_idx, (orig_array_idx, comp_idx) in enumerate(flat_to_original):
        orig_array_name = array_names[orig_array_idx]
        flat_name = flat_array_names[flat_idx]

        if comp_idx is None:
            # Scalar array - direct assignment
            new_scalars[orig_array_name] = flat_new_scalars[flat_name]
        else:
            # Vector/tensor array - assign to specific component
            if array_shapes[orig_array_name] == (3,):  # Common case: 3D vectors
                new_scalars[orig_array_name][:, comp_idx] = flat_new_scalars[flat_name]
            else:
                # General case: reshape component index back to original structure
                orig_shape = array_shapes[orig_array_name]
                flat_shape = (new_mesh.GetNumberOfPoints(), np.prod(orig_shape))
                reshaped = new_scalars[orig_array_name].reshape(flat_shape)
                reshaped[:, comp_idx] = flat_new_scalars[flat_name]
                new_scalars[orig_array_name] = reshaped.reshape(
                    (new_mesh.GetNumberOfPoints(),) + orig_shape
                )

    if return_mesh is False:
        # Convert only categorical arrays to int, preserve float arrays as float
        for array_name in array_names:
            if categorical_flags[array_name]:
                new_scalars[array_name] = new_scalars[array_name].astype(int)
        return new_scalars
    else:
        if create_new_mesh is True:
            # create a new memory/vtk object.
            new_mesh_ = vtk_deep_copy(new_mesh)
        else:
            # this just makes a reference to the existing new_mesh in memory.
            new_mesh_ = new_mesh
        for array_name, scalars in new_scalars.items():
            # Convert only categorical arrays to int before creating VTK array
            if categorical_flags[array_name]:
                scalars = scalars.astype(int)
            new_array = numpy_to_vtk(scalars)
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
        if (
            scalars[idx] > 0
        ):  # don't smooth nodes with thickness == 0 (or negative? if that were to happen)
            point = mesh.GetPoint(idx)
            closest_ids = vtk.vtkIdList()
            kDTree.FindPointsWithinRadius(
                max_dist, point, closest_ids
            )  # This will return a value ( 0 or 1). Can use that for debudding.

            list_scalars = []
            list_distances = []
            for closest_pt_idx in range(closest_ids.GetNumberOfIds()):
                pt_idx = closest_ids.GetId(closest_pt_idx)
                _point = mesh.GetPoint(pt_idx)
                list_scalars.append(scalars[pt_idx])
                list_distances.append(
                    np.sqrt(np.sum(np.square(np.asarray(point) - np.asarray(_point) + epsilon)))
                )

            distances_weighted = (max_dist - np.asarray(list_distances)) ** order
            scalars_weights = distances_weighted * np.asarray(list_scalars)
            normalized_value = np.sum(scalars_weights) / np.sum(distances_weighted)
            thickness_smoothed[idx] = normalized_value
    return thickness_smoothed


def gaussian_smooth_surface_scalars(
    mesh, sigma=1.0, idx_coords_to_smooth=None, array_name="thickness (mm)", array_idx=None
):
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

    from pymskt.mesh import BoneMesh, CartilageMesh, Mesh

    point_coordinates = get_mesh_physical_point_coords(mesh)
    if idx_coords_to_smooth is not None:
        point_coordinates = point_coordinates[idx_coords_to_smooth, :]
    kernel = gaussian_kernel(point_coordinates, point_coordinates, sigma=sigma)

    original_array = mesh.GetPointData().GetArray(
        array_idx if array_idx is not None else array_name
    )
    original_scalars = np.copy(vtk_to_numpy(original_array))

    if idx_coords_to_smooth is not None:
        smoothed_scalars = np.sum(kernel * original_scalars[idx_coords_to_smooth], axis=1)
        original_scalars[idx_coords_to_smooth] = smoothed_scalars
        smoothed_scalars = original_scalars
    else:
        smoothed_scalars = np.sum(kernel * original_scalars, axis=1)

    if isinstance(mesh, (Mesh, BoneMesh, CartilageMesh, pv.PolyData)):
        mesh.point_data[original_array.GetName()] = smoothed_scalars
    else:
        smoothed_scalars = numpy_to_vtk(smoothed_scalars)
        smoothed_scalars.SetName(original_array.GetName())
        original_array.DeepCopy(smoothed_scalars)  # Assign the scalars back to the original mesh

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


def get_largest_connected_component(mesh):
    """
    Get the largest connected component of a mesh.

    Parameters
    ----------
    mesh : vtk.vtkPolyData or pyvista.PolyData or Mesh or BoneMesh or CartilageMesh
        The mesh that we want to get the largest connected component of.

    Returns
    -------
    vtk.vtkPolyData
        The largest connected component of the input mesh.
    """
    mesh_ = check_mesh_types(mesh)

    largest = mesh_.extract_largest()
    largest.point_data.clear()
    largest.cell_data.clear()

    return largest


def check_mesh_types(mesh, return_type="pyvista"):
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
    from pymskt.mesh import BoneMesh, CartilageMesh, Mesh

    if isinstance(mesh, (Mesh, BoneMesh, CartilageMesh, pv.PolyData)):
        pass

    elif isinstance(mesh, vtk.vtkPolyData):
        mesh = pv.wrap(mesh)
    else:
        raise TypeError(f"Mesh type not recognized: {type(mesh)}")

    return mesh


def fix_mesh(
    mesh,
    method="meshfix",
    treat_as_single_component=False,
    resolution=50000,
    project_onto_surface=True,
    verbose=True,
):
    """

    Parameters
    ----------
    mesh : vtk.vtkPolyData or pyvista.PolyData or Mesh or BoneMesh or CartilageMesh
        The mesh to be fixed.
    method : str, optional
        The method to use for mesh repair. Options are 'meshfix' or 'pcu', by default 'meshfix'.
    treat_as_single_component : bool, optional
        If True, then treat the mesh as a single component for use with meshfix. Otherwise,
        treat each connected component as a separate mesh, by default False
    resolution : int, optional
        The resolution to use for the pcu method, by default 50000
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
    from pymskt.mesh import BoneMesh, CartilageMesh, Mesh

    mesh_ = check_mesh_types(mesh)

    if method == "pcu":
        new_object = meshfix_pcu(
            mesh_, resolution=resolution, project_onto_surface=project_onto_surface
        )

    elif method == "meshfix":
        if treat_as_single_component is True:
            new_object = meshfix_pymeshfix(
                mesh_, joincomp=True, remove_smallest_components=False, verbose=verbose
            )
        else:
            connected = mesh_.connectivity(largest=False)

            cell_ids = np.unique(connected["RegionId"])
            for idx in cell_ids:
                obj = connected.threshold([idx - 0.5, idx + 0.5])
                obj = obj.extract_surface()

                if method == "meshfix":
                    obj = meshfix_pymeshfix(obj, verbose=verbose)

                if idx == 0:
                    # if first iteration create new object
                    new_object = obj
                else:
                    # if not first iteration append to new object
                    new_object += obj

    if isinstance(mesh, (Mesh, BoneMesh, CartilageMesh)):
        mesh.mesh = new_object
        return mesh
    else:
        return new_object


def meshfix_pymeshfix(obj, joincomp=False, remove_smallest_components=True, verbose=True):
    meshfix = mf.MeshFix(obj)
    meshfix.repair(
        joincomp=joincomp, remove_smallest_components=remove_smallest_components, verbose=verbose
    )
    return meshfix.mesh


def get_faces_vertices(mesh):
    """
    Get the faces and vertices of a mesh.
    """

    faces = vtk_to_numpy(mesh.GetPolys().GetData())
    faces = faces.reshape(-1, 4)
    faces = np.delete(faces, 0, 1)
    points = vtk_to_numpy(mesh.GetPoints().GetData())

    return faces, points


def project_point_onto_line(P0, P1, P2):
    """
    Project a 3D point onto a line defined by two 3D points.

    Parameters:
        P0 (array-like): The point to be projected. Should be a list or array of length 3.
        P1 (array-like): A point on the line. Should be a list or array of length 3.
        P2 (array-like): Another point on the line. Should be a list or array of length 3.

    Returns:
        numpy.ndarray: The projection of P0 onto the line through P1 and P2.
    """

    # Convert the input to numpy arrays
    P0, P1, P2 = np.array(P0), np.array(P1), np.array(P2)

    # Define the vectors
    v = P2 - P1
    w = P0 - P1

    # Compute the projection of w onto v
    proj_v_w = np.dot(w, v) / np.dot(v, v) * v

    # The projection of P0 onto the line P1P2 is then P1 + proj_v_w
    P_proj = P1 + proj_v_w

    return P_proj


def meshfix_pcu(obj, resolution=50000, project_onto_surface=True):
    """
    this is a wrapper for point cloud utils method of getting watertight manifold for shapenet models
    """
    # get faces and points
    faces, points = get_faces_vertices(obj)
    points_wt, faces_wt = pcu.make_mesh_watertight(points, faces, resolution)

    # add a column of 3s to faces_wt (vtk uses this to know how many points to use for each face)
    faces_wt = np.hstack((np.ones((faces_wt.shape[0], 1), dtype=int) * 3, faces_wt))

    if project_onto_surface is True:
        # project points onto original mesh
        dists, fid, bc = pcu.closest_points_on_mesh(points_wt, points, faces)

        closest_pts = pcu.interpolate_barycentric_coords(faces, fid, bc, points)

        # Get nan bc/closest_pts & fix them
        nans_rows_bc = np.unique(np.where(np.isnan(bc))[0])
        nans_rows_pts = np.unique(np.where(np.isnan(closest_pts))[0])
        nans_rows = np.unique(np.concatenate((nans_rows_bc, nans_rows_pts)))

        # if 2 points are identical, then interpolate between them
        # if all 3 points are identical, then just use that point
        for nan_row in nans_rows:
            nan_pts = points[faces[fid[nan_row]]].squeeze()
            # get unique points
            unique_pts = np.unique(nan_pts, axis=0)
            n_unique_pts = unique_pts.shape[0]

            if n_unique_pts == 1:
                new_pt = unique_pts[0, :]
            elif n_unique_pts == 2:
                new_pt = project_point_onto_line(
                    points_wt[nan_row, :], unique_pts[0, :], unique_pts[1, :]
                )
            elif n_unique_pts == 3:
                bc_ = bc[nan_row]
                A, B, C = unique_pts
                bc[np.isinf(bc)] = np.nan
                if np.isnan(bc_).all():  # All barycentric coords are NaN
                    print("All barycentric coords are NaN - using centroid")
                    new_pt = compute_centroid(A, B, C)
                else:
                    print("Some barycentric coords are NaN - using closest point")
                    corrected_bc = np.where(np.isnan(bc_), 0, bc_)
                    new_pt = compute_point_from_barycentric(corrected_bc, A, B, C)
            else:
                raise ValueError("Has 3 unique points but still nan error.... ")

            closest_pts[nan_row, :] = new_pt

        points_wt = closest_pts

    # create new mesh
    new_mesh = pv.PolyData(points_wt, faces_wt)

    return new_mesh


def compute_centroid(A, B, C):
    print(A, B, C)
    return (A + B + C) / 3.0


def compute_point_from_barycentric(bc, A, B, C):
    print(bc)
    u, v, w = bc
    return u * A + v * B + w * C


def consistent_normals(mesh):
    """
    update faces of mesh to be consistently oriented using pcu
    """
    # get faces and points
    faces, points = get_faces_vertices(mesh)
    # get consistent normals
    faces_consitent, _ = pcu.orient_mesh_faces(faces)
    # add a column of 3s to faces_consitent (vtk uses this to know how many points to use for each face)
    faces_consitent = np.hstack((np.ones((faces_consitent.shape[0], 1)) * 3, faces_consitent))
    # create new mesh
    new_mesh = pv.PolyData(points, faces_consitent.astype(int))

    return new_mesh


def rand_sample_pts_mesh(mesh, n_pts, method="bluenoise"):
    """
    Randomly sample points from a mesh
    """
    # get faces and points
    faces, points = get_faces_vertices(mesh)

    if method == "random":
        fid, bc = pcu.sample_mesh_random(points, faces, n_pts)
    elif method == "bluenoise":
        fid, bc = pcu.sample_mesh_poisson_disk(points, faces, num_samples=n_pts)

    rand_pts = pcu.interpolate_barycentric_coords(faces, fid, bc, points)

    return rand_pts


def vtk_sdf(pts, mesh):
    """
    Calculates the signed distance functions (SDFs) for a set of points
    given a mesh using VTK.

    Args:
        pts (np.ndarray): (n_pts, 3) array of points
        mesh (vtkPolyData or mskt.mesh.Mesh): VTK mesh

    Returns:
        np.ndarray: (n_pts, ) array of SDFs
    """
    implicit_distance = vtk.vtkImplicitPolyDataDistance()
    implicit_distance.SetInput(mesh)

    # Convert the numpy array to a vtkPoints object
    vtk_pts = numpy_to_vtk(pts)
    # Pre allocate (vtk) where store SDFs
    sdfs = numpy_to_vtk(np.zeros(pts.shape[0]))
    # calculate SDFs
    implicit_distance.FunctionValue(vtk_pts, sdfs)
    # Convert back to numpy array
    sdfs = vtk_to_numpy(sdfs)

    return sdfs


def pcu_sdf(pts, mesh):
    """
    Calculates the signed distance functions (SDFs) for a set of points
    given a mesh using Point Cloud Utils (PCU).

    Args:
        pts (np.ndarray): (n_pts, 3) array of points
        mesh (vtkPolyData or mskt.mesh.Mesh): VTK mesh

    Returns:
        np.ndarray: (n_pts, ) array of SDFs
    """

    faces = vtk_to_numpy(mesh.GetPolys().GetData())
    faces = faces.reshape(-1, 4)
    faces = np.delete(faces, 0, 1)
    points = vtk_to_numpy(mesh.GetPoints().GetData())
    sdfs, face_ids, barycentric_coords = pcu.signed_distance_to_mesh(pts, points, faces)

    return sdfs


def get_rand_samples(pts1, pts2, num_samples):
    """
    Randomly sample points from two point clouds.

    Args:
    - pts1 (numpy.ndarray): The first point cloud.
    - pts2 (numpy.ndarray): The second point cloud.
    - num_samples (int): The number of points to sample from each point cloud.

    Returns:
    - pts1 (numpy.ndarray): The first point cloud with num_samples points randomly sampled.
    - pts2 (numpy.ndarray): The second point cloud with num_samples points randomly sampled.
    """

    sample1 = np.random.choice(
        pts1.shape[0], size=num_samples, replace=True if pts1.shape[0] < num_samples else False
    )
    pts1 = pts1[sample1, :]

    sample2 = np.random.choice(
        pts2.shape[0], size=num_samples, replace=True if pts2.shape[0] < num_samples else False
    )
    pts2 = pts2[sample2, :]

    return pts1, pts2


def get_pt_cloud_distances(pts1, pts2, num_samples=None):
    """
    Compute the distances between two point clouds.

    Args:
    - pts1 (numpy.ndarray): The first point cloud.
    - pts2 (numpy.ndarray): The second point cloud.
    - num_samples (int, optional): The number of points to randomly sample from each point cloud. If None, all points are used.

    Returns:
    - d1 (numpy.ndarray): The distances from each point in pts1 to its nearest neighbor in pts2.
    - d2 (numpy.ndarray): The distances from each point in pts2 to its nearest neighbor in pts1.
    """

    if num_samples is not None:
        pts1, pts2 = get_rand_samples(pts1, pts2, num_samples)

    kd1 = cKDTree(pts1)
    kd2 = cKDTree(pts2)

    d1, _ = kd1.query(pts2)
    d2, _ = kd2.query(pts1)

    return d1, d2


def compute_assd_between_point_clouds(
    pts1,
    pts2,
    num_samples=None,
):
    """
    Compute the average symmetric surface distance (ASSD) between two point clouds.

    Args:
    - pts1 (numpy.ndarray): The first point cloud.
    - pts2 (numpy.ndarray): The second point cloud.
    - num_samples (int, optional): The number of points to randomly sample from each point cloud. If None, all points are used.

    Returns:
    - assd (float): The average symmetric surface distance between the two point clouds.
    """
    d1, d2 = get_pt_cloud_distances(pts1, pts2, num_samples)

    return (np.sum(d1) + np.sum(d2)) / (pts1.shape[0] + pts2.shape[0])


def decimate_mesh_pcu(mesh, percent_orig_faces=0.5):
    # get faces and points
    faces, points = get_faces_vertices(mesh)

    print(type(points), points.shape)
    print(type(faces), faces.shape)

    points_, faces_, corr_qv, corr_qf = pcu.decimate_triangle_mesh(
        points, faces, max_faces=int(faces.shape[0] * percent_orig_faces)
    )

    print(type(points_), points_.shape)
    print(type(faces_), faces_.shape)

    faces_ = np.hstack((np.ones((faces_.shape[0], 1)) * 3, faces_))

    new_mesh = pv.PolyData(points_, faces_.astype(int))

    return new_mesh


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

    edge_lengths = np.sqrt(
        np.sum(np.square(points[edges[:, 0], :] - points[edges[:, 1], :]), axis=1)
    )

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
