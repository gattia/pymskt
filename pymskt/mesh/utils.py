import numpy as np
import pyvista as pv
import vtk
from matplotlib import pyplot as plt
from vtk.util.numpy_support import numpy_to_vtk, vtk_to_numpy

import pymskt
from pymskt.utils import sigma2fwhm

# Some functions were originally based on the tutorial on ray casting in python + vtk
# by Adamos Kyriakou @:
# https://pyscience.wordpress.com/2014/09/21/ray-casting-with-python-and-vtk-intersecting-linesrays-with-surface-meshes/


def is_hit(obb_tree, source, target):
    """
    Return True if line intersects mesh (`obb_tree`). The line starts at `source` and ends at `target`.

    Parameters
    ----------
    obb_tree : vtk.vtkOBBTree
        OBBTree of a surface mesh.
    source : list
        x/y/z position of starting point of ray (to find intersection)
    target : list
        x/y/z position of ending point of ray (to find intersection)

    Returns
    -------
    bool
        Telling if the line (source to target) intersects the obb_tree.
    """

    code = obb_tree.IntersectWithLine(source, target, None, None)
    if code == 0:
        return False
    else:
        return True


def get_intersect(obbTree, pSource, pTarget):
    """
    Get intersecting points on the obbTree between a line from pSource to pTarget.

    Parameters
    ----------
    obb_tree : vtk.vtkOBBTree
        OBBTree of a surface mesh.
    pSource : list
        x/y/z position of starting point of ray (to find intersection)
    pTarget : list
        x/y/z position of ending point of ray (to find intersection)

    Returns
    -------
    tuple (list1, list2)
        list1 is of the intersection points
        list2 is the idx of the cells that were intersected.
    """
    # Create an empty 'vtkPoints' object to store the intersection point coordinates
    points = vtk.vtkPoints()
    # Create an empty 'vtkIdList' object to store the ids of the cells that intersect
    # with the cast rays
    cell_ids = vtk.vtkIdList()

    # Perform intersection
    code = obbTree.IntersectWithLine(pSource, pTarget, points, cell_ids)

    # Get point-data
    point_data = points.GetData()
    # Get number of intersection points found
    n_points = point_data.GetNumberOfTuples()
    # Get number of intersected cell ids
    n_Ids = cell_ids.GetNumberOfIds()

    assert n_points == n_Ids

    # Loop through the found points and cells and store
    # them in lists
    points_inter = []
    cell_ids_inter = []
    for idx in range(n_points):
        points_inter.append(point_data.GetTuple3(idx))
        cell_ids_inter.append(cell_ids.GetId(idx))

    return points_inter, cell_ids_inter


def get_surface_normals(surface, point_normals_on=True, cell_normals_on=True):
    """
    Get the surface normals of a mesh (`surface`

    Parameters
    ----------
    surface : vtk.vtkPolyData
        surface mesh to get normals from
    point_normals_on : bool, optional
        Whether or not to get normals of points (vertices), by default True
    cell_normals_on : bool, optional
        Whether or not to get normals from cells (faces?), by default True

    Returns
    -------
    vtk.vtkPolyDataNormals
        Normval vectors for points/cells.
    """

    normals = vtk.vtkPolyDataNormals()
    normals.SetInputData(surface)

    # Disable normal calculation at cell vertices
    if point_normals_on is True:
        normals.ComputePointNormalsOn()
    elif point_normals_on is False:
        normals.ComputePointNormalsOff()
    # Enable normal calculation at cell centers
    if cell_normals_on is True:
        normals.ComputeCellNormalsOn()
    elif cell_normals_on is False:
        normals.ComputeCellNormalsOff()
    # Disable splitting of sharp edges
    normals.SplittingOff()
    # Disable global flipping of normal orientation
    normals.FlipNormalsOff()
    # Enable automatic determination of correct normal orientation
    normals.AutoOrientNormalsOn()
    # Perform calculation
    normals.Update()

    return normals


def get_obb_surface(surface):
    """
    Get vtk.vtkOBBTree for a surface mesh
    Get obb of a surface mesh. This can be queried to see if a line etc. intersects a surface.

    Parameters
    ----------
    surface : vtk.vtkPolyData
        The surface mesh to get an OBBTree for.

    Returns
    -------
    vtk.vtkOBBTree
        The OBBTree to be used to find intersections for calculating cartilage thickness etc.
    """

    obb = vtk.vtkOBBTree()
    obb.SetDataSet(surface)
    obb.BuildLocator()
    return obb


def vtk_deep_copy(mesh):
    """
    "Deep" copy a vtk.vtkPolyData so that they are not connected in any way.

    Parameters
    ----------
    mesh : vtk.vtkPolyData
        Mesh to copy.

    Returns
    -------
    vtk.vtkPolyData
        Copy of the input mesh.
    """
    new_mesh = vtk.vtkPolyData()
    new_mesh.DeepCopy(mesh)
    return new_mesh


def estimate_mesh_scalars_FWHMs(mesh, scalar_name="thickness_mm"):
    """
    Calculate the Full Width Half Maximum (FWHM) based on surface mesh scalars.

    Parameters
    ----------
    mesh : vtk.vtkPolyData
        Surface mesh to estimate FWHM of the scalars from.
    scalar_name : str, optional
        Name of the scalars to calcualte FWHM for, by default 'thickness_mm'

    Returns
    -------
    list
        List of the FWHM values. Assuming they are for X/Y/Z
    """
    gradient_filter = vtk.vtkGradientFilter()
    gradient_filter.SetInputData(mesh)
    gradient_filter.Update()
    gradient_mesh = vtk.vtkPolyData()
    gradient_mesh.DeepCopy(gradient_filter.GetOutput())

    scalars = vtk_to_numpy(mesh.GetPointData().GetScalars())
    location_non_zero = np.where(scalars != 0)
    gradient_scalars = vtk_to_numpy(gradient_mesh.GetPointData().GetAbstractArray("Gradients"))
    cartilage_gradients = gradient_scalars[location_non_zero, :][0]

    thickness_scalars = vtk_to_numpy(gradient_mesh.GetPointData().GetAbstractArray(scalar_name))
    cartilage_thicknesses = thickness_scalars[location_non_zero]

    V0 = np.mean((cartilage_thicknesses - np.mean(cartilage_thicknesses)) ** 2)
    V1 = np.mean((cartilage_gradients - np.mean(cartilage_gradients)) ** 2, axis=0)
    sigma2s = -1 / (4 * np.log(1 - (V1 / (2 * V0))))
    sigmas = np.sqrt(sigma2s)
    FWHMs = [sigma2fwhm(x) for x in sigmas]

    return FWHMs


def get_surface_distance(surface_1, surface_2, return_RMS=True, return_individual_distances=False):
    if (return_RMS is True) & (return_individual_distances is True):
        raise Exception(
            "Nothing to return - either return_RMS or return_individual_distances must be `True`"
        )

    pt_locator = vtk.vtkPointLocator()
    pt_locator.SetDataSet(surface_2)
    pt_locator.AutomaticOn()
    pt_locator.BuildLocator()

    distances = np.zeros(surface_1.GetNumberOfPoints())

    for pt_idx in range(surface_1.GetNumberOfPoints()):
        point_1 = np.asarray(surface_1.GetPoint(pt_idx))
        pt_idx_2 = pt_locator.FindClosestPoint(point_1)
        point_2 = np.asarray(surface_2.GetPoint(pt_idx_2))
        distances[pt_idx] = np.sqrt(np.sum(np.square(point_2 - point_1)))

    RMS = np.sqrt(np.mean(np.square(distances)))

    if return_individual_distances is True:
        if return_RMS is True:
            return RMS, distances
        else:
            return distances
    else:
        if return_RMS is True:
            return RMS


def get_symmetric_surface_distance(surface_1, surface_2):
    surf1_to_2_distances = get_surface_distance(
        surface_1, surface_2, return_RMS=False, return_individual_distances=True
    )
    surf2_to_1_distances = get_surface_distance(
        surface_2, surface_1, return_RMS=False, return_individual_distances=True
    )

    symmetric_distance = (np.sum(surf1_to_2_distances) + np.sum(surf2_to_1_distances)) / (
        len(surf1_to_2_distances) + len(surf2_to_1_distances)
    )

    return symmetric_distance


class GIF:
    """
    Class for generating GIF of surface meshes.

    Parameters
    ----------
    plotter : pyvista.Plotter
        Plotter to use for plotting.
    color: str, optional
        Color to use for object, by default 'orange'
    show_edges: bool, optional
        Whether to show edges on mesh, by default True
    edge_color: str, optional
        Color to use for edges, by default 'black'
    camera_position: list or string, optional
        Camera position to use, by default 'xz'
    window_size: list, optional
        Window size to use for GIF, by default [3000, 4000]
    background_color: str, optional
        Background color to use, by default 'white'
    path_save: str, optional
        Path to save GIF, by default '~/Downloads/ssm.gif'

    Attributes
    ----------
    _plotter : pyvista.Plotter
        Plotter to use for plotting.
    _color : str
        Color to use for object.
    _show_edges : bool
        Whether to show edges on mesh.
    _edge_color : str
        Color to use for edges.
    _camera_position : list or string
        Camera position to use.
    _window_size : list
        Window size to use for GIF.
    _background_color : str
        Background color to use.
    _path_save : str
        Path to save GIF.

    Methods
    -------
    add_mesh_frame(mesh)
        Add a mesh to the GIF.
    update_view()
        Update the view of the plotter.
    done()
        Close the plotter.


    """

    def __init__(
        self,
        plotter=None,
        color="orange",
        show_edges=True,
        edge_color="black",
        camera_position="xz",
        window_size=[3000, 4000],
        background_color="white",
        path_save="~/Downloads/ssm.gif",
        scalar_bar_range=[0, 4],
        cmap="viridis",
        lighting=False,
    ):
        """
        Initialize the GIF class.

        Parameters
        ----------
        plotter : pyvista.Plotter, optional
            Plotter to use for plotting, by default None
        color: str, optional
            Color to use for object, by default 'orange'
        show_edges: bool, optional
            Whether to show edges on mesh, by default True
        edge_color: str, optional
            Color to use for edges, by default 'black'
        camera_position: list or string, optional
            Camera position to use, by default 'xz'
        window_size: list, optional
            Window size to use for GIF, by default [3000, 4000]
        background_color: str, optional
            Background color to use, by default 'white'
        path_save: str, optional
            Path to save GIF, by default '~/Downloads/ssm.gif'

        """
        if plotter is None:
            self._plotter = pv.Plotter(notebook=False, off_screen=True)
        else:
            self._plotter = plotter

        if path_save[-3:] != "gif":
            raise Exception("path must be to a file ending with suffix `.gif`")

        self.counter = 0

        self._plotter.open_gif(path_save)

        self._color = color
        self._show_edges = show_edges
        self._edge_color = edge_color
        self._camera_position = camera_position
        self._window_size = window_size
        self._background_color = background_color
        self._path_save = path_save
        self._scalar_bar_range = scalar_bar_range
        self._cmap = plt.cm.get_cmap(cmap)
        self._lighting = lighting

    def update_view(self):
        self._plotter.camera_position = self._camera_position
        self._plotter.window_size = self._window_size
        self._plotter.set_background(color=self._background_color)

    def add_mesh_frame(self, mesh):
        if type(mesh) in (list, tuple):
            actors = []
            for mesh_ in mesh:
                actors.append(
                    self._plotter.add_mesh(
                        mesh_,
                        render=False,
                        color=self._color,
                        edge_color=self._edge_color,
                        show_edges=self._show_edges,
                        cmap=self._cmap,
                        clim=self._scalar_bar_range,
                        lighting=self._lighting,
                        n_colors=1000,
                        # pbr=True,
                    )
                )
        else:
            actor = self._plotter.add_mesh(
                mesh,
                render=False,
                color=self._color,
                edge_color=self._edge_color,
                show_edges=self._show_edges,
                cmap=self._cmap,
                clim=self._scalar_bar_range,
                lighting=False,
                # n_colors=1000,
                # pbr=True,
            )

        if self.counter == 0:
            self.update_view()
        # self._plotter.update_scalar_bar_range(clim=self._scalar_bar_range)
        self._plotter.write_frame()

        if type(mesh) in (list, tuple):
            for actor in actors:
                self._plotter.remove_actor(actor)
        else:
            self._plotter.remove_actor(actor)
        self.counter += 1

    def done(self):
        self._plotter.close()

    @property
    def scalar_bar_range(self):
        return self._scalar_bar_range

    @scalar_bar_range.setter
    def scalar_bar_range(self, scalar_bar_range):
        self._scalar_bar_range = scalar_bar_range
        self._plotter.update_scalar_bar_range(clim=scalar_bar_range)

    @property
    def color(self):
        return self._color

    @color.setter
    def color(self, color):
        self._color = color

    @property
    def show_edges(self):
        return self._show_edges

    @show_edges.setter
    def show_edges(self, show_edges):
        self._show_edges = show_edges

    @property
    def edge_color(self):
        return self._edge_color

    @edge_color.setter
    def edge_color(self, edge_color):
        self._edge_color = edge_color

    @property
    def camera_position(self):
        return self._camera_position

    @camera_position.setter
    def camera_position(self, camera_position):
        self._camera_position = camera_position

    @property
    def window_size(self):
        return self._window_size

    @window_size.setter
    def window_size(self, window_size):
        self._window_size = window_size

    @property
    def background_color(self):
        return self._background_color

    @background_color.setter
    def background_color(self, background_color):
        self._background_color = background_color

    @property
    def path_save(self):
        return self._path_save


def get_arrow(
    direction,
    origin,
    scale=100,
    tip_length=0.25,
    tip_radius=0.1,
    tip_resolution=20,
    shaft_radius=0.05,
    shaft_resolution=20,
):
    arrow = vtk.vtkArrowSource()
    arrow.SetTipLength(tip_length)
    arrow.SetTipRadius(tip_radius)
    arrow.SetTipResolution(tip_resolution)
    arrow.SetShaftRadius(shaft_radius)
    arrow.SetShaftResolution(shaft_resolution)
    arrow.Update()

    arrow = arrow.GetOutput()
    points = arrow.GetPoints().GetData()
    array = vtk_to_numpy(points)
    array *= scale
    arrow.GetPoints().SetData(numpy_to_vtk(array))

    normx = np.array(direction) / np.linalg.norm(direction)
    normz = np.cross(normx, [0, 1.0, 0.0001])
    normz /= np.linalg.norm(normz)
    normy = np.cross(normz, normx)

    four_by_four = np.identity(4)
    four_by_four[:3, 0] = normx
    four_by_four[:3, 1] = normy
    four_by_four[:3, 2] = normz
    four_by_four[:3, 3] = origin

    transform = pymskt.mesh.meshTransform.create_transform(four_by_four)
    arrow = pymskt.mesh.meshTransform.apply_transform(arrow, transform)

    return arrow
