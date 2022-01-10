import vtk

# Some functions were originally based on the tutorial on ray casting in python + vtk 
# by Adamos Kyriakou @:
# https://pyscience.wordpress.com/2014/09/21/ray-casting-with-python-and-vtk-intersecting-linesrays-with-surface-meshes/
#
#

def is_hit(obb_tree, source, target):
    """
    Return True if line intersects mesh (`obb_tree`). The line starts at `source` and ends at `target`.

    Parameters
    ----------
    obb_tree
    source
    target

    Returns
    -------

    """
    "Returns True if the line intersects with the mesh in 'obbTree'"
    code = obb_tree.IntersectWithLine(source, target, None, None)
    if code == 0:
        return False
    else: 
        return True


def get_intersect(obbTree, pSource, pTarget):
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

    assert (n_points == n_Ids)

    # Loop through the found points and cells and store
    # them in lists
    points_inter = []
    cell_ids_inter = []
    for idx in range(n_points):
        points_inter.append(point_data.GetTuple3(idx))
        cell_ids_inter.append(cell_ids.GetId(idx))

    return points_inter, cell_ids_inter


def get_surface_normals(surface,
                        point_normals_on=True,
                        cell_normals_on=True):
    """
    Get the surface normals of a mesh (`surface`)

    Parameters
    ----------
    surface
    point_normals_on
    cell_normals_on

    Returns
    -------

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
    Get obb of a surface mesh. This can be queried to see if a line etc. intersects a surface.

    Parameters
    ----------
    surface

    Returns
    -------

    """
    obb = vtk.vtkOBBTree()
    obb.SetDataSet(surface)
    obb.BuildLocator()
    return obb


def vtk_deep_copy(mesh):
    new_mesh = vtk.vtkPolyData()
    new_mesh.DeepCopy(mesh)
    return new_mesh
