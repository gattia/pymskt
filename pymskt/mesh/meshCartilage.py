import logging

import numpy as np
import pyvista as pv
import SimpleITK as sitk
import vtk

from pymskt.mesh.utils import as_mesh
from pymskt.utils import timed_stage

logger = logging.getLogger(__name__)


# extract the articular surfaces from the cartilages
def remove_intersecting_vertices(mesh1, mesh2, ray_length=1.0, overlap_buffer=0.1):
    """
    This function takes in two meshes: mesh1 and mesh2.
    Rays are cast from each vertex of mesh1 in the negative direction of the normal to the surface of mesh1.
    If a ray intersects mesh2, the vertex from which the ray was cast is marked for removal.
    A version of mesh1 with the marked vertices removed is returned.

    Parameters:
    - ray_length: The length of the ray. Default is 1.0.
    - overlap_buffer: Distance along the (negative) normal at which the ray starts, so that
      a vertex sitting exactly on mesh2 is not counted as an intersection. Default is 0.1.
    """

    # Compute point normals for mesh1
    mesh1.compute_normals(point_normals=True, cell_normals=False, inplace=True)

    # Build OBBTree ONCE for mesh2
    obb_tree = vtk.vtkOBBTree()
    obb_tree.SetDataSet(mesh2)
    obb_tree.BuildLocator()

    mesh1_points = mesh1.points
    mesh1_normals = mesh1.point_data["Normals"]
    n_points = len(mesh1_points)

    # Reusable VTK objects
    points_intersect = vtk.vtkPoints()
    cell_ids = vtk.vtkIdList()

    vertex_mask = np.ones(n_points, dtype=bool)

    for idx in range(n_points):
        vertex = mesh1_points[idx]
        normal = mesh1_normals[idx]
        start_point = vertex - overlap_buffer * normal
        end_point = vertex - ray_length * normal

        # Clear and reuse
        points_intersect.Reset()
        cell_ids.Reset()

        # Ray trace
        obb_tree.IntersectWithLine(start_point, end_point, points_intersect, cell_ids)

        if points_intersect.GetNumberOfPoints() > 0:
            vertex_mask[idx] = False

    logger.debug(
        "remove_intersecting_vertices: %d of %d vertices intersect", (~vertex_mask).sum(), n_points
    )

    # Use the mask to filter out the vertices and the associated cells from mesh1
    mesh1.point_data["vertex_mask"] = vertex_mask
    cleaned_mesh = mesh1.threshold(0.5, scalars="vertex_mask", invert=True)

    return cleaned_mesh.extract_surface()


def get_n_largest(surface, n=1):
    """
    Get the n largest regions from a surface mesh.

    Parameters:
    -----------
    surface : pyvista.PolyData
        The surface mesh to get the n largest regions from.
    n : int, optional
        The number of largest regions to get, by default 1.

    Returns:
    --------
    pyvista.PolyData
        The surface mesh with the n largest regions.
    """
    subregions = surface.connectivity("all")
    unique_regions = np.unique(subregions["RegionId"])
    # getting the first "n" because the outputs are sorted by # of cells
    # assume all cells are ~ the same size, therefore largest # cells ~= largest areas
    largest_n = unique_regions[:n]

    assert isinstance(surface, pv.PolyData), f"surface is not a PolyData object: {type(surface)}"
    assert isinstance(
        subregions, pv.PolyData
    ), f"subregions is not a PolyData object: {type(subregions)}"

    return subregions.connectivity(extraction_mode="specified", variable_input=largest_n)


def remove_cart_in_bone(cartilage_mesh, bone_mesh):
    """
    Remove cartilage points that are inside the bone and clean up the resulting mesh.

    Args:
    cartilage_mesh (Mesh, pyvista.PolyData, or vtk.vtkPolyData): The articular surface mesh
    bone_mesh (Mesh, pyvista.PolyData, or vtk.vtkPolyData): The bone surface mesh

    Returns:
    Mesh: The cleaned cartilage mesh
    """
    cartilage_mesh = as_mesh(cartilage_mesh, "cartilage mesh")
    bone_mesh = as_mesh(bone_mesh, "bone mesh")

    # Ensure both meshes have the same dtype (use the higher precision one)
    target_dtype = np.promote_types(cartilage_mesh.point_coords.dtype, bone_mesh.point_coords.dtype)
    cartilage_mesh.point_coords = cartilage_mesh.point_coords.astype(target_dtype)
    bone_mesh.point_coords = bone_mesh.point_coords.astype(target_dtype)

    # Signed distance from each cartilage point to the bone surface (negative = inside bone).
    cart_copy = cartilage_mesh.copy()
    cart_copy.calc_surface_error(bone_mesh)
    surf_error = cart_copy.get_scalar("surface_error")
    cart_copy.set_scalar("surface_error", surf_error * -1)

    # Keep only points outside the bone (surface_error > 0), then clean up.
    cleaned = cart_copy.threshold(0, scalars="surface_error", invert=True).extract_surface()
    cart_copy.deep_copy(cleaned.clean())

    return cart_copy


def _triangle_edge_neighbor_counts(faces, n_points):
    """
    Number of edge-neighbouring triangles for every triangle of an (n, 3) face array.

    Vectorized equivalent of ``len(mesh.cell_neighbors(i, connections="edges"))`` for every
    cell of a triangle mesh: every other triangle sharing an edge counts once per shared edge.
    """
    edges = np.sort(faces[:, [[0, 1], [1, 2], [2, 0]]].reshape(-1, 2), axis=1)
    keys = edges[:, 0].astype(np.int64) * n_points + edges[:, 1]
    _, inverse, counts = np.unique(keys, return_inverse=True, return_counts=True)
    return (counts[inverse] - 1).reshape(-1, 3).sum(axis=1)


def remove_isolated_cells(input_mesh):
    """
    Remove isolated cells from a mesh that have only one edge neighbor.

    Cells are removed iteratively until no cell has exactly one edge neighbour
    (removing a cell can leave its neighbour with a single neighbour). Non-triangle
    polygons are triangulated first.

    Parameters:
    -----------
    input_mesh : Mesh, pyvista.PolyData, or vtk.vtkPolyData
        The input mesh to clean.

    Returns:
    --------
    Mesh
        The cleaned mesh with isolated cells removed (a new object; the input is not modified).
    """
    from pymskt.mesh import Mesh

    # nothing below mutates `mesh`: every step returns a new object
    mesh = as_mesh(input_mesh, "input_mesh")

    if mesh.n_cells != mesh.n_faces_strict:
        # vertices / lines / strips are not polygons; keep only the polygon cells
        logger.warning(
            "remove_isolated_cells: dropping %d non-polygon cells",
            mesh.n_cells - mesh.n_faces_strict,
        )
        polys_only = pv.PolyData(mesh.points, faces=mesh.faces)
        polys_only.point_data.update(mesh.point_data)
        mesh = polys_only
    if not mesh.is_all_triangles:
        mesh = mesh.triangulate()

    n_removed_total = 0
    while True:
        faces = mesh.faces.reshape(-1, 4)[:, 1:]
        n_neighbors = _triangle_edge_neighbor_counts(faces, mesh.n_points)
        isolated = np.flatnonzero(n_neighbors == 1)
        if len(isolated) == 0:
            break
        n_removed_total += len(isolated)
        mesh = mesh.remove_cells(isolated, inplace=False)
    logger.debug("remove_isolated_cells: removed %d cells", n_removed_total)

    # clean the mesh (removes points orphaned by the cell removal)
    return Mesh(mesh.clean())


def extract_articular_surface(bone_mesh, ray_length=10.0, smooth_iter=100, n_largest=1):
    """
    Extract the articular surface from the cartilage meshes.

    Parameters:
    -----------
    bone_mesh : pymskt.mesh.Mesh
        The bone mesh to extract the articular surface from.
    ray_length : float, optional
        The length of the ray to cast from each vertex of the cartilage mesh, by default 10.0.
    smooth_iter : int, optional
        The number of iterations to smooth the articular surface, by default 100.
    n_largest : int, optional
        The number of largest regions to get, by default 1.
    """
    list_articular_surfaces = []

    bone_mesh.compute_normals(
        point_normals=True, cell_normals=False, auto_orient_normals=True, inplace=True
    )

    for cart_mesh in bone_mesh.list_cartilage_meshes:
        cart_mesh.compute_normals(
            point_normals=True, cell_normals=False, auto_orient_normals=True, inplace=True
        )
        logger.debug(
            "extract_articular_surface: cartilage %d pts, bone %d pts",
            cart_mesh.n_points,
            bone_mesh.n_points,
        )
        with timed_stage("  remove_intersecting_vertices (ray cast)", logger):
            articular_surface = remove_intersecting_vertices(
                cart_mesh,
                bone_mesh,
                ray_length=ray_length,
            )
        assert isinstance(
            articular_surface, pv.PolyData
        ), f"articular_surface is not a PolyData object: {type(articular_surface)}"

        with timed_stage("  get_n_largest", logger):
            articular_surface = get_n_largest(articular_surface, n=n_largest)
        if not isinstance(articular_surface, pv.PolyData):
            articular_surface = articular_surface.extract_surface()
        assert isinstance(
            articular_surface, pv.PolyData
        ), f"articular_surface is not a PolyData object: {type(articular_surface)}"

        # remove articular surface points that are inside the bone
        with timed_stage("  remove_cart_in_bone", logger):
            articular_surface = remove_cart_in_bone(articular_surface, bone_mesh)
        # remove isolated cells at the boundaries
        with timed_stage("  remove_isolated_cells", logger):
            articular_surface = remove_isolated_cells(articular_surface)

        # smooth the articular surface...
        #   boundary_smoothing=False will enable smoothing at the boundary - which can fix
        #   some of the issues with errors at the edges (boundaries)
        with timed_stage("  smooth", logger):
            articular_surface = articular_surface.smooth(
                n_iter=smooth_iter, boundary_smoothing=False
            )

        list_articular_surfaces.append(articular_surface)

    return list_articular_surfaces


def _label_voxel_coords(seg_arr, labels):
    """(n, 3) array-index coordinates (z, y, x) of every voxel whose value is in `labels`."""
    mask = np.isin(seg_arr, labels)
    return np.argwhere(mask)


def _voxel_coords_to_world(voxel_coords, image):
    """
    Convert (n, 3) numpy array-index coordinates (z, y, x) into physical (x, y, z)
    coordinates using the image origin / spacing / direction.
    """
    origin = np.array(image.GetOrigin())
    rotation_matrix = np.array(image.GetDirection()).reshape(3, 3)
    scale = np.array(image.GetSpacing())

    transform = np.eye(4)
    transform[:3, :3] = rotation_matrix * scale
    transform[:3, 3] = origin

    # numpy (z, y, x) -> image (x, y, z), padded with ones for the 4x4 transform
    coords_image = np.hstack([voxel_coords[:, ::-1], np.ones((voxel_coords.shape[0], 1))])
    return np.ascontiguousarray((transform @ coords_image.T).T[:, :3])


def _log_array_stats(name, values):
    if not logger.isEnabledFor(logging.DEBUG):
        return
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        logger.warning("No finite %s values found.", name)
        return
    logger.debug(
        "%s stats: Min=%.4f, Max=%.4f, Mean=%.4f, Median=%.4f (%d non-finite)",
        name,
        finite.min(),
        finite.max(),
        finite.mean(),
        np.median(finite),
        values.size - finite.size,
    )


def break_cartilage_into_superficial_deep(
    bone_mesh,
    seg_image=None,
    list_cartilage_labels=None,
    rel_depth_thresh=0.5,
    resample_cartilage_surface=10_000,
    return_rel_depth=False,
    deep_label=100,
    superficial_label=200,
    sdf_method="vtk",  # "pcu" or "vtk"
    cartilage_fix_method="pcu",
    resample_subdivisions=2,
):
    """
    Break the cartilage into superficial and deep regions based on the relative depth
    from the bone surface.

    For every cartilage voxel the relative depth is
    ``d_bone / (d_bone + d_articular)`` where ``d_bone`` is the distance to the bone surface
    and ``d_articular`` the distance to the (extracted) articular surface. Voxels with
    relative depth below ``rel_depth_thresh`` are labelled ``deep_label``, the rest
    ``superficial_label``.

    Parameters:
    -----------
    bone_mesh : pymskt.mesh.BoneMesh
        The bone mesh to extract the articular surface from.
    seg_image : SimpleITK.Image, optional
        The segmentation image to break into superficial and deep regions, by default None.
        Only used if ``bone_mesh.seg_image`` is None.
    list_cartilage_labels : list of int, optional
        The labels of the cartilage to break into superficial and deep regions, by default None.
        Only used if ``bone_mesh.list_cartilage_labels`` is None.
    rel_depth_thresh : float, optional
        The relative depth threshold to break the cartilage into superficial and deep regions, by default 0.5.
    resample_cartilage_surface : int, optional
        The number of points to resample the cartilage surface to before extracting the
        articular surface, by default 10_000. Only the (temporary) resampled copy is used for
        the articular surface extraction; ``bone_mesh.list_cartilage_meshes`` is left at full
        resolution. ``None`` disables resampling (much slower).
    resample_subdivisions : int, optional
        ``subdivisions`` passed to ``Mesh.resample_surface`` for the cartilage resampling,
        by default 2. Each subdivision quadruples the faces before clustering; 1 (or 0) is
        considerably faster for these dense meshes at a small cost in vertex uniformity.
    return_rel_depth : bool, optional
        Whether to also return the relative depth image, by default False. The image is
        float32 with ``d_bone / (d_bone + d_articular)`` at the cartilage voxels (0 at the
        bone, 1 at the articular surface; slightly negative where a voxel centre lies inside
        the bone surface) and 0 elsewhere.
    deep_label : int, optional
        The label to assign to the deep regions, by default 100.
    superficial_label : int, optional
        The label to assign to the superficial regions, by default 200.
    sdf_method : str, optional
        Backend for the voxel-to-bone distances, "vtk" or "pcu", by default "vtk". The
        voxel-to-articular-surface distances always use "vtk": the articular surfaces are
        open meshes and ``pcu.signed_distance_to_mesh`` returns wrong magnitudes for those.
    cartilage_fix_method : str or None, optional
        ``fix_method`` passed to ``BoneMesh.create_cartilage_meshes`` if the cartilage meshes
        do not exist yet, by default "pcu" (watertight remeshing; slow). Ignored if
        ``bone_mesh.list_cartilage_meshes`` already exists.

    Returns
    -------
    SimpleITK.Image
        Image with ``deep_label`` / ``superficial_label`` at the cartilage voxels and 0
        elsewhere (uint16).
    SimpleITK.Image, optional
        Relative depth image (see ``return_rel_depth``).
    """
    from pymskt.mesh import Mesh

    with timed_stage("bone compute_normals", logger):
        bone_mesh.compute_normals(auto_orient_normals=True, inplace=True)

    # the seg_image might be in the bone_mesh, or provided as input. Check, and raise
    # errors if its not provided.
    if bone_mesh.seg_image is not None:
        seg_image = bone_mesh.seg_image
    elif seg_image is None:
        raise ValueError("seg_image is not provided and not in bone_mesh")

    # make sure the seg_image is actually a SimpleITK image so we can properly
    # place it in 3D space for extracting voxel locations.
    assert isinstance(
        seg_image, sitk.Image
    ), f"seg_image is not a SimpleITK image: {type(seg_image)}"

    # make sure that the list_cartilage_labels is provided somewhere, either
    # directly or in the bone_mesh, or as an input argument.
    if bone_mesh.list_cartilage_labels is not None:
        list_cartilage_labels = bone_mesh.list_cartilage_labels
    elif list_cartilage_labels is None:
        raise ValueError("list_cartilage_labels is not provided and not in bone_mesh")

    # if the cartilage meshes don't exist yet, create them.
    with timed_stage("create_cartilage_meshes", logger):
        if bone_mesh.list_cartilage_meshes is None:
            bone_mesh.create_cartilage_meshes(fix_method=cartilage_fix_method)

    # if the articular surfaces don't exist yet, create them - from a resampled (lower
    # resolution) copy of the cartilage meshes because extraction cost scales with the
    # number of cartilage vertices. The full resolution meshes are restored afterwards.
    if bone_mesh.list_articular_surfaces is None:
        orig_cart_meshes = [cart_mesh_.copy() for cart_mesh_ in bone_mesh.list_cartilage_meshes]
        try:
            with timed_stage("resample cartilage surfaces", logger):
                if resample_cartilage_surface is not None:
                    for cartilage_mesh in bone_mesh.list_cartilage_meshes:
                        cartilage_mesh.resample_surface(
                            subdivisions=resample_subdivisions,
                            clusters=resample_cartilage_surface,
                        )

            # fix normals of cartilage mesh & fix mesh
            with timed_stage("fix cartilage meshes", logger):
                for cart_mesh in bone_mesh.list_cartilage_meshes:
                    cart_mesh.fix_mesh()
                    cart_mesh.compute_normals(auto_orient_normals=True, inplace=True)

            with timed_stage("extract_articular_surfaces", logger):
                bone_mesh.extract_articular_surfaces()
        finally:
            # re-assign the full resolution cartilage meshes to the bone_mesh object.
            bone_mesh.list_cartilage_meshes = orig_cart_meshes

    # voxel locations (numpy z, y, x order) of the cartilage labels, in world coordinates
    with timed_stage("voxel coordinates", logger):
        seg_arr = sitk.GetArrayFromImage(seg_image)
        voxel_coords = _label_voxel_coords(seg_arr, list_cartilage_labels)
        voxel_coords_world = _voxel_coords_to_world(voxel_coords, seg_image)
    logger.debug("%d cartilage voxels", len(voxel_coords))

    # get the distances for the cartilage coordinates from the bone and articular surfaces.
    # The articular surfaces are open meshes; only the vtk backend is correct for those.
    with timed_stage("articular surface distances (vtk)", logger):
        articular_cart_distances = np.min(
            [
                np.abs(Mesh(surface).get_sdf_pts(voxel_coords_world, method="vtk"))
                for surface in bone_mesh.list_articular_surfaces
            ],
            axis=0,
        )
    with timed_stage(f"bone distances ({sdf_method})", logger):
        bone_distance = bone_mesh.get_sdf_pts(voxel_coords_world, method=sdf_method)

    _log_array_stats("Bone distance", bone_distance)
    _log_array_stats("Articular distance", articular_cart_distances)

    with np.errstate(divide="ignore", invalid="ignore"):
        rel_depth = bone_distance / (bone_distance + articular_cart_distances)
    _log_array_stats("Relative depth", rel_depth)

    # combine the existing seg labels into a single label, then break that into
    # superficial and deep based on the rel_depth threshold. (Voxels with a non-finite
    # relative depth - both distances zero - are left as 0.)
    with timed_stage("label assignment", logger):
        # uint16 to avoid overflow (many segs are int8 causing an issue)
        new_seg_array = np.zeros_like(seg_arr, dtype=np.uint16)
        zyx = tuple(voxel_coords.T)
        new_labels = np.zeros(len(voxel_coords), dtype=np.uint16)
        new_labels[rel_depth < rel_depth_thresh] = deep_label
        new_labels[rel_depth >= rel_depth_thresh] = superficial_label
        new_seg_array[zyx] = new_labels

        new_seg_image = sitk.GetImageFromArray(new_seg_array)
        new_seg_image.CopyInformation(seg_image)

    if not return_rel_depth:
        return new_seg_image

    # relative depth at the cartilage voxels (0 at the bone surface, 1 at the articular
    # surface), 0 elsewhere; non-finite values (both distances zero) are written as 0.
    rel_depth_array = np.zeros_like(seg_arr, dtype=np.float32)
    rel_depth_array[zyx] = np.where(np.isfinite(rel_depth), rel_depth, 0.0)
    rel_depth_image = sitk.GetImageFromArray(rel_depth_array)
    rel_depth_image.CopyInformation(seg_image)

    return new_seg_image, rel_depth_image
