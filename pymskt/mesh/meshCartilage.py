import numpy as np
import pyvista as pv
import SimpleITK as sitk
import vtk


# extract the articular surfaces from the cartilages
def remove_intersecting_vertices(mesh1, mesh2, ray_length=1.0, overlap_buffer=0.1):
    """
    This function takes in two meshes: mesh1 and mesh2.
    Rays are cast from each vertex of mesh1 in the negative direction of the normal to the surface of mesh1.
    If a ray intersects mesh2, the vertex from which the ray was cast is marked for removal.
    A version of mesh1 with the marked vertices removed is returned.

    Parameters:
    - ray_length: The length of the ray. Default is 1.0.
    - invert: If True, the vertices marked for removal are kept and the rest are removed.
              If False, the vertices marked for removal are removed and the rest are kept.
              Default is True.
    """

    # Compute point normals for mesh1
    mesh1_normals = mesh1.compute_normals(point_normals=True, cell_normals=False)

    vertex_mask = np.ones(mesh1.n_points, dtype=bool)

    for idx, (vertex, normal) in enumerate(zip(mesh1.points, mesh1_normals.point_data["Normals"])):
        start_point = vertex - overlap_buffer * normal
        end_point = vertex - ray_length * normal

        intersections = mesh2.ray_trace(start_point, end_point)

        # If there's any intersection
        if len(intersections[1]) > 0:
            vertex_mask[idx] = False

    print(f"number of intersections: {sum(~vertex_mask)}")

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
    Mesh or pyvista.PolyData: The cleaned cartilage mesh (type matches input)
    """

    # Check and convert input types
    def convert_to_mesh(mesh, mesh_name):
        from pymskt.mesh import Mesh

        if isinstance(mesh, Mesh):
            return mesh, "Mesh"
        elif isinstance(mesh, pv.PolyData):
            return Mesh(mesh), "pyvista"
        elif isinstance(mesh, vtk.vtkPolyData):
            return Mesh(pv.PolyData(mesh)), "vtk"
        else:
            raise TypeError(
                f"The {mesh_name} is not of type pymskt.mesh.Mesh, pyvista.PolyData, or vtk.vtkPolyData"
            )

    cartilage_mesh, cart_type = convert_to_mesh(cartilage_mesh, "cartilage mesh")
    bone_mesh, bone_type = convert_to_mesh(bone_mesh, "bone mesh")

    # Ensure both meshes have the same dtype (use the higher precision one)
    target_dtype = np.promote_types(cartilage_mesh.point_coords.dtype, bone_mesh.point_coords.dtype)
    cartilage_mesh.point_coords = cartilage_mesh.point_coords.astype(target_dtype)
    bone_mesh.point_coords = bone_mesh.point_coords.astype(target_dtype)

    # Create a copy of the cartilage mesh
    cart_copy = cartilage_mesh.copy()
    # cart_copy.mesh = pv.PolyData(cart_copy.mesh)

    # Calculate the surface error
    cart_copy.calc_surface_error(bone_mesh)
    surf_error = cart_copy.get_scalar("surface_error")

    # Invert the surface error values
    cart_copy.set_scalar("surface_error", surf_error * -1)

    # Threshold the surface to keep only points outside the bone (surface_error > 0)
    cart_copy.mesh = cart_copy.mesh.threshold(
        0, scalars="surface_error", invert=True
    ).extract_surface()
    # Clean up the resulting mesh
    cart_copy.mesh = cart_copy.mesh.clean()

    # Return the appropriate type
    if cart_type == "Mesh":
        return cart_copy
    else:  # 'pyvista' or 'vtk'
        return cart_copy


def remove_isolated_cells(input_mesh):
    """
    Remove isolated cells from a mesh that have only one edge neighbor.

    Parameters:
    -----------
    input_mesh : Mesh, pyvista.PolyData, or vtk.vtkPolyData
        The input mesh to clean.

    Returns:
    --------
    cleaned_mesh : Same type as input_mesh or pyvista.PolyData
        The cleaned mesh with isolated cells removed.
    """
    # Type checking and conversion
    from pymskt.mesh import Mesh

    if isinstance(input_mesh, Mesh):
        mesh = pv.PolyData(input_mesh.mesh)
        return_type = "Mesh"
    elif isinstance(input_mesh, pv.PolyData):
        mesh = input_mesh.copy()
        return_type = "pyvista"
    elif isinstance(input_mesh, vtk.vtkPolyData):
        mesh = pv.PolyData(input_mesh)
        return_type = "pyvista"
    else:
        raise TypeError("Input mesh must be of type Mesh, pyvista.PolyData, or vtk.vtkPolyData")

    n_cells_removed = 1
    while n_cells_removed > 0:
        n_cells = mesh.n_cells
        cell_mask = np.ones(n_cells, dtype=bool)

        for i in range(n_cells):
            cell_neighbors = mesh.cell_neighbors(i, connections="edges")
            if len(cell_neighbors) == 1:
                cell_mask[i] = False

        mesh.cell_data["cell_mask"] = cell_mask
        mesh = mesh.threshold(0.5, scalars="cell_mask", invert=False).extract_surface()
        n_cells_removed = n_cells - mesh.n_cells

    # clean the mesh
    mesh = mesh.clean()

    # Return the cleaned mesh in the appropriate type

    if return_type == "Mesh":
        cleaned_mesh = Mesh()
        cleaned_mesh.mesh = mesh
    else:
        cleaned_mesh = mesh

    return cleaned_mesh


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

    for cart_mesh in bone_mesh.list_cartilage_meshes:
        print(cart_mesh.point_coords.shape)
        print(bone_mesh.point_coords.shape)
        articular_surface = remove_intersecting_vertices(
            pv.PolyData(cart_mesh.mesh),
            pv.PolyData(bone_mesh.mesh),
            ray_length=ray_length,
        )
        assert isinstance(
            articular_surface, pv.PolyData
        ), f"articular_surface is not a PolyData object: {type(articular_surface)}"

        articular_surface = get_n_largest(articular_surface, n=n_largest)
        if not isinstance(articular_surface, pv.PolyData):
            articular_surface = articular_surface.extract_surface()
        assert isinstance(
            articular_surface, pv.PolyData
        ), f"articular_surface is not a PolyData object: {type(articular_surface)}"

        # remove articular surface points that are inside the bone
        articular_surface = remove_cart_in_bone(articular_surface, bone_mesh)
        # remove isolated cells at the boundaries
        articular_surface = remove_isolated_cells(articular_surface)

        # smooth the articular surface...
        #   boundary_smoothing=False will enable smoothing at the boundary - which can fix
        #   some of the issues with errors at the edges (boundaries)
        articular_surface = articular_surface.smooth(n_iter=smooth_iter, boundary_smoothing=False)

        list_articular_surfaces.append(articular_surface)

    return list_articular_surfaces


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
):
    """
    Break the cartilage into superficial and deep regions based on the relative depth
    from the bone surface.

    Parameters:
    -----------
    bone_mesh : pymskt.mesh.Mesh
        The bone mesh to extract the articular surface from.
    seg_image : SimpleITK.Image, optional
        The segmentation image to break into superficial and deep regions, by default None.
    list_cartilage_labels : list of int, optional
        The labels of the cartilage to break into superficial and deep regions, by default None.
    rel_depth_thresh : float, optional
        The relative depth threshold to break the cartilage into superficial and deep regions, by default 0.5.
    resample_cartilage_surface : int, optional
        The number of points to resample the cartilage surface to, by default 10_000 (speeds up the process).
    return_rel_depth : bool, optional
        Whether to return the relative depth, by default False.
    deep_label : int, optional
        The label to assign to the deep regions, by default 100.
    superficial_label : int, optional
        The label to assign to the superficial regions, by default 200.
    """
    from pymskt.mesh import Mesh

    bone_mesh.compute_normals(auto_orient_normals=True, inplace=True)

    # the seg_image might be in the bone_mesh, or provided as input. Check, and raise
    # errors if its not provided.
    if bone_mesh.seg_image is None:
        if seg_image is None:
            raise ValueError("seg_image is not provided and not in bone_mesh")
    else:
        seg_image = bone_mesh.seg_image

    # make sure the seg_image is actually a SimpleITK image so we can properly
    # place it in 3D space for extracting voxel locations.
    assert isinstance(
        seg_image, sitk.Image
    ), f"seg_image is not a SimpleITK image: {type(seg_image)}"

    # make sure that the list_cartilage_labels is provided somewhere, either
    # directly or in the bone_mesh, or as an input argument.
    if bone_mesh.list_cartilage_labels is None:
        if list_cartilage_labels is None:
            raise ValueError("list_cartilage_labels is not provided and not in bone_mesh")
    else:
        list_cartilage_labels = bone_mesh.list_cartilage_labels

    # if the cartilage meshes don't exist yet, create them.
    if bone_mesh.list_cartilage_meshes is None:
        bone_mesh.create_cartilage_meshes()

    orig_cart_meshes = [cart_mesh_.copy() for cart_mesh_ in bone_mesh.list_cartilage_meshes]

    # if the cartilage surfaces are not resampled yet, and we want them to be,
    # do that now.
    if resample_cartilage_surface is not None:
        for cartilage_mesh in bone_mesh.list_cartilage_meshes:
            cartilage_mesh.resample_surface(clusters=resample_cartilage_surface)

    # fix normals of cartilage mesh & fix mesh
    for cart_mesh in bone_mesh.list_cartilage_meshes:
        cart_mesh.fix_mesh()
        cart_mesh.compute_normals(auto_orient_normals=True, inplace=True)

    # if the articular surfaces don't exist yet, create them.
    if bone_mesh.list_articular_surfaces is None:
        bone_mesh.extract_articular_surfaces()

    # re-assign the full resolution cartilage meshes to the bone_mesh object.
    bone_mesh.list_cartilage_meshes = orig_cart_meshes

    # convert the seg_image to a numpy array, and get the voxel locations of the
    # cartilage labels.
    seg_arr = sitk.GetArrayFromImage(seg_image)
    voxel_coords = []
    for label in list_cartilage_labels:
        voxel_coords.append(np.array(np.where(seg_arr == label)).T)
    voxel_coords = np.concatenate(voxel_coords, axis=0)

    # create the transform to apply to the voxels to place them in 3D space.
    origin = np.array(seg_image.GetOrigin())
    rotation_matrix = np.array(seg_image.GetDirection()).reshape(3, 3)
    scale = np.array(seg_image.GetSpacing())

    transform = np.eye(4)

    transform[:3, :3] = rotation_matrix
    transform[:3, :3] *= scale
    transform[:3, 3] = origin

    # swap the axes so that the first axis is the z-axis, and the last axis is the
    # x-axis. This is the orientation of the image.
    voxel_coords_image = voxel_coords[:, ::-1]

    # pad the voxel coordinates with ones to allow use with 4x4 transform
    # that includes translation.
    voxel_coords_image = np.hstack([voxel_coords_image, np.ones((voxel_coords_image.shape[0], 1))])

    # transform these coordinates to world space
    voxel_coords_world = np.ascontiguousarray(transform @ voxel_coords_image.T).T[:, :3]

    # get the SDFs for the cartilage coordinates from the bone and articular surfaces.
    articular_surfaces = bone_mesh.list_articular_surfaces
    articular_cart_distances = []
    for surface in articular_surfaces:
        articular_cart_distances.append(
            abs(Mesh(surface).get_sdf_pts(voxel_coords_world, method=sdf_method))
        )
    articular_cart_distances = np.min(articular_cart_distances, axis=0)

    bone_distance = bone_mesh.get_sdf_pts(voxel_coords_world, method=sdf_method)
    # bone_distance[bone_distance < 0] = 0 # Clip negative distances (inside bone) to 0

    # === DEBUG: Log distance statistics ===
    try:
        # Use standard logging for library code
        import logging

        lib_logger = logging.getLogger("pymskt.mesh.meshCartilage")

        finite_bone_dist = bone_distance[np.isfinite(bone_distance)]
        if finite_bone_dist.size > 0:
            lib_logger.debug(
                f"Bone distance stats: Min={np.min(finite_bone_dist):.4f}, Max={np.max(finite_bone_dist):.4f}, Mean={np.mean(finite_bone_dist):.4f}, Median={np.median(finite_bone_dist):.4f}"
            )
        else:
            lib_logger.warning("No finite bone distance values found.")

        finite_articular_dist = articular_cart_distances[np.isfinite(articular_cart_distances)]
        if finite_articular_dist.size > 0:
            lib_logger.debug(
                f"Articular distance stats: Min={np.min(finite_articular_dist):.4f}, Max={np.max(finite_articular_dist):.4f}, Mean={np.mean(finite_articular_dist):.4f}, Median={np.median(finite_articular_dist):.4f}"
            )
        else:
            lib_logger.warning("No finite articular distance values found.")

    except Exception as dbg_e:
        lib_logger.error(f"Error calculating distance stats: {dbg_e}")
    # === END DEBUG ===

    rel_depth = bone_distance / (bone_distance + articular_cart_distances)

    # print the dype of bone_distance and articular_cart_distances
    print(f"bone_distance dtype: {bone_distance.dtype}")
    print(f"articular_cart_distances dtype: {articular_cart_distances.dtype}")
    print(f"rel_depth dtype: {rel_depth.dtype}")

    # === DEBUG: Log relative depth statistics ===
    try:
        # Ignore potential NaNs or Infs from division by zero if distances are zero
        finite_rel_depth = rel_depth[np.isfinite(rel_depth)]
        if finite_rel_depth.size > 0:
            min_rd = np.min(finite_rel_depth)
            max_rd = np.max(finite_rel_depth)
            mean_rd = np.mean(finite_rel_depth)
            median_rd = np.median(finite_rel_depth)
            # Use standard logging for library code
            lib_logger = logging.getLogger("pymskt.mesh.meshCartilage")
            lib_logger.debug(
                f"Relative depth stats: Min={min_rd:.4f}, Max={max_rd:.4f}, Mean={mean_rd:.4f}, Median={median_rd:.4f}"
            )
        else:
            lib_logger.warning("No finite relative depth values found.")
    except Exception as dbg_e:
        lib_logger.error(f"Error calculating relative depth stats: {dbg_e}")
    # === END DEBUG ===

    # combine the existing seg labels into a single label
    # then break that into superficial and deep based on the rel_depth threshold.

    new_seg_array = np.zeros_like(
        seg_arr, dtype=np.uint16
    )  # set to uint16 to avoid overflow (many segs are int8 causing an issue)

    # break the combined label into superficial and deep based on the rel_depth threshold.
    deep_idx = voxel_coords[rel_depth < rel_depth_thresh].astype(int)
    superficial_idx = voxel_coords[rel_depth >= rel_depth_thresh].astype(int)
    new_seg_array[deep_idx[:, 0], deep_idx[:, 1], deep_idx[:, 2]] = deep_label
    new_seg_array[superficial_idx[:, 0], superficial_idx[:, 1], superficial_idx[:, 2]] = (
        superficial_label
    )

    new_seg_image = sitk.GetImageFromArray(new_seg_array)
    new_seg_image.CopyInformation(seg_image)

    rel_depth = bone_distance
    rel_depth_array = np.zeros_like(seg_arr, dtype=np.float32)
    rel_depth_array[voxel_coords[:, 0], voxel_coords[:, 1], voxel_coords[:, 2]] = rel_depth
    rel_depth_image = sitk.GetImageFromArray(rel_depth_array)
    rel_depth_image.CopyInformation(seg_image)

    if return_rel_depth:
        return new_seg_image, rel_depth_image
    else:
        return new_seg_image
