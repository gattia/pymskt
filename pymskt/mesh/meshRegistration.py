import sys

import cycpd
import pyvista as pv
import vtk

from pymskt.mesh.meshTools import transfer_mesh_scalars_get_weighted_average_n_closest
from pymskt.mesh.meshTransform import create_transform, get_linear_transform_matrix

try:
    import pyfocusr
except ModuleNotFoundError:
    print("pyfocusr not found")
    print("If you are not using the registration tools, you can ignore this message.")
    print("install pyfocusr as described in the README: https://github.com/gattia/pymskt")
    print("or visit the pyfocusr github repo: https://github.com/gattia/pyfocusr")

import numpy as np


def get_icp_transform(source, target, max_n_iter=1000, n_landmarks=1000, reg_mode="similarity"):
    """
    Get the Interative Closest Point (ICP) transformation from the `source` mesh to the
    `target` mesh.

    Parameters
    ----------
    source : vtk.vtkPolyData
        Source mesh that we want to transform onto the target mesh.
    target : vtk.vtkPolyData
        Target mesh that we want to transform the source mesh onto.
    max_n_iter : int, optional
        Max number of iterations for the registration algorithm to perform, by default 1000
    n_landmarks : int, optional
        How many landmarks to sample when determining distance between meshes &
        solving for the optimal transformation, by default 1000
    reg_mode : str, optional
        The type of registration to perform. The options are:
            - 'rigid': true rigid, translation only
            - 'similarity': rigid + equal scale
        by default 'similarity'

    Returns
    -------
    vtk.vtkIterativeClosestPointTransform
        The actual transform object after running the registration.
    """

    icp = vtk.vtkIterativeClosestPointTransform()
    icp.SetSource(source)
    icp.SetTarget(target)
    if reg_mode == "rigid":
        icp.GetLandmarkTransform().SetModeToRigidBody()
    elif reg_mode == "similarity":
        icp.GetLandmarkTransform().SetModeToSimilarity()
    icp.SetMaximumNumberOfIterations(max_n_iter)
    icp.StartByMatchingCentroidsOn()
    icp.SetMaximumNumberOfLandmarks(n_landmarks)
    icp.Modified()
    icp.Update()

    # there was a bug where the ICP transform was getting overwritten to identity
    # after it was applied once to a mesh. Converting to own base transform seems to fix this.
    transform_array = get_linear_transform_matrix(icp)
    transform = create_transform(transform_array)

    return transform


def cpd_register(
    target_mesh,
    source_mesh,
    reg_type="non_rigid",
    max_iterations=1_000,
    alpha=1,  # 0.1
    beta=7,  # 3
    num_eig=100,
    n_samples=1_000,
    icp_register_first=True,  # Get bones/objects into roughly the same alignment first
    icp_registration_mode="similarity",  # similarity = rigid + scaling (isotropic), ("rigid", "similarity", "affine")
    icp_reg_target_to_source=True,
    icp_iterations=100,
    icp_n_landmarks=1000,
    transfer_scalars=False,
    return_icp_transform=False,
):
    from pymskt.mesh import Mesh

    assert isinstance(target_mesh, Mesh), "target_mesh must be inherited from pymskt.mesh.Mesh"
    assert isinstance(source_mesh, Mesh), "source_mesh must be inherited from pymskt.mesh.Mesh"

    if icp_register_first is True:
        if icp_reg_target_to_source is True:
            icp_source = source_mesh.copy()
            icp_target = target_mesh.copy()
        else:
            icp_source = target_mesh.copy()
            icp_target = source_mesh.copy()

        print("icp_source")
        print(icp_source)
        print("icp_target")
        print(icp_target)

        icp_source = icp_source.rigidly_register(
            other_mesh=icp_target,
            as_source=True,
            apply_transform_to_mesh=True,
            return_transformed_mesh=True,
            return_transform=False,
            max_n_iter=icp_iterations,
            n_landmarks=icp_n_landmarks,
            reg_mode=icp_registration_mode,
        )

        print("icp_source")
        print(icp_source)

        if icp_reg_target_to_source is True:
            source_mesh = icp_source
        else:
            target_mesh = icp_source

    if n_samples is not None:
        rand_idxs_source = np.random.choice(source_mesh.points.shape[0], n_samples, replace=False)
        rand_idxs_target = np.random.choice(target_mesh.points.shape[0], n_samples, replace=False)

    if reg_type == "non_rigid":
        reg = cycpd.deformable_registration(
            X=np.asarray(
                target_mesh.points if n_samples is None else target_mesh.points[rand_idxs_target, :]
            ),
            Y=np.asarray(
                source_mesh.points if n_samples is None else source_mesh.points[rand_idxs_source, :]
            ),
            max_iterations=max_iterations,
            alpha=alpha,
            beta=beta,
            num_eig=num_eig,
            tolerance=1e-5,
        )
    elif reg_type == "rigid":
        # if rigid, then we can turn the result into the vtk transform
        # on the other side of things.
        raise NotImplementedError("Rigid registration not implemented yet")
    elif reg_type == "affine":
        raise NotImplementedError("Affine registration not implemented yet")
    else:
        raise ValueError(f"Registration type {reg_type} not recognized")

    transformed_source, reg_params = reg.register()

    if n_samples is not None:
        transformed_source = reg.transform_point_cloud(source_mesh.points)

    source_mesh.points = transformed_source

    print("source_mesh")
    print(source_mesh)

    if transfer_scalars is True:
        source_mesh = transfer_mesh_scalars_get_weighted_average_n_closest(
            source_mesh,
            target_mesh,
            n=3,
            return_mesh=True,
            create_new_mesh=True,
        )

    print("source_mesh")
    print(source_mesh)

    # if return_icp_transform is True:
    #     return mesh_transformed_to_target, reg.icp_transform

    return source_mesh, reg_params


def non_rigidly_register(
    target_mesh=None,
    source_mesh=None,
    final_pt_location="weighted_average",  # 'weighted_average' or 'nearest_neighbour'
    icp_register_first=True,  # Get bones/objects into roughly the same alignment first
    icp_registration_mode="similarity",  # similarity = rigid + scaling (isotropic), ("rigid", "similarity", "affine")
    icp_reg_target_to_source=True,  # For shape models, the source is usually the reference so we want target in its space (true)
    n_spectral_features=3,
    n_extra_spectral=3,  # For ensuring we have the right spec coords - determined using wasserstein distances.
    target_eigenmap_as_reference=True,
    get_weighted_spectral_coords=False,
    list_features_to_calc=[
        "curvature"
    ],  # 'curvature', min_curvature' 'max_curvature' (other features for registration)
    use_features_as_coords=True,  # During registraiton - do we want to use curvature etc.
    rigid_reg_max_iterations=100,
    non_rigid_alpha=0.01,
    non_rigid_beta=50,
    non_rigid_n_eigens=100,  # number of eigens for low rank CPD registration
    non_rigid_max_iterations=500,
    rigid_before_non_rigid_reg=False,  # This is of the spectral coordinates - not the x/y/z used in icp_register_first
    projection_smooth_iterations=30,  # Used for distributing registered points onto target surface - helps preserve diffeomorphism
    graph_smoothing_iterations=300,  # For smoothing the target mesh before final point correspondence
    feature_smoothing_iterations=30,  # how much should features (curvature) be smoothed before registration
    include_points_as_features=False,  # Do we want to incldue x/y/z positions in registration?
    norm_physical_and_spectral=True,  # set standardized mean and variance for each feature
    feature_weights=np.diag(
        [0.1, 0.1]
    ),  # should we weight the extra features (curvature) more/less than spectral
    n_coords_spectral_ordering=20000,  # How many points on mesh to use for ordering spectral coordinates ()
    n_coords_spectral_registration=1000,  # How many points to use for spectral registrtaion (usually random subsample)
    initial_correspondence_type="kd",  # kd = nearest neightbor, hungarian = minimum cost of assigning between graphs (more compute heavy)
    final_correspondence_type="kd",  # kd = nearest neightbor, hungarian = minimum cost of assigning between graphs (more compute heavy)
    transfer_scalars=False,
    return_icp_transform=False,
    verbose=False,
):
    if "pyfocusr" not in sys.modules:
        raise ModuleNotFoundError(
            "pyfocusr is not installed & is necessary for non-rigid registration."
        )

    if final_pt_location not in ["weighted_average", "nearest_neighbour"]:
        raise Exception(
            'Did not specify appropriate final_pt_location, must be either "weighted_average", or "nearest_neighbour"'
        )

    # Test if mesh is a vtk mesh, or a pymsky.Mesh object.
    if isinstance(target_mesh, vtk.vtkPolyData):
        vtk_mesh_target = target_mesh
    else:
        try:
            vtk_mesh_target = target_mesh.mesh
        except:
            raise Exception(
                f"expected type vtk.vtkPolyData or pymskt.mesh.Mesh, got: {type(target_mesh)}"
            )

    if isinstance(source_mesh, vtk.vtkPolyData):
        vtk_mesh_source = source_mesh
    else:
        try:
            vtk_mesh_source = source_mesh.mesh
        except:
            raise Exception(
                f"expected type vtk.vtkPolyData or pymskt.mesh.Mesh, got: {type(target_mesh)}"
            )

    reg = pyfocusr.Focusr(
        vtk_mesh_target=vtk_mesh_target,
        vtk_mesh_source=vtk_mesh_source,
        icp_register_first=icp_register_first,
        icp_registration_mode=icp_registration_mode,
        icp_reg_target_to_source=icp_reg_target_to_source,
        n_spectral_features=n_spectral_features,
        n_extra_spectral=n_extra_spectral,
        target_eigenmap_as_reference=target_eigenmap_as_reference,
        get_weighted_spectral_coords=get_weighted_spectral_coords,
        list_features_to_calc=list_features_to_calc,
        use_features_as_coords=use_features_as_coords,
        rigid_reg_max_iterations=rigid_reg_max_iterations,
        non_rigid_alpha=non_rigid_alpha,
        non_rigid_beta=non_rigid_beta,
        non_rigid_n_eigens=non_rigid_n_eigens,
        non_rigid_max_iterations=non_rigid_max_iterations,
        rigid_before_non_rigid_reg=rigid_before_non_rigid_reg,
        projection_smooth_iterations=projection_smooth_iterations,
        graph_smoothing_iterations=graph_smoothing_iterations,
        feature_smoothing_iterations=feature_smoothing_iterations,
        include_points_as_features=include_points_as_features,
        norm_physical_and_spectral=norm_physical_and_spectral,
        feature_weights=feature_weights,
        n_coords_spectral_ordering=n_coords_spectral_ordering,
        n_coords_spectral_registration=n_coords_spectral_registration,
        initial_correspondence_type=initial_correspondence_type,
        final_correspondence_type=final_correspondence_type,
        verbose=verbose,
    )
    reg.align_maps()

    if final_pt_location == "weighted_average":
        reg.get_source_mesh_transformed_weighted_avg()
        mesh_transformed_to_target = reg.weighted_avg_transformed_mesh
    elif final_pt_location == "nearest_neighbour":
        reg.get_source_mesh_transformed_nearest_neighbour()
        mesh_transformed_to_target = reg.nearest_neighbour_transformed_mesh

    if transfer_scalars is True:
        mesh_transformed_to_target = transfer_mesh_scalars_get_weighted_average_n_closest(
            mesh_transformed_to_target,
            reg.graph_target.vtk_mesh,
            n=3,
            return_mesh=True,
            create_new_mesh=False,
        )

    if return_icp_transform is True:
        return mesh_transformed_to_target, reg.icp_transform
    return mesh_transformed_to_target
