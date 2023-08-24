import sys
import vtk
from pymskt.mesh.meshTools import transfer_mesh_scalars_get_weighted_average_n_closest
try:
     import pyfocusr
except ModuleNotFoundError:
    print('pyfocusr not found')
    print('If you are not using the registration tools, you can ignore this message.')
    print('install pyfocusr as described in the README: https://github.com/gattia/pymskt')
    print('or visit the pyfocusr github repo: https://github.com/gattia/pyfocusr')

import numpy as np

def get_icp_transform(source, target, max_n_iter=1000, n_landmarks=1000, reg_mode='similarity'):
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
    if reg_mode == 'rigid':
        icp.GetLandmarkTransform().SetModeToRigidBody()
    elif reg_mode == 'similarity':
        icp.GetLandmarkTransform().SetModeToSimilarity()
    icp.SetMaximumNumberOfIterations(max_n_iter)
    icp.StartByMatchingCentroidsOn()
    icp.Modified()
    icp.Update()
    icp.SetMaximumNumberOfLandmarks(n_landmarks)
    return icp

def non_rigidly_register(
    target_mesh=None,
    source_mesh=None,
    final_pt_location='weighted_average',   # 'weighted_average' or 'nearest_neighbour'
    icp_register_first=True,                # Get bones/objects into roughly the same alignment first
    icp_registration_mode='similarity',     # similarity = rigid + scaling (isotropic), ("rigid", "similarity", "affine")
    icp_reg_target_to_source=True,          # For shape models, the source is usually the reference so we want target in its space (true)
    n_spectral_features=3,
    n_extra_spectral=3,                     # For ensuring we have the right spec coords - determined using wasserstein distances. 
    target_eigenmap_as_reference=True,
    get_weighted_spectral_coords=False,
    list_features_to_calc=['curvature'],    # 'curvature', min_curvature' 'max_curvature' (other features for registration)
    use_features_as_coords=True,            # During registraiton - do we want to use curvature etc. 
    rigid_reg_max_iterations=100,
    non_rigid_alpha=0.01,
    non_rigid_beta=50,
    non_rigid_n_eigens=100,                 # number of eigens for low rank CPD registration
    non_rigid_max_iterations=500,
    rigid_before_non_rigid_reg=False,       # This is of the spectral coordinates - not the x/y/z used in icp_register_first
    projection_smooth_iterations=30,        # Used for distributing registered points onto target surface - helps preserve diffeomorphism
    graph_smoothing_iterations=300,         # For smoothing the target mesh before final point correspondence
    feature_smoothing_iterations=30,        # how much should features (curvature) be smoothed before registration 
    include_points_as_features=False,       # Do we want to incldue x/y/z positions in registration? 
    norm_physical_and_spectral=True,        # set standardized mean and variance for each feature
    feature_weights=np.diag([.1,.1]),       # should we weight the extra features (curvature) more/less than spectral
    n_coords_spectral_ordering=20000,       # How many points on mesh to use for ordering spectral coordinates ()
    n_coords_spectral_registration=1000,    # How many points to use for spectral registrtaion (usually random subsample)
    initial_correspondence_type='kd',       # kd = nearest neightbor, hungarian = minimum cost of assigning between graphs (more compute heavy)
    final_correspondence_type='kd',          # kd = nearest neightbor, hungarian = minimum cost of assigning between graphs (more compute heavy)
    transfer_scalars=False,
    return_icp_transform=False,
    verbose=False
):
    
    if 'pyfocusr' not in sys.modules:
        raise ModuleNotFoundError('pyfocusr is not installed & is necessary for non-rigid registration.')

    if final_pt_location not in ['weighted_average', 'nearest_neighbour']:
        raise Exception('Did not specify appropriate final_pt_location, must be either "weighted_average", or "nearest_neighbour"')

    # Test if mesh is a vtk mesh, or a pymsky.Mesh object. 
    if isinstance(target_mesh, vtk.vtkPolyData):
        vtk_mesh_target = target_mesh
    else:
        try:
            vtk_mesh_target = target_mesh.mesh
        except:
            raise Exception(f'expected type vtk.vtkPolyData or pymskt.mesh.Mesh, got: {type(target_mesh)}')
    
    if isinstance(source_mesh, vtk.vtkPolyData):
        vtk_mesh_source = source_mesh
    else:
        try:
            vtk_mesh_source = source_mesh.mesh
        except:
            raise Exception(f'expected type vtk.vtkPolyData or pymskt.mesh.Mesh, got: {type(target_mesh)}')
    
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
        verbose=verbose
    ) 
    reg.align_maps()

    if final_pt_location == 'weighted_average':
        reg.get_source_mesh_transformed_weighted_avg()
        mesh_transformed_to_target = reg.weighted_avg_transformed_mesh
    elif final_pt_location == 'nearest_neighbour':
        reg.get_source_mesh_transformed_nearest_neighbour()
        mesh_transformed_to_target = reg.nearest_neighbour_transformed_mesh
    
    if transfer_scalars is True:
        mesh_transformed_to_target = transfer_mesh_scalars_get_weighted_average_n_closest(
            mesh_transformed_to_target, 
            reg.graph_target.vtk_mesh,
            n=3,
            return_mesh=True,
            create_new_mesh=False)
    
    if return_icp_transform is True:
        return mesh_transformed_to_target, reg.icp_transform
    return mesh_transformed_to_target    