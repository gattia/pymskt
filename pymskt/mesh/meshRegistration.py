import pyfocusr

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


# reg = pyfocusr.Focusr(vtk_mesh_target=mesh, 
#                           vtk_mesh_source=ref_mesh,  # WE ARE TRANSFORMING THE REF_MESH TO THE NEW MESH(es)
#                           icp_register_first=True,
#                           icp_registration_mode='similarity',
#                           icp_reg_target_to_source=True,
#                           n_spectral_features=3,
#                           n_extra_spectral=3,
#                           get_weighted_spectral_coords=False,
#                           list_features_to_calc=['curvature'], # 'curvature', min_curvature' 'max_curvature'
#                           use_features_as_coords=True,
#                           rigid_reg_max_iterations=100,
#                           non_rigid_alpha=0.01,
#                           non_rigid_beta=50,
#                           non_rigid_n_eigens=100,
#                           non_rigid_max_iterations=500,
#                           rigid_before_non_rigid_reg=False,
#                           projection_smooth_iterations=30,
#                           graph_smoothing_iterations=300,
#                           feature_smoothing_iterations=30,
#                           include_points_as_features=False,
#                           norm_physical_and_spectral=True,
#                           feature_weights=np.diag([.1,.1]),
#                           n_coords_spectral_ordering=20000,
#                           n_coords_spectral_registration=1000,
#                           initial_correspondence_type='kd',
#                           final_correspondence_type='kd')  #'kd' 'hungarian'
#     reg.align_maps()
#     reg.get_source_mesh_transformed_weighted_avg()
#     ref_mesh_transformed_to_target = reg.weighted_avg_transformed_mesh
#     scalars = vtk_functions.transfer_mesh_scalars_get_weighted_average_n_closest(
#         ref_mesh_transformed_to_target,                                                               
#         reg.graph_target.vtk_mesh,
#         n=3
#     )
    
#     scalars[np.isnan(scalars)] = 0
#     thickness_scalars = numpy_to_vtk(scalars)
#     thickness_scalars.SetName('thickness (mm)')
#     ref_mesh_transformed_to_target.GetPointData().AddArray(thickness_scalars)
#     ref_mesh_transformed_to_target.GetPointData().SetActiveScalars('thickness (mm)')

#     save_filename = '{}_{}_reg_to_{}.vtk'.format(leg, 
#                                                  bone, 
#                                                  'mean_mesh_round_1')
    
#     nsvtk.write_vtk(ref_mesh_transformed_to_target,
#               save_filename,
#               folder_name)