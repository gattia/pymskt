from tracemalloc import start
import numpy as np
from scipy.linalg import svd
import vtk
from vtk.util.numpy_support import numpy_to_vtk
from pymskt.mesh.utils import GIF
import os
from pymskt.mesh.io import write_vtk

def pca_svd(data):
    """
    Calculate eigenvalues & eigenvectors of `data` using Singular Value Decomposition (SVD)

    Parameters
    ----------
    data : numpy.ndarray
        MxN matrix 
        M = # of features / dimensions of data
        N = # of trials / participants in dataset

    Returns
    -------
    tuple (PC = numpy.ndarray, V = numpy.ndarray)
        PC - each volumn is a principal component (eigenvector)
        V - Mx1 matrix of variances (coinciding with each PC)

    Notes
    -----
    Adapted from:
    "A Tutorial on Principal Component Analysis by Jonathon Shlens"
    https://arxiv.org/abs/1404.1100
    Inputs
    data = MxN matrix (M dimensions, N trials)
    Returns
    PC - each column is a PC
    V - Mx1 matrix of variances
    """
    M, N = data.shape
    mn = np.mean(data, axis=1)
    data = data - mn[:, None]  # produce centered data. If already centered this shouldnt be harmful.

    Y = data.T / np.sqrt(N - 1)

    U, S, V = svd(Y, full_matrices=False)
    PC = V.T  # V are the principle components (PC)
    V = S ** 2  # The squared singular values are the variances (V)

    return PC, V

def get_ssm_deformation(PCs, Vs, mean_coords, pc=0, n_sds=3):
    """
    Function to Statistical Shape Model (SSM) deformed along given Principal Component.

    Parameters
    ----------
    PCs : numpy.ndarray
        NxM ndarray; N = number of points on surface, M = number of principal components in model
        Each column is a principal component.
    Vs : numpy.ndarray
        M ndarray; M = number of principal components in model
        Each entry is the variance for the coinciding principal component in PCs
    mean_coords : numpy.ndarray
        3xN ndarray; N = number of points on surface. 
    pc : int, optional
        The principal component of the SSM to deform, by default 0
    n_sds : int, optional
        The number of standard deviations (sd) to deform the SSM. 
        This can be positive or negative to scale the model in either direction. , by default 3

    Returns
    -------
    numpy.ndarray
        3xN ndarray; N=number of points on mesh surface. 
        This includes the x/y/z position of each surface node after deformation using the SSM and
        the specified characteristics (pc, n_sds)
    """

    pc_vector = PCs[:, pc]
    pc_vector_scale = np.sqrt(Vs[pc]) * n_sds # convert Variances to SDs & multiply by n_sds (negative/positive important)
    coords_deformation = pc_vector * pc_vector_scale
    deformed_coords = (mean_coords.flatten() + coords_deformation).reshape(mean_coords.shape)
    return deformed_coords

def get_rand_bone_shape(PCs, Vs, mean_coords, n_pcs=100, n_samples=1, mean_=0., sd_=1.0):
    """
    Function to get random bones using a Statistical Shape Model (SSM).

    Parameters
    ----------
    PCs : numpy.ndarray
        NxM ndarray; N = number of points on surface, M = number of principal components in model
        Each column is a principal component.
    Vs : numpy.ndarray
        M ndarray; M = number of principal components in model
        Each entry is the variance for the coinciding principal component in PCs
    mean_coords : numpy.ndarray
        3xN ndarray; N = number of points on surface.
    n_pcs : int, optional
        Number of PCs to randomly sample from (sequentially), by default 100
    n_samples : int, optional
        number of bones to create, by default 1
    mean_ : float, optional
        Mean of the normal distribution to sample PCs from, by default 0.
    sd_ : float, optional
        Standard deviation of the normal distribution to sample PCs from, by default 1.0

    Returns
    -------
    numpy.ndarray
        nx(3xN) ndarray; N=number of points on mesh surface, n=number of new meshes
        This includes the x/y/z position of each surface node(N) for the random bones(n).
    """

    rand_pc_scores = np.random.normal(mean_, sd_, size=[n_samples, n_pcs])
    rand_pc_weights = rand_pc_scores * np.sqrt(Vs[:n_pcs])
    rand_data = rand_pc_weights @ PCs[:, :n_pcs].T
    rand_data = mean_coords.flatten() + rand_data
    
    return rand_data   

def create_vtk_mesh_from_deformed_points(mean_mesh, new_points):
    """
    Create new vtk mesh (polydata) from a set of points (ndarray) deformed using the SSM. 

    Parameters
    ----------
    mean_mesh : vtk.PolyData
        vtk polydata of the mean mesh
    new_points : numpy.ndarray
        3xN ndarray; N=number of points on mesh surface (same as number of points on mean_mesh).
        This includes the x/y/z position of each surface node should be deformed to.

    Returns
    -------
    vtk.PolyData
        vtk polydata of the deformed mesh
    """

    new_mesh = vtk.vtkPolyData()
    new_mesh.DeepCopy(mean_mesh)
    new_mesh.GetPoints().SetData(numpy_to_vtk(new_points))
    
    return new_mesh

def save_gif(
    path_save,
    PCs,
    Vs,
    mean_coords,  # mean_coords could be extracted from mean mesh...?
    mean_mesh,
    pc=0,
    min_sd=-3.,
    max_sd=3.,
    step=0.25,
    color='orange', 
    show_edges=True, 
    edge_color='black',
    camera_position='xz',
    window_size=[3000, 4000],
    background_color='white',
    verbose=False,
):
    """
    Function to save a gif of the SSM deformation.

    Parameters
    ----------
    path_save : str
        Path to save the gif to.
    PCs : numpy.ndarray
        SSM Principal Components.
    Vs : numpy.ndarray
        SSM Variances.
    mean_coords : numpy.ndarray
        NxM ndarray; N = number of meshes, M = number of points x n_dimensions
    mean_mesh : vtk.PolyData
        vtk polydata of the mean mesh
    pc : int, optional
        The principal component of the SSM to deform, by default 0
    min_sd : float, optional
        The lower bound (minimum) standard deviations (sd) to deform the SSM from
        This can be positive or negative to scale the model in either direction. , by default -3.
    max_sd : float, optional
        The upper bound (maximum) standard deviations (sd) to deform the SSM from
        This can be positive or negative to scale the model in either direction. , by default 3.
    step : float, optional
        The step size (sd) to deform the SSM by, by default 0.25
    color : str, optional
        The color of the SSM surface during rendering, by default 'orange'
    show_edges : bool, optional
        Whether to show the edges of the SSM surface during rendering, by default True
    edge_color : str, optional
        The color of the edges of the SSM surface during rendering, by default 'black'
    camera_position : str, optional
        The camera position to use during rendering, by default 'xz'
    window_size : list, optional
        The window size to use during rendering, by default [3000, 4000]
    background_color : str, optional
        The background color to use during rendering, by default 'white'
    verbose : bool, optional
        Whether to print progress to console, by default False


    """
    # ALTERNATIVELY... could pass a bunch of the above parameters as kwargs..?? but thats less clear
    gif = GIF(
        path_save=path_save,
        color=color, 
        show_edges=show_edges, 
        edge_color=edge_color,
        camera_position=camera_position,
        window_size=window_size,
        background_color=background_color,
    )

    for idx, sd in enumerate(np.arange(min_sd, max_sd + step, step)):
        if verbose is True:
            print(f'Deforming SSM with idx={idx} sd={sd}')
        pts = get_ssm_deformation(PCs, Vs, mean_coords, pc=pc, n_sds=sd)
        
        if type(mean_mesh) == dict:
            mesh = []
            start_idx = 0
            for mesh_name, mesh_params in mean_mesh.items():
                mesh.append(
                    create_vtk_mesh_from_deformed_points(
                        mesh_params['mesh'], 
                        pts[start_idx:start_idx+mesh_params['n_points'], :],
                    )
                )
                start_idx += mesh_params['n_points']
        if type(mean_mesh) in (list, tuple):
            mesh = []
            start_idx = 0
            for mesh_ in mean_mesh:
                n_pts = mesh_.GetNumberOfPoints()
                mesh.append(
                    create_vtk_mesh_from_deformed_points(
                        mesh_, 
                        pts[start_idx:start_idx+n_pts, :],
                    )
                )
                start_idx += n_pts
        
        else:
            mesh = create_vtk_mesh_from_deformed_points(mean_mesh, pts)
        
        gif.add_mesh_frame(mesh)

    gif.done()

def save_meshes_across_pc(
    mesh,
    mean_coords,
    PCs,
    Vs,
    pc=0, 
    min_sd=-3, 
    max_sd=3, 
    step_size=1,
    loc_save='./', 
    mesh_name='bone', #['femur', 'tibia', 'patella'],
    save_filename='{mesh_name}_{sd}.vtk',
    verbose=True
):
    os.makedirs(loc_save, exist_ok=True)
    for idx, sd in enumerate(np.arange(min_sd, max_sd + step_size, step_size)):
        if verbose is True:
            print(f'Deforming SSM with idx={idx} sd={sd}')
        pts = get_ssm_deformation(PCs, Vs, mean_coords, pc=pc, n_sds=sd)
        
        if type(mesh) in [list, tuple]:
            start_idx = 0
            for mesh_idx, mesh_ in enumerate(mesh):
                mesh_name_ = mesh_name[mesh_idx]
                # load bone specific mesh
                n_pts = mesh_.GetNumberOfPoints()                

                updated_mesh = create_vtk_mesh_from_deformed_points(
                    mesh_, 
                    pts[start_idx:start_idx+n_pts, :],
                )
                write_vtk(mesh=updated_mesh, filepath=os.path.join(loc_save, save_filename.format(mesh_name=mesh_name_, sd=sd)))
                start_idx += n_pts
        
        elif type(mesh) is dict:
            start_idx = 0
            for mesh_name_, mesh_params in mesh.items():
                updated_mesh = create_vtk_mesh_from_deformed_points(
                    mesh_params['mesh'], 
                    pts[start_idx:start_idx+mesh_params['n_points'], :],
                )
                write_vtk(mesh=updated_mesh, filepath=os.path.join(loc_save, save_filename.format(mesh_name=mesh_name_, sd=sd)))                
                start_idx += mesh_params['n_points']
        else:
            updated_mesh = create_vtk_mesh_from_deformed_points(mesh, pts)
            write_vtk(mesh=updated_mesh, filepath=os.path.join(loc_save, save_filename.format(mesh_name=mesh_name, sd=sd)))                
                                       
def save_gif_vec_2_vec(
    path_save,
    PCs,
    Vs,
    mean_coords,  # mean_coords could be extracted from mean mesh...?
    mean_mesh,
    vec_1,
    vec_2,
    n_steps=24,
    color='orange', 
    show_edges=True, 
    edge_color='black',
    camera_position='xz',
    window_size=[900, 1200], #[3000, 4000],
    background_color='white',
    verbose=False,
):
    """
    Function to save a gif of the SSM from vec_1 to vec_2. All PCs that are not defined
    are assumed to not be included in the midel. 

    Parameters
    ----------
    path_save : str
        Path to save the gif to.
    PCs : numpy.ndarray
        SSM Principal Components.
    Vs : numpy.ndarray
        SSM Variances.
    mean_coords : numpy.ndarray
        NxM ndarray; N = number of meshes, M = number of points x n_dimensions
    mean_mesh : vtk.PolyData
        vtk polydata of the mean mesh
    vec_1 : np.ndarray
        Starting point for mesh deformation
    vec_2 : np.ndarray
        Ending point for mesh deformation
    n_steps : int, optional
        The number of steps to take between the two vectors, by default 24
    color : str, optional
        The color of the SSM surface during rendering, by default 'orange'
    show_edges : bool, optional
        Whether to show the edges of the SSM surface during rendering, by default True
    edge_color : str, optional
        The color of the edges of the SSM surface during rendering, by default 'black'
    camera_position : str, optional
        The camera position to use during rendering, by default 'xz'
    window_size : list, optional
        The window size to use during rendering, by default [3000, 4000]
    background_color : str, optional
        The background color to use during rendering, by default 'white'
    verbose : bool, optional
        Whether to print progress to console, by default False


    """
    # ALTERNATIVELY... could pass a bunch of the above parameters as kwargs..?? but thats less clear
    gif = GIF(
        path_save=path_save,
        color=color, 
        show_edges=show_edges, 
        edge_color=edge_color,
        camera_position=camera_position,
        window_size=window_size,
        background_color=background_color,
    )
    
    if PCs.shape[0] == np.product(mean_coords.shape):
        PCs = PCs.T
    elif PCs.shape[1] == np.product(mean_coords.shape):
        pass
    else:
        raise Exception('PCs should be the same length as the mean vector')
    
    # convert vec_1 and vec_2 to unnormalized scale
    # currently they are expected to be put in as "standard deviations" along each PC
    SDs = np.sqrt(Vs[:len(vec_1)])
    vec_1 = SDs * vec_1
    vec_2 = SDs * vec_2       
    
    if len(vec_1) != len(vec_2):
        raise Exception('Two vectors must be equal sized. ')

    # calculated the vector between point 1 and point 2. 
    # and then determine what step vector we must take for each
    # step to end up at vec_2.  
    vec_diff = vec_2 - vec_1
    vec_step = vec_diff/n_steps
    
    for step in range(n_steps + 1):
        if verbose is True:
            print(f'Deforming step = {step}')
        
        # calcualte the vector we are plotting for the current step. 
        vec_ = vec_1 + (step * vec_step)
        
        # calculated the coordinate deformation. & deformed points
        coords_deformation = vec_ @ PCs[:len(vec_), :]
        deformed_coords = (mean_coords.flatten() + coords_deformation).reshape(mean_coords.shape)
        
        # the following will add the deformation(s) to the GIF
        # properly handles single meshes, lists of meshes or a dict with meshes
        if type(mean_mesh) == dict:
            mesh = []
            start_idx = 0
            for mesh_name, mesh_params in mean_mesh.items():
                mesh.append(
                    create_vtk_mesh_from_deformed_points(
                        mesh_params['mesh'], 
                        deformed_coords[start_idx:start_idx+mesh_params['n_points'], :],
                    )
                )
                start_idx += mesh_params['n_points']
        if type(mean_mesh) in (list, tuple):
            mesh = []
            start_idx = 0
            for mesh_ in mean_mesh:
                n_pts = mesh_.GetNumberOfPoints()
                mesh.append(
                    create_vtk_mesh_from_deformed_points(
                        mesh_, 
                        deformed_coords[start_idx:start_idx+n_pts, :],
                    )
                )
                start_idx += n_pts
        else:
            mesh = create_vtk_mesh_from_deformed_points(mean_mesh, deformed_coords)
        
        gif.add_mesh_frame(mesh)

    gif.done()