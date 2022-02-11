import numpy as np
from scipy.linalg import svd
import vtk
from vtk.util.numpy_support import numpy_to_vtk

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

    U, S, V = svd(Y)
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