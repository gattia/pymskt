from logging import error
from posixpath import supports_unicode_filenames
import warnings
import numpy as np
import vtk
from pymskt.image.main import apply_transform_retain_array
from vtk.util.numpy_support import numpy_to_vtk, vtk_to_numpy
import pyacvd
import pyvista as pv
import SimpleITK as sitk
import os
import random
import string
import warnings
import tempfile

# import pyfocusr     # MAKE THIS AN OPTIONAL IMPORT? 

import pymskt
from pymskt.mesh import createMesh
from pymskt.utils import safely_delete_tmp_file, copy_image_transform_to_mesh
from pymskt.image import read_nrrd, crop_bone_based_on_width
from pymskt.mesh.utils import vtk_deep_copy
from pymskt.mesh.meshTools import (gaussian_smooth_surface_scalars, 
                                   get_mesh_physical_point_coords, 
                                   get_cartilage_properties_at_points,
                                   smooth_scalars_from_second_mesh_onto_base,
                                   transfer_mesh_scalars_get_weighted_average_n_closest,
                                   resample_surface,
                                   get_distance_other_surface_at_points,
                                   fix_mesh,
                                   consistent_normals,
                                   get_mesh_edge_lengths,
                                   rand_sample_pts_mesh,
                                   vtk_sdf,
                                   pcu_sdf,
                                   decimate_mesh_pcu,
                                   compute_assd_between_point_clouds,
                                   get_largest_connected_component
                                   )
from pymskt.mesh.createMesh import create_surface_mesh
from pymskt.mesh.meshTransform import (SitkVtkTransformer, 
                                       get_versor_from_transform, 
                                       break_versor_into_center_rotate_translate_transforms,
                                       apply_transform,
                                       create_transform)
from pymskt.mesh.meshRegistration import non_rigidly_register, get_icp_transform
import pymskt.mesh.io as io

class Mesh:
    """
    An object to contain surface meshes for musculoskeletal anatomy. Includes helper
    functions to build surface meshes, to process them, and to save them. 

    Parameters
    ----------
    mesh : vtk.vtkPolyData, optional
        vtkPolyData object that is basis of surface mesh, by default None
    seg_image : SimpleITK.Image, optional
        Segmentation image that can be used to create surface mesh - used 
        instead of mesh, by default None
    path_seg_image : str, optional
        Path to a medical image (.nrrd) to load and create mesh from, 
        by default None
    label_idx : int, optional
        Label of anatomy of interest, by default None
    min_n_pixels : int, optional
        All islands smaller than this size are dropped, by default 5000


    Attributes
    ----------
    _mesh : vtk.vtkPolyData
        Item passed from __init__, or created during life of class. 
        This is the main surface mesh of this class. 
    _seg_image : SimpleITK.Image
        Segmentation image that can be used to create mesh. This is optional.
    path_seg_image : str
        Path to medical image (.nrrd) that can be loaded to create `_seg_image`
        and then creat surface mesh `_mesh` 
    label_idx : int
        Integer of anatomy to create surface mesh from `_seg_image`
    min_n_pixels : int
        Minimum number of pixels for an isolated island of a segmentation to be
        retained
    list_applied_transforms : list
        A list of transformations applied to a surface mesh. 
        This list allows for undoing of most recent transform, or undoing
        all of them by iterating over the list in reverse. 

    Methods
    ----------

    """    
    def __init__(self,
                 mesh=None,
                 seg_image=None,
                 path_seg_image=None,
                 label_idx=None,
                 min_n_pixels=5000
                 ):
        """
        Initialize Mesh class

        Parameters
        ----------
        mesh : vtk.vtkPolyData, optional
            vtkPolyData object that is basis of surface mesh, by default None
        seg_image : SimpleITK.Image, optional
            Segmentation image that can be used to create surface mesh - used 
            instead of mesh, by default None
        path_seg_image : str, optional
            Path to a medical image (.nrrd) to load and create mesh from, 
            by default None
        label_idx : int, optional
            Label of anatomy of interest, by default None
        min_n_pixels : int, optional
            All islands smaller than this size are dropped, by default 5000
        """      
        if isinstance(mesh, str): #accept path like objects?  
            print('mesh string passed, loading mesh from disk')
            self._mesh = io.read_vtk(mesh)
        else:
            self._mesh = mesh
        self._seg_image = seg_image
        self._path_seg_image = path_seg_image
        self._label_idx = label_idx
        self._min_n_pixels = min_n_pixels
        self._mesh_scalars = []
        self._n_scalars = 0
        if self._mesh is not None:
            self.load_mesh_scalars()

        self._list_applied_transforms = []
    
    def copy(self):
        """
        Create a copy of the mesh object.

        Returns
        -------
        Mesh
            A copy of the mesh object
        """

        mesh = Mesh(
            mesh=vtk_deep_copy(self._mesh),
            seg_image=self._seg_image,
            path_seg_image=self._path_seg_image,
            label_idx=self._label_idx,
            min_n_pixels=self.min_n_pixels,
        )

        mesh._list_applied_transforms = self._list_applied_transforms

        return mesh

    def read_seg_image(self,
                       path_seg_image=None):
        """
        Read segmentation image from disk. Must be a single file (e.g., nrrd, 3D dicom)

        Parameters
        ----------
        path_seg_image : str, optional
            Path to the medical image file to be loaded in, by default None

        Raises
        ------
        Exception
            If path_seg_image does not exist, exception is raised. 
        """        
        # If passing new location/seg image name, then update variables. 
        if path_seg_image is not None:
            self._path_seg_image = path_seg_image
        
        # If seg image location / name exist, then load image else raise exception
        if (self._path_seg_image is not None):
            self._seg_image = sitk.ReadImage(self._path_seg_image)
        else:
            raise Exception('No file path (self._path_seg_image) provided.')
    
    def create_mesh(self,
                    smooth_image=True,
                    smooth_image_var=0.3125 / 2,
                    marching_cubes_threshold=0.5,
                    label_idx=None,
                    min_n_pixels=None,
                    set_seg_border_to_zeros=True,
                    use_discrete_marching_cubes=False
                    ):
        """
        Create a surface mesh from the classes `_seg_image`. If `_seg_image`
        does not exist, then read it in using `read_seg_image`. 

        Parameters
        ----------
        smooth_image : bool, optional
            Should the `_seg_image` be gaussian filtered, by default True
        smooth_image_var : float, optional
            Variance of gaussian filter to apply to `_seg_image`, by default 0.3125/2
        marching_cubes_threshold : float, optional
            Threshold contour level to create surface mesh on, by default 0.5
        label_idx : int, optional
            Label value / index to create mesh from, by default None
        min_n_pixels : int, optional
            Minimum number of continuous pixels to include segmentation island
            in the surface mesh creation, by default None

        Raises
        ------
        Exception
            If the total number of pixels segmentated (`n_pixels_labelled`) is
            < `min_n_pixels` then there is no object in the image.  
        Exception
            If no `_seg_image` and no `label_idx` then we don't know what tissue to create the 
            surface mesh from. 
        Exception
            If no `_seg_image` or `path_seg_image` then we have no image to create mesh from. 
        """        
        # allow assigning label idx during mesh creation step. 
        if label_idx is not None:
            self._label_idx = label_idx
        
        if self._seg_image is None:
            self.read_seg_image()
        
        # Ensure the image has a certain number of pixels with the label of interest, otherwise there might be an issue.
        if min_n_pixels is None: min_n_pixels = self._min_n_pixels
        seg_view = sitk.GetArrayViewFromImage(self._seg_image)
        n_pixels_labelled = sum(seg_view[seg_view == self._label_idx])

        if n_pixels_labelled < min_n_pixels:
            raise Exception('The mesh does not exist in this segmentation!, only {} pixels detected, threshold # is {}'.format(n_pixels_labelled, 
                                                                                                                               marching_cubes_threshold))
        tmp_filename = ''.join(random.choice(string.ascii_lowercase) for i in range(10)) + '.nrrd'
        self._mesh = create_surface_mesh(self._seg_image,
                                         self._label_idx,
                                         smooth_image_var,
                                         loc_tmp_save=tempfile.gettempdir(),
                                         tmp_filename=tmp_filename,
                                         mc_threshold=marching_cubes_threshold,
                                         filter_binary_image=smooth_image,
                                         set_seg_border_to_zeros=set_seg_border_to_zeros,
                                         use_discrete_marching_cubes=use_discrete_marching_cubes
                                         )

        self.load_mesh_scalars()
        safely_delete_tmp_file(tempfile.gettempdir(),
                               tmp_filename)
    
    def save_mesh(self,
                  filepath,
                  write_binary=False):
        """
        Save the surface mesh from this class to disk. 

        Parameters
        ----------
        filepath : str
            Location & filename to save the surface mesh (vtk.vtkPolyData) to. 
        write_binary : bool, optional
            Should the mesh be saved as a binary or ASCII format, by default False
        """        
        io.write_vtk(self._mesh, filepath, write_binary=write_binary)
    
    def fix_mesh(self, method='meshfix', treat_as_single_component=False, resolution=50_000, project_onto_surface=True, verbose=True):
        """
        Fix the surface mesh by removing duplicate points and cells.

        Parameters
        ----------
        method : str, optional
            Method to use to fix the mesh, by default 'meshfix'
        resolution : int, optional
            Resolution to use if pcu watertight method is used, by default 50000
        verbose : bool, optional
            Should the function print out information about the mesh fixing
            process, by default True
        """        
        self._mesh = fix_mesh(self._mesh, method=method, treat_as_single_component=treat_as_single_component, resolution=resolution, project_onto_surface=project_onto_surface, verbose=verbose)
    
    def consistent_faces(self):
        """
        Make the faces of the mesh consistent. 
        """
        self._mesh = consistent_normals(self._mesh)
    
    def decimate(self, percent_orig_faces=0.5):
        """
        Decimate the mesh to reduce the number of faces/points.
        """
        self._mesh = decimate_mesh_pcu(self._mesh, percent_orig_faces=percent_orig_faces)
    
    def get_largest(self):
        """
        Get the largest connected component of the mesh. 
        """
        self._mesh = get_largest_connected_component(self._mesh)
    
    def rand_surface_pts(self, n_pts=100_000, method='bluenoise'):
        """
        Sample points from the surface of the mesh. 
        """
        return rand_sample_pts_mesh(self._mesh, n_pts=n_pts, method=method)

    def rand_pts_around_surface(self, n_pts=100_000, surface_method='bluenoise', distribution='normal', sigma=1.0):
        """
        Sample points around the surface of the mesh. For SDF sampling & neural implicit representation models.
        """
        if distribution == 'normal':
            rand_gen = np.random.default_rng().multivariate_normal
        elif distribution =='laplace':
            rand_gen = np.random.default_rng().laplace
        
        base_pts = self.rand_surface_pts(n_pts=n_pts, method=surface_method)
        mean = [0, 0, 0]

        if (distribution == 'normal') and (sigma is not None):
            cov = np.identity(len(mean)) * sigma**2
            rand_pts = rand_gen(mean, cov, n_pts)
        elif distribution == 'laplace':
            rand_pts = np.tile(mean, [n_pts, 1])
            rand_pts = rand_gen(rand_pts, sigma, n_pts)
        
        samples = base_pts + rand_pts

        return samples

    def get_sdf_pts(self, pts, method='pcu'):
        """
        Calculates the signed distances (SDFs) for a set of points.

        Args:
            pts (np.ndarray): (n_pts, 3) array of points
            method (str, optional): Method to use. Defaults to 'pcu' as its faster
        
        Returns:
            np.ndarray: (n_pts, ) array of SDFs
        """
        if method == 'pcu':
            sdfs = pcu_sdf(pts, self._mesh)
        elif method == 'vtk':
            sdfs = vtk_sdf(pts, self._mesh)
        
        return sdfs
    
    def get_assd_mesh(self, other_mesh):
        if isinstance(other_mesh, Mesh):
            pass
        elif isinstance(other_mesh, (vtk.vtkPolyData, pv.PolyData, str)):
            other_mesh = Mesh(other_mesh)
        else:
            raise TypeError('other_mesh must be of type Mesh, vtk.vtkPolyData, pv.PolyData, or str, and received: {}'.format(type(other_mesh)))
        
        distances1 = np.abs(pcu_sdf(self.point_coords, other_mesh.mesh))
        distances2 = np.abs(pcu_sdf(other_mesh.point_coords, self.mesh))

        assd = (np.sum(distances1) + np.sum(distances2)) / (len(distances1) + len(distances2))

        return assd


    def get_assd(self, point_cloud):
        if isinstance(point_cloud, Mesh):
            point_cloud = point_cloud.point_coords
        elif isinstance(point_cloud, vtk.vtkPolyData):
            point_cloud = get_mesh_physical_point_coords(point_cloud)
        elif isinstance(point_cloud, np.ndarray):
            pass
        else:
            raise TypeError('point_cloud must be of type Mesh, vtk.vtkPolyData, or np.ndarray, and received: {}'.format(type(point_cloud)))

        return compute_assd_between_point_clouds(self.point_coords, point_cloud)
    
    def load_mesh_scalars(self):
        """
        Retrieve scalar names from mesh & store as Mesh attribute. 
        """
        n_scalars = self._mesh.GetPointData().GetNumberOfArrays()
        array_names = [self._mesh.GetPointData().GetArray(array_idx).GetName() for array_idx in range(n_scalars)]
        self._scalar_names = array_names
        self._n_scalars = n_scalars

    # def add_mesh_scalars(self, scalar_name, scalar_array):
    #     """
    #     Add a scalar array to the mesh. 

    #     Parameters
    #     ----------
    #     scalar_name : str
    #         Name of scalar array
    #     scalar_array : numpy.ndarray
    #         Array of scalars to add to mesh. 
    #     """
    #     array = numpy_to_vtk(scalar_array)
    #     array.SetName(scalar_name)
    #     self._mesh.GetPointData().AddArray(array)
    #     self.load_mesh_scalars() # Do this because it also updates the number of scalars.
    
    def set_active_scalars(self, scalar_name):
        """
        Set the active scalar array of the mesh. 

        Parameters
        ----------
        scalar_name : str
            Name of scalar array to set as active. 
        """
        self._mesh.GetPointData().SetActiveScalars(scalar_name)
    
    def fill_holes(self, max_size=100):
        """
        Fill holes in the mesh. 
        """
        filler = vtk.vtkFillHolesFilter()
        filler.SetInputData(self._mesh)
        filler.SetHoleSize(max_size)
        filler.Update()

        self._mesh = filler.GetOutput()
        

    def resample_surface(self,
                         subdivisions=2,
                         clusters=10000
                         ):
        """
        Resample a surface mesh using the ACVD algorithm: 
        Version used: 
        - https://github.com/pyvista/pyacvd
        Original version w/ more references: 
        - https://github.com/valette/ACVD

        Parameters
        ----------
        subdivisions : int, optional
            Subdivide the mesh to have more points before clustering, by default 2
            Probably not necessary for very dense meshes.
        clusters : int, optional
            The number of clusters (points/vertices) to create during resampling 
            surafce, by default 10000
            - This is not exact, might have slight differences. 
        """
        self._mesh = resample_surface(self._mesh, subdivisions=subdivisions, clusters=clusters)

    def apply_transform_to_mesh(self,
                                transform=None,
                                transformer=None,
                                save_transform=True):
        """
        Apply a transformation to the surface mesh. 

        Parameters
        ----------
        transform : vtk.vtkTransform, optional
            Transformation to apply to mesh, by default None
        transformer : vtk.vtkTransformFilter, optional
            Can supply transformFilter directly, by default None
        save_transform : bool, optional
            Should transform be saved to list of applied transforms, by default True

        Raises
        ------
        Exception
            No `transform` or `transformer` supplied - have not transformation
            to apply. 
        """
        if isinstance(transform, np.ndarray):
            transform = create_transform(transform)
        if (transform is not None) & (transformer is None):
            transformer = vtk.vtkTransformPolyDataFilter()
            transformer.SetTransform(transform)

        elif (transform is None) & (transformer is not None):
            transform = transformer.GetTransform()

        if transformer is not None:
            transformer.SetInputData(self._mesh)
            transformer.Update()
            self._mesh = vtk_deep_copy(transformer.GetOutput())

            if save_transform is True:
                self._list_applied_transforms.append(transform)
                
        else:
            raise Exception('No transform or transformer provided')

    def reverse_most_recent_transform(self):
        """
        Function to undo the most recent transformation stored in self._list_applied_transforms
        """
        transform = self._list_applied_transforms.pop()
        transform.Inverse()
        self.apply_transform_to_mesh(transform=transform, save_transform=False)

    def reverse_all_transforms(self):
        """
        Function to iterate over all of the self._list_applied_transforms (in reverse order) and undo them.
        """
        while len(self._list_applied_transforms) > 0:
            self.reverse_most_recent_transform()
    
    def non_rigidly_register(
        self,
        other_mesh,
        as_source=True,
        apply_transform_to_mesh=True,
        return_transformed_mesh=False,
        **kwargs
    ):  
        """
        Function to perform non rigid registration between this mesh and another mesh. 

        Parameters
        ----------
        other_mesh : pymskt.mesh.Mesh or vtk.vtkPolyData
            Other mesh to use in registration process
        as_source : bool, optional
            Should the current mesh (in this object) be the source or the target, by default True
        apply_transform_to_mesh : bool, optional
            If as_source is True should we apply transformation to internal mesh, by default True
        return_transformed_mesh : bool, optional
            Should we return the registered mesh, by default False

        Returns
        -------
        _type_
            _description_
        """
        # Setup the source & target meshes based on `as_source``
        if as_source is True:
            source = self._mesh
            target = other_mesh
        elif as_source is False:
            source = other_mesh
            target = self._mesh

        # Get registered mesh (source to target)
        source_transformed_to_target = non_rigidly_register(
                target_mesh=target,
                source_mesh=source,
                **kwargs
            ) 

        # If current mesh is source & apply_transform_to_mesh is true then replace current mesh. 
        if (as_source is True) & (apply_transform_to_mesh is True):
            self._mesh = source_transformed_to_target
        
        # curent mesh is target, or is source & want to return mesh, then return it.  
        if (as_source is False) or ((as_source is True) & (return_transformed_mesh is True)):
            return source_transformed_to_target

    def rigidly_register(
        self,
        other_mesh,
        as_source=True,
        apply_transform_to_mesh=True,
        return_transformed_mesh=False,
        return_transform=False,
        max_n_iter=100,
        n_landmarks=1000,
        reg_mode='similarity'

    ):
        """
        Function to perform rigid registration between this mesh and another mesh. 

        Parameters
        ----------
        other_mesh : pymskt.mesh.Mesh or vtk.vtkPolyData
            Other mesh to use in registration process
        as_source : bool, optional
            Should the current mesh (in this object) be the source or the target, by default True
        apply_transform_to_mesh : bool, optional
            If as_source is True should we apply transformation to internal mesh, by default True
        return_transformed_mesh : bool, optional
            Should we return the registered mesh, by default False
        max_n_iter : int, optional
            Maximum number of iterations to perform, by default 100
        n_landmarks : int, optional
            Number of landmarks to use in registration, by default 1000
        reg_mode : str, optional
            Mode of registration to use, by default 'similarity' (similarity, rigid, or affine)

        Returns
        -------
        _type_
            _description_
        """

        if (return_transform is True) & (return_transformed_mesh is True):
            raise Exception('Cannot return both transformed mesh and transform')

        if type(other_mesh) in (pymskt.mesh.meshes.BoneMesh, pymskt.mesh.meshes.Mesh):
            other_mesh = other_mesh.mesh

        # Setup the source & target meshes based on `as_source``
        if as_source is True:
            source = self._mesh
            target = other_mesh
        elif as_source is False:
            source = other_mesh
            target = self._mesh
        
        icp_transform = get_icp_transform(
            source=source,
            target=target,
            max_n_iter=max_n_iter,
            n_landmarks=n_landmarks,
            reg_mode=reg_mode
        )

        # If current mesh is source & apply_transform_to_mesh is true then replace current mesh. 
        if (as_source is True) & (apply_transform_to_mesh is True):
            self.apply_transform_to_mesh(transform=icp_transform)

            if return_transformed_mesh is True:
                return self._mesh
        
        # curent mesh is target, or is source & want to return mesh, then return it.  
        elif (as_source is False) & (return_transformed_mesh is True):
            return apply_transform(source=source, transform=icp_transform)
        
        if return_transform is True:
            return icp_transform
        else:
            raise Exception('Nothing to return from rigid registration.')

    def copy_scalars_from_other_mesh_to_current(
        self,
        other_mesh,
        orig_scalars_name=None,             # Default None - therefore will use all scalars from other mesh
        new_scalars_name=None,              # Defaule None - therefore will use the original scalar names
        weighted_avg=True,                  # Use weighted average, otherwise guassian smooth transfer
        n_closest=3,
        sigma=1.,
        idx_coords_to_smooth_base=None,
        idx_coords_to_smooth_other=None,
        set_non_smoothed_scalars_to_zero=True,
    ):
        """
        Convenience function to enable easy transfer scalars from another mesh to the current. 
        Can use either a gaussian smoothing function, or transfer using nearest neighbours. 

        ** This function requires that the `other_mesh` is non-rigidly registered to the surface
            of the mesh inside of this class. Or rigidly registered but using the same anatomy that
            VERY closely matches. Otherwise, the transfered scalars will be bad (really bad!).  

        Parameters
        ----------
        other_mesh : pymskt.mesh.Mesh or vtk.vtkPolyData
            Mesh we want to copy 
        orig_scalars_name : str or list of str, optional
            Name(s) of the scalar(s) to copy from `other_mesh`, by default None (copy all scalars)
        new_scalars_name : str or list of str, optional
            Name(s) to give to the copied scalar(s), by default None (use original scalar names)
        weighted_avg : bool, optional
            Should we use `weighted average` or `gaussian smooth` methods for transfer, by default True
        n_closest : int, optional
            If `weighted_avg` True, the number of nearest neighbours to use, by default 3
        sigma : float, optional
            If `weighted_avg` False, the standard deviation of gaussian kernel, by default 1.
        idx_coords_to_smooth_base : list, optional
            If `weighted_avg` False, list of indices from current mesh to use in transfer, by default None
        idx_coords_to_smooth_other : list, optional
            If `weighted_avg` False, list of indices from `other_mesh` to use in transfer, by default None
        set_non_smoothed_scalars_to_zero : bool, optional
            Should all other indices (not included in idx_coords_to_smooth_other) be set to 0, by default True
        """
        n_scalars_at_start = self._n_scalars

        if isinstance(other_mesh, Mesh):
            other_mesh = other_mesh.mesh
        elif isinstance(other_mesh, vtk.vtkPolyData):
            pass
        else:
            raise TypeError(f'other_mesh must be type `pymskt.mesh.Mesh` or `vtk.vtkPolyData` and received: {type(other_mesh)}')

        if orig_scalars_name is None:
            orig_scalars_name = [other_mesh.GetPointData().GetArray(array_idx).GetName() for array_idx in range(other_mesh.GetPointData().GetNumberOfArrays())]
        elif isinstance(orig_scalars_name, str):
            orig_scalars_name = [orig_scalars_name]
        if new_scalars_name is None:
            new_scalars_name = orig_scalars_name
        elif isinstance(new_scalars_name, str):
            new_scalars_name = [new_scalars_name]
        
        array_names = [other_mesh.GetPointData().GetArray(array_idx).GetName() for array_idx in range(other_mesh.GetPointData().GetNumberOfArrays())]
        
        if len(orig_scalars_name) != len(new_scalars_name):
            raise ValueError("orig_scalars_name and new_scalars_name must have the same length")

        if weighted_avg is True:
            transferred_scalars = transfer_mesh_scalars_get_weighted_average_n_closest(
                self._mesh,
                other_mesh,
                n=n_closest
            )
        else:
            raise Exception('Gaussian smoothing only implemented for active scalars')
            transferred_scalars = smooth_scalars_from_second_mesh_onto_base(
                self._mesh,
                other_mesh,
                sigma=sigma,
                idx_coords_to_smooth_base=idx_coords_to_smooth_base,
                idx_coords_to_smooth_second=idx_coords_to_smooth_other,
                set_non_smoothed_scalars_to_zero=set_non_smoothed_scalars_to_zero
            )
        
        # for array_name in array_names:
        for scalars_idx, scalars_name in enumerate(orig_scalars_name):
            vtk_transferred_scalars = numpy_to_vtk(transferred_scalars[scalars_name])
            vtk_transferred_scalars.SetName(new_scalars_name[scalars_idx])
            self._mesh.GetPointData().AddArray(vtk_transferred_scalars)
                
        self.load_mesh_scalars()
        return transferred_scalars
    
    def calc_distance_to_other_mesh(self,
                                    list_other_meshes=[],
                                    ray_cast_length=10.0,
                                    percent_ray_length_opposite_direction=0.25,
                                    name='thickness (mm)'
                                    ):
        """
        Using bone mesh (`_mesh`) and the list of cartilage meshes (`list_cartilage_meshes`)
        calcualte the cartilage thickness for each node on the bone surface. 

        Parameters
        ----------
        list_cartilage_labels : list, optional
            Cartilag labels to be used to create cartilage meshes (if they dont
            exist), by default None
        list_cartilage_meshes : list, optional
            Cartilage meshes to be used for calculating cart thickness, by default None
        image_smooth_var_cart : float, optional
            Variance of gaussian filter to be applied to binary cartilage masks, 
            by default 0.3125/2
        marching_cubes_threshold : float, optional
            Threshold to create bone surface at, by default 0.5
        ray_cast_length : float, optional
            Length (mm) of ray to cast from bone surface when trying to find cartilage (inner &
            outter shell), by default 10.0
        percent_ray_length_opposite_direction : float, optional
            How far to project ray inside of the bone. This is done just in case the cartilage
            surface ends up slightly inside of (or coincident with) the bone surface, by default 0.25

        Raises
        ------
        Exception
            No cartilage available (either `list_cartilage_meshes` or `list_cartilage_labels`)
        """
        if not isinstance(list_other_meshes, (list, tuple)):
            list_other_meshes = [list_other_meshes,]
        
        # pre-allocate empty thicknesses so that as labels are iterated over, they can all be appended to the same bone. 
        distances = np.zeros(self._mesh.GetNumberOfPoints())
        
        # iterate over meshes and add their thicknesses to the thicknesses list. 
        for other_mesh in list_other_meshes:
            if isinstance(other_mesh, Mesh):
                other_mesh = other_mesh.mesh

            node_data = get_distance_other_surface_at_points(
                self._mesh,
                other_mesh,
                ray_cast_length=ray_cast_length,
                percent_ray_length_opposite_direction=percent_ray_length_opposite_direction,
            )

            distances += node_data
        
        # Assign the thickness scalars to the bone mesh surface. 
        distance_scalars = numpy_to_vtk(distances)
        distance_scalars.SetName(name)
        self._mesh.GetPointData().AddArray(distance_scalars)
        self.set_active_scalars(name)
        self.load_mesh_scalars() # Re load mesh scalars to include the newly calculated distances. 
    
    def calc_surface_error(self, other_mesh, new_scalar_name='surface_error'):
        """
        Calculate the surface error between this mesh and another mesh.
        Assign the surface error as a new scalar to this mesh.

        Parameters
        ----------
        other_mesh : pymskt.mesh.Mesh or vtk.vtkPolyData
            Other mesh to calculate surface error with. 

        Returns
        -------
        None
        """
        # get point coordinates for other mesh
        if isinstance(other_mesh, (vtk.vtkPolyData)):
            other_mesh = Mesh(other_mesh)
        elif isinstance(other_mesh, Mesh):
            pass
        else:
            raise TypeError('other_mesh must be of type vtk.vtkPolyData or pymskt.mesh.Mesh and received: {}'.format(type(other_mesh)))

        # get sdf for other_pts
        sdf = other_mesh.get_sdf_pts(self.point_coords)
        
        # add sdf as new scalar to current mesh
        self.set_scalar(new_scalar_name, sdf)

    @property
    def seg_image(self):
        """
        Return the `_seg_image` object

        Returns
        -------
        SimpleITK.Image
            Segmentation image used to build the surface mesh
        """        
        return self._seg_image

    @seg_image.setter
    def seg_image(self, new_seg_image):
        """
        Set the `_seg_image` of the class to be the inputted `new_seg_image`. 

        Parameters
        ----------
        new_seg_image : SimpleITK.Image
            New image to use for creating surface mesh. This can be used to provide image to
            class if it was not provided during `__init__`
        """        
        self._seg_image = new_seg_image

    @property
    def mesh(self):
        """
        Return the `_mesh` object

        Returns
        -------
        vtk.vtkPolyData
            The main mesh of this class. 
        """        
        return self._mesh

    @mesh.setter
    def mesh(self, new_mesh):
        """
        Set the `_mesh` of the class to be the inputted `new_mesh`

        Parameters
        ----------
        new_mesh : vtk.vtkPolyData
            New mesh for this class - or a method to provide a mesh to the class
            after `__init__` has already been run. 
        """        
        self._mesh = new_mesh
    
    @property
    def point_coords(self):
        """
        Convenience function to return the vertices (point coordinates) for the surface mesh. 

        Returns
        -------
        numpy.ndarray
            Mx3 numpy array containing the x/y/z position of each vertex of the mesh. 
        """        
        return get_mesh_physical_point_coords(self._mesh)
    
    @point_coords.setter
    def point_coords(self, new_point_coords):
        """
        Convenience function to change/update the vertices/points locations

        Parameters
        ----------
        new_point_coords : numpy.ndarray
            n_points X 3 numpy array to replace exisiting point coordinate locations
            This can be used to easily/quickly update the x/y/z position of a set of points on a surface mesh. 
            The `new_point_coords` must include the same number of points as the mesh contains. 
        """        
        orig_point_coords = get_mesh_physical_point_coords(self._mesh)
        if new_point_coords.shape == orig_point_coords.shape:
            self._mesh.GetPoints().SetData(numpy_to_vtk(new_point_coords))

    @property
    def scalar_names(self):
        """
        Convenience function to return the names of the scalars in the mesh

        Returns
        -------
        list
            List of strings containing the names of the scalars in the mesh
        """        
        return self._scalar_names
    
    @property
    def n_scalars(self):
        """
        Convenience function to return the number of scalars in the mesh

        Returns
        -------
        int
            Number of scalars in the mesh
        """        
        return self._n_scalars
    
    # @property
    # def scalar(self, scalar_name):
    def get_scalar(self, scalar_name):
        """
        Convenience function to return the array of a scalar in the mesh

        Parameters
        ----------
        scalar_name : str
            Name of the scalar to return

        Returns
        -------
        numpy.ndarray
            Numpy array containing the scalars in the mesh
        """        
        return vtk_to_numpy(self._mesh.GetPointData().GetArray(scalar_name))
    
    # @scalar.setter
    def set_scalar(self, scalar_name, scalar_array):
        """
        Add a scalar array to the mesh. 

        Parameters
        ----------
        scalar_name : str
            Name of scalar array
        scalar_array : numpy.ndarray
            Array of scalars to add to mesh. 
        """
        array = numpy_to_vtk(scalar_array)
        array.SetName(scalar_name)
        self._mesh.GetPointData().AddArray(array)
        self.load_mesh_scalars() # Do this because it also updates the number of scalars.
    
    def remove_scalar(self, scalar_name):
        """
        Remove a scalar array from the mesh. 

        Parameters
        ----------
        scalar_name : str
            Name of scalar array
        """
        self._mesh.GetPointData().RemoveArray(scalar_name)
        self.load_mesh_scalars()
    
    @property
    def path_seg_image(self):
        """
        Convenience function to get the `path_seg_image`

        Returns
        -------
        str
            Path to the segmentation image
        """        
        return self._path_seg_image
    
    @path_seg_image.setter
    def path_seg_image(self, new_path_seg_image):
        """
        Convenience function to set the `path_seg_image`

        Parameters
        ----------
        new_path_seg_image : str
            String to where segmentation image that should be loaded is. 
        """        
        self._path_seg_image = new_path_seg_image
    
    @property
    def label_idx(self):
        """
        Convenience function to get `label_idx`

        Returns
        -------
        int
            Integer indeicating the index/value of the tissues in `seg_image` associated with this mesh. 
        """        
        return self._label_idx
    
    @label_idx.setter
    def label_idx(self, new_label_idx):
        """
        Convenience function to set `label_idx`

        Parameters
        ----------
        new_label_idx : int
            Integer indeicating the index/value of the tissues in `seg_image` associated with this mesh. 
        """        
        self._label_idx = new_label_idx

    @property
    def min_n_pixels(self):
        """
        Convenience function to get the minimum number of pixels for a segmentation region to be created as a mesh. 

        Returns
        -------
        int
            Minimum number of pixels needed to create a mesh. Less than this and it will be skipped / error raised. 
        """        
        return self._min_n_pixels
    
    @min_n_pixels.setter
    def min_n_pixels(self, new_min_n_pixels):
        """
        Convenience function to set the minimum number of pixels for a segmentation region to be created as a mesh. 

        Parameters
        ----------
        new_min_n_pixels : int
            Minimum number of pixels needed to create a mesh. Less than this and it will be skipped / error raised. 
        """        
        self._min_n_pixels = new_min_n_pixels

    @property
    def list_applied_transforms(self):
        """
        Convenience function to get the list of transformations that have been applied to this mesh. 

        Returns
        -------
        list
            List of vtk.vtkTransform objects that have been applied to the current mesh. 
        """        
        return self._list_applied_transforms
    
    @property
    def edge_lengths(self):
        """
        Convenience function to get the edge lengths of the mesh

        Returns
        -------
        numpy.ndarray
            Numpy array containing the edge lengths of the mesh
        """
        return get_mesh_edge_lengths(self._mesh)


class CartilageMesh(Mesh):
    """
    Class to create, store, and process cartilage meshes

    Parameters
    ----------
    mesh : vtk.vtkPolyData, optional
        vtkPolyData object that is basis of surface mesh, by default None
    seg_image : SimpleITK.Image, optional
        Segmentation image that can be used to create surface mesh - used 
        instead of mesh, by default None
    path_seg_image : str, optional
        Path to a medical image (.nrrd) to load and create mesh from, 
        by default None
    label_idx : int, optional
        Label of anatomy of interest, by default None
    min_n_pixels : int, optional
        All islands smaller than this size are dropped, by default 5000


    Attributes
    ----------
    _mesh : vtk.vtkPolyData
        Item passed from __init__, or created during life of class. 
        This is the main surface mesh of this class. 
    _seg_image : SimpleITK.Image
        Segmentation image that can be used to create mesh. This is optional.
    path_seg_image : str
        Path to medical image (.nrrd) that can be loaded to create `_seg_image`
        and then creat surface mesh `_mesh` 
    label_idx : int
        Integer of anatomy to create surface mesh from `_seg_image`
    min_n_pixels : int
        Minimum number of pixels for an isolated island of a segmentation to be
        retained
    list_applied_transforms : list
        A list of transformations applied to a surface mesh. 
        This list allows for undoing of most recent transform, or undoing
        all of them by iterating over the list in reverse. 

    Methods
    ----------

    """

    def __init__(self,
                 mesh=None,
                 seg_image=None,
                 path_seg_image=None,
                 label_idx=None,
                 min_n_pixels=1000
                 ):
        super().__init__(mesh=mesh,
                         seg_image=seg_image,
                         path_seg_image=path_seg_image,
                         label_idx=label_idx,
                         min_n_pixels=min_n_pixels)


class BoneMesh(Mesh):
    """
    Class to create, store, and process bone meshes

    Intention is that this class includes functions to process other data & assign it to the bone surface.
    It might be possible that instead this class & a cartilage class or, this class and image data etc. are
    provided to another function or class that does those analyses.

    Parameters
    ----------
    mesh : vtk.vtkPolyData, optional
        vtkPolyData object that is basis of surface mesh, by default None
    seg_image : SimpleITK.Image, optional
        Segmentation image that can be used to create surface mesh - used 
        instead of mesh, by default None
    path_seg_image : str, optional
        Path to a medical image (.nrrd) to load and create mesh from, 
        by default None
    label_idx : int, optional
        Label of anatomy of interest, by default None
    min_n_pixels : int, optional
        All islands smaller than this size are dropped, by default 5000
    list_cartilage_meshes : list, optional
        List object which contains 1+ `CartilageMesh` objects that wrap
        a vtk.vtkPolyData surface mesh of cartilage, by default None
    list_cartilage_labels : list, optional
        List of `int` values that represent the different cartilage
        regions of interest appropriate for a single bone, by default None
    crop_percent : float, optional
        Proportion value to crop long-axis of bone so it is proportional
        to the width of the bone for standardization purposes, by default 1.0
    bone : str, optional
        String indicating what bone is being analyzed so that cropping
        can be applied appropriatey. {'femur', 'tibia'}, by default 'femur'.
        Patella is not an option because we do not need to crop for the patella. 


    Attributes
    ----------
    _mesh : vtk.vtkPolyData
        Item passed from __init__, or created during life of class. 
        This is the main surface mesh of this class. 
    _seg_image : SimpleITK.Image
        Segmentation image that can be used to create mesh. This is optional.
    path_seg_image : str
        Path to medical image (.nrrd) that can be loaded to create `_seg_image`
        and then creat surface mesh `_mesh` 
    label_idx : int
        Integer of anatomy to create surface mesh from `_seg_image`
    min_n_pixels : int
        Minimum number of pixels for an isolated island of a segmentation to be
        retained
    list_applied_transforms : list
        A list of transformations applied to a surface mesh. 
        This list allows for undoing of most recent transform, or undoing
        all of them by iterating over the list in reverse.
    crop_percent : float
        Percent of width to crop along long-axis of bone
    bone : str
        A string indicating what bone is being represented by this class. 
    list_cartilage_meshes : list
        List of cartialge meshes assigned to this bone. 
    list_cartilage_labels : list
        List of cartilage labels for the `_seg_image` that are associated
        with this bone. 

    Methods
    ----------

    """

    def __init__(self,
                 mesh=None,
                 seg_image=None,
                 path_seg_image=None,
                 label_idx=None,
                 min_n_pixels=5000,
                 list_cartilage_meshes=None,
                 list_cartilage_labels=None,
                 crop_percent=None,
                 bone='femur',
                 ):
        """
        Class initialization

        Parameters
        ----------
        mesh : vtk.vtkPolyData, optional
            vtkPolyData object that is basis of surface mesh, by default None
        seg_image : SimpleITK.Image, optional
            Segmentation image that can be used to create surface mesh - used 
            instead of mesh, by default None
        path_seg_image : str, optional
            Path to a medical image (.nrrd) to load and create mesh from, 
            by default None
        label_idx : int, optional
            Label of anatomy of interest, by default None
        min_n_pixels : int, optional
            All islands smaller than this size are dropped, by default 5000
        list_cartilage_meshes : list, optional
            List object which contains 1+ `CartilageMesh` objects that wrap
            a vtk.vtkPolyData surface mesh of cartilage, by default None
        list_cartilage_labels : list, optional
            List of `int` values that represent the different cartilage
            regions of interest appropriate for a single bone, by default None
        crop_percent : float, optional
            Proportion value to crop long-axis of bone so it is proportional
            to the width of the bone for standardization purposes, by default 1.0
        bone : str, optional
            String indicating what bone is being analyzed so that cropping
            can be applied appropriatey. {'femur', 'tibia'}, by default 'femur'.
            Patella is not an option because we do not need to crop for the patella. 
        """        
        self._crop_percent = crop_percent
        self._bone = bone
        self._list_cartilage_meshes = list_cartilage_meshes
        self._list_cartilage_labels = list_cartilage_labels

        super().__init__(mesh=mesh,
                         seg_image=seg_image,
                         path_seg_image=path_seg_image,
                         label_idx=label_idx,
                         min_n_pixels=min_n_pixels)

        
    def create_mesh(self, 
                    smooth_image=True, 
                    smooth_image_var=0.3125 / 2, 
                    marching_cubes_threshold=0.5, 
                    label_idx=None, 
                    min_n_pixels=None,
                    crop_percent=None
                    ):
        """
        This is an extension of `Mesh.create_mesh` that enables cropping of bones. 
        Bones might need to be cropped (this isnt necessary for cartilage)
        So, adding this functionality to the processing steps before the bone mesh is created.

        All functionality, except for that relevant to `crop_percent` is the same as:
        `Mesh.create_mesh`. 
        
        Create a surface mesh from the classes `_seg_image`. If `_seg_image`
        does not exist, then read it in using `read_seg_image`. 

        Parameters
        ----------
        smooth_image : bool, optional
            Should the `_seg_image` be gaussian filtered, by default True
        smooth_image_var : float, optional
            Variance of gaussian filter to apply to `_seg_image`, by default 0.3125/2
        marching_cubes_threshold : float, optional
            Threshold contour level to create surface mesh on, by default 0.5
        label_idx : int, optional
            Label value / index to create mesh from, by default None
        min_n_pixels : int, optional
            Minimum number of continuous pixels to include segmentation island
            in the surface mesh creation, by default None
        crop_percent : [type], optional
            [description], by default None

        Raises
        ------
        Exception
            If cropping & bone is not femur or tibia, then raise an error. 
        Exception
            If the total number of pixels segmentated (`n_pixels_labelled`) is
            < `min_n_pixels` then there is no object in the image.  
        Exception
            If no `_seg_image` and no `label_idx` then we don't know what tissue to create the 
            surface mesh from. 
        Exception
            If no `_seg_image` or `path_seg_image` then we have no image to create mesh from. 
        """     
        
        if self._seg_image is None:
            self.read_seg_image()
        
        # Bones might need to be cropped (this isnt necessary for cartilage)
        # So, adding this functionality to the processing steps before the bone mesh is created
        if crop_percent is not None:
            self._crop_percent = crop_percent
        if (self._crop_percent is not None) and (('femur' in self._bone) or ('tibia' in self._bone)):
            if 'femur' in self._bone:
                bone_crop_distal = True
            elif 'tibia' in self._bone:
                bone_crop_distal = False
            else:
                raise Exception('var bone should be "femur" or "tiba" got: {} instead'.format(self._bone))

            self._seg_image = crop_bone_based_on_width(self._seg_image,
                                                       self._label_idx,
                                                       percent_width_to_crop_height=self._crop_percent,
                                                       bone_crop_distal=bone_crop_distal)
        elif self._crop_percent is not None:
            warnings.warn(f'Trying to crop bone, but {self._bone} specified and only bones `femur`',
                          'or `tibia` currently supported for cropping. If using another bone, consider',
                          'making a pull request. If cropping not desired, set `crop_percent=None`.'
                )
        super().create_mesh(smooth_image=smooth_image, smooth_image_var=smooth_image_var, marching_cubes_threshold=marching_cubes_threshold, label_idx=label_idx, min_n_pixels=min_n_pixels)

    def create_cartilage_meshes(self,
                                image_smooth_var_cart=0.3125 / 2,
                                marching_cubes_threshold=0.5):
        """
        Helper function to create the list of cartilage meshes from the list of cartilage
        labels. 

        Parameters
        ----------
        image_smooth_var_cart : float
            Variance to smooth cartilage segmentations before finding surface using continuous
            marching cubes.             
        marching_cubes_threshold : float
            Threshold value to create cartilage surface at from segmentation images. 

        Notes
        -----
        ?? Should this function just be everything inside the for loop and then that 
        function gets called somewhere else? 
        """        

        self._list_cartilage_meshes = []
        for cart_label_idx in self._list_cartilage_labels:
            seg_array_view = sitk.GetArrayViewFromImage(self._seg_image)
            n_pixels_with_cart = np.sum(seg_array_view == cart_label_idx)
            if n_pixels_with_cart == 0:
                warnings.warn(
                    f"Not analyzing cartilage for label {cart_label_idx} because it doesnt have any pixels!",
                    UserWarning
                )
            else:
                cart_mesh = CartilageMesh(seg_image=self._seg_image,
                                            label_idx=cart_label_idx)
                cart_mesh.create_mesh(smooth_image_var=image_smooth_var_cart,
                                        marching_cubes_threshold=marching_cubes_threshold)
                self._list_cartilage_meshes.append(cart_mesh)


    def calc_cartilage_thickness(self,
                                 list_cartilage_labels=None,
                                 list_cartilage_meshes=None,
                                 image_smooth_var_cart=0.3125 / 2,
                                 marching_cubes_threshold=0.5,
                                 ray_cast_length=10.0,
                                 percent_ray_length_opposite_direction=0.25
                                 ):
        """
        Using bone mesh (`_mesh`) and the list of cartilage meshes (`list_cartilage_meshes`)
        calcualte the cartilage thickness for each node on the bone surface. 

        Parameters
        ----------
        list_cartilage_labels : list, optional
            Cartilag labels to be used to create cartilage meshes (if they dont
            exist), by default None
        list_cartilage_meshes : list, optional
            Cartilage meshes to be used for calculating cart thickness, by default None
        image_smooth_var_cart : float, optional
            Variance of gaussian filter to be applied to binary cartilage masks, 
            by default 0.3125/2
        marching_cubes_threshold : float, optional
            Threshold to create bone surface at, by default 0.5
        ray_cast_length : float, optional
            Length (mm) of ray to cast from bone surface when trying to find cartilage (inner &
            outter shell), by default 10.0
        percent_ray_length_opposite_direction : float, optional
            How far to project ray inside of the bone. This is done just in case the cartilage
            surface ends up slightly inside of (or coincident with) the bone surface, by default 0.25

        Raises
        ------
        Exception
            No cartilage available (either `list_cartilage_meshes` or `list_cartilage_labels`)
        """        
        # If new cartilage infor/labels are provided, then replace existing with these ones. 
        if list_cartilage_meshes is not None: self._list_cartilage_meshes = list_cartilage_meshes
        if list_cartilage_labels is not None: self._list_cartilage_labels = list_cartilage_labels

        # If no cartilage stuff provided, then cant do this function - raise exception. 
        if (self._list_cartilage_meshes is None) & (self._list_cartilage_labels is None):
            raise Exception('No cartilage meshes or list of cartilage labels are provided!  - These can be provided either to the class function `calc_cartilage_thickness` directly, or can be specified at the time of instantiating the `BoneMesh` class.')

        # if cartilage meshes don't exist yet, then make them. 
        if self._list_cartilage_meshes is None:
            self.create_cartilage_meshes(image_smooth_var_cart=image_smooth_var_cart,
                                         marching_cubes_threshold=marching_cubes_threshold)
        
        # pre-allocate empty thicknesses so that as labels are iterated over, they can all be appended to the same bone. 
        thicknesses = np.zeros(self._mesh.GetNumberOfPoints())
        
        # iterate over meshes and add their thicknesses to the thicknesses list. 
        for cart_mesh in self._list_cartilage_meshes:
            node_data = get_cartilage_properties_at_points(self._mesh,
                                                           cart_mesh._mesh,
                                                           t2_vtk_image=None,
                                                           #   seg_vtk_image=vtk_seg if assign_seg_label_to_bone is True else None,
                                                           seg_vtk_image=None,
                                                           ray_cast_length=ray_cast_length,
                                                           percent_ray_length_opposite_direction=percent_ray_length_opposite_direction
                                                           )
            thicknesses += node_data
        
        # Assign the thickness scalars to the bone mesh surface. 
        thickness_scalars = numpy_to_vtk(thicknesses)
        thickness_scalars.SetName('thickness (mm)')
        self._mesh.GetPointData().SetScalars(thickness_scalars)
    
    def assign_cartilage_regions(self,
                                 image_smooth_var_cart=0.3125 / 2,
                                 marching_cubes_threshold=0.5,
                                 ray_cast_length=10.0,
                                 percent_ray_length_opposite_direction=0.25,
                                 n_intersections=2):
        """
        Assign cartilage regions to the bone surface (e.g. medial/lateral tibial cartilage)
        - Can also be used for femur sub-regions (anterior, medial weight-bearing, etc.)

        Parameters
        ----------
        image_smooth_var_cart : float, optional
            Variance of gaussian filter to be applied to binary cartilage masks, 
            by default 0.3125/2
        marching_cubes_threshold : float, optional
            Threshold to create bone surface at, by default 0.5
        ray_cast_length : float, optional
            Length (mm) of ray to cast from bone surface when trying to find cartilage (inner &
            outter shell), by default 10.0
        percent_ray_length_opposite_direction : float, optional
            How far to project ray inside of the bone. This is done just in case the cartilage
            surface ends up slightly inside of (or coincident with) the bone surface, by default 0.25
        n_intersections : int, optional
            Number of intersections to use when ray casting to find cartilage regions, by default 2
            Only use 1 if not using for cartilage and just want to see what object is closest to the bone.
        """        
        tmp_filename = ''.join(random.choice(string.ascii_lowercase) for i in range(10)) + '.nrrd'
        path_save_tmp_file = os.path.join(tempfile.gettempdir(), tmp_filename)
        # if self._bone == 'femur':
        #     new_seg_image = qc.get_knee_segmentation_with_femur_subregions(seg_image,
        #                                                                    fem_cart_label_idx=1)
        #     sitk.WriteImage(new_seg_image, path_save_tmp_file)
        # else:
        sitk.WriteImage(self._seg_image, path_save_tmp_file)
        vtk_seg_reader = read_nrrd(path_save_tmp_file,
                                   set_origin_zero=True
                                   )
        vtk_seg = vtk_seg_reader.GetOutput()

        seg_transformer = SitkVtkTransformer(self._seg_image)

        # Delete tmp files
        safely_delete_tmp_file(tempfile.gettempdir(),
                               tmp_filename)
        
        self.apply_transform_to_mesh(transform=seg_transformer.get_inverse_transform())
        labels = np.zeros(self._mesh.GetNumberOfPoints(), dtype=int)

        # if cartilage meshes don't exist yet, then make them. 
        if self._list_cartilage_meshes is None:
            self.create_cartilage_meshes(image_smooth_var_cart=image_smooth_var_cart,
                                         marching_cubes_threshold=marching_cubes_threshold)
        
        # iterate over meshes and add their label (region) 
        for cart_mesh in self._list_cartilage_meshes:
            cart_mesh.apply_transform_to_mesh(transform=seg_transformer.get_inverse_transform())
            node_data = get_cartilage_properties_at_points(self._mesh,
                                                           cart_mesh._mesh,
                                                           t2_vtk_image=None,
                                                           seg_vtk_image=vtk_seg,
                                                           ray_cast_length=ray_cast_length,
                                                           percent_ray_length_opposite_direction=percent_ray_length_opposite_direction,
                                                           n_intersections=n_intersections
                                                           )
            labels += node_data[1]
            cart_mesh.reverse_all_transforms()

        # Assign the label (region) scalars to the bone mesh surface. 
        label_scalars = numpy_to_vtk(labels)
        label_scalars.SetName('labels')
        self._mesh.GetPointData().AddArray(label_scalars)

        self.reverse_all_transforms()
    
    def get_cart_thickness_mean(self,
                                region_idx):
        """
        Calculate the mean thickness of a given cartilage region.

        Parameters
        ----------
        region_idx : int
            The index of the cartilage region to calculate the mean thickness for.

        Returns
        -------
        float
            The mean thickness of the specified cartilage region.
        """
        region_array = self.get_scalar('labels')
        thickness_array = self.get_scalar('thickness (mm)')

        mean = np.nanmean(thickness_array[region_array==region_idx])
        return mean

    def get_cart_thickness_std(self,
                               region_idx):
        """
        Calculate the standard deviation of the thickness of a given cartilage region.

        Parameters
        ----------
        region_idx : int
            The index of the cartilage region to calculate the standard deviation for.

        Returns
        -------
        float
            The standard deviation of the thickness of the specified cartilage region.
        """
        region_array = self.get_scalar('labels')
        thickness_array = self.get_scalar('thickness (mm)')

        std = np.nanstd(thickness_array[region_array==region_idx])
        return std

    def get_cart_thickness_percentile(self,
                                      region_idx,
                                      percentile):
        """
        Calculate the thickness percentile of a given cartilage region.

        Parameters
        ----------
        region_idx : int
            The index of the cartilage region to calculate the thickness percentile for.
        percentile : float
            The percentile to calculate the thickness for. Should be between 0-100.

        Returns
        -------
        float
            The thickness percentile of the specified cartilage region.
        """
        region_array = self.get_scalar('labels')
        thickness_array = self.get_scalar('thickness (mm)')

        if percentile < 1:
            warnings.warn("Percentiles should be between 0-100 and not 0-1", UserWarning)
            
        data = thickness_array[region_array==region_idx]
        percentile = np.percentile(data, percentile)
        
        return percentile

    def calc_cartilage_t2(self,
                          path_t2_nrrd,
                          path_seg_to_t2_transform=None,
                          ray_cast_length=10.0,
                          percent_ray_length_opposite_direction=0.25):
        """
        Apply cartilage T2 values to bone surface. 

        Parameters
        ----------
        path_t2_nrrd : str
            Path to nrrd image of T2 map to load / use. 
        path_seg_to_t2_transform : str, optional
            Path to a transform file to be used for aligning T2 map with segmentations, 
            by default None
        ray_cast_length : float, optional
            Length (mm) of ray to cast from bone surface when trying to find cartilage (inner &
            outter shell), by default 10.0
        percent_ray_length_opposite_direction : float, optional
            How far to project ray inside of the bone. This is done just in case the cartilage
            surface ends up slightly inside of (or coincident with) the bone surface, by default 0.25
        """        
        print('Not yet implemented')
        # if self._list_cartilage_meshes is None:
        #     raise('Should calculate cartialge thickness before getting T2')
        #     # ALTERNATIVELY - COULD ALLOW PASSING OF CARTILAGE REGIONS IN HERE
        #     # THOUGH, DOES THAT JUST COMPLICATE THINGS? 

        # if path_seg_transform is not None:
        #     # this is in case there is a transformation needed to align the segmentation with the
        #     # underlying T2 image
        #     seg_transform = sitk.ReadTransform(path_seg_transform)
        #     seg_image = apply_transform_retain_array(self._seg_image,
        #                                              seg_transform,
        #                                              interpolator=sitk.sitkNearestNeighbor)
            
            
        #     versor = get_versor_from_transform(seg_transform)
        #     center_transform, rotate_transform, translate_transform = break_versor_into_center_rotate_translate_transforms(versor)
        #     # first apply negative of center of rotation to mesh
        #     self._mesh.apply_transform_to_mesh(transform=center_transform.GetInverse())
        #     # now apply the transform (rotation then translation)
        #     self._mesh.apply_transform_to_mesh(transform=rotate_transform.GetInverse())
        #     self._mesh.apply_transform_to_mesh(transform=translate_transform.GetInverse())
        #     #then undo the center of rotation
        #     self._mesh.apply_transform_to_mesh(transform=center_transform)

        # # Read t2 map (vtk format)
        # vtk_t2map_reader = read_nrrd(path_t2_nrrd,
        #                              set_origin_zero=True)
        # vtk_t2map = vtk_t2map_reader.GetOutput()
        # sitk_t2map = sitk.ReadImage(path_t2_nrrd)
        # t2_transformer = SitkVtkTransformer(sitk_t2map)

        # self._mesh.apply_transform_to_mesh(transform=t2_transformer.get_inverse_transform())

        # t2 = np.zeros(self._mesh.GetNumberOfPoints())
        # # iterate over meshes and add their t2 to the t2 list. 
        # for cart_mesh in self._list_cartilage_meshes:
        #     if path_seg_to_t2_transform is not None:
        #         # first apply negative of center of rotation to mesh
        #         cart_mesh.apply_transform_to_mesh(transform=center_transform.GetInverse())
        #         # now apply the transform (rotation then translation)
        #         cart_mesh.apply_transform_to_mesh(transform=rotate_transform.GetInverse())
        #         cart_mesh.apply_transform_to_mesh(transform=translate_transform.GetInverse())
        #         #then undo the center of rotation
        #         cart_mesh.apply_transform_to_mesh(transform=center_transform)

        #     cart_mesh.apply_transform_to_mesh(transform=t2_transformer.get_inverse_transform())
        #     _, t2_data = get_cartilage_properties_at_points(self._mesh,
        #                                                     cart_mesh._mesh,
        #                                                     t2_vtk_image=vtk_t2map,
        #                                                     ray_cast_length=ray_cast_length,
        #                                                     percent_ray_length_opposite_direction=percent_ray_length_opposite_direction
        #                                                     )
        #     t2 += t2_data
        #     cart_mesh.reverse_all_transforms()
        print('NOT DONE!!!')
        


    def smooth_surface_scalars(self,
                               smooth_only_cartilage=True,
                               scalar_sigma=1.6986436005760381,  # This is a FWHM = 4
                               scalar_array_name='thickness (mm)',
                               scalar_array_idx=None,
                               ):
                               
        """
        Function to smooth the scalars with name `scalar_array_name` on the bone surface. 

        Parameters
        ----------
        smooth_only_cartilage : bool, optional
            Should we only smooth where there is cartialge & ignore everywhere else, by default True
        scalar_sigma : float, optional
            Smoothing sigma (standard deviation or sqrt(variance)) for gaussian filter, by default 1.6986436005760381
            default is based on a Full Width Half Maximum (FWHM) of 4mm. 
        scalar_array_name : str
            Name of scalar array to smooth, default 'thickness (mm)'.
        scalar_array_idx : int, optional
            Index of the scalar array to smooth (alternative to using `scalar_array_name`) , by default None
        """
        if smooth_only_cartilage is True:
            loc_cartilage = np.where(vtk_to_numpy(self._mesh.GetPointData().GetArray('thickness (mm)')) > 0.01)[0]
        else:
            loc_cartilage = None
        self._mesh = gaussian_smooth_surface_scalars(self._mesh,
                                                     sigma=scalar_sigma,
                                                     idx_coords_to_smooth=loc_cartilage,
                                                     array_name=scalar_array_name,
                                                     array_idx=scalar_array_idx)

    @property
    def list_cartilage_meshes(self):
        """
        Convenience function to get the list of cartilage meshes

        Returns
        -------
        list
            A list of `CartilageMesh` objects associated with this bone
        """        
        return self._list_cartilage_meshes
    
    @list_cartilage_meshes.setter
    def list_cartilage_meshes(self, new_list_cartilage_meshes):
        """
        Convenience function to set the list of cartilage meshes

        Parameters
        ----------
        new_list_cartilage_meshes : list
            A list of `CartilageMesh` objects associated with this bone
        """
        if isinstance(new_list_cartilage_meshes, list):
            for mesh in new_list_cartilage_meshes:
                if type(mesh) != pymskt.mesh.meshes.CartilageMesh:
                    raise TypeError('Item in `list_cartilage_meshes` is not a `CartilageMesh`')
        elif isinstance(new_list_cartilage_meshes, pymskt.mesh.meshes.CartilageMesh):
            new_list_cartilage_meshes = [new_list_cartilage_meshes,]
        self._list_cartilage_meshes = new_list_cartilage_meshes
    
    @property
    def list_cartilage_labels(self):
        """
        Convenience function to get the list of labels for cartilage tissues associated
        with this bone. 

        Returns
        -------
        list
            list of `int`s for the cartilage tissues associated with this bone. 
        """        
        return self._list_cartilage_labels
    
    @list_cartilage_labels.setter
    def list_cartilage_labels(self, new_list_cartilage_labels):
        """
        Convenience function to set the list of labels for cartilage tissues associated
        with this bone

        Parameters
        ----------
        new_list_cartilage_labels : list
            list of `int`s for the cartilage tissues associated with this bone. 
        """
        if isinstance(new_list_cartilage_labels, list):
            for label in new_list_cartilage_labels:
                if not isinstance(label, int):
                    raise TypeError(f'Item in `list_cartilage_labels` is not a `int` - got {type(label)}')
        elif isinstance(new_list_cartilage_labels, int):
            new_list_cartilage_labels = [new_list_cartilage_labels,]
        self._list_cartilage_labels = new_list_cartilage_labels
    
    @property
    def crop_percent(self):
        """
        Convenience function to get the value that `crop_percent` is set to. 

        Returns
        -------
        float
            Floating point > 0.0 indicating how much of the length of the bone should be included
            when cropping - expressed as a proportion of the width. 
        """        
        return self._crop_percent
    
    @crop_percent.setter
    def crop_percent(self, new_crop_percent):
        """
        Convenience function to set the value that `crop_percent` is set to. 

        Parameters
        ----------
        new_crop_percent : float
            Floating point > 0.0 indicating how much of the length of the bone should be included
            when cropping - expressed as a proportion of the width. 
        """
        if not isinstance(new_crop_percent, float):
            raise TypeError(f'New `crop_percent` provided is type {type(new_crop_percent)} - expected `float`')
        self._crop_percent = new_crop_percent
    
    @property
    def bone(self):
        """
        Convenience function to get the name of the bone in this object. 

        Returns
        -------
        str
            Name of the bone in this object - used to help identify how to crop the bone. 
        """        
        return self._bone

    @bone.setter
    def bone(self, new_bone):
        """
        Convenience function to set the name of the bone in this object.         

        Parameters
        ----------
        new_bone : str
            Name of the bone in this object - used to help identify how to crop the bone. 
        """
        if not isinstance(new_bone, str):
            raise TypeError(f'New bone provided is type {type(new_bone)} - expected `str`')   
        self._bone = new_bone   
        


                     


