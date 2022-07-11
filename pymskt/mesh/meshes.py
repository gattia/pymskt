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
                                   resample_surface
                                   )
from pymskt.mesh.createMesh import create_surface_mesh
from pymskt.mesh.meshTransform import (SitkVtkTransformer, 
                                       get_versor_from_transform, 
                                       break_versor_into_center_rotate_translate_transforms)
from pymskt.mesh.meshRegistration import non_rigidly_register
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
        self._mesh = mesh
        self._seg_image = seg_image
        self._path_seg_image = path_seg_image
        self._label_idx = label_idx
        self._min_n_pixels = min_n_pixels

        self._list_applied_transforms = []

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
                    min_n_pixels=None):
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
                                         loc_tmp_save='/tmp',
                                         tmp_filename=tmp_filename,
                                         mc_threshold=marching_cubes_threshold,
                                         filter_binary_image=smooth_image
                                         )
        safely_delete_tmp_file('/tmp',
                               tmp_filename)
    
    def save_mesh(self,
                  filepath):
        """
        Save the surface mesh from this class to disk. 

        Parameters
        ----------
        filepath : str
            Location & filename to save the surface mesh (vtk.vtkPolyData) to. 
        """        
        io.write_vtk(self._mesh, filepath)

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

    def copy_scalars_from_other_mesh_to_currect(
        self,
        other_mesh,
        new_scalars_name='scalars_from_other_mesh',
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
            VERY closely matches. Otherwise, the transfered scalars will be bad.  

        Parameters
        ----------
        other_mesh : pymskt.mesh.Mesh or vtk.vtkPolyData
            Mesh we want to copy 
        new_scalars_name : str, optional
           What to name the scalars being transfered to this current mesh, by default 'scalars_from_other_mesh'
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
        if type(other_mesh) is Mesh:
            other_mesh = other_mesh.mesh
        elif type(other_mesh) is vtk.vtkPolyData:
            pass
        else:
            raise TypeError(f'other_mesh must be type `pymskt.mesh.Mesh` or `vtk.vtkPolyData` and received: {type(other_mesh)}')

        if weighted_avg is True:
            transferred_scalars = transfer_mesh_scalars_get_weighted_average_n_closest(
                self._mesh,
                other_mesh,
                sigma=sigma,
                idx_coords_to_smooth_base=idx_coords_to_smooth_base,
                idx_coords_to_smooth_second=idx_coords_to_smooth_other,
                set_non_smoothed_scalars_to_zero=set_non_smoothed_scalars_to_zero
            )
        else:
            transferred_scalars = smooth_scalars_from_second_mesh_onto_base(
                self._mesh,
                other_mesh,
                n=n_closest
            )
        vtk_transferred_scalars = numpy_to_vtk(transferred_scalars)
        vtk_transferred_scalars.SetName(new_scalars_name)
        self._mesh.GetPointData().AddArray(vtk_transferred_scalars)

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
                 crop_percent=1.0,
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
        if self._crop_percent != 1.0:
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
                                 percent_ray_length_opposite_direction=0.25):
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
        """        
        tmp_filename = ''.join(random.choice(string.ascii_lowercase) for i in range(10)) + '.nrrd'
        path_save_tmp_file = os.path.join('/tmp', tmp_filename)
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
        safely_delete_tmp_file('/tmp',
                               tmp_filename)
        
        self.apply_transform_to_mesh(transform=seg_transformer.get_inverse_transform())
        labels = np.zeros(self._mesh.GetNumberOfPoints(), dtype=np.int)

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
                                                           percent_ray_length_opposite_direction=percent_ray_length_opposite_direction
                                                           )
            labels += node_data[1]
            cart_mesh.reverse_all_transforms()

        # Assign the label (region) scalars to the bone mesh surface. 
        label_scalars = numpy_to_vtk(labels)
        label_scalars.SetName('labels')
        self._mesh.GetPointData().AddArray(label_scalars)

        self.reverse_all_transforms()

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
        loc_cartilage = np.where(vtk_to_numpy(self._mesh.GetPointData().GetArray('thickness (mm)')) > 0.01)[0]
        self._mesh = gaussian_smooth_surface_scalars(self._mesh,
                                                     sigma=scalar_sigma,
                                                     idx_coords_to_smooth=loc_cartilage if smooth_only_cartilage is True else None,
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
        if type(new_list_cartilage_meshes) is list:
            for mesh in new_list_cartilage_meshes:
                if type(mesh) != pymskt.mesh.meshes.CartilageMesh:
                    raise TypeError('Item in `list_cartilage_meshes` is not a `CartilageMesh`')
        elif type(new_list_cartilage_meshes) is pymskt.mesh.meshes.CartilageMesh:
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
        if type(new_list_cartilage_labels) == list:
            for label in new_list_cartilage_labels:
                if type(label) != int:
                    raise TypeError(f'Item in `list_cartilage_labels` is not a `int` - got {type(label)}')
        elif type(new_list_cartilage_labels) == int:
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
        if type(new_crop_percent) != float:
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
        if type(new_bone) != str:
            raise TypeError(f'New bone provided is type {type(new_bone)} - expected `str`')   
        self._bone = new_bone   
        


                     


