from logging import error
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

from pymskt.mesh import createMesh
from pymskt.utils import safely_delete_tmp_file, copy_image_transform_to_mesh
from pymskt.image import read_nrrd, crop_bone_based_on_width
from pymskt.mesh.utils import vtk_deep_copy
from pymskt.mesh.meshTools import gaussian_smooth_surface_scalars, get_mesh_physical_point_coords, get_cartilage_properties_at_points
from pymskt.mesh.createMesh import create_surface_mesh
from pymskt.mesh.meshTransform import (SitkVtkTransformer, 
                                      get_versor_from_transform, 
                                      break_versor_into_center_rotate_translate_transforms)
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
        self.path_seg_image = path_seg_image
        self.label_idx = label_idx
        self.min_n_pixels=min_n_pixels

        self.list_applied_transforms = []

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
            self.path_seg_image = path_seg_image
        
        # If seg image location / name exist, then load image else raise exception
        if (self.path_seg_image is not None):
            self._seg_image = sitk.ReadImage(self.path_seg_image)
        else:
            raise Exception('No file path (self.path_seg_image) provided.')
    
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
            self.label_idx = label_idx
        
        if self._seg_image is None:
            self.read_seg_image()
        
        # Ensure the image has a certain number of pixels with the label of interest, otherwise there might be an issue.
        if min_n_pixels is None: min_n_pixels = self.min_n_pixels
        seg_view = sitk.GetArrayViewFromImage(self._seg_image)
        n_pixels_labelled = sum(seg_view[seg_view == self.label_idx])

        if n_pixels_labelled < min_n_pixels:
            raise Exception('The mesh does not exist in this segmentation!, only {} pixels detected, threshold # is {}'.format(n_pixels_labelled, 
                                                                                                                               self.bone_label_threshold))
        tmp_filename = ''.join(random.choice(string.ascii_lowercase) for i in range(10)) + '.nrrd'
        self._mesh = create_surface_mesh(self._seg_image,
                                         self.label_idx,
                                         smooth_image_var,
                                         loc_tmp_save='/tmp',
                                         tmp_filename=tmp_filename,
                                         mc_threshold=marching_cubes_threshold,
                                         filter_binary_image=smooth_image
                                         )
        safely_delete_tmp_file('/tmp',
                               tmp_filename)
        # if smooth_image is True:
        #     tmp_filename = ''.join(random.choice(string.ascii_lowercase) for i in range(10)) + '.nrrd'
        #     self._mesh = create_surface_mesh(self._seg_image,
        #                                      self.label_idx,
        #                                      smooth_image_var,
        #                                      loc_tmp_save='/tmp',
        #                                      tmp_filename=tmp_filename,
        #                                      mc_threshold=marching_cubes_threshold
        #                                      )

        #     # Delete tmp files
        #     safely_delete_tmp_file('/tmp',
        #                            tmp_filename)
        # else:
        #     # If location & name of seg provided, read into nrrd_reader
        #     if (self.path_seg_image is not None):
        #         nrrd_reader = read_nrrd(self.path_seg_image, set_origin_zero=True)
        #         if self._seg_image is None:
        #             self.read_seg_image()
        #     # If no file data provided, but sitk image provided, save to disk and read to nrrd_reader.
        #     elif self._seg_image is not None:
        #         if self.label_idx is None:
        #             raise Exception('self.label_idx not specified and is necessary to extract isosurface')
        #         tmp_filename = ''.join(random.choice(string.ascii_lowercase) for i in range(10)) + '.nrrd'
        #         tmp_filepath = os.path.join('/tmp', tmp_filename)
        #         sitk.WriteImage(self._seg_image, tmp_filepath)
        #         nrrd_reader = read_nrrd(tmp_filepath, set_origin_zero=True)

        #         # Delete tmp files
        #         safely_delete_tmp_file('/tmp',
        #                                tmp_filename)
        #     # If neither of above provided, dont have enough data and raise error.
        #     else:
        #         raise Exception('Neither location/name (self.path_seg_image) of segmentation file '
        #                         'or a sitk image (self._seg_image) of the segmentation are provided! ')
        #     # Get surface mesh using discrete marching cubes
        #     self._mesh = createMesh.discrete_marching_cubes(nrrd_reader,
        #                                                     n_labels=1,
        #                                                     start_label=self.label_idx,
        #                                                     end_label=self.label_idx)
        #     # & then apply image transform for it to be in right loc.
        #     self._mesh = copy_image_transform_to_mesh(self._mesh, self._seg_image)
    
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
        pv_smooth_mesh = pv.wrap(self._mesh)
        clus = pyacvd.Clustering(pv_smooth_mesh)
        clus.subdivide(subdivisions)
        clus.cluster(clusters)
        self._mesh = clus.create_mesh()

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
                self.list_applied_transforms.append(transform)
            else:
                pass
        else:
            raise Exception('No transform or transformer provided')

    def reverse_most_recent_transform(self):
        """
        Function to undo the most recent transformation stored in self.list_applied_transforms
        """
        transform = self.list_applied_transforms.pop()
        transform.Inverse()
        self.apply_transform_to_mesh(transform=transform, save_transform=False)

    def reverse_all_transforms(self):
        """
        Function to iterate over all of the self.list_applied_transforms (in reverse order) and undo them.
        """
        while len(self.list_applied_transforms) > 0:
            self.reverse_most_recent_transform()

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
        self.crop_percent = crop_percent
        self.bone = bone
        self.list_cartilage_meshes = list_cartilage_meshes
        self.list_cartilage_labels = list_cartilage_labels

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
            self.crop_percent = crop_percent
        if self.crop_percent != 1.0:
            if 'femur' in self.bone:
                bone_crop_distal = True
            elif 'tibia' in self.bone:
                bone_crop_distal = False
            else:
                raise Exception('var bone should be "femur" or "tiba" got: {} instead'.format(self.bone))

            self._seg_image = crop_bone_based_on_width(self._seg_image,
                                                       self.label_idx,
                                                       percent_width_to_crop_height=self.crop_percent,
                                                       bone_crop_distal=bone_crop_distal)
           
        super().create_mesh(smooth_image=smooth_image, smooth_image_var=smooth_image_var, marching_cubes_threshold=marching_cubes_threshold, label_idx=label_idx, min_n_pixels=min_n_pixels)

    def create_cartilage_meshes(self,
                                image_smooth_var_cart,
                                marching_cubes_threshold):
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

        self.list_cartilage_meshes = []
        for cart_label_idx in self.list_cartilage_labels:
            seg_array_view = sitk.GetArrayViewFromImage(self._seg_image)
            n_pixels_with_cart = np.sum(seg_array_view == cart_label_idx)
            if n_pixels_with_cart == 0:
                print("Not analyzing cartilage for label {} because it doesnt have any pixels!".format(cart_label_idx))
                continue

            cart_mesh = CartilageMesh(seg_image=self._seg_image,
                                        label_idx=cart_label_idx)
            cart_mesh.create_mesh(smooth_image_var=image_smooth_var_cart,
                                    marching_cubes_threshold=marching_cubes_threshold)
            self.list_cartilage_meshes.append(cart_mesh)


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
        if list_cartilage_meshes is not None: self.list_cartilage_meshes = list_cartilage_meshes
        if list_cartilage_labels is not None: self.list_cartilage_labels = list_cartilage_labels

        # If no cartilage stuff provided, then cant do this function - raise exception. 
        if (self.list_cartilage_meshes is None) & (self.list_cartilage_labels is None):
            raise Exception('No cartilage meshes or list of cartilage labels are provided!  - These can be provided either to the class function `calc_cartilage_thickness` directly, or can be specified at the time of instantiating the `BoneMesh` class.')

        # if cartilage meshes don't exist yet, then make them. 
        if self.list_cartilage_meshes is None:
            self.create_cartilage_meshes(image_smooth_var_cart=image_smooth_var_cart,
                                         marching_cubes_threshold=marching_cubes_threshold)
        
        # pre-allocate empty thicknesses so that as labels are iterated over, they can all be appended to the same bone. 
        thicknesses = np.zeros(self._mesh.GetNumberOfPoints())
        
        # iterate over meshes and add their thicknesses to the thicknesses list. 
        for cart_mesh in self.list_cartilage_meshes:
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
        # if self.bone == 'femur':
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
        if self.list_cartilage_meshes is None:
            self.create_cartilage_meshes(image_smooth_var_cart=image_smooth_var_cart,
                                         marching_cubes_threshold=marching_cubes_threshold)
        
        # iterate over meshes and add their label (region) 
        for cart_mesh in self.list_cartilage_meshes:
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
        # if self.list_cartilage_meshes is None:
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
        # for cart_mesh in self.list_cartilage_meshes:
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
        
        


                     


