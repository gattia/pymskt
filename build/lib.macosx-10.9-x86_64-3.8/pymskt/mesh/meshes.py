from logging import error
import numpy as np
import vtk
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
from pymskt.mesh.createMesh import create_surface_mesh_smoothed
from pymskt.mesh.meshTransform import SitkVtkTransformer
import pymskt.mesh.io as io

class Mesh:
    def __init__(self,
                 mesh=None,
                 seg_image=None,
                 path_seg_image=None,
                 label_idx=None,
                 min_n_pixels=5000
                 ):
        self._mesh = mesh
        self._seg_image = seg_image
        self.path_seg_image = path_seg_image
        self.label_idx = label_idx
        self.min_n_pixels=min_n_pixels

        self.list_applied_transforms = []

    def read_seg_image(self,
                       path_seg_image=None):
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

        if smooth_image is True:
            tmp_filename = ''.join(random.choice(string.ascii_lowercase) for i in range(10)) + '.nrrd'
            self._mesh = create_surface_mesh_smoothed(self._seg_image,
                                                      self.label_idx,
                                                      smooth_image_var,
                                                      loc_tmp_save='/tmp',
                                                      tmp_filename=tmp_filename,
                                                      mc_threshold=marching_cubes_threshold
                                                      )

            # Delete tmp files
            safely_delete_tmp_file('/tmp',
                                   tmp_filename)
        else:
            # If location & name of seg provided, read into nrrd_reader
            if (self.path_seg_image is not None):
                nrrd_reader = read_nrrd(self.path_seg_image, set_origin_zero=True)
                if self._seg_image is None:
                    self.read_seg_image()
            # If no file data provided, but sitk image provided, save to disk and read to nrrd_reader.
            elif self._seg_image is not None:
                if self.label_idx is None:
                    raise Exception('self.label_idx not specified and is necessary to extract isosurface')
                tmp_filename = ''.join(random.choice(string.ascii_lowercase) for i in range(10)) + '.nrrd'
                tmp_filepath = os.path.join('/tmp', tmp_filename)
                sitk.WriteImage(self._seg_image, tmp_filepath)
                nrrd_reader = read_nrrd(tmp_filepath, set_origin_zero=True)

                # Delete tmp files
                safely_delete_tmp_file('/tmp',
                                       tmp_filename)
            # If neither of above provided, dont have enough data and raise error.
            else:
                raise Exception('Neither location/name (self.path_seg_image) of segmentation file '
                                'or a sitk image (self._seg_image) of the segmentation are provided! ')
            # Get surface mesh using discrete marching cubes
            self._mesh = createMesh.discrete_marching_cubes(nrrd_reader,
                                                            n_labels=1,
                                                            start_label=self.label_idx,
                                                            end_label=self.label_idx)
            # & then apply image transform for it to be in right loc.
            self._mesh = copy_image_transform_to_mesh(self._mesh, self._seg_image)
    
    def save_mesh(self,
                  filepath):
        io.write_vtk(self._mesh, filepath)

    def resample_surface(self,
                         subdivisions=2,
                         clusters=10000
                         ):
        pv_smooth_mesh = pv.wrap(self._mesh)
        clus = pyacvd.Clustering(pv_smooth_mesh)
        # mesh is not dense enough for uniform remeshing
        clus.subdivide(subdivisions)
        clus.cluster(clusters)
        self._mesh = clus.create_mesh()

    def apply_transform_to_mesh(self,
                                transform=None,
                                transformer=None,
                                save_transform=True):
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
        :return:
        """
        transform = self.list_applied_transforms.pop()
        transform.Inverse()
        self.apply_transform_to_mesh(transform=transform, save_transform=False)

    def reverse_all_transforms(self):
        """
        Function to iterate over all of the self.list_applied_transforms (in reverse order) and undo them.
        :return:
        """
        while len(self.list_applied_transforms) > 0:
            self.reverse_most_recent_transform()

    @property
    def seg_image(self):
        return self._seg_image

    @seg_image.setter
    def seg_image(self, new_seg_image):
        self._seg_image = new_seg_image

    @property
    def mesh(self):
        return self._mesh

    @mesh.setter
    def mesh(self, new_mesh):
        self._mesh = new_mesh
    
    @property
    def point_coords(self):
        return get_mesh_physical_point_coords(self._mesh)


class CartilageMesh(Mesh):
    """
    Class to create, store, and process cartilage meshes
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
           
        return super().create_mesh(smooth_image=smooth_image, smooth_image_var=smooth_image_var, marching_cubes_threshold=marching_cubes_threshold, label_idx=label_idx, min_n_pixels=min_n_pixels)

    def smooth_surface_scalars(self,
                               scalar_label='thickness (mm)', # alternates = 't2 (ms)'
                               smooth_only_cartilage=True,
                               ):
        if smooth_only_cartilage is True:
            loc_cartilage = np.where(vtk_to_numpy(self._mesh.GetPointData().GetArray(scalar_label)) > 0.01)[0]
        


    def create_cartilage_meshes(self,
                                image_smooth_var_cart,
                                marching_cubes_threshold):
        """
        Helper function to create the list of cartilage meshes from the list of cartilage
        labels. 

        ?? Should this function just be everything inside the for loop and then that function gets called somewhere else? 
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
        
        # iterate over meshes and add their thicknesses to the thicknesses list. 
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

        # Assign the thickness scalars to the bone mesh surface. 
        label_scalars = numpy_to_vtk(labels)
        label_scalars.SetName('labels')
        self._mesh.GetPointData().AddArray(label_scalars)

        self.reverse_all_transforms()
        
    def calc_cartilage_t2():
        print('Not yet implemented')      

    def smooth_surface_scalars(self,
                               smooth_only_cartilage=True,
                               scalar_sigma=1.6986436005760381,  # This is a FWHM = 4
                               scalar_array_name='thickness (mm)',
                               scalar_array_idx=None,
                               ):
        loc_cartilage = np.where(vtk_to_numpy(self._mesh.GetPointData().GetArray('thickness (mm)')) > 0.01)[0]
        gaussian_smooth_surface_scalars(self._mesh,
                                        sigma=scalar_sigma,
                                        idx_coords_to_smooth=loc_cartilage if smooth_only_cartilage is True else None,
                                        array_name=scalar_array_name,
                                        array_idx=scalar_array_idx)
        
        


                     


