import vtk
import pyacvd
import pyvista as pv
import SimpleITK as sitk
import os
import random
import string

from pymskt.mesh import createMesh
from pymskt.utils import safely_delete_tmp_file, copy_image_transform_to_mesh
from pymskt.image import read_nrrd
from pymskt.mesh.utils import vtk_deep_copy
from pymskt.mesh.meshTools import get_mesh_physical_point_coords
from pymskt.mesh.createMesh import create_surface_mesh_smoothed
import pymskt.mesh.io as io

class Mesh:
    def __init__(self,
                 mesh=None,
                 seg_image=None,
                 path_seg_image=None,
                 label_idx=None,
                 ):
        self._mesh = mesh
        self._seg_image = seg_image
        self.path_seg_image = path_seg_image
        self.label_idx = label_idx

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
                    label_idx=None):
        # allow assigning label idx during mesh creation step. 
        if label_idx is not None:
            self.label_idx = label_idx
        
        if self._seg_image is None:
            self.read_seg_image()

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
                 ):
        super().__init__(mesh=mesh,
                         seg_image=seg_image,
                         path_seg_image=path_seg_image,
                         label_idx=label_idx)


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
                 list_cart_meshes=None,
                 list_cart_labels=None
                 ):
        super().__init__(mesh=mesh,
                         seg_image=seg_image,
                         path_seg_image=path_seg_image,
                         label_idx=label_idx)

        self.list_cart_meshes = list_cart_meshes
        self.list_cart_labels = list_cart_labels
