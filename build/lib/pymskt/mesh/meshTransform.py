import vtk
import numpy as np
from pymskt.mesh.utils import vtk_deep_copy
from pymskt.utils import create_4x4_from_3x3

def apply_transform(source, transform):
    transform_filter = vtk.vtkTransformPolyDataFilter()
    transform_filter.SetInputData(source)
    transform_filter.SetTransform(transform)
    transform_filter.Update()
    return transform_filter.GetOutput()

def copy_image_transform_to_mesh(mesh, image, verbose=False):
    transform_array = create_4x4_from_3x3(image.GetDirection(), translation=image.GetOrigin())
    transform = vtk.vtkTransform()
    transform.SetMatrix(transform_array.flatten())

    if verbose is True:
        print(transform)

    return apply_transform(source=mesh, transform=transform)

def get_icp_transform(source, target, max_n_iter=1000, n_landmarks=1000, reg_mode='similarity'):
    """
    transform = ('rigid': true rigid, translation only; similarity': rigid + equal scale)
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

class SitkVtkTransformer:
    def __init__(self,
                 sitk_image=None,
                 three_by_three=None,
                 translation=None,
                 center=None
                 ):
        if sitk_image is not None:
            self.transform_array = create_4x4_from_3x3(sitk_image.GetDirection(), translation=sitk_image.GetOrigin())
        # Need to setup below code to receive 3x3, translation, and center to apply a transform. 

        # elif three_by_three is not None:
        #     self.transform_array = create_4x4_from_3x3(three_by_three)
        #     # if translation is not None:
        #     #     # self.transform_array[:3, 3] = translation
        # self.translation = translation

        # # Center is in this, but is not currently used. Below is commented out to show how it could be automatically applied. 
        # self.center = center 


        # Easier to rotate mesh to image instead of image to mesh. So take inverse of transform.
        self.inverse_transform_array = np.linalg.inv(self.transform_array)

    def get_transform(self):
        # create vtk transform.
        transform = vtk.vtkTransform()
        # if self.center is not None:
        #     transform.PostMultiply()
        #     transform.Translate(-self.center[0], -self.center[1], -self.center[2])
        transform.SetMatrix(self.transform_array.flatten())
        # if self.translation is not None:
        #     transform.Translate(self.translation)
        # if self.center is not None:
        #     transform.Translate(self.center[0], self.center[1], self.center[2])
        return transform

    def get_inverse_transform(self):
        transform = vtk.vtkTransform()
        transform.SetMatrix(self.inverse_transform_array.flatten())
        # if self.translation is not None:
        #     transform.Translate(tuple([-x for x in self.translation]))
        return transform

    def get_transformer(self):
        # Get transform filter and apply transform to bone mesh.
        transformer = vtk.vtkTransformPolyDataFilter()
        transformer.SetTransform(self.get_transform())
        # Now, can use this later to transform the bone/
        return transformer

    def get_inverse_transformer(self):
        inverse_transformer = vtk.vtkTransformPolyDataFilter()
        inverse_transformer.SetTransform(self.get_inverse_transform())
        return inverse_transformer

    def apply_transform_to_mesh(self,
                                mesh):
        transformer = self.get_transformer()
        transformer.SetInputData(mesh)
        transformer.Update()
        return vtk_deep_copy(transformer.GetOutput())

    def apply_inverse_transform_to_mesh(self,
                                        mesh):
        inverse_transformer = self.get_inverse_transformer()
        inverse_transformer.SetInputData(mesh)
        inverse_transformer.Update()
        return vtk_deep_copy(inverse_transformer.GetOutput())