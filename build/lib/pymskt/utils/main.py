import vtk
import numpy as np
import os
import errno

def l2n(l):
    return np.array(l)


def n2l(n):
    return list(n)


def create_4x4_from_3x3(three_by_three, translation=None):
    if len(three_by_three) == 9:
        three_by_three = np.reshape(three_by_three, (3, 3))
    four_by_four = np.identity(4)
    four_by_four[:3, :3] = three_by_three

    # if translation is provided, put it in the last column
    if translation is not None:
        four_by_four[:3, 3] = translation
    return four_by_four


def copy_image_transform_to_mesh(mesh, image, verbose=False):
    transform_array = create_4x4_from_3x3(image.GetDirection())
    transform_array[:3, 3] = image.GetOrigin()
    transform = vtk.vtkTransform()
    transform.SetMatrix(transform_array.flatten())
    #     transform.Translate(image.GetOrigin())

    if verbose is True:
        print(transform)

    transformer = vtk.vtkTransformPolyDataFilter()
    transformer.SetTransform(transform)
    transformer.SetInputData(mesh)
    transformer.Update()
    return transformer.GetOutput()


def sigma2fwhm(sigma):
    return sigma * np.sqrt(8 * np.log(2))

def fwhm2sigma(fwhm):
    return fwhm / np.sqrt(8 * np.log(2))

def safely_delete_tmp_file(location,
                           filename):
    try:
        os.remove(os.path.join(location, filename))
    except OSError as exc:
        if exc.errno != errno.ENOENT:
            raise
        pass