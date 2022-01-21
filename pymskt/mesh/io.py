import vtk
import os
from vtk.util.numpy_support import vtk_to_numpy, numpy_to_vtk

def read_vtk(filepath):
    """
    Function to read surface mesh using into `vtk.vtkPolydata`.

    Parameters
    ----------
    filepath : str
        String of filepath to read. 

    Returns
    -------
    vtk.PolyData
        The surface mesh at `filepath`.
    """    
    reader = vtk.vtkPolyDataReader()
    reader.SetFileName(filepath)
    reader.Update()
    return reader.GetOutput()


def write_vtk(mesh, filepath, scalar_name=None, points_dtype=float):
    """
    Save vtk polydata to disk. 

    Parameters
    ----------
    mesh : vtk.vtkPolyData
        PolyData of surface mesh to be saved to disk. 
    filepath : str
        Location & filename to save mesh to. 
    scalar_name : str, optional
        A name to assign to the active scalars of the surface mesh before saving it, by default None
        meaning that no name will be applied.
    points_dtype : type, optional
        Specifies the datatype that the points should be. 
        If they are the wrong datatype then they are changed to the requisite
        type before savinge (writing) the file. 

    
    Notes
    ----------
    Fileversion is the old legacy version because new tools (Slicer, Paraview) dont support vtk
    version 5.1 which shipped with VTK 9. 
    https://discourse.vtk.org/t/legacy-polydata-file-compatibility/5354
    https://discourse.vtk.org/t/can-we-write-out-the-old-vtk-4-2-file-format-with-vtk-9/5066/17
    https://gitlab.kitware.com/vtk/vtk/-/merge_requests/7652/diffs?commit_id=7f76b9e97b1a05cfe4fcd5f9af58f0d7a385b639#528e66f324b988666af9696641f935da71b6f670
    """

    points = mesh.GetPoints()
    if vtk_to_numpy(points.GetData()).dtype != points_dtype:
        points.SetData(numpy_to_vtk(vtk_to_numpy(points.GetData()).astype(float)))

    writer = vtk.vtkPolyDataWriter()
    writer.SetFileName(filepath)
    writer.SetInputData(mesh)
    if scalar_name is not None:
        writer.SetScalarsName(scalar_name)
    writer.Write()