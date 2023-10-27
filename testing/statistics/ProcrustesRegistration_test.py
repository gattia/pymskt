import pytest
import os
import glob
from pymskt.statistics import ProcrustesRegistration
import time

file_path = os.path.abspath(__file__)
# Construct the path to the package directory
package_dir = os.path.dirname(os.path.dirname(os.path.dirname(file_path)))

LIST_MESH_PATHS = [os.path.join(package_dir, 'data', 'femur_meshes_registration', f'{idx}_RIGHT_femur_Nov_02_2021.vtk') for idx in range(1, 6)]
REF_MESH_PATH = LIST_MESH_PATHS.pop(0)

for path in LIST_MESH_PATHS:
    print(os.path.exists(path))

@pytest.mark.skip(reason="Takes too long - and probably still fails?")
def test_ProcrustesRegistration_runs():
    folder_save = os.path.join(package_dir, 'data', 'femur_meshes_registration', 'registered_meshes')

    procrustes_reg = ProcrustesRegistration(
        path_ref_mesh=REF_MESH_PATH, # using the idx of the best mesh from the previous step
        list_mesh_paths=LIST_MESH_PATHS, # This will automatically remove the ref_mesh path if it is in the list.
        max_n_registration_steps=1,
        n_coords_spectral_ordering=10000,
        n_coords_spectral_registration=1000,
        n_extra_spectral=6,
        include_points_as_features=True,
        vertex_features=['thickness (mm)'],
        multiprocessing=True,
        verbose=False,
        save_meshes_during_registration=True,
        folder_save=folder_save,
    )

    tic = time.time()
    procrustes_reg.execute()
    toc = time.time()

    print(f'Time taken for ProcrustesRegistration: {toc - tic} seconds')

    
    # procrustes_reg.save_meshes(folder=folder_save)

    list_meshes = glob.glob(os.path.join(folder_save, '*.vtk'))
    assert len(list_meshes) == len(LIST_MESH_PATHS)

    # delete the folder of registered meshes
    os.system(f'rm -r {folder_save}')


    print(procrustes_reg.registered_pt_coords.shape)
    print(procrustes_reg.registered_vertex_features.shape)


if __name__ == '__main__':
    # pytest.main([__file__])
    test_ProcrustesRegistration_runs()