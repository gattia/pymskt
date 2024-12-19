import os
import time
import warnings
from datetime import date
from multiprocessing import Pool
from re import sub

import numpy as np
import pyacvd
import pyvista as pv
from vtk.util.numpy_support import vtk_to_numpy

from pymskt.mesh import io
from pymskt.mesh.meshRegistration import get_icp_transform, non_rigidly_register
from pymskt.mesh.meshTools import (
    get_mesh_physical_point_coords,
    get_mesh_point_features,
    resample_surface,
    set_mesh_physical_point_coords,
    set_mesh_point_features,
    transfer_mesh_scalars_get_weighted_average_n_closest,
)
from pymskt.mesh.meshTransform import apply_transform, read_linear_transform, write_linear_transform
from pymskt.mesh.utils import get_symmetric_surface_distance, vtk_deep_copy

today = date.today()


class FindReferenceMeshICP:
    """
    For list of meshes perform all possible ICP registrations to identify mesh with smallest
    surface error to all other meshes.

    Parameters
    ----------
    list_meshes : _type_
        _description_
    """

    def __init__(
        self,
        list_mesh_paths,
        max_n_iter=1000,
        n_landmarks=1000,
        reg_mode="similarity",
        verbose=True,
    ):
        """
        Perform ICP registration between all pairs of meshes. Calculate
        symmetric surface distance for all registered meshes. Find target
        mesh with smallest mean surface error to all other meshes.

        This smallest error mesh is the refrence mesh for the next step of
        SSM pipelines (procrustes using non-rigid registration)

        Parameters
        ----------
        list_mesh_paths : _type_
            _description_
        max_n_iter : int, optional
            _description_, by default 1000
        n_landmarks : int, optional
            _description_, by default 1000
        reg_mode : str, optional
            _description_, by default 'similarity'
        verbose : bool, optional
            _description_, by default True
        """
        self.list_mesh_paths = list_mesh_paths
        self.n_meshes = len(list_mesh_paths)
        self._symm_surface_distances = np.zeros((self.n_meshes, self.n_meshes), dtype=float)
        self._mean_errors = None

        self.max_n_iter = max_n_iter
        self.n_landmarks = n_landmarks
        self.reg_mode = reg_mode

        self.verbose = verbose

        self._ref_idx = None
        self._ref_path = None

    def register_meshes(self, idx1_target, idx2_source):
        target = io.read_vtk(self.list_mesh_paths[idx1_target])
        source = io.read_vtk(self.list_mesh_paths[idx2_source])

        icp_transform = get_icp_transform(
            source,
            target,
            max_n_iter=self.max_n_iter,
            n_landmarks=self.n_landmarks,
            reg_mode=self.reg_mode,
        )

        transformed_source = apply_transform(source, icp_transform)

        symmetric_surf_distance = get_symmetric_surface_distance(target, transformed_source)

        self._symm_surface_distances[idx1_target, idx2_source] = symmetric_surf_distance

    def get_template_idx(self):
        self._mean_errors = np.mean(self._symm_surface_distances, axis=1)
        self._ref_idx = np.argmin(self._mean_errors)
        self._ref_path = self.list_mesh_paths[self._ref_idx]

    def execute(self):
        if self.verbose is True:
            print(f"Starting registrations, there are {len(self.list_mesh_paths)} meshes")
        for idx1_target, target_path in enumerate(self.list_mesh_paths):
            if self.verbose is True:
                print(f"\tStarting target mesh {idx1_target}")
            for idx2_source, source_path in enumerate(self.list_mesh_paths):
                if self.verbose is True:
                    print(f"\t\tStarting source mesh {idx2_source}")
                # If the target & mesh are same skip, errors = 0
                if idx1_target == idx2_source:
                    continue
                else:
                    self.register_meshes(idx1_target, idx2_source)
        if self.verbose is True:
            print("Finished all registrations!")

        self.get_template_idx()

    @property
    def ref_idx(self):
        return self._ref_idx

    @property
    def ref_path(self):
        return self._ref_path

    @property
    def symm_surface_distances(self):
        return self._symm_surface_distances

    @property
    def mean_errors(self):
        return self._mean_errors


class ProcrustesRegistration:
    # https://en.wikipedia.org/wiki/Generalized_Procrustes_analysis
    def __init__(
        self,
        path_ref_mesh,
        list_mesh_paths,
        tolerance1=2e-1,
        tolerance2=1e-2,
        max_n_registration_steps=10,
        verbose=True,
        remesh_each_step=False,
        patience=2,
        ref_mesh_eigenmap_as_reference=True,
        registering_secondary_bone=False,  # True if registering secondary bone of joint, after
        # primary already used for initial registration. E.g.,
        # already did femur for knee, now applying to tibia/patella
        vertex_features=None,
        include_ref_in_sample=True,
        save_meshes_during_registration=False,
        folder_save=None,
        save_mesh_suffix=f'procrustes_registered_{today.strftime("%b")}_{today.day}_{today.year}',
        multiprocessing=True,
        num_processes=None,
        remove_temp_icp=True,
        **kwargs,
    ):
        self.path_ref_mesh = path_ref_mesh
        self.list_mesh_paths = list_mesh_paths
        # Ensure that path_ref_mesh is in list & at index 0
        if self.path_ref_mesh in self.list_mesh_paths:
            path_ref_idx = self.list_mesh_paths.index(self.path_ref_mesh)
            self.list_mesh_paths.pop(path_ref_idx)
        self.include_ref_in_sample = include_ref_in_sample
        if self.include_ref_in_sample is True:
            self.list_mesh_paths.insert(0, self.path_ref_mesh)

        self._ref_mesh = io.read_vtk(self.path_ref_mesh)
        self.n_points = self._ref_mesh.GetNumberOfPoints()
        self.ref_mesh_eigenmap_as_reference = ref_mesh_eigenmap_as_reference

        self.mean_mesh = None

        self.tolerance1 = tolerance1
        self.tolerance2 = tolerance2
        self.max_n_registration_steps = max_n_registration_steps

        self.remove_temp_icp = remove_temp_icp

        self.kwargs = kwargs
        # ORIGINALLY THIS WAS THE LOGIC:
        # Ensure that the source mesh (mean, or reference) is the base mesh
        # We want all meshes aligned with this reference. Then we want
        # to apply a "warp" of the ref/mean mesh to make it
        # EXCETION - if we are registering a secondary bone in a joint model
        # E.g., for registering tibia/patella in knee model.
        self.kwargs["icp_register_first"] = True
        if registering_secondary_bone is False:
            self.kwargs["icp_reg_target_to_source"] = True
        elif registering_secondary_bone is True:
            self.kwargs["icp_reg_target_to_source"] = False

        self.vertex_features = vertex_features

        self._registered_pt_coords = np.zeros((len(list_mesh_paths), self.n_points, 3), dtype=float)
        if self.include_ref_in_sample is True:
            self._registered_pt_coords[0, :, :] = get_mesh_physical_point_coords(self._ref_mesh)

        if self.vertex_features is not None:
            self._registered_vertex_features = np.zeros(
                (len(self.list_mesh_paths), self.n_points, len(self.vertex_features)), dtype=float
            )
        else:
            self._registered_vertex_features = None

        self.sym_error = 100
        self.list_errors = []
        self.list_ref_meshes = []
        self.reg_idx = 0

        self.patience = patience
        self.patience_idx = 0
        self._best_score = 100

        self.verbose = verbose

        self.remesh_each_step = remesh_each_step

        self.error_2_error_change = 100

        self.save_meshes_during_registration = save_meshes_during_registration
        self.folder_save = folder_save
        self.save_mesh_suffix = save_mesh_suffix
        self.multiprocessing = multiprocessing
        self.num_processes = num_processes

        if self.save_meshes_during_registration is True:
            if (self.folder_save is not None) and (os.path.exists(self.folder_save)):
                os.makedirs(self.folder_save, exist_ok=True)

    # def register(self, ref_mesh_source, other_mesh_idx):
    #     target_mesh = io.read_vtk(self.list_mesh_paths[other_mesh_idx])

    #     registered_mesh, icp_transform = non_rigidly_register(
    #         target_mesh=target_mesh,
    #         source_mesh=ref_mesh_source,
    #         target_eigenmap_as_reference=not self.ref_mesh_eigenmap_as_reference,
    #         transfer_scalars=True if self.vertex_features is not None else False,
    #         return_icp_transform=True,
    #         verbose=self.verbose,
    #         **self.kwargs
    #     )

    #     coords = get_mesh_physical_point_coords(registered_mesh)

    #     n_points = coords.shape[0]

    #     if self.vertex_features is not None:
    #         features = get_mesh_point_features(registered_mesh, self.vertex_features)
    #     else:
    #         features = None

    #     return coords, features, icp_transform

    # def registration_step(
    #     self,
    #     idx,
    #     path,
    #     ref_mesh
    # ):
    #     # This is a helper function to allow for multiprocessing to work
    #     # becuase cant pickle vtk objects, instead we read them from disk.
    #     if type(ref_mesh) is str:
    #         ref_mesh = io.read_vtk(ref_mesh)

    #     if self.verbose is True:
    #         print(f'\tRegistering to mesh # {idx}')
    #     # skip the first mesh in the list if its the first round (its the reference)
    #     if (self.reg_idx == 0) & (idx == 0) & (self.include_ref_in_sample is True):
    #         # first iteration & ref mesh, just use points as they are.
    #         registered_pt_coords = get_mesh_physical_point_coords(ref_mesh)
    #         if self.vertex_features is not None:
    #             registered_vertex_features = get_mesh_point_features(ref_mesh, self.vertex_features)
    #         registered_icp_transform = None
    #     else:
    #         # register & save registered coordinates in the pre-allocated array
    #         registered_pt_coords, features, icp_transform = self.register(vtk_deep_copy(ref_mesh), idx)
    #         if self.vertex_features is not None:
    #             registered_vertex_features = features
    #         registered_icp_transform = icp_transform

    #     # SAVE EACH ITERATION OF THE REGISTRATION PROCESS???
    #     if self.save_meshes_during_registration is True:
    #         path_to_save = self.get_path_save_mesh(path, idx=None, mesh_suffix=None)  # use global suffix, and no idx
    #         self.save_mesh(self.ref_mesh, registered_pt_coords, registered_vertex_features, path_to_save)

    #     return registered_pt_coords, registered_vertex_features, registered_icp_transform

    def execute(self):
        # create placeholder to store registered point clouds & update inherited one only if also storing
        registered_pt_coords = np.zeros_like(self._registered_pt_coords)
        if self.vertex_features is not None:
            registered_vertex_features = np.zeros_like(self._registered_vertex_features)
        registered_icp_transforms = []

        # keep doing registrations until max# is hit, or the minimum error between registrations is hit.
        while (
            (self.reg_idx < self.max_n_registration_steps)
            & (self.sym_error > self.tolerance1)
            & (self.error_2_error_change > self.tolerance2)
        ):
            if self.verbose is True:
                print(f"Starting registration round {self.reg_idx}")

            # If its not the very first iteration - check whether or not we want to re-mesh after every iteration.
            if (self.reg_idx != 0) & (self.remesh_each_step is True):
                n_points = self._ref_mesh.GetNumberOfPoints()
                self._ref_mesh = resample_surface(self._ref_mesh, subdivisions=2, clusters=n_points)
                if n_points != self.n_points:
                    print(
                        f"Updating n_points for mesh from {self.n_points} to {self._ref_mesh.GetNumberOfPoints()}"
                    )
                    # re-create the array to store registered points as the # vertices might change after re-meshing.
                    # also update n_points.
                    self.n_points = n_points
                    registered_pt_coords = np.zeros(
                        (len(self.list_mesh_paths), self.n_points, 3), dtype=float
                    )

            # register the reference mesh to all other meshes
            if self.multiprocessing is True:
                # get temp folder to save reference mesh
                temp_folder = os.path.join(os.path.dirname(self.list_mesh_paths[0]), "temp")
                os.makedirs(temp_folder, exist_ok=True)
                temp_ref_path = os.path.join(temp_folder, "temp_ref_mesh.vtk")
                io.write_vtk(self._ref_mesh, temp_ref_path)

                args_list = []

                # for idx, path in enumerate(self.list_mesh_paths):
                #     path_to_save = self.get_path_save_mesh(path)  # use global suffix, and no idx
                #     args_list.append((idx, temp_ref_path, self.get_path_save_mesh(path)) + constant_args)
                constant_args = (
                    self.reg_idx,
                    self.include_ref_in_sample,
                    self.vertex_features,
                    self.save_meshes_during_registration,
                    self.list_mesh_paths,
                    self.ref_mesh_eigenmap_as_reference,
                    self.kwargs,
                    self.verbose,
                )
                args_list = [
                    (idx, temp_ref_path, self.get_path_save_mesh(path)) + constant_args
                    for idx, path in enumerate(self.list_mesh_paths)
                ]
                # args_list = [(idx, temp_ref_path, self.reg_idx, self.include_ref_in_sample, self.vertex_features, self.list_mesh_paths, self.ref_mesh_eigenmap_as_reference, self.kwargs, self.verbose) for idx, path in enumerate(self.list_mesh_paths)]

                with Pool(processes=self.num_processes) as pool:
                    results = pool.starmap(registration_step, args_list)

                # delete the temp reference mesh
                os.remove(temp_ref_path)

            # Close the pool and wait for worker processes to finish
            else:
                results = [
                    registration_step(
                        idx,
                        self._ref_mesh,
                        self.get_path_save_mesh(path),
                        self.reg_idx,
                        self.include_ref_in_sample,
                        self.vertex_features,
                        self.save_meshes_during_registration,
                        self.list_mesh_paths,
                        self.ref_mesh_eigenmap_as_reference,
                        self.kwargs,
                        self.verbose,
                    )
                    for idx, path in enumerate(self.list_mesh_paths)
                ]

            # for idx, path in enumerate(self.list_mesh_paths):
            for idx, (
                registered_pt_coords_,
                registered_vertex_features_,
                registered_icp_transform_,
            ) in enumerate(results):
                # registered_pt_coords_, registered_vertex_features_, registered_icp_transform = self.registration_step(idx, path)
                if registered_pt_coords_ is None:
                    # warning that there was an error in registration and the mesh was skipped - e.g.,
                    # nans were used in its place.
                    warnings.warn(
                        f"WARNING: mesh has no points, skipping registration\n\tPath: {self.list_mesh_paths[idx]}",
                        RuntimeWarning,
                    )
                    registered_pt_coords[idx, :, :] = np.nan
                    if self.vertex_features is not None:
                        registered_vertex_features[idx, :, :] = np.nan
                    registered_icp_transforms.append(None)
                else:
                    if type(registered_icp_transform_) is str:
                        # if the registered_icp_transform is a string, then it is a path to a temp file
                        # that was created to save the transform.
                        # read the transform from disk & delete the temp file.
                        registered_icp_transform = read_linear_transform(registered_icp_transform_)
                        if self.remove_temp_icp is True:
                            os.remove(registered_icp_transform_)
                    else:
                        registered_icp_transform = registered_icp_transform_

                    registered_pt_coords[idx, :, :] = registered_pt_coords_
                    if self.vertex_features is not None:
                        registered_vertex_features[idx, :, :] = registered_vertex_features_
                    registered_icp_transforms.append(registered_icp_transform)

            # if there is a nan in registered pt corrds, let user know - we handle with nanmean
            if np.isnan(registered_pt_coords).sum() > 0:
                warnings.warn(f"WARNING: registered_pt_coords has nans", RuntimeWarning)
            # Calculate the mean bone shape & create new mean bone shape mesh
            mean_shape = np.nanmean(registered_pt_coords, axis=0)
            mean_mesh = vtk_deep_copy(self._ref_mesh)
            set_mesh_physical_point_coords(mean_mesh, mean_shape)
            if self.vertex_features is not None:
                mean_features = np.nanmean(registered_vertex_features, axis=0)
                set_mesh_point_features(
                    mesh=mean_mesh, features=mean_features, feature_names=self.vertex_features
                )

            # store in list of reference meshes
            self.list_ref_meshes.append(mean_mesh)

            # Get surface distance between previous reference mesh and the new mean
            # TODO: Update below to get real ASSD - look at Mesh get assd_mesh code.
            sym_error = get_symmetric_surface_distance(self._ref_mesh, mean_mesh)
            self.error_2_error_change = np.abs(sym_error - self.sym_error)
            self.sym_error = sym_error
            if self.sym_error >= self._best_score:
                # if the error isnt going down, then keep track of that and done save the
                # new reference mesh.
                self.patience_idx += 1
            else:
                self.patience_idx = 0
                # ONLY UPDATE THE REF_MESH or the REGISTERED_PTS WHEN THE INTER-MESH (REF) ERROR GETS BETTER
                # NOT SURE IF THIS IS A GOOD IDEA - MIGHT WANT A BETTER CRITERIA?
                self._ref_mesh = mean_mesh
                self._registered_pt_coords = registered_pt_coords
                if self.vertex_features is not None:
                    self._registered_vertex_features = registered_vertex_features
                self._registered_icp_transforms = registered_icp_transforms
            # Store the symmetric error values so they can be plotted later
            self.list_errors.append(self.sym_error)
            if self.verbose is True:
                print(f"\t\tSymmetric surface error: {self.sym_error}")

            if self.patience_idx >= self.patience:
                print(
                    f"Early stopping initiated - no improvment for {self.patience_idx} iterations, patience is: {self.patience}"
                )
                break

            self.reg_idx += 1

    def get_path_save_mesh(
        self,
        path,
        idx=None,
        mesh_suffix=None,
    ):
        if mesh_suffix is not None:
            self.save_mesh_suffix = mesh_suffix
        orig_folder = os.path.dirname(path)
        orig_filename = os.path.basename(path)
        base_filename = orig_filename[: orig_filename.rfind(".")]
        if idx is not None:
            filename = f"{base_filename}_{self.save_mesh_suffix}_{idx}.vtk"
        else:
            filename = f"{base_filename}_{self.save_mesh_suffix}.vtk"

        if self.folder_save is None:
            path_to_save = os.path.join(orig_folder, filename)
        else:
            path_to_save = os.path.join(os.path.abspath(self.folder_save), filename)

        return path_to_save

    def save_meshes(
        self,
        mesh_suffix=f'procrustes_registered_{today.strftime("%b")}_{today.day}_{today.year}',
        folder=None,
    ):
        if folder is not None:
            self.folder_save = folder

        if self.folder_save is not None:
            os.makedirs(self.folder_save, exist_ok=True)

        mesh = vtk_deep_copy(self._ref_mesh)
        for idx, path in enumerate(self.list_mesh_paths):
            path_to_save = self.get_path_save_mesh(path, idx=idx, mesh_suffix=mesh_suffix)
            reg_vert_feat = (
                self._registered_vertex_features[idx, :, :]
                if self.vertex_features is not None
                else None
            )
            save_mesh(
                mesh,
                self._registered_pt_coords[idx, :, :],
                reg_vert_feat,
                path_to_save,
                self.vertex_features,
            )

    def save_icp_transforms(
        self,
        transform_suffix=f'icp_transforms_{today.strftime("%b")}_{today.day}_{today.year}',
        folder=None,
    ):
        if folder is not None:
            os.makedirs(folder, exist_ok=True)

        for idx, path in enumerate(self.list_mesh_paths):
            # parse folder / filename for saving
            orig_folder = os.path.dirname(path)
            orig_filename = os.path.basename(path)
            base_filename = orig_filename[: orig_filename.rfind(".")]
            filename = f"{base_filename}_{transform_suffix}_{idx}.txt"
            if folder is None:
                path_to_save = os.path.join(orig_folder, filename)
            else:
                path_to_save = os.path.join(os.path.abspath(folder), filename)

            if self._registered_icp_transforms[idx] is None:
                # if there was no ICP transform, then just save an empty file
                with open(path_to_save, "w") as f:
                    f.write("NO ICP TRANSFORM - THIS WAS THE REFERENCE MESH")
            else:
                write_linear_transform(self._registered_icp_transforms[idx], path_to_save)

    def save_ref_mesh(self, path):
        io.write_vtk(self._ref_mesh, path)

    @property
    def ref_mesh(self):
        return self._ref_mesh

    @property
    def registered_pt_coords(self):
        return self._registered_pt_coords

    @property
    def registered_vertex_features(self):
        return self._registered_vertex_features

    @property
    def registered_icp_transforms(self):
        return self._registered_icp_transforms


def save_mesh(mesh, registered_pts, registered_features, path_to_save, vertex_features):
    mesh_ = vtk_deep_copy(mesh)
    # Keep recycling the same base mesh, just move the x/y/z point coords around.
    set_mesh_physical_point_coords(mesh_, registered_pts)
    if registered_features is not None:
        set_mesh_point_features(
            mesh=mesh_, features=registered_features, feature_names=vertex_features
        )
    # save mesh to disk
    io.write_vtk(mesh_, path_to_save)


def register(
    ref_mesh_source,
    path_other_mesh,
    ref_mesh_eigenmap_as_reference,
    vertex_features,
    verbose,
    kwargs,
):
    target_mesh = io.read_vtk(path_other_mesh)

    if (
        (target_mesh.points.shape[0] == 0)
        or (np.mean(target_mesh.points) == 0)
        or (np.isnan(target_mesh.points).sum() > 0)
    ):
        print("ERROR: target mesh has no points, skipping registration")
        print(f"\tPath: {path_other_mesh}")
        return None, None, None

    registered_mesh, icp_transform = non_rigidly_register(
        target_mesh=target_mesh,
        source_mesh=ref_mesh_source,
        target_eigenmap_as_reference=not ref_mesh_eigenmap_as_reference,
        transfer_scalars=True if vertex_features is not None else False,
        return_icp_transform=True,
        verbose=verbose,
        **kwargs,
    )

    coords = get_mesh_physical_point_coords(registered_mesh)

    n_points = coords.shape[0]

    if vertex_features is not None:
        features = get_mesh_point_features(registered_mesh, vertex_features)
    else:
        features = None

    return coords, features, icp_transform


def registration_step(
    idx,
    ref_mesh,
    path_save_mesh,
    reg_idx,
    include_ref_in_sample,
    vertex_features,
    save_meshes_during_registration,
    list_mesh_paths,
    ref_mesh_eigenmap_as_reference,
    kwargs,
    verbose,
):
    tic = time.time()
    # This is a helper function to allow for multiprocessing to work
    # becuase cant pickle vtk objects, instead we read them from disk.
    if type(ref_mesh) is str:
        mp = True
        ref_mesh = io.read_vtk(ref_mesh)
    else:
        mp = False

    if verbose is True:
        print(f"\tRegistering to mesh # {idx}")
        print(f"\t\tPath save mesh: {path_save_mesh}")
    # skip the first mesh in the list if its the first round (its the reference)
    if (reg_idx == 0) & (idx == 0) & (include_ref_in_sample is True):
        # first iteration & ref mesh, just use points as they are.
        registered_pt_coords = get_mesh_physical_point_coords(ref_mesh)
        if vertex_features is not None:
            registered_vertex_features = get_mesh_point_features(ref_mesh, vertex_features)
        else:
            registered_vertex_features = None
        registered_icp_transform = None
    else:
        # register & save registered coordinates in the pre-allocated array
        registered_pt_coords, features, icp_transform = register(
            vtk_deep_copy(ref_mesh),
            path_other_mesh=list_mesh_paths[idx],
            vertex_features=vertex_features,
            ref_mesh_eigenmap_as_reference=ref_mesh_eigenmap_as_reference,
            verbose=verbose,
            kwargs=kwargs,
        )
        print(registered_pt_coords)
        if registered_pt_coords is None:
            # if the registered_pt_coords is None, then there was an error reading the mesh
            # so just return None for everything.
            with open(
                "/dataNAS/people/aagatti/projects/OAI_Segmentation/CVPR_Data_Curation/ssm_registrations/output.log",
                "a",
            ) as f:
                # add a line to the log file of the mesh that errored
                f.write(f"ERROR: mesh has no points, skipping registration\n")
                f.write(f"\tPath: {list_mesh_paths[idx]}\n")
            return None, None, None
        if vertex_features is not None:
            registered_vertex_features = features
        else:
            registered_vertex_features = None
        registered_icp_transform = icp_transform

    # SAVE EACH ITERATION OF THE REGISTRATION PROCESS???
    if save_meshes_during_registration is True:
        save_mesh(
            ref_mesh,
            registered_pt_coords,
            registered_vertex_features,
            path_save_mesh,
            vertex_features,
        )

    if (mp is True) and (registered_icp_transform is not None):
        # write the registered_icp_transform to disk & return the path
        # get temp path to save the transform
        filename = os.path.basename(list_mesh_paths[idx]).split(".vtk")[0]
        temp_filename = filename + "_temp_icp_transform.json"
        temp_path = os.path.join(os.path.dirname(list_mesh_paths[idx]), temp_filename)
        write_linear_transform(registered_icp_transform, temp_path)
        registered_icp_transform = temp_path

    toc = time.time()
    print(f"\t\tTime taken for registration step: {toc - tic} seconds")
    return registered_pt_coords, registered_vertex_features, registered_icp_transform
