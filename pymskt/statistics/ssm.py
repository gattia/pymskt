import json
import os
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import vtk

from pymskt.mesh import Mesh, io
from pymskt.mesh.meshRegistration import non_rigidly_register
from pymskt.mesh.meshTools import get_mesh_physical_point_coords, get_mesh_point_features
from pymskt.mesh.utils import vtk_deep_copy
from pymskt.statistics import ProcrustesRegistration
from pymskt.statistics.pca import (
    create_vtk_mesh_from_deformed_points,
    pca_svd,
    save_gif,
    save_gif_vec_2_vec,
    save_mesh_vec_2_vec,
    save_meshes_across_pc,
)

# import multiprocessing


class SSM:
    def __init__(
        self,
        list_mesh_paths=None,
        path_ref_mesh=None,
        points_already_correspond=False,
        max_n_procrustes_steps=1,
        # SSM Options
        vertex_features=None,
        # Defaults ProcrustesRegistration:
        n_coords_spectral_ordering=10000,
        n_coords_spectral_registration=1000,
        n_spectral_features=3,
        n_extra_spectral=6,
        include_points_as_features=True,
        save_registered_meshes=False,
        folder_save_registered_meshes=None,
        feature_norm_ignore_zero=True,
        feature_norm_include_pt_if_any_mesh_has_value=True,
        include_ref_mesh=True,
        # Multiprocessing
        multiprocessing=True,
        num_processes=None,
        verbose=False,
    ):
        # Pre-process list(s) of meshes & related info.
        self._list_mesh_paths = list_mesh_paths
        self._path_ref_mesh = path_ref_mesh

        self.include_ref_mesh = include_ref_mesh

        if self._list_mesh_paths is not None:
            self.parse_list_mesh_paths()

        # SSM Options
        self.vertex_features = vertex_features
        self.points_already_correspond = points_already_correspond
        self.feature_norm_ignore_zero = feature_norm_ignore_zero
        self.feature_norm_include_pt_if_any_mesh_has_value = (
            feature_norm_include_pt_if_any_mesh_has_value
        )

        # ProcrustrsRegistration
        self.max_n_procrustes_steps = max_n_procrustes_steps
        self.n_coords_spectral_ordering = n_coords_spectral_ordering
        self.n_coords_spectral_registration = n_coords_spectral_registration
        self.n_spectral_features = n_spectral_features
        self.n_extra_spectral = n_extra_spectral
        self.include_points_as_features = include_points_as_features
        self.save_registered_meshes = save_registered_meshes
        self.folder_save_registered_meshes = folder_save_registered_meshes

        self.n_meshes = len(self._list_mesh_paths) if self._list_mesh_paths is not None else 0

        self.parse_ref_mesh_and_params()

        # Pre-allocate registered points
        self.points = np.zeros((self.n_meshes, self.n_points * self.n_features))

        self._points_loaded = False
        self._points_normalized = False

        # Multiprocessing
        self.multiprocessing = multiprocessing
        self.num_processes = num_processes

        self.verbose = verbose

        self.scores = None
        self.scores_raw = None
        self.dict_threshold_n_pcs = {}
        self.absolute_variance_explained = None
        self.percent_variance_explained = None

    def parse_list_mesh_paths(self):
        """Parse list of mesh paths"""
        # Prepare list of meshes to load for registration.
        # if path_ref_mesh not provided, get random mesh from list_mesh_paths
        if (self._path_ref_mesh is None) & (self._list_mesh_paths is not None):
            path_ref_idx = np.random.randint(0, len(self._list_mesh_paths))
            self._path_ref_mesh = self._list_mesh_paths[path_ref_idx]

        # Ensure that path_ref_mesh is in list & at index 0
        if self._path_ref_mesh in self._list_mesh_paths:
            path_ref_idx = self._list_mesh_paths.index(self._path_ref_mesh)
            self._list_mesh_paths.pop(path_ref_idx)

        if self.include_ref_mesh is True:
            self._list_mesh_paths.insert(0, self._path_ref_mesh)
        else:
            pass

    def parse_ref_mesh_and_params(self):
        """Get reference mesh parameters"""
        if (hasattr(self, "_ref_mesh") is False) or (self._ref_mesh is None):
            if self._path_ref_mesh is not None:
                self._ref_mesh = io.read_vtk(self._path_ref_mesh)
            else:
                self._ref_mesh = None
                self.n_points = 0
                self.n_features = 0
                return
        self.n_points = self._ref_mesh.GetNumberOfPoints()
        self.n_features = 3 + (len(self.vertex_features) if self.vertex_features is not None else 0)

    def find_point_correspondence(
        self,
        #   path_ref_mesh=None,
        #   list_mesh_paths=None
    ):
        """Find point correspondence between meshes"""
        # if path_ref_mesh is not None:
        #     self._path_ref_mesh = path_ref_mesh
        # if list_mesh_paths is not None:
        #     self._list_mesh_paths = list_mesh_paths
        procrustes_reg = ProcrustesRegistration(
            path_ref_mesh=self._path_ref_mesh,  # using the idx of the best mesh from the previous step
            list_mesh_paths=self._list_mesh_paths,  # This will automatically remove the ref_mesh path if it is in the list.
            max_n_registration_steps=self.max_n_procrustes_steps,
            n_coords_spectral_ordering=self.n_coords_spectral_ordering,
            n_coords_spectral_registration=self.n_coords_spectral_registration,
            n_extra_spectral=self.n_extra_spectral,
            include_points_as_features=self.include_points_as_features,
            vertex_features=self.vertex_features,
            multiprocessing=self.multiprocessing,
            num_processes=self.num_processes,
        )

        procrustes_reg.execute()

        if self.save_registered_meshes is True:
            procrustes_reg.save_meshes(folder=self.folder_save_registered_meshes)

        # fill pre-allocated points array

        for idx in range(self.n_meshes):
            xyz = procrustes_reg.registered_pt_coords[idx, :, :].flatten()
            if self.vertex_features is None:
                features = np.zeros(0)
            else:
                features = procrustes_reg.registered_vertex_features[idx, :, :].flatten()
            self.points[idx, :] = np.concatenate((xyz, features))

    def prepare_points(self):
        """Prepare points"""
        print("Preparing points...")
        if self.points_already_correspond is False:
            print("Finding point correspondences...")
            self.find_point_correspondence()

        elif self.points_already_correspond is True:
            print("Loading points...")
            for idx, path in enumerate(self._list_mesh_paths):
                mesh = io.read_vtk(path)
                xyz = get_mesh_physical_point_coords(mesh).flatten()
                if self.vertex_features is None:
                    features = np.zeros(0)
                else:
                    features = get_mesh_point_features(mesh, self.vertex_features).flatten()
                self.points[idx, :] = np.concatenate((xyz, features))

        self._points_loaded = True

    def normalize_points(self):
        """Get points info"""
        print("Beginning Point Normalization")
        self._mean = np.mean(self.points, axis=0)
        self._std = np.std(self.points, axis=0, ddof=1)

        self._centered = self.points - self._mean

        self._std_geometric = np.std(self._centered[:, : 3 * self.n_points], ddof=1)
        if self.verbose is True:
            print("Geometric std: {}".format(self._std_geometric))

        if self.vertex_features is not None:
            self._std_features = []
            for idx in range(len(self.vertex_features)):
                data = self._centered[:, (3 + idx) * self.n_points : (3 + idx + 1) * self.n_points]
                if self.feature_norm_ignore_zero is True:
                    # NORMALIZE TO INCLUDE ANY VERTEX WITH AT LEAST ONE PERSON THAT IS NON-ZERO
                    # THIS MIGHT NOT BE THE BEST WAY TO DO THIS
                    # ALTERNATIVELY, COULD JUST THROW OUT ALL ZEROS... E.G.:
                    # data = data[np.where(data != 0)]
                    if self.feature_norm_include_pt_if_any_mesh_has_value is True:
                        data = data[:, np.where(np.abs(data).max(axis=0) > 0)]
                    else:
                        data = data[np.where(data != 0)]
                self._std_features.append(np.std(data, ddof=1))

                if self.verbose is True:
                    print(f"Feature {self.vertex_features[idx]} std: {self._std_features[idx]}")

        self._centered = self.apply_normalization(self.points)
        # self._centered[:, :3*self.n_points] = self._centered[:, :3*self.n_points] / self._std_geometric
        # if self.vertex_features is not None:
        #     for idx in range(len(self.vertex_features)):
        #         data = self._centered[:, (3+idx)*self.n_points:(3+idx+1)*self.n_points]
        #         self._centered[:, (3+idx)*self.n_points:(3+idx+1)*self.n_points] = data / self._std_features[idx]

        self._points_normalized = True

    def fit_model(self):
        """Fit PCA based SSM model"""
        self.n_pcs = min(
            self.points.shape
        )  # MAX N "relevant" PCs is the smaller of the number of points/examples.

        if self._points_loaded is False:
            self.prepare_points()
        if self._points_normalized is False:
            self.normalize_points()
        print("Fitting PCA-based model...")
        self._PCs, self._Vs = pca_svd(self._centered.T)

    def get_dict_model_params(self, PCs_filename, Vs_filename):
        """Get dict of model parameters"""
        # Add generic model information
        dict_dump = {
            "date": datetime.now().strftime("%b-%d-%Y"),
            "n_pcs": self.n_pcs,
            "n_points": self.n_points,
            "n_meshes": self.n_meshes,
            "PCs_filename": f"{PCs_filename}.npy",
            "Vs_filename": f"{Vs_filename}.npy",
            "geometric_std": self._std_geometric,
            "list_vertex_features": self.vertex_features,
            "list_vertex_features_stds": (
                None if self.vertex_features is None else self._std_features
            ),
            "feature_norm_ignore_zero": self.feature_norm_ignore_zero,
            "dict_threshold_n_pcs": self.dict_threshold_n_pcs,
            "absolute_variance_explained": self.absolute_variance_explained,
            "percent_variance_explained": self.percent_variance_explained,
        }

        # add feature specific stds for normalization
        if self.vertex_features is not None:
            for idx, vertex_feature in enumerate(self.vertex_features):
                dict_dump[f"{vertex_feature}_std"] = self._std_features[idx]

        # Add whether points already corresponding... and registration parameters
        # registration parameters should be used for future registrations/using the model.
        dict_dump["points_already_correspond"] = self.points_already_correspond
        dict_dump["n_coords_spectral_ordering"] = self.n_coords_spectral_ordering
        dict_dump["n_coords_spectral_registration"] = self.n_coords_spectral_registration
        dict_dump["n_spectral_features"] = self.n_spectral_features
        dict_dump["n_extra_spectral"] = self.n_extra_spectral
        dict_dump["include_points_as_features"] = self.include_points_as_features

        dict_dump["list_mesh_locations"] = self._list_mesh_paths

        return dict_dump

    def save_model(self, folder=None, PCs_filename="PCs", Vs_filename="Vs", save_points=False):
        """
        Save PCA-based model
        Notes
        -----
        To decode the model:
            1.
        """
        print("Saving model...")
        if os.path.isdir(folder) is False:
            os.makedirs(folder, exist_ok=True)
        np.save(os.path.join(folder, f"{PCs_filename}.npy"), self._PCs)
        np.save(os.path.join(folder, f"{Vs_filename}.npy"), self._Vs)

        # Save mean mesh
        io.write_vtk(self._ref_mesh, os.path.join(folder, "ref_mesh.vtk"))

        io.write_vtk(self._mean_mesh().mesh, os.path.join(folder, "mean_mesh.vtk"))

        # save mean features / everything?
        np.save(os.path.join(folder, "mean_features.npy"), self._mean)
        # save std of geometric points / features:
        with open(os.path.join(folder, "ssm_model_information.json"), "w") as f:
            dict_model_params = self.get_dict_model_params(PCs_filename, Vs_filename)
            json.dump(dict_model_params, f, indent=4)

        if save_points is True:
            np.save(os.path.join(folder, "points.npy"), self.points)

    def load_model(self, folder):
        """Load PCA-based model"""
        with open(os.path.join(folder, "ssm_model_information.json"), "r") as f:
            dict_model_params = json.load(f)

        PCs_path = os.path.join(folder, dict_model_params["PCs_filename"])
        Vs_path = os.path.join(folder, dict_model_params["Vs_filename"])

        self._PCs = np.load(PCs_path)
        self._Vs = np.load(Vs_path)
        self._ref_mesh = io.read_vtk(os.path.join(folder, "ref_mesh.vtk"))
        self._mean = np.load(os.path.join(folder, "mean_features.npy"))

        self.n_pcs = dict_model_params.get("n_pcs", self._PCs.shape[1])
        self.n_points = dict_model_params["n_points"]
        self.n_meshes = None  # TODO: Parse this from self._list_mesh_paths?
        self._std_geometric = dict_model_params["geometric_std"]
        self.vertex_features = dict_model_params["list_vertex_features"]
        self._std_features = dict_model_params["list_vertex_features_stds"]
        self.feature_norm_ignore_zero = dict_model_params["feature_norm_ignore_zero"]

        # Registration parameters
        self.n_coords_spectral_ordering = dict_model_params["n_coords_spectral_ordering"]
        self.n_coords_spectral_registration = dict_model_params["n_coords_spectral_registration"]
        self.n_spectral_features = dict_model_params["n_spectral_features"]
        self.n_extra_spectral = dict_model_params["n_extra_spectral"]
        self.include_points_as_features = dict_model_params["include_points_as_features"]

        self._list_mesh_paths = dict_model_params["list_mesh_locations"]

        self.dict_threshold_n_pcs = dict_model_params.get("dict_threshold_n_pcs", None)
        if self.dict_threshold_n_pcs is not None:
            # convert keys back to integers
            self.dict_threshold_n_pcs = {
                int(key): value for key, value in self.dict_threshold_n_pcs.items()
            }
        self.absolute_variance_explained = dict_model_params.get(
            "absolute_variance_explained", None
        )
        self.percent_variance_explained = dict_model_params.get("percent_variance_explained", None)

        if os.path.exists(os.path.join(folder, "points.npy")):
            self.points = np.load(os.path.join(folder, "points.npy"))
            self._points_loaded = True

    def perform_variance_analysis(self):
        """Perform variance analysis"""
        self.calculate_total_variance()
        self.calculate_variance_explained_per_pc()
        self.calculate_n_pcs_explain_variance()

    def calculate_total_variance(self):
        """Calculate total variance"""
        self.total_variance = np.sum(np.square(self.points - self.mean)) / self.points.shape[0]

    def calc_pc_scores_internal(self):
        """Calculate PC scores"""
        self.scores_raw = self.PCs.T @ self.centered.T
        self.scores = self.scores_raw / (np.sqrt(self.Vs)[:, None])

    def calculate_variance_explained_per_pc(self):
        """Calculate variance explained per PC"""

        if self.scores_raw is None:
            self.calc_pc_scores_internal()

        self.absolute_variance_explained = []
        self.percent_variance_explained = []
        for pc in range(self.PCs.shape[1]):
            pred_ = self.scores_raw[pc, :][None, :].T @ self.PCs[:, pc][None, :]
            pred_[:, : 3 * self.n_points] *= self.std_geometric
            if self.vertex_features is not None:
                for idx in range(len(self.vertex_features)):
                    pred_[:, 3 * self.n_points :] *= self.std_features[0]
            pred_ = self.mean + pred_
            variance = np.sum(np.square(pred_ - self.points)) / self.points.shape[0]
            explained_var = self.total_variance - variance
            self.absolute_variance_explained.append(explained_var)
            self.percent_variance_explained.append(explained_var / self.total_variance * 100)

        self.cumulative_explained_variance = np.cumsum(self.percent_variance_explained)

    def calculate_n_pcs_explain_variance(self, thresholds=None):
        if thresholds is None:
            thresholds = [10, 20, 30, 40, 50, 60, 70, 80, 90, 95, 98, 99]

        self.dict_threshold_n_pcs = {}
        for threshold in thresholds:
            self.dict_threshold_n_pcs[threshold] = int(
                np.where(self.cumulative_explained_variance > threshold)[0][0] + 1
            )

    def plot_variance_explained(self, save_path=None):
        plt.figure()
        plt.title("SSM - Explained Variance")
        plt.plot(self.percent_variance_explained, label="PC Percent Variance Explained")
        plt.plot(np.cumsum(self.percent_variance_explained), label="Cumulative Variance Explained")
        plt.plot(
            (0, self.n_pcs),
            (90, 90),
            label=f"90% Explained Variance (N={self.dict_threshold_n_pcs[90]})",
        )
        plt.plot(
            (0, self.n_pcs),
            (95, 95),
            label=f"95% Explained Variance (N={self.dict_threshold_n_pcs[95]})",
        )

        # Set x and y axis titles
        plt.xlabel("Principal Component", fontsize=14)
        plt.ylabel("Explained Variance (%)", fontsize=14)

        # Increase font size on x and y axis
        plt.xticks(fontsize=12)
        plt.yticks(fontsize=12)

        # Plot the legend
        plt.legend(fontsize=12)

    def plot_hists_pc_scores(
        self,
        pc,
    ):
        """Plot histograms of PC scores"""
        if self.scores is None:
            self.calc_pc_scores_internal()
        plt.figure()
        plt.title(f"PC {pc} Scores")
        plt.hist(self.scores[pc, :])
        plt.xlabel("PC Score", fontsize=14)
        plt.ylabel("Count", fontsize=14)
        # Increase font size on x and y axis
        plt.xticks(fontsize=12)
        plt.yticks(fontsize=12)

    def plot_hists_n_pc_scores(self, n_pcs=5):
        """Plot histograms of PC scores"""
        for pc in range(n_pcs):
            self.plot_hists_pc_scores(pc)

    def save_meshes_across_pc(self, folder_save, pc, std, step_size=1, mesh_name="bone"):
        """Save meshes across PC"""
        # TODO: Eventually update all functions to take multiple bones
        # Will need to unpack self._ref_mesh to be multiple meshes.

        save_meshes_across_pc(
            mesh=self._ref_mesh,
            mean_coords=self._mean,
            PCs=self._PCs,
            Vs=self._Vs,
            pc=pc,
            min_sd=-abs(std),
            max_sd=abs(std),
            step_size=step_size,
            loc_save=folder_save,
            mesh_name=f"{mesh_name}_{pc}",  # ['tibia', 'patella', 'femur'], #['femur', 'tibia', 'patella'],
            save_filename="{mesh_name}_{sd}.vtk",  # specifically not with `f` so we can fill in later.
        )

    def save_gif_across_pc(
        self,
        path_save,
        pc,
        std,
        step_size=0.25,
        camera_position="xz",
        scalar_bar_range=[0, 4],
        background_color="white",
        cmap=None,
        **kwargs,
    ):
        """Save gif"""
        save_gif(
            path_save=path_save,
            PCs=self._PCs,
            Vs=self._Vs,
            mean_coords=self._mean,  # mean_coords could be extracted from mean mesh...?
            mean_mesh=self._ref_mesh,
            features=self.vertex_features,
            pc=pc,
            min_sd=-abs(std),
            max_sd=abs(std),
            step=step_size,
            color="orange" if self.vertex_features is None else None,
            show_edges=True,
            edge_color="black",
            camera_position=camera_position,
            window_size=[3000, 4000],
            background_color=background_color,
            scalar_bar_range=scalar_bar_range,
            cmap=cmap,
            verbose=False,
            **kwargs,
        )

    def save_gif_vector(
        self,
        path_save,
        vec_start,
        vec_end,
        # n_steps=24,
        # camera_position='xz',
        # window_size=[900, 1200],
        # background_color='white',
        # scalar_bar_range=[0, 4],
        # cmap=None,
        **kwargs,
    ):
        if (self.vertex_features is None) and ("color" not in kwargs):
            kwargs["color"] = "orange"

        save_gif_vec_2_vec(
            path_save,
            PCs=self._PCs,
            Vs=self._Vs,
            mean_coords=self._mean,  # mean_coords could be extracted from mean mesh...?
            mean_mesh=self._ref_mesh,
            vec_1=vec_start,
            vec_2=vec_end,
            # color='orange' if self.vertex_features is None else None,
            # show_edges=True,
            # edge_color='black',
            # n_steps=n_steps,
            # camera_position=camera_position,
            # window_size=window_size, #[3000, 4000],
            # background_color=background_color,
            # scalar_bar_range=scalar_bar_range,
            # cmap=cmap,
            **kwargs,
            # features=None,
            # verbose=False,
        )

    def save_meshes_vector(
        self,
        path_save,
        vec_start,
        vec_end,
        # n_steps=24,
        # camera_position='xz',
        # window_size=[900, 1200],
        # background_color='white',
        # scalar_bar_range=[0, 4],
        # cmap=None,
        **kwargs,
    ):
        save_mesh_vec_2_vec(
            path_save=path_save,
            PCs=self._PCs,
            Vs=self._Vs,
            mean_coords=self._mean,  # mean_coords could be extracted from mean mesh...?
            mean_mesh=self._ref_mesh,
            vec_1=vec_start,
            vec_2=vec_end,
            **kwargs,
        )

    def register_ref_to_mesh(self, mesh):
        registered_mesh = non_rigidly_register(
            target_mesh=mesh,
            source_mesh=self._ref_mesh,
            target_eigenmap_as_reference=False,
            n_coords_spectral_ordering=self.n_coords_spectral_ordering,
            n_coords_spectral_registration=self.n_coords_spectral_registration,
            n_spectral_features=self.n_spectral_features,
            n_extra_spectral=self.n_extra_spectral,
            include_points_as_features=self.include_points_as_features,
            transfer_scalars=True if self.vertex_features is not None else False,
            verbose=self.verbose,
        )

        return registered_mesh

    def apply_normalization(self, array):
        array = array.copy()
        if len(array.shape) == 1:
            assert array.shape[0] == self._mean.shape[0]
            array = np.expand_dims(array, axis=0)
        else:
            assert array.shape[1] == self._mean.shape[0]

        array -= self._mean

        array[:, : 3 * self.n_points] = array[:, : 3 * self.n_points] / self._std_geometric

        if self.vertex_features is not None:
            for idx, vertex_feature in enumerate(self.vertex_features):
                array[:, (3 + idx) * self.n_points : (3 + idx + 1) * self.n_points] = (
                    array[:, (3 + idx) * self.n_points : (3 + idx + 1) * self.n_points]
                    / self._std_features[idx]
                )

        return array

    def get_mesh_point_features(self, mesh, registered=False):
        if isinstance(mesh, str):
            # load mesh
            mesh = io.read_vtk(mesh)
        elif isinstance(mesh, Mesh):
            mesh = mesh.mesh

        if registered is False:
            # get point correspondences
            mesh = self.register_ref_to_mesh(mesh)

        # get the xyz points (the primary features)
        features = get_mesh_physical_point_coords(mesh).flatten()

        if self.vertex_features is not None:
            # if there are additional features, get them
            features_ = get_mesh_point_features(mesh, self.vertex_features).flatten()
            features = np.concatenate((features, features_))

        # normalize the features
        features = self.apply_normalization(features)

        return features

    def get_score(self, mesh=None, pc=None, max_pc=None, registered=False, normalize=True):
        """Get score"""

        features = self.get_mesh_point_features(mesh, registered=registered)

        if max_pc is not None:
            # return all PCs upto a maximum
            scores = self.PCs[:, :max_pc].T @ features.T
            if normalize is True:
                scores /= np.sqrt(self.Vs)[:max_pc, None]
        elif isinstance(pc, int):
            # return a single PC
            scores = self.PCs[:, pc : pc + 1].T @ features.T
            if normalize is True:
                scores /= np.sqrt(self.Vs)[pc, None]
        elif isinstance(pc, list):
            # return a list of specific PCs
            scores = self.PCs[:, pc].T @ features.T
            if normalize is True:
                scores /= np.sqrt(self.Vs)[pc, None]
        return scores

    def deform_model_using_pc_scores(self, scores, normalized=False):
        """Deform model"""
        scores = scores.squeeze()
        n_pcs = scores.shape[0]
        if normalized is True:
            scores = scores * np.sqrt(self.Vs)[:n_pcs]

        # get normalized deformation
        deformation = scores @ self.PCs[:, :n_pcs].T
        # unnormalize the deformation
        deformation[: 3 * self.n_points] *= self._std_geometric
        if self.vertex_features is not None:
            for idx in range(len(self.vertex_features)):
                deformation[
                    (3 + idx) * self.n_points : (3 + idx + 1) * self.n_points
                ] *= self._std_features[idx]

        new_points = self.mean + deformation

        return new_points

    def reconstruct_mesh(self, mesh, n_pcs, corresponding_points=False):
        # load mesh as Mesh
        if isinstance(mesh, Mesh):
            mesh = mesh
        else:
            mesh = Mesh(mesh)

        registered_mesh = mesh.copy()

        if corresponding_points is False:
            # rigidly register mesh to ref mesh & save transform
            icp_transform = registered_mesh.rigidly_register(
                self.mean_mesh,
                as_source=True,
                apply_transform_to_mesh=True,
                return_transformed_mesh=False,
                return_transform=True,
                max_n_iter=100,
                n_landmarks=1000,
                reg_mode="similarity",
            )

        # get scores & new mesh coordinates
        scores = self.get_score(
            mesh=registered_mesh, max_pc=n_pcs, registered=corresponding_points, normalize=False
        )[:, 0]

        # get normalized deformation
        # deformation = scores @ self.PCs[:,:n_pcs].T
        # # unnormalize the deformation
        # deformation[:3*self.n_points] *= self._std_geometric
        # if self.vertex_features is not None:
        #     for idx in range(len(self.vertex_features)):
        #         deformation[(3+idx)*self.n_points:(3+idx+1)*self.n_points] *= self._std_features[idx]
        # # add deformation to mean to get the new shape
        # new_shape = self._mean + deformation

        new_shape = self.deform_model_using_pc_scores(scores, normalized=False)

        # TODO: Extend to multiple bones

        # reconstruct mesh
        reconstructed_mesh = Mesh(
            create_vtk_mesh_from_deformed_points(
                mean_mesh=self.mean_mesh.mesh, new_points=new_shape, features=self.vertex_features
            )
        )

        if corresponding_points is False:
            # apply inverse of icp transform
            icp_transform.Inverse()
            reconstructed_mesh.apply_transform_to_mesh(icp_transform)

        return reconstructed_mesh

    def reconstruct_mesh_least_squares(self, mesh, n_pcs, registered=False):
        raise NotImplementedError
        # # load mesh as Mesh
        # if isinstance(mesh, Mesh):
        #     mesh = mesh
        # else:
        #     mesh = Mesh(mesh)

        # registered_mesh = mesh.copy()

        # # rigidly register mesh to ref mesh & save transform
        # icp_transform = registered_mesh.rigidly_register(
        #     self.mean_mesh,
        #     as_source=True,
        #     apply_transform_to_mesh=True,
        #     return_transformed_mesh=False,
        #     return_transform=True,
        #     max_n_iter=100,
        #     n_landmarks=1000,
        #     reg_mode='similarity'
        # )

        # # get features
        # features = self.get_mesh_point_features(mesh, registered=registered)

        # # use least squares to find the best fit between the SSM and these features.

        """
        BELOW IS CODE BRIEFLY WRITTEN TO DO TIHS TPYE OF REGISRATION. 
        NEED TO FIGURE OUT L1/L2 REGULARIZATION
        DO YOU DO REGISTRATION IN RAW OR NORMALIZED SPACE?
        DO YOU ADD WEIGHTING FACTORS TO PCS BASED ON VARIANCE EXPLAINED?
        """
        # from scipy.optimize import least_squares

        # def model(PC_scores, PCs, Vs, mean):
        #     n_pcs = PC_scores.shape[0]
        #     PC_scores *= np.sqrt(Vs[:n_pcs])
        #     deformation = PC_scores @ PCs[:,:n_pcs].T
        #     recon = (mean + deformation).squeeze()
        #     return recon

        # def residuals(PC_scores, PCs, Vs, mean, Y, l1_penalty=0, l2_penalty=0.1):
        #     # This function should compute the difference between your model's predictions and the actual data
        #     # 'params' are the parameters of your model that you are trying to optimize
        #     # 'X' is your input data
        #     # 'Y' is your output data

        #     Y = Y.squeeze()
        #     predicted_Y = model(PC_scores, PCs, Vs, mean)  # Replace this with your model

        #     n_pcs = PC_scores.shape[0]
        # #     Vs = Vs[:n_pcs]
        # #     PC_scores = PC_scores * 1/Vs

        #     l1_term = l1_penalty * np.abs(PC_scores).sum()
        #     l2_term = l2_penalty * (PC_scores**2).sum()

        #     return predicted_Y - Y + l1_term + l2_term

        # N_PCs = ssm.dict_threshold_n_pcs[95]

        # PCs = ssm.PCs
        # Vs = ssm.Vs
        # mean = ssm.mean
        # Y = features

        # initial_params = initial_params = np.random.normal(0, 0.025, size=N_PCs)

        # # Perform the optimization
        # result = least_squares(residuals, initial_params, args=(PCs, Vs, mean, Y))

        # from pymskt.statistics.pca import create_vtk_mesh_from_deformed_points
        # deformation = result['x'] @ ssm.PCs[:,:len(result['x'])].T
        # new_shape = ssm.mean + deformation

        # #TODO: Extend to multiple bones

        # # reconstruct mesh
        # reconstructed_mesh_LS = Mesh(create_vtk_mesh_from_deformed_points(mean_mesh=ssm.mean_mesh.mesh, new_points=new_shape, features=ssm.vertex_features))

        # # apply inverse of icp transform
        # # icp_transform.Inverse()
        # reconstructed_mesh_LS.apply_transform_to_mesh(icp_transform)

    def _mean_mesh(self):
        xyz = self._mean[: 3 * self.n_points].reshape(-1, 3)
        mesh = Mesh(vtk_deep_copy(self._ref_mesh))
        mesh.point_coords = xyz

        if self.vertex_features is not None:
            for idx, vertex_feature in enumerate(self.vertex_features):
                mesh.set_scalar(
                    vertex_feature,
                    self._mean[(3 + idx) * self.n_points : (3 + idx + 1) * self.n_points],
                )

        return mesh

    # GETTERS
    @property
    def list_mesh_paths(self):
        return self._list_mesh_paths

    @property
    def path_ref_mesh(self):
        return self._path_ref_mesh

    @property
    def mean(self):
        return self._mean

    @property
    def mean_mesh(self):
        return self._mean_mesh()

    @property
    def std(self):
        return self._std

    @property
    def std_geometric(self):
        return self._std_geometric

    @property
    def std_features(self):
        return self._std_features

    @property
    def centered(self):
        if self._points_normalized is False:
            self.normalize_points()
        return self._centered

    @property
    def ref_mesh(self):
        return self._ref_mesh

    @property
    def PCs(self):
        return self._PCs

    @property
    def Vs(self):
        return self._Vs

    # SETTERS
    @PCs.setter
    def PCs(self, value):
        self._PCs = value

    @Vs.setter
    def Vs(self, value):
        self._Vs = value

    @list_mesh_paths.setter
    def list_mesh_paths(self, value):
        assert isinstance(value, list)
        self._list_mesh_paths = value

    @path_ref_mesh.setter
    def path_ref_mesh(self, value):
        assert isinstance(value, str)
        self._path_ref_mesh = value

    @ref_mesh.setter
    def ref_mesh(self, value):
        self._ref_mesh = value


# def register_meshes(args):
#     idx, path_ref_mesh, list_mesh_paths, max_n_procrustes_steps, n_coords_spectral_ordering, n_coords_spectral_registration, n_extra_spectral, include_points_as_features, vertex_features = args
#     procrustes_reg = ProcrustesRegistration(
#         path_ref_mesh=path_ref_mesh,
#         list_mesh_paths=[list_mesh_paths[idx]],
#         max_n_registration_steps=max_n_procrustes_steps,
#         n_coords_spectral_ordering=n_coords_spectral_ordering,
#         n_coords_spectral_registration=n_coords_spectral_registration,
#         n_extra_spectral=n_extra_spectral,
#         include_points_as_features=include_points_as_features,
#         vertex_features=vertex_features,
#     )
#     procrustes_reg.execute()
#     return procrustes_reg.registered_pt_coords[idx,:,:], procrustes_reg.registered_vertex_features[idx,:,:]

# def find_point_correspondence(self):
#     # ...
#     pool = multiprocessing.Pool()
#     args_list = [(idx, self._path_ref_mesh, self._list_mesh_paths, self.max_n_procrustes_steps, self.n_coords_spectral_ordering, self.n_coords_spectral_registration, self.n_extra_spectral, self.include_points_as_features, self.vertex_features) for idx in range(self.n_meshes)]
#     results = pool.map(register_meshes, args_list)
#     pool.close()
#     pool.join()
#     for idx, (pt_coords, vertex_features) in enumerate(results):
#         self.registered_pt_coords[idx,:,:] = pt_coords
#         self.registered_vertex_features[idx,:,:] = vertex_features
#     # ...
