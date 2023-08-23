from pymskt.statistics import ProcrustesRegistration
from pymskt.mesh import io
from pymskt.mesh.meshTools import get_mesh_physical_point_coords, get_mesh_point_features
from pymskt.mesh.meshRegistration import non_rigidly_register
from pymskt.statistics.pca import pca_svd, save_meshes_across_pc, save_gif, save_gif_vec_2_vec, save_mesh_vec_2_vec
from pymskt.mesh import Mesh
from pymskt.mesh.utils import vtk_deep_copy
import numpy as np
import os
import json
from datetime import datetime


class SSM:
    def __init__(self,
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
                 verbose=False,
                ):
        # Pre-process list(s) of meshes & related info. 
        self._list_mesh_paths = list_mesh_paths
        self._path_ref_mesh = path_ref_mesh

        if self._list_mesh_paths is not None:
           self.parse_list_mesh_paths()

        # SSM Options
        self.vertex_features = vertex_features
        self.points_already_correspond = points_already_correspond
        self.feature_norm_ignore_zero = feature_norm_ignore_zero
        self.feature_norm_include_pt_if_any_mesh_has_value = feature_norm_include_pt_if_any_mesh_has_value
        
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
        self.points = np.zeros((
            self.n_meshes, 
            self.n_points * self.n_features
        ))

        self._points_loaded = False
        self._points_normalized = False
        self.verbose = verbose
    
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
        self._list_mesh_paths.insert(0, self._path_ref_mesh)
    
    def parse_ref_mesh_and_params(self):
        """Get reference mesh parameters"""
        if (hasattr(self, '_ref_mesh') is False) or (self._ref_mesh is None):
            if self._path_ref_mesh is not None:
                self._ref_mesh = io.read_vtk(self._path_ref_mesh)
            else:
                self._ref_mesh = None
                self.n_points = 0
                self.n_features = 0
                return
        self.n_points = self._ref_mesh.GetNumberOfPoints()
        self.n_features = 3 + (len(self.vertex_features) if self.vertex_features is not None else 0)
            

    def find_point_correspondence(self, 
                                #   path_ref_mesh=None, 
                                #   list_mesh_paths=None
                                 ):
        """Find point correspondence between meshes"""
        # if path_ref_mesh is not None:
        #     self._path_ref_mesh = path_ref_mesh
        # if list_mesh_paths is not None:
        #     self._list_mesh_paths = list_mesh_paths
        procrustes_reg = ProcrustesRegistration(
            path_ref_mesh=self._path_ref_mesh, # using the idx of the best mesh from the previous step
            list_mesh_paths=self._list_mesh_paths, # This will automatically remove the ref_mesh path if it is in the list.
            max_n_registration_steps=self.max_n_procrustes_steps,
            n_coords_spectral_ordering=self.n_coords_spectral_ordering,
            n_coords_spectral_registration=self.n_coords_spectral_registration,
            n_extra_spectral=self.n_extra_spectral,
            include_points_as_features=self.include_points_as_features,
            vertex_features=self.vertex_features,    
        )

        procrustes_reg.execute()

        if self.save_registered_meshes is True:
            procrustes_reg.save_meshes(folder=self.folder_save_registered_meshes)
        
        # fill pre-allocated points array

        for idx in range(self.n_meshes):
            xyz = procrustes_reg.registered_pt_coords[idx,:,:].flatten()
            features = procrustes_reg.registered_vertex_features[idx,:,:].flatten()
            self.points[idx, :] = np.concatenate((xyz, features))
    
    def prepare_points(self):
        """Prepare points"""
        print('Preparing points...')
        if self.points_already_correspond is False:
            print('Finding point correspondences...')
            self.find_point_correspondence()
        
        elif self.points_already_correspond is True:
            print('Loading points...')
            for idx, path in enumerate(self._list_mesh_paths):
                mesh = io.read_vtk(path)
                xyz = get_mesh_physical_point_coords(mesh).flatten()
                features = get_mesh_point_features(mesh, self.vertex_features).flatten()
                self.points[idx, :] = np.concatenate((xyz, features))
        
        self._points_loaded = True

    def normalize_points(self):
        """Get points info"""
        print('Beginning Point Normalization')
        self._mean = np.mean(self.points, axis=0)
        self._std = np.std(self.points, axis=0, ddof=1)
       
        self._centered = self.points - self._mean

        self._std_geometric = np.std(self._centered[:, :3*self.n_points], ddof=1)
        if self.verbose is True:
            print('Geometric std: {}'.format(self._std_geometric))

        if self.vertex_features is not None:
            self._std_features = []
            for idx in range(len(self.vertex_features)):
                data = self._centered[:, (3+idx)*self.n_points:(3+idx+1)*self.n_points]
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
                    print(f'Feature {self.vertex_features[idx]} std: {self._std_features[idx]}')
        
        self._centered = self.apply_normalization(self.points)
        # self._centered[:, :3*self.n_points] = self._centered[:, :3*self.n_points] / self._std_geometric
        # if self.vertex_features is not None:
        #     for idx in range(len(self.vertex_features)):
        #         data = self._centered[:, (3+idx)*self.n_points:(3+idx+1)*self.n_points]
        #         self._centered[:, (3+idx)*self.n_points:(3+idx+1)*self.n_points] = data / self._std_features[idx]

        self._points_normalized = True

    def fit_model(self):
        """Fit PCA based SSM model"""
        if self._points_loaded is False:
            self.prepare_points()
        if self._points_normalized is False:
            self.normalize_points()
        print('Fitting PCA-based model...')
        self._PCs, self._Vs = pca_svd(self._centered.T)
    
    def get_dict_model_params(self, PCs_filename, Vs_filename):
        """Get dict of model parameters"""
        # Add generic model information
        dict_dump = {
            'date': datetime.now().strftime("%b-%d-%Y"),
            'n_points': self.n_points,
            'n_meshes': self.n_meshes,
            'PCs_filename': f'{PCs_filename}.npy',
            'Vs_filename': f'{Vs_filename}.npy',
            'geometric_std': self._std_geometric,
            'list_vertex_features': self.vertex_features,
            'list_vertex_features_stds': self._std_features,
            'feature_norm_ignore_zero': self.feature_norm_ignore_zero,
        }

        # add feature specific stds for normalization
        for idx, vertex_feature in enumerate(self.vertex_features):
            dict_dump[f'{vertex_feature}_std'] = self._std_features[idx] 

        # Add whether points already corresponding... and registration parameters
        # registration parameters should be used for future registrations/using the model. 
        dict_dump['points_already_correspond'] = self.points_already_correspond
        dict_dump['n_coords_spectral_ordering'] = self.n_coords_spectral_ordering
        dict_dump['n_coords_spectral_registration'] = self.n_coords_spectral_registration
        dict_dump['n_spectral_features'] = self.n_spectral_features
        dict_dump['n_extra_spectral'] = self.n_extra_spectral
        dict_dump['include_points_as_features'] = self.include_points_as_features

        dict_dump['list_mesh_locations'] = self._list_mesh_paths

        return dict_dump

    def save_model(self, folder=None, PCs_filename='PCs', Vs_filename='Vs', save_points=False):
        """
        Save PCA-based model
        Notes
        -----
        To decode the model: 
            1. 
        """
        print('Saving model...')
        if os.path.isdir(folder) is False:
            os.makedirs(folder, exist_ok=True)
        np.save(os.path.join(folder, f'{PCs_filename}.npy'), self._PCs)
        np.save(os.path.join(folder, f'{Vs_filename}.npy'), self._Vs)

        # Save mean mesh
        io.write_vtk(self._ref_mesh, os.path.join(folder, 'ref_mesh.vtk'))

        io.write_vtk(self._mean_mesh().mesh, os.path.join(folder, 'mean_mesh.vtk'))

        # save mean features / everything? 
        np.save(os.path.join(folder, 'mean_features.npy'), self._mean)
        # save std of geometric points / features:
        with open(os.path.join(folder, 'ssm_model_information.json'), 'w') as f:
            dict_model_params = self.get_dict_model_params(PCs_filename, Vs_filename)
            json.dump(dict_model_params, f, indent=4)
        
        if save_points is True:
            np.save(os.path.join(folder, 'points.npy'), self.points)

    def load_model(self, folder):
        """Load PCA-based model"""
        with open(os.path.join(folder, 'ssm_model_information.json'), 'r') as f:
            dict_model_params = json.load(f)
        
        PCs_path = os.path.join(folder, dict_model_params['PCs_filename'])
        Vs_path = os.path.join(folder, dict_model_params['Vs_filename'])

        self._PCs = np.load(PCs_path)
        self._Vs = np.load(Vs_path)
        self._ref_mesh = io.read_vtk(os.path.join(folder, 'ref_mesh.vtk'))
        self._mean = np.load(os.path.join(folder, 'mean_features.npy'))

        self.n_points = dict_model_params['n_points']
        self.n_meshes = None #TODO: Parse this from self._list_mesh_paths?
        self._std_geometric = dict_model_params['geometric_std']
        self.vertex_features = dict_model_params['list_vertex_features']
        self._std_features = dict_model_params['list_vertex_features_stds']
        self.feature_norm_ignore_zero = dict_model_params['feature_norm_ignore_zero']

        # Registration parameters
        self.n_coords_spectral_ordering = dict_model_params['n_coords_spectral_ordering'] 
        self.n_coords_spectral_registration = dict_model_params['n_coords_spectral_registration'] 
        self.n_spectral_features = dict_model_params['n_spectral_features']
        self.n_extra_spectral = dict_model_params['n_extra_spectral']
        self.include_points_as_features = dict_model_params['include_points_as_features']

        self._list_mesh_paths = dict_model_params['list_mesh_locations']

        if os.path.exists(os.path.join(folder, 'points.npy')):
            self.points = np.load(os.path.join(folder, 'points.npy'))
            self._points_loaded = True

    def calculate_total_variance(self):
        """Calculate total variance"""
        pass
    
    def calculate_variance_explained_per_pc(self):
        """Calculate variance explained per PC"""
        pass
    
    def deform_model(self, pc=0, std=3):
        """Deform model"""
        pass
    
    def save_meshes_across_pc(self, folder_save, pc, std, step_size=1, mesh_name='bone'):
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
            mesh_name=f'{mesh_name}_{pc}',   #['tibia', 'patella', 'femur'], #['femur', 'tibia', 'patella'],
            save_filename='{mesh_name}_{sd}.vtk' #specifically not with `f` so we can fill in later. 
        )
    
    def save_gif_across_pc(
            self, 
            path_save, 
            pc, 
            std, 
            step_size=0.25, 
            camera_position='xz',
            scalar_bar_range=[0, 4],
            background_color='white',
            cmap=None
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
            color='orange' if self.vertex_features is None else None, 
            show_edges=True, 
            edge_color='black',
            camera_position=camera_position,
            window_size=[3000, 4000],
            background_color=background_color,
            scalar_bar_range=scalar_bar_range,
            cmap=cmap,
            verbose=False,
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
        **kwargs
    ):
        if (self.vertex_features is None) and ('color' not in kwargs):
            kwargs['color'] = 'orange'

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
            **kwargs
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
        **kwargs    
    ):
        save_mesh_vec_2_vec(
            path_save=path_save,
            PCs=self._PCs,
            Vs=self._Vs,
            mean_coords=self._mean,  # mean_coords could be extracted from mean mesh...?
            mean_mesh=self._ref_mesh,
            vec_1=vec_start,
            vec_2=vec_end,
            **kwargs
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
        )
        return registered_mesh

    def apply_normalization(self, array):
        array = array.copy()
        if len(array.shape) == 1:
            assert(array.shape[0] == self._mean.shape[0])
            array = np.expand_dims(array, axis=0)
        else:
            assert(array.shape[1] == self._mean.shape[0])

        array -= self._mean

        array[:, :3*self.n_points] = array[:, :3*self.n_points] / self._std_geometric
        
        if self.vertex_features is not None:
            for idx, vertex_feature in enumerate(self.vertex_features):
                array[:, (3+idx)*self.n_points:(3+idx+1)*self.n_points] = \
                    array[:, (3+idx)*self.n_points:(3+idx+1)*self.n_points] / self._std_features[idx]
        
        return array

    def get_score(self, mesh=None, pc=None, max_pc=None, registered=False):
        """Get score"""
        if isinstance(mesh, str):
            # load mesh
            mesh = io.read_vtk(mesh)
        if registered is False:
            # get point correspondences
            mesh = self.register_ref_to_mesh(mesh)

        # get the xyz points (the primary features)
        features  = get_mesh_physical_point_coords(mesh).flatten()

        if self.vertex_features is not None:
            # if there are additional features, get them
            features_ = get_mesh_point_features(mesh, self.vertex_features).flatten()
            features = np.concatenate((features, features_))
        
        # normalize the features
        features = self.apply_normalization(features)

        if max_pc is not None:
            # return all PCs upto a maximum
            scores = self.PCs[:,:max_pc].T @ features.T
            scores /= (np.sqrt(self.Vs)[:max_pc, None])
        elif isinstance(pc, int):
            # return a single PC
            scores = self.PCs[:,pc:pc+1].T @ features.T
            scores /= (np.sqrt(self.Vs)[pc, None])
        elif isinstance(pc, list):
            # return a list of specific PCs
            scores = self.PCs[:,pc].T @ features.T
            scores /= (np.sqrt(self.Vs)[pc, None])
        return scores

    def _mean_mesh(self):
        xyz = self._mean[: 3 * self.n_points].reshape(-1, 3)
        mesh = Mesh(vtk_deep_copy(self._ref_mesh))
        mesh.point_coords = xyz

        if self.vertex_features is not None:
            for idx, vertex_feature in enumerate(self.vertex_features):
                mesh.set_scalar(vertex_feature, self._mean[(3+idx)*self.n_points:(3+idx+1)*self.n_points])
        
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
        assert(isinstance(value, list))
        self._list_mesh_paths = value
    
    @path_ref_mesh.setter
    def path_ref_mesh(self, value):
        assert(isinstance(value, str))
        self._path_ref_mesh = value
    
    @ref_mesh.setter
    def ref_mesh(self, value):
        self._ref_mesh = value

    
        
        