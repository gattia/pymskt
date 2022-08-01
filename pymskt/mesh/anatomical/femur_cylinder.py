from scipy.optimize import least_squares
from vtk.util.numpy_support import vtk_to_numpy, numpy_to_vtk
import numpy as np
from pymskt.statistics.pca import pca_svd

class FitCylinderFemur:
    def __init__(
        self,
        femur,
        labels_name='labels',
        labels=(12, 13, 14, 15),
        z_resolution=50,
        theta_resolution=50,
        cylinder_percent_bone_width=0.9,
#         copy_femur=True,
        ftol=1e-4
        
    ):
        # not using the copy method becuase the femur is a pymskt object not vtk so it 
        # doesnt work with vtk_deep_copy - would need to create new femur object. 
#         if copy_femur is True:
#             self.femur = mskt.mesh.utils.vtk_deep_copy(femur)
#         else:
#             self.femur = femur
        self.femur = femur
        self.labels_name = labels_name
        if (type(labels) == int) or (type(labels) == float):
            self.labels = [labels,]
        else:
            self.labels = labels
        self.z_resolution = z_resolution
        self.theta_resolution = theta_resolution
        self.cylinder_percent_bone_width = cylinder_percent_bone_width
        self.ftol = ftol
        
        self.pts_articular_cylinder = None
        self.inertial_matrix_artic_surf = None
        self.inertial_aligned_pts_articular_cylinder = None
        self._height = None
        self._origin = None
        self._vector = None
        self._radius = None
        self.bounds = None
        self.params = None

    def get_initial_parameters(self):
        self.get_articular_surf_points()
        self.guess_height()
        self.guess_origin()
        self.guess_vector()
        self.guess_radius()
        
    def get_articular_surf_points(self):
        label_idx = vtk_to_numpy(self.femur.mesh.GetPointData().GetArray(self.labels_name))
        cylinder_labels = label_idx == self.labels[0] 
        if len(self.labels) > 1:
            for idx in range(1, len(self.labels)):
                cylinder_labels += (label_idx == self.labels[idx])
        cylinder_labels = np.asarray(cylinder_labels, dtype=int)
        cylinder_scalars = numpy_to_vtk(cylinder_labels)
        cylinder_scalars.SetName('cylinder labels')
        self.femur.mesh.GetPointData().AddArray(cylinder_scalars)
        
        self.pts_articular_cylinder = self.femur.point_coords[cylinder_labels == 1, :]
    
    def get_inertial_matrix_articular_surface(self):
        self.inertial_matrix_artic_surf, _ = pca_svd(self.pts_articular_cylinder.T)
        self.inv_inertial_matrix_artic_surf = np.linalg.inv(self.inertial_matrix_artic_surf)
    
    def get_artic_pts_aligned_inertial_matrix(self):
        if self.inertial_matrix_artic_surf is None:
            self.get_inertial_matrix_articular_surface()
        self.inertial_aligned_pts_articular_cylinder = self.inertial_matrix_artic_surf @ self.pts_articular_cylinder.T
    
    def guess_height(self):
        if self.inertial_aligned_pts_articular_cylinder is None:
            self.get_artic_pts_aligned_inertial_matrix()
        height_guess = self.inertial_aligned_pts_articular_cylinder[0,:].max() - self.inertial_aligned_pts_articular_cylinder[0,:].min()
        self._height = self.cylinder_percent_bone_width * height_guess

    def guess_origin(self):
        if self.inertial_matrix_artic_surf is None:
            self.get_inertial_matrix_articular_surface()
        min_x = self.inertial_aligned_pts_articular_cylinder[0,:].min() # this is going to be fully medial or laterl
        max_x = self.inertial_aligned_pts_articular_cylinder[0,:].max() # this is going to be fully medial or laterl (opposite above)
        mean_y = self.inertial_aligned_pts_articular_cylinder[1,:].mean() # use this as the origin y 
        mean_z = self.inertial_aligned_pts_articular_cylinder[2,:].mean() # I think this is going to be too close to the articular surface... but maybe good enought start? 

        # Get points in roughly the center of the cylinder of the condyle on the medial & lateral sides. 
        origin1 = np.asarray([[min_x, mean_y, mean_z],])
        origin2 = np.asarray([[max_x, mean_y, mean_z],])

        origin1 = self.inv_inertial_matrix_artic_surf @ origin1.T
        origin1 = np.squeeze(origin1.T)
        origin2 = self.inv_inertial_matrix_artic_surf @ origin2.T
        origin2 = np.squeeze(origin2.T)

        # Set the origin to a point just inside of the extreme on the min_x side (whether thats medial or lateral)
        origin = (origin2 - origin1) * 0.05 + origin1
        
        self._origin = origin

    def guess_vector(self):
        if self.inertial_matrix_artic_surf is None:
            self.get_inertial_matrix_articular_surface()
        vector = np.asarray([
            self.inertial_matrix_artic_surf[0,0], # vector X
            self.inertial_matrix_artic_surf[1,0], # vector Y
            self.inertial_matrix_artic_surf[2,0], # vector Z
        ], dtype=float)
        
        if np.linalg.norm(vector) != 1:
            vector = vector/np.linalg.norm(vector)
        self._vector = vector

    def guess_radius(self):
        if self.inertial_aligned_pts_articular_cylinder is None:
            self.get_artic_pts_aligned_inertial_matrix()

        range_y = self.inertial_aligned_pts_articular_cylinder[1,:].max() - self.inertial_aligned_pts_articular_cylinder[1,:].min()
        radius = range_y/2
        
        self._radius = radius
    
    @staticmethod
    def get_unit_cylinder(z_resolution, theta_resolution):
        theta = np.linspace(0, 2*np.pi, theta_resolution)
        
        unit_cylinder = np.zeros((theta_resolution * z_resolution, 3))  
        unit_cylinder[:, 0] = np.tile(np.cos(theta), z_resolution)
        unit_cylinder[:, 1] = np.tile(np.sin(theta), z_resolution)

    #     for i in range(z_resolution):
        unit_cylinder[:, 2] = np.repeat(np.linspace(0, 1, z_resolution), theta_resolution)

        return unit_cylinder

    def cylinder_function(self, origin, height, radius, vector, z_resolution=None, theta_resolution=None):
        if z_resolution is None:
            z_resolution = self.z_resolution
        if theta_resolution is None:
            theta_resolution = self.theta_resolution
        
        # ensure vector is np array of floats. 
        vector = np.asarray(vector, dtype=float)
        if np.linalg.norm(vector) != 1:
            vector = vector/np.linalg.norm(vector)

        # scale the size of the cylinder
        unit_cylinder = FitCylinderFemur.get_unit_cylinder(z_resolution=z_resolution, theta_resolution=theta_resolution)
        unit_cylinder[:, 0] = unit_cylinder[:, 0] * radius
        unit_cylinder[:, 1] = unit_cylinder[:, 1] * radius
        unit_cylinder[:, 2] = unit_cylinder[:, 2] * height

        # Create rotation matrix to rotate cylinder axis
        #make some vector not in the same direction as v
        not_v = np.array([1, 0, 0])
        if (vector == not_v).all():
            not_v = np.array([0, 1, 0])
        #make vector perpendicular to v
        norm1 = np.cross(vector, not_v)
        #normalize n1
        norm1 /= np.linalg.norm(norm1)
        #make unit vector perpendicular to v and n1
        norm2 = np.cross(vector, norm1)

        rot_matrix = np.zeros((3,3))
        rot_matrix[:,0] = norm1
        rot_matrix[:,1] = norm2
        rot_matrix[:,2] = vector

        # rotate the cylinder along the vector axis
        unit_cylinder = rot_matrix @ unit_cylinder.T
        unit_cylinder = unit_cylinder.T
        unit_cylinder += origin

        return unit_cylinder
    
    @staticmethod
    def residuals(points, cylinder):
        """
        Find closest point on cylinder for each point. Calcualte 
        """

        diff = points[None, :, :] - cylinder[:, None, :]
        diff = np.sqrt(np.sum(diff **2, axis=-1))
        resids = diff.min(axis=0)
        return resids

    def get_func(self):
        """
        Function to create the function that we want to minimize. The returned function
        returns the residuals of the points vs the generated cylinder. 
        """
        def func(params):
            cylinder = self.cylinder_function(
                origin=[params[0], params[1], params[2]],
                height=params[3],
                radius=params[4],
                vector=np.asarray([params[5],params[6],params[7]], dtype=float),
                z_resolution=self.z_resolution, 
                theta_resolution=self.theta_resolution
            )

            resid = FitCylinderFemur.residuals(self.pts_articular_cylinder, cylinder)

            return resid
        return func
    
    def get_bounds(self):
        self.bounds = [
            [
                self.pts_articular_cylinder[:,0].min(),
                self.pts_articular_cylinder[:,1].min(),
                self.pts_articular_cylinder[:,2].min(),
                self._height - self._height * 0.2,
                self._radius - self._radius * 0.2,
                -1,
                -1,
                -1

            ],
            [
                self.pts_articular_cylinder[:,0].max(),
                self.pts_articular_cylinder[:,1].max(),
                self.pts_articular_cylinder[:,2].max(),
                self._height + self._height * 0.2,
                self._radius + self._radius * 0.2,
                1,
                1,
                1 
            ]
        ]
    
    def get_params(self):
        if (self._origin is None) or (self._height is None) or (self._radius is None) or (self._vector is None):
            self.get_initial_parameters()
        self.params = [
            self._origin[0],
            self._origin[1],
            self._origin[2],
            self._height,
            self._radius,
            self._vector[0],
            self._vector[1],
            self._vector[2]
        ]
    def fit(self):
        if self.params is None:
            self.get_params()
        if self.bounds is None:
            self.get_bounds()
            
        func = self.get_func()
        
        result = least_squares(
            func,
            self.params,
            bounds=self.bounds,
            ftol=self.ftol                    
        )
        
        self.optimization_success = result['success']
        self.params = result['x']
        self._origin = np.array([self.params[0], self.params[1], self.params[2]])
        self._height = self.params[3]
        self._radius = self.params[4]
        self._vector = np.array([self.params[5], self.params[6], self.params[7]])
        self._vector /= np.linalg.norm(self._vector)
        
        if self.optimization_success is True:
            print('Fitting cylinder to condyles completed successfully!')
        else:
            print('Fitting cylinder to condyles did not converge properly:\n', result)
    
    @property
    def height(self):
        return self._height
    
    @property
    def radius(self):
        return self._radius
    
    @property
    def origin(self):
        return self._origin
    
    @property
    def vector(self):
        return self._vector
    
    @property
    def cylinder(self):
        return self.cylinder_function(
            origin=self._origin, 
            height=self._height, 
            radius=self._radius, 
            vector=self._vector
        )