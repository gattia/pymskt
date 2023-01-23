from .femur_cylinder import FitCylinderFemur
from .femur_long_axis import FitLongAxisFemur
import numpy as np

class FemurACS:
    def __init__(
        self,
        femur,
        labels_name='labels',
        cart_label=1,
        condyle_cart_labels=(12, 13, 14, 15),
        bone_label=0,
        percent_epiph_higher=0.3,
        n_pts=100,
        buffer=1,
        use_center_pts_only=True,
        theta_resolution=50,
        z_resolution=50,
        cylinder_percent_bone_width=0.9,
        ftol=1e-3
    ):
        self.femur = femur
        self.labels_name = labels_name
        self.cart_label = cart_label
        self.condyle_cart_labels = condyle_cart_labels
        self.bone_label = bone_label
        self.percent_epiph_higher = percent_epiph_higher
        self.n_pts = n_pts
        self.buffer = buffer
        self.use_center_pts_only = use_center_pts_only
        self.theta_resolution = theta_resolution
        self.z_resolution = z_resolution
        self.cylinder_percent_bone_width = cylinder_percent_bone_width
        self.ftol = ftol
    
        self._fit_longaxis = None
        self._fit_cylinder = None
        self._ml_axis = None
        self._is_axis = None
        self._ap_axis = None
        self._origin = None
    
    def get_axes(self):
        self._ml_axis = self._fit_cylinder.vector
        self._is_axis = self._fit_longaxis.vector

        self._ap_axis = np.cross(self._ml_axis, self._is_axis)
        self._is_axis = np.cross(self._ap_axis, self._ml_axis)

        cart_pts = self._fit_longaxis.cart_pts
        proj_pts = self._ml_axis @ (cart_pts - self._fit_cylinder.origin).T
        self._origin = self._fit_cylinder.origin + np.mean(proj_pts) * self._fit_cylinder.vector

    def fit(self):
        
        self._fit_longaxis = FitLongAxisFemur(
            self.femur, 
            labels_name=self.labels_name, 
            cart_label=self.cart_label,
            bone_label=self.bone_label,
            percent_epiph_higher=self.percent_epiph_higher,
            n_pts=self.n_pts,
            buffer=self.buffer,
            use_center_pts_only=self.use_center_pts_only
        )
        self._fit_longaxis.fit()
        
        self._fit_cylinder = FitCylinderFemur(
            self.femur,
            labels_name=self.labels_name,
            labels=self.condyle_cart_labels,
            z_resolution=self.z_resolution,
            theta_resolution=self.theta_resolution,
            cylinder_percent_bone_width=self.cylinder_percent_bone_width,
            ftol=self.ftol
        )
        self._fit_cylinder.fit()
        
        self.get_axes()
    
    @property
    def origin(self):
        return self._origin
    
    @property
    def ml_axis(self):
        return self._ml_axis
    
    @property
    def is_axis(self):
        return self._is_axis
    
    @property
    def ap_axis(self):
        return self._ap_axis
    
    @property
    def fit_cylinder(self):
        return self._fit_cylinder
    
    @property
    def fit_longaxis(self):
        return self._fit_longaxis
    