import numpy as np
import pyvista as pv
from vtk.util.numpy_support import vtk_to_numpy

from pymskt.statistics.pca import pca_svd


class FitLongAxisFemur:
    def __init__(
        self,
        femur,
        labels_name="labels",
        cart_label=1,
        bone_label=0,
        percent_epiph_higher=0.3,
        n_pts=100,
        buffer=1,
        use_center_pts_only=True,
    ):
        self.femur = femur
        self.labels_name = labels_name
        self.cart_label = cart_label
        self.bone_label = bone_label
        self.percent_epiph_higher = percent_epiph_higher
        self.n_pts = n_pts
        self.buffer = buffer
        self.use_center_pts_only = use_center_pts_only

        self._bone_pts = None
        self._cart_pts = None
        self.min_cart = None
        self.min_bone = None
        self.max_cart = None
        self.max_bone = None

        self.long_axis = None
        self.proximal = None

        self.origin = None

        self._diaph_points = None
        self._epiph_points = None

        self._vector = None

    def get_bone_cart_points(self):
        labels = self.femur.point_data[self.labels_name]
        if (type(self.cart_label) == list) or (type(self.cart_label) == tuple):
            cart_indices = None
            for cart_label in self.cart_label:
                if cart_indices is None:
                    cart_indices = labels == cart_label
                cart_indices += labels == cart_label
        else:
            cart_indices = labels == self.cart_label
        cart_ids = np.where(cart_indices)
        cart_pts = np.squeeze(self.femur.point_coords[cart_ids, :])
        bone_ids = np.where(labels == self.bone_label)
        bone_pts = np.squeeze(self.femur.point_coords[bone_ids, :])

        self._bone_pts = bone_pts
        self._cart_pts = cart_pts

    def get_min_max_bone_cart(self):
        if (self._bone_pts is None) or (self._cart_pts is None):
            self.get_bone_cart_points()

        self.max_cart = np.max(self._cart_pts, axis=0)
        self.max_bone = np.max(self._bone_pts, axis=0)

        self.min_cart = np.min(self._cart_pts, axis=0)
        self.min_bone = np.min(self._bone_pts, axis=0)

    def get_long_axis_and_proximal_direction(self):
        if (self._bone_pts is None) or (self._cart_pts is None):
            self.get_bone_cart_points()
        if (
            (self.max_cart is None)
            or (self.max_bone is None)
            or (self.min_cart is None)
            or (self.min_bone is None)
        ):
            self.get_min_max_bone_cart()

        cart_range = self.max_cart - self.min_cart
        bone_range = self.max_bone - self.min_bone
        long_axis = np.argmax(np.abs(cart_range - bone_range))

        if np.abs(self.max_cart[2] - self.max_bone[2]) > np.abs(
            self.min_cart[2] - self.min_bone[2]
        ):
            proximal = int(1)
        else:
            proximal = int(-1)

        self.proximal = proximal
        self.long_axis = long_axis

    def guess_origin(self):
        if (
            (self.max_cart is None)
            or (self.max_bone is None)
            or (self.min_cart is None)
            or (self.min_bone is None)
        ):
            self.get_min_max_bone_cart()

        epiphysis_height = self.max_cart[self.long_axis] - self.min_cart[self.long_axis]

        origin = np.mean(self.femur.point_coords, axis=0)

        if self.proximal == 1:
            origin[self.long_axis] = (
                self.max_cart[self.long_axis] + epiphysis_height * self.percent_epiph_higher
            )  # add 5mm buffer?
        if self.proximal == -1:
            origin[self.long_axis] = (
                self.min_cart[self.long_axis] - epiphysis_height * self.percent_epiph_higher
            )  # add 5mm buffer?

        self.origin = origin

    def get_diaph_epiph_points(self):
        if (self.long_axis is None) or (self.proximal is None):
            self.get_long_axis_and_proximal_direction()
        if self.origin is None:
            self.guess_origin()

        # NORMAL MUST HAVE MAGNITUDE OF 1
        normal = np.zeros(3)
        normal[self.long_axis] = 1

        # create a random point on the plane we want to slice from.
        point_on_plane = (
            np.random.random_sample(size=3) * 20 - 10
        )  # make random sample in range [-10, 10]
        point_on_plane[self.long_axis] = self.origin[self.long_axis]

        side = normal @ (self.femur.point_coords - point_on_plane).T

        diaph_points = self.femur.point_coords[side > 0, :]
        epiph_points = self.femur.point_coords[side < 0, :]

        pv_diaph, _ = self.femur.remove_points(side < 0)

        self.pv_diaph = pv_diaph
        self._diaph_points = diaph_points
        self._epiph_points = epiph_points

    def get_diaph_vector(self):
        if (self._diaph_points is None) or (self._epiph_points is None) or (self.pv_diaph):
            self.get_diaph_epiph_points()

        min_long = np.min(self._diaph_points[:, self.long_axis])
        max_long = np.max(self._diaph_points[:, self.long_axis])
        if self.proximal == 1:
            min_ = min_long + self.buffer
            max_ = max_long - self.buffer
        elif self.proximal == -1:
            max_ = max_long - self.buffer
            min_ = min_long + self.buffer

        normal = [0, 0, 0]
        normal[self.long_axis] = 1

        centers = []

        for idx, depth in enumerate(np.linspace(min_, max_, self.n_pts)):
            # get slice
            origin_ = [0, 0, 0]
            origin_[self.long_axis] = depth
            slice_ = self.pv_diaph.slice(normal=normal, origin=origin_)
            # get points from slice
            slice_pts = slice_.points

            if self.use_center_pts_only is True:
                ptp = np.ptp(slice_pts, axis=0)
                min_ = np.min(slice_pts, axis=0)
                middle = min_ + ptp / 2
                lower = middle - 0.05 * ptp
                upper = middle + 0.05 * ptp

                pts = None
                for axis in range(3):
                    if ptp[axis] == 0:
                        pass
                    else:
                        indices = (slice_pts[:, axis] < upper[axis]) * (
                            slice_pts[:, axis] > lower[axis]
                        )
                        pts_ = slice_pts[indices, :]
                        if pts is None:
                            pts = pts_
                        else:
                            pts = np.append(pts, pts_, axis=0)
            #             pts.append()
            elif self.use_center_pts_only is False:
                pts = slice_.points

            centroid = np.mean(pts, axis=0)
            # append Y-position of that point to the `Ys` list
            centers.append(centroid)
        centers = np.asarray(centers)

        inertial_matrix, _ = pca_svd(centers.T)

        vector = inertial_matrix[:, 0]

        self._vector = vector

    def fit(self):
        self.get_diaph_vector()

    @property
    def vector(self):
        return self._vector

    @property
    def diaph_points(self):
        return self._diaph_points

    @property
    def epiph_points(self):
        return self._epiph_points

    @property
    def cart_pts(self):
        return self._cart_pts

    @property
    def bone_pts(self):
        return self._bone_pts
