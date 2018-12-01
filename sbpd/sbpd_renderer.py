from mp_env.render import swiftshader_renderer as sr
from mp_env import sbpd, map_utils as mu
from utils import depth_utils as du
import numpy as np


class SBPDRenderer():
    """
    An image renderer to render images from the
    SBPD dataset.
    """
    renderer = None

    def __init__(self, params):
        self.p = params

        d = sbpd.get_dataset(self.p.dataset_name, 'all')
        self.building = d.load_data(self.p.building_name, self.p.robot_params, self.p.flip)

        assert(len(self.p.camera_params.modalities) == 1)
        # Instantiating a camera/ shader object is only needed
        # for rgb and depth images
        if 'rgb' in self.p.camera_params.modalities or 'depth' in self.p.camera_params.modalities:
            r_obj = sr.get_r_obj(self.p.camera_params)
            self.building.set_r_obj(r_obj)
            self.building.load_building_into_scene()
        elif 'occupancy_grid' in self.p.camera_params.modalities:
            # MP Env only allows for square top views to be generated currently
            assert(self.p.camera_params.width == self.p.camera_params.height)
        else:
            assert(False)

    @classmethod
    def get_renderer(cls, params):
        """
        Used to instantiate a renderer object. Ensures that only one renderer
        object ever exists as they are very memory intensive.
        """
        r = cls.renderer
        if r is not None:
            dn, bn, f, c = r.p.dataset_name, r.p.building_name, r.p.flip, r.p.modalities
            if dn == params.dataset_name and bn == params.building_name and f == params.flip and c == params.modalities:
                return r
            else:
                assert(False, "Renderer settings are different than previously instantiated renderer")

        cls.renderer = cls(params)
        return cls.renderer

    def render_images(self, starts_n2, thetas_n1):
        """
        Render the corresponding image from
        the x, y positions in starts_2n facing heading
        thetas_1n
        """
        p = self.p.camera_params
        if 'occupancy_grid' in p.modalities:
            imgs = self._get_topview(starts_n2, thetas_n1, crop_size=p.width)
        elif 'rgb' in p.modalities:
            imgs = self._get_rgb_image(starts_n2, thetas_n1)
        elif 'depth' in p.modalities:
            raise NotImplementedError
        else:
            assert(False)
        return imgs

    def _get_rgb_image(self, starts, thetas):
        """
        Render rgb image(s) from the x, y, theta
        location in starts and thetas.
        """
        loc = starts * 1.
        # Scale thetas by 1/delta_theta as the building object
        # internally scales theta by delta_theta
        nodes = np.concatenate([loc, thetas / self.building.robot.delta_theta], axis=1)
        imgs = self.building.render_nodes(nodes)
        return imgs

    def _get_topview(self, starts, thetas, crop_size=80):
        """
        Render crop_size x crop_size topview(s) from the x, y, theta locations
        in starts and thetas.
        """
        traversible_map = self.building.map.traversible * 1.
        
        # In the topview the positive x axis points to the right and 
        # the positive y axis points up. The robot is located at
        # (0, (crop_size-1)/2) (in pixel coordinates) facing directly to the right
        x_axis = np.concatenate([np.cos(thetas), np.sin(thetas)], axis=1)
        y_axis = -np.concatenate([np.cos(thetas + np.pi / 2.), np.sin(thetas + np.pi / 2.)], axis=1)
        robot_loc = np.array([0, (crop_size-1.)/2.])
        crops_n1mk = mu.generate_egocentric_maps([traversible_map], [1.0], [crop_size],
                                                 starts, x_axis, y_axis, dst_theta=0.,
                                                 dst_loc=robot_loc)

        # Invert the crops so that 1.0 corresponds to occupied space
        # and 0.0 corresponds to free space
        crops_nmk1 = [np.logical_not(crop_1mk[0, :, :, None])*1.0 for crop_1mk in crops_n1mk]
        return crops_nmk1

    def _get_depth_image(self, starts, thetas, xy_resolution, map_size):
        """
        Render analytically projected depth images at the locations in
        starts, thetas. Bin data inside bins in a resolution of xy_resolution along x and y axis and
        z_bins in the z direction. Z Direction is the vertical z = 0 is floor. """
        r_obj = self.building.r_obj
        robot = self.building.robot
        z_bins = [-10, robot.base, robot.base + robot.height]

        loc = starts * 1.
        nodes = np.concatenate([loc, thetas / self.building.robot.delta_theta], axis=1)
        imgs = self.building.render_nodes(nodes)
        tt = np.array(imgs)

        assert (r_obj.fov_horizontal == r_obj.fov_vertical)
        cm = du.get_camera_matrix(r_obj.width, r_obj.height, r_obj.fov_vertical)
        XYZ = du.get_point_cloud_from_z(100. / tt[..., 0], cm)
        XYZ = XYZ * 100.  # convert to centimeters
        XYZ = du.make_geocentric(XYZ, robot.sensor_height, robot.camera_elevation_degree)
        count, isvalid = du.bin_points(XYZ * 1., map_size, z_bins, xy_resolution)
        count = [x[0, ...] for x in np.split(count, count.shape[0], 0)]
        isvalid = [x[0, ...] for x in np.split(isvalid, isvalid.shape[0], 0)]
        return imgs, count, isvalid

    def get_config(self):
        resolution, traversible = self.building.env.resolution, self.building.traversible
        return resolution, traversible
