import tensorflow as tf

from utils.angle_utils import angle_normalize
from objectives.objective_function import Objective


class AngleDistance(Objective):
    """
    Compute the angular distance to the optimal path.
    """
    def __init__(self, params, fmm_map):
        self.p = params
        self.fmm_map = fmm_map
        self.tag = 'angular_distance_to_optimal_direction'

    def evaluate_objective(self, trajectory):
        optimal_angular_orientation_nk = self.fmm_map.fmm_angle_map.compute_voxel_function(trajectory.position_nk2())
        angular_dist_to_optimal_path_nk = angle_normalize(
            trajectory.heading_nk1()[:, :, 0] - optimal_angular_orientation_nk)
        return self.p.angle_cost*tf.pow(tf.abs(angular_dist_to_optimal_path_nk), self.p.power)
