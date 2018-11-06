import utils.utils as utils
import os
import numpy as np
import tensorflow as tf
from optCtrl.lqr import LQRSolver
from trajectory.trajectory import Trajectory, SystemConfig
from control_pipelines.base import ControlPipelineBase
from control_pipelines.control_pipeline_v0_helper import ControlPipelineV0Helper


class ControlPipelineV0(ControlPipelineBase):
    """
    A control pipeline that generate dynamically feasible spline trajectories of varying horizon.
    """

    def __init__(self, params):
        self.start_velocities = np.linspace(
            0.0, params.binning_parameters.max_speed, params.binning_parameters.num_bins)
        self.helper = ControlPipelineV0Helper()
        super().__init__(params)

    def plan(self, start_config):
        """Computes which velocity bin start_config belongs to
        and returns the corresponding waypoints, horizons, lqr_trajectories,
        and LQR controllers."""
        idx = tf.squeeze(self._compute_bin_idx_for_start_velocities(
            start_config.speed_nk1()[:, :, 0])).numpy()
        self.waypt_configs_world[idx] = self.system_dynamics.to_world_coordinates(start_config, self.waypt_configs[idx],
                                                                                  self.waypt_configs_world[idx], mode='assign')
        self.trajectories_world[idx] = self.system_dynamics.to_world_coordinates(start_config, self.lqr_trajectories[idx],
                                                                                 self.trajectories_world[idx], mode='assign')

        # TODO: K & k are currently in egocentric coordinates
        controllers = {'K_array': self.K_arrays[idx], 'k_array': self.k_arrays[idx]}
        return self.waypt_configs_world[idx], self.horizons[idx], self.trajectories_world[idx], controllers

    def generate_control_pipeline(self, params=None):
        p = self.params
        # Initialize spline, cost function, lqr solver
        waypoints_egocentric = self._sample_egocentric_waypoints(vf=0.)
        self._init_pipeline()
        pipeline_data = self.helper.empty_data_dictionary()

        with tf.name_scope('generate_control_pipeline'):
            if not self._incorrectly_binned_data_exists():
                for v0 in self.start_velocities:
                    start_config = self.system_dynamics.init_egocentric_robot_config(dt=p.system_dynamics_params.dt,
                                                                                     n=self.waypoint_grid.n,
                                                                                     v=v0)
                    goal_config = waypoints_egocentric.copy()
                    start_config, goal_config, horizons_n1 = self._dynamically_fit_spline(
                        start_config, goal_config)
                    lqr_trajectory, K_array, k_array = self._lqr(start_config)
                    data_bin = {'start_configs': start_config,
                                'waypt_configs': goal_config,
                                'start_speeds': self.spline_trajectory.speed_nk1()[:, 0],
                                'spline_trajectories': self.spline_trajectory.copy(),
                                'horizons': horizons_n1,
                                'lqr_trajectories': lqr_trajectory,
                                'K_arrays': K_array,
                                'k_arrays': k_array}
                    self.helper.append_data_bin_to_pipeline_data(pipeline_data, data_bin)
                # This data is incorrectly binned by velocity
                # so collapse it all into one bin before saving it
                pipeline_data = self.helper.concat_data_across_binning_dim(pipeline_data)
                self._save_incorrectly_binned_data(pipeline_data)
            else:
                pipeline_data = self._load_incorrectly_binned_data()

            pipeline_data = self._rebin_data_by_initial_velocity(pipeline_data)
            self._set_instance_variables(pipeline_data)

        for i, v0 in enumerate(self.start_velocities):
            filename = self._data_file_name(v0=v0)
            data_bin = self.helper.prepare_data_for_saving(pipeline_data, i)
            self.save_control_pipeline(data_bin, filename)

    def _dynamically_fit_spline(self, start_config, goal_config):
        """Fit a spline between start_config and goal_config only keeping
        points that are dynamically feasible within the planning horizon."""
        p = self.params
        times_nk = tf.tile(tf.linspace(0., p.planning_horizon_s, p.planning_horizon)[
                           None], [self.waypoint_grid.n, 1])
        final_times_n1 = tf.ones((self.waypoint_grid.n, 1), dtype=tf.float32) * p.planning_horizon_s
        self.spline_trajectory.fit(start_config, goal_config,
                                   final_times_n1=final_times_n1)
        self.spline_trajectory.eval_spline(times_nk, calculate_speeds=True)
        self.spline_trajectory.rescale_spline_horizon_to_dynamically_feasible_horizon(speed_max_system=self.system_dynamics.v_bounds[1],
                                                                                      angular_speed_max_system=self.system_dynamics.w_bounds[1])

        valid_idxs = self.spline_trajectory.find_trajectories_within_a_horizon(p.planning_horizon_s)
        horizons_n1 = self.spline_trajectory.final_times_n1

        # Only keep the valid problems and corresponding
        # splines and horizons
        start_config.gather_across_batch_dim(valid_idxs)
        goal_config.gather_across_batch_dim(valid_idxs)
        horizons_n1 = tf.gather(horizons_n1, valid_idxs)
        self.spline_trajectory.gather_across_batch_dim(valid_idxs)

        return start_config, goal_config, horizons_n1

    def _lqr(self, start_config):
        # Update the shape of the cost function
        # as the batch dim of spline may have changed
        self.lqr_solver.cost.update_shape()
        lqr_res = self.lqr_solver.lqr(start_config, self.spline_trajectory,
                                      verbose=False)
        return lqr_res['trajectory_opt'], lqr_res['K_array_opt'], lqr_res['k_array_opt']

    def _init_pipeline(self):
        p = self.params
        self.spline_trajectory = p.spline_params.spline(dt=p.system_dynamics_params.dt,
                                                        n=p.waypoint_params.n,
                                                        k=p.planning_horizon,
                                                        params=p.spline_params)

        self.cost_fn = p.lqr_params.cost_fn(trajectory_ref=self.spline_trajectory,
                                            system=self.system_dynamics,
                                            params=p.lqr_params)
        self.lqr_solver = LQRSolver(T=p.planning_horizon - 1,
                                    dynamics=self.system_dynamics,
                                    cost=self.cost_fn)

    def _load_control_pipeline(self, params=None):
        # Initialize a dictionary with keys corresponding to
        # instance variables of the control pipeline and
        # values corresponding to empty lists
        pipeline_data = self.helper.empty_data_dictionary()

        for v0, expected_filename in zip(self.start_velocities, self.pipeline_files):
            filename = self._data_file_name(v0=v0)
            assert(filename == expected_filename)
            data_bin = self.helper.load_and_process_data(filename)
            self.helper.append_data_bin_to_pipeline_data(pipeline_data, data_bin)
        self._set_instance_variables(pipeline_data)

    def _set_instance_variables(self, data):
        """Set the control pipelines instance variables from
        a data dictionary."""
        self.start_configs = data['start_configs']
        self.waypt_configs = data['waypt_configs']
        self.start_speeds = data['start_speeds']
        self.spline_trajectories = data['spline_trajectories']
        self.horizons = data['horizons']
        self.lqr_trajectories = data['lqr_trajectories']
        self.K_arrays = data['K_arrays']
        self.k_arrays = data['k_arrays']

        # Initialize variable tensors for waypoints and trajectories in world coordinates
        dt = self.params.system_dynamics_params.dt
        self.waypt_configs_world = [SystemConfig(
            dt=dt, n=config.n, k=1, variable=True) for config in data['start_configs']]
        self.trajectories_world = [Trajectory(
            dt=dt, n=config.n, k=self.params.planning_horizon, variable=True) for config in
            data['start_configs']]

        if self.params.verbose:
            N = self.params.waypoint_params.n
            for v0, start_config in zip(self.start_velocities, self.start_configs):
                print('Velocity: {:.3f}, {:.3f}% of goals kept({:d}).'.format(v0,
                                                                              100.*start_config.n/N,
                                                                              start_config.n))

    def _rebin_data_by_initial_velocity(self, data):
        """Take incorrecly binned data and rebins
        it according to the dynamically feasible initial
        velocity of the robot."""
        pipeline_data = self.helper.empty_data_dictionary()
        # This data has been incorrectly binned and thus collapsed into one
        # bin. Extract this singular bin for rebinning.
        data = self.helper.extract_data_bin(data, idx=0)
        bin_idxs = self._compute_bin_idx_for_start_velocities(data['start_speeds'])

        for i in range(len(self.start_velocities)):
            idxs = tf.where(tf.equal(bin_idxs, i))[:, 0]
            data_bin = self.helper.empty_data_dictionary()
            data_bin['start_configs'] = SystemConfig.gather_across_batch_dim_and_create(data['start_configs'], idxs)
            data_bin['waypt_configs'] = SystemConfig.gather_across_batch_dim_and_create(data['waypt_configs'], idxs)
            data_bin['start_speeds'] = tf.gather(data['start_speeds'], idxs, axis=0)
            data_bin['spline_trajectories'] = Trajectory.gather_across_batch_dim_and_create(data['spline_trajectories'], idxs)
            data_bin['horizons'] = tf.gather(data['horizons'], idxs, axis=0)
            data_bin['lqr_trajectories'] = Trajectory.gather_across_batch_dim_and_create(data['lqr_trajectories'], idxs)
            data_bin['K_arrays'] = tf.gather(data['K_arrays'], idxs, axis=0)
            data_bin['k_arrays'] = tf.gather(data['k_arrays'], idxs, axis=0)
            self.helper.append_data_bin_to_pipeline_data(pipeline_data, data_bin)

        return pipeline_data

    def _compute_bin_idx_for_start_velocities(self, start_speeds_n1):
        """Computes the closest starting velocity bin to each speed
        in start_speeds."""
        diff = tf.abs(self.start_velocities - start_speeds_n1)
        bin_idxs = tf.argmin(diff, axis=1)
        return bin_idxs

    def valid_file_names(self, file_format='.pkl'):
        filenames = []
        for v0 in self.start_velocities:
            filenames.append(self._data_file_name(v0=v0, file_format=file_format))
        return filenames

    def _save_incorrectly_binned_data(self, data):
        data_to_save = self.helper.prepare_data_for_saving(data, idx=0)
        filename = self._data_file_name(incorrectly_binned=True)
        self.save_control_pipeline(data_to_save, filename)

    def _load_incorrectly_binned_data(self):
        filename = self._data_file_name(incorrectly_binned=True)
        return self._load_and_process_data(filename)

    def _incorrectly_binned_data_exists(self):
        filename = self._data_file_name(incorrectly_binned=True)
        return os.path.isfile(filename)

    def _data_file_name(self, file_format='.pkl', v0=None, incorrectly_binned=True):
        """Returns the unique file name given either a starting velocity
        or incorrectly binned=True."""

        # One of these must be True
        assert(v0 is not None or incorrectly_binned)

        p = self.params
        base_dir = os.path.join(p.dir, 'control_pipeline_v0')
        base_dir = os.path.join(base_dir, 'planning_horizon_{:d}_dt_{:.2f}'.format(
            p.planning_horizon, p.system_dynamics_params.dt))

        utils.mkdir_if_missing(base_dir)
        filename = 'n_{:d}'.format(p.waypoint_params.n)
        filename += '_theta_bins_{:d}'.format(p.waypoint_params.num_theta_bins)
        filename += '_bound_min_{:.2f}_{:.2f}_{:.2f}'.format(
            *p.waypoint_params.bound_min)
        filename += '_bound_max_{:.2f}_{:.2f}_{:.2f}'.format(
            *p.waypoint_params.bound_max)

        if v0 is not None:
            filename += '_velocity_{:.3f}{:s}'.format(v0, file_format)
        elif incorrectly_binned:
            filename += '_incorrectly_binned{:s}'.format(file_format)
        else:
            assert(False)
        filename = os.path.join(base_dir, filename)
        return filename

    def _sample_egocentric_waypoints(self, vf=0.):
        """ Uniformly samples an egocentric waypoint grid
        over which to plan trajectories."""
        p = self.params.waypoint_params

        self.waypoint_grid = p.grid(p)
        waypoints_egocentric = self.waypoint_grid.sample_egocentric_waypoints(
            vf=vf)
        waypoints_egocentric = self._ensure_waypoints_valid(
            waypoints_egocentric)
        wx_n11, wy_n11, wtheta_n11, wv_n11, ww_n11 = waypoints_egocentric
        waypt_pos_n12 = np.concatenate([wx_n11, wy_n11], axis=2)
        waypoint_egocentric_config = SystemConfig(dt=self.params.dt, n=self.waypoint_grid.n, k=1,
                                                  position_nk2=waypt_pos_n12, speed_nk1=wv_n11,
                                                  heading_nk1=wtheta_n11, angular_speed_nk1=ww_n11,
                                                  variable=True)
        return waypoint_egocentric_config

    def _ensure_waypoints_valid(self, waypoints_egocentric):
        """Ensure that a unique spline exists between start_x=0.0, start_y=0.0
        goal_x, goal_y, goal_theta. If a unique spline does not exist
        wx_n11, wy_n11, wt_n11 are modified such that one exists."""
        p = self.params
        wx_n11, wy_n11, wtheta_n11, wv_n11, ww_n11 = waypoints_egocentric
        wx_n11, wy_n11, wtheta_n11 = p.spline_params.spline.ensure_goals_valid(
            0.0, 0.0, wx_n11, wy_n11, wtheta_n11, epsilon=p.spline_params.epsilon)
        return [wx_n11, wy_n11, wtheta_n11, wv_n11, ww_n11]
