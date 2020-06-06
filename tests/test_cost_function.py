import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
tf.enable_eager_execution()
from obstacles.sbpd_map import SBPDMap
from objectives.obstacle_avoidance import ObstacleAvoidance
from objectives.goal_distance import GoalDistance
from objectives.angle_distance import AngleDistance
from objectives.objective_function import ObjectiveFunction
from trajectory.trajectory import Trajectory
from systems.dubins_v3 import DubinsV3
# Testing with test_dynamics
from tests.test_dynamics import create_system_dynamics_params

# Testing with test_lqr
from tests.test_lqr import create_params_2
from costs.quad_cost_with_wrapping import QuadraticRegulatorRef
from optCtrl.lqr import LQRSolver
from systems.dubins_v1 import DubinsV1

# Testing with test_splines
from trajectory.trajectory import SystemConfig
from trajectory.spline.spline_3rd_order import Spline3rdOrder

from utils.fmm_map import FmmMap
from dotmap import DotMap


def create_renderer_params():
    from params.renderer_params import get_traversible_dir, get_sbpd_data_dir
    p = DotMap()
    p.dataset_name = 'sbpd'
    p.building_name = 'area3'
    p.flip = False

    p.camera_params = DotMap(modalities=['occupancy_grid'],  # occupancy_grid, rgb, or depth
                             width=64,
                             height=64)

    # The robot is modeled as a solid cylinder
    # of height, 'height', with radius, 'radius',
    # base at height 'base' above the ground
    # The robot has a camera at height
    # 'sensor_height' pointing at 
    # camera_elevation_degree degrees vertically
    # from the horizontal plane.
    p.robot_params = DotMap(radius=18,
                            base=10,
                            height=100,
                            sensor_height=80,
                            camera_elevation_degree=-45,  # camera tilt
                            delta_theta=1.0)

    # Traversible dir
    p.traversible_dir = get_traversible_dir()

    # SBPD Data Directory
    p.sbpd_data_dir = get_sbpd_data_dir()

    return p


def create_params():
    p = DotMap()
    # Obstacle avoidance parameters
    p.avoid_obstacle_objective = DotMap(obstacle_margin0=0.3,
                                        obstacle_margin1=.5,
                                        power=2,
                                        obstacle_cost=25.0)
    # Angle Distance parameters
    p.goal_angle_objective = DotMap(power=1,
                                    angle_cost=25.0)
    # Goal Distance parameters
    p.goal_distance_objective = DotMap(power=2,
                                       goal_cost=25.0,
                                       goal_margin=0.0)

    p.objective_fn_params = DotMap(obj_type='mean')
    p.obstacle_map_params = DotMap(obstacle_map=SBPDMap,
                                   map_origin_2=[0, 0],
                                   sampling_thres=2,
                                   plotting_grid_steps=100)
    p.obstacle_map_params.renderer_params = create_renderer_params()
    return p


def test_cost_function(plot=False):
    """
    Creating objective points maually, plotting them in the ObjectiveFunction
    class, and then asserting that combined, their sum adds up to the same
    objective cost as the sum of the individual trajectories
    """
    # Create parameters
    p = create_params()

    obstacle_map = SBPDMap(p.obstacle_map_params)
    obstacle_occupancy_grid = obstacle_map.create_occupancy_grid_for_map()
    map_size_2 = obstacle_occupancy_grid.shape[::-1]

    # Define a goal position and compute the corresponding fmm map
    goal_pos_n2 = np.array([[20., 16.5], [8., 16.5]])
    fmm_map = FmmMap.create_fmm_map_based_on_goal_position(goal_positions_n2=goal_pos_n2,
                                                           map_size_2=map_size_2,
                                                           dx=0.05,
                                                           map_origin_2=[0., 0.],
                                                           mask_grid_mn=obstacle_occupancy_grid)
    # Define the cost function
    objective_function = ObjectiveFunction(p.objective_fn_params)
    objective_function.add_objective(ObstacleAvoidance(params=p.avoid_obstacle_objective, obstacle_map=obstacle_map))
    objective_function.add_objective(GoalDistance(params=p.goal_distance_objective, fmm_map=fmm_map))
    objective_function.add_objective(AngleDistance(params=p.goal_angle_objective, fmm_map=fmm_map))
    
    # Define each objective separately
    objective1 = ObstacleAvoidance(params=p.avoid_obstacle_objective, obstacle_map=obstacle_map)
    objective2 = GoalDistance(params=p.goal_distance_objective, fmm_map=fmm_map)
    objective3 = AngleDistance(params=p.goal_angle_objective, fmm_map=fmm_map)
    
    # Define a set of positions and evaluate objective
    pos_nk2 = tf.constant([[[8., 12.5], [8., 16.], [18., 16.5]]], dtype=tf.float32)
    heading_nk2 = tf.constant([[[np.pi/2.0], [0.1], [0.1]]], dtype=tf.float32)
    trajectory = Trajectory(dt=0.1, n=1, k=3, position_nk2=pos_nk2, heading_nk1=heading_nk2)

    #Testing with splines
    np.random.seed(seed=1)
    n = 5    # Batch size (unused now)
    dt = .01 # delta-t: time intervals
    k = 100  # Number of time-steps (should be 1/dt)

    # States represents each individual tangent point that the spline will pass through
    # states[0] = initial state, and states[len(states) - 1] = terminal state
    states = [
            (8, 12, np.pi/2.0, 0.5),    # Start State (x, y, theta, vel)
            (15, 14.5, np.pi/2.0, 0.8), # Middle State (x, y, theta, vel)
            (10, 15, -np.pi/2.0, 1),    # Middle State (x, y, theta, vel)
            (18, 16.5, np.pi/2.0, 0.2)  # Goal State (x, y, theta, vel)
        ]

    p = DotMap(spline_params=DotMap(epsilon=1e-5))
    ts_nk = tf.tile(tf.linspace(0., dt*k, k)[None], [n, 1])
    splines = []
    prev_config = None
    next_config = None
    # Generate all two-point splines from the states
    for s in states:
        spline_traj = Spline3rdOrder(dt=dt, k=k, n=n, params=p.spline_params)
        # Keep track of old trajectory
        if(next_config is not None):
            # Rewrite the previous state config with the 'old' next one
            prev_config = next_config
        # Generate position
        s_posx_nk1 = tf.ones((n, 1, 1), dtype=tf.float32) * s[0] # X position matrix
        s_posy_nk1 = tf.ones((n, 1, 1), dtype=tf.float32) * s[1] # Y position matrix
        s_pos_nk2 = tf.concat([s_posx_nk1, s_posy_nk1], axis=2)  # combined matrix of (X,Y)
        # Generate speed and heading
        heading_nk1 = tf.ones((n, 1, 1), dtype=tf.float32) * s[2]# Theta angle matrix
        speed_nk1 = tf.ones((n, 1, 1), dtype=tf.float32) * s[3]  # Speed matrix
        next_config = SystemConfig(dt, n, 1, 
                                    position_nk2=s_pos_nk2, 
                                    speed_nk1=speed_nk1, 
                                    heading_nk1=heading_nk1, 
                                    variable=False
                                  )
        # Append to the trajectory if a new trajectory can be constructed 
        # Note that any spline needs a 'previous' and 'next' state
        if(prev_config is not None):
            spline_traj.fit(prev_config, next_config, factors=None)
            spline_traj.eval_spline(ts_nk, calculate_speeds=True)
            splines.append(spline_traj)
    
    # Loop through all calculated splines to combine them together into a singular one
    final_spline = None
    for i, s in enumerate(splines):
        if(final_spline is not None):
            final_spline.append_along_time_axis(s)
        if(i == 0):
            # For first spline
            final_spline = s

    fig = plt.figure()
    fig, ax = plt.subplots(4,1, figsize=(5,15), squeeze=False)
    final_spline.render(ax, freq=4, plot_heading=True, plot_velocity=True, label_start_and_end=True)
    # trajectory.render(ax, freq=1, plot_heading=True, plot_velocity=True, label_start_and_end=True)
    fig.savefig('./tests/cost/trajectory.png', bbox_inches='tight', pad_inches=0)

    # Compute the objective function
    values_by_objective = objective_function.evaluate_function_by_objective(trajectory)
    overall_objective = objective_function.evaluate_function(trajectory)
    
    # Expected objective values
    expected_objective1 = objective1.evaluate_objective(trajectory)
    expected_objective2 = objective2.evaluate_objective(trajectory)
    expected_objective3 = objective3.evaluate_objective(trajectory)
    # expected_overall_objective = tf.reduce_mean(expected_objective1 + expected_objective2 + expected_objective3, axis=1)
    expected_overall_objective = np.mean(expected_objective1 + expected_objective2 + expected_objective3, axis=1)
    assert len(values_by_objective) == 3
    assert values_by_objective[0][1].shape == (1, 3)
    assert overall_objective.shape == (1,)
    # assert np.allclose(overall_objective.numpy(), expected_overall_objective.numpy(), atol=1e-2)
    assert np.allclose(overall_objective.numpy(), expected_overall_objective, atol=1e-2)

    # Optionally visualize the traversable and the points on which
    # we compute the objective function
    if plot:
        fig = plt.figure()
        ax = fig.add_subplot(1,1,1)
        obstacle_map.render(ax)
        ax.plot(pos_nk2[0, :, 0].numpy(), pos_nk2[0, :, 1].numpy(), 'r.')
        ax.plot(goal_pos_n2[0, 0], goal_pos_n2[0, 1], 'k*')
        fig.savefig('./tests/cost/test_cost_function.png', bbox_inches='tight', pad_inches=0)


if __name__ == '__main__':
    test_cost_function(plot=True)
