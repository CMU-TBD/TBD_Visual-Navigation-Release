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
    # Create parameters
    p = create_params()

    obstacle_map = SBPDMap(p.obstacle_map_params)
    obstacle_occupancy_grid = obstacle_map.create_occupancy_grid_for_map()
    map_size_2 = obstacle_occupancy_grid.shape[::-1]

    # Define a goal position and compute the corresponding fmm map
    goal_pos_n2 = np.array([[20., 16.5]])
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
    n = 5
    dt = .01
    k = 100

    target_state = (18, 16.5, np.pi/2.0)   # Goal State (x, y, theta)
    v0 = np.random.uniform(0., 0.5, 1)[0]  # Initial speed
    vf = 0.

    # Initial SystemConfig is [0, 0, 0, v0, 0]
    start_speed_nk1 = tf.ones((n, 1, 1), dtype=tf.float32)*v0
    start_heading_nk1 = tf.ones((n, 1, 1), dtype=tf.float32)*np.pi/2.
    # start_heading_nk1 = tf.constant([[[np.pi/2.]]], dtype=tf.float32)

    print(target_state)
    print(v0)
    
    goal_posx_nk1 = tf.ones((n, 1, 1), dtype=tf.float32) * target_state[0]
    goal_posy_nk1 = tf.ones((n, 1, 1), dtype=tf.float32) * target_state[1]
    goal_pos_nk2 = tf.concat([goal_posx_nk1, goal_posy_nk1], axis=2)
    goal_heading_nk1 = tf.ones((n, 1, 1), dtype=tf.float32) * target_state[2]
    goal_speed_nk1 = tf.ones((n, 1, 1), dtype=tf.float32) * vf

    start_config = SystemConfig(dt, n, 1, speed_nk1=start_speed_nk1, heading_nk1=start_heading_nk1, variable=False)
    goal_config = SystemConfig(dt, n, 1, position_nk2=goal_pos_nk2,
                               speed_nk1=goal_speed_nk1, heading_nk1=goal_heading_nk1,
                               variable=True)

    start_nk5 = start_config.position_heading_speed_and_angular_speed_nk5()
    start_n5 = start_nk5[:, 0]

    goal_nk5 = goal_config.position_heading_speed_and_angular_speed_nk5()
    goal_n5 = goal_nk5[:, 0]
    p = DotMap(spline_params=DotMap(epsilon=1e-5))
    ts_nk = tf.tile(tf.linspace(0., dt*k, k)[None], [n, 1])
    spline_traj = Spline3rdOrder(dt=dt, k=k, n=n, params=p.spline_params)
    spline_traj.fit(start_config, goal_config, factors=None)
    spline_traj.eval_spline(ts_nk, calculate_speeds=True)

    pos_nk3 = spline_traj.position_and_heading_nk3()
    v_nk1 = spline_traj.speed_nk1()

    #Trying with the lqr tests
    # p = create_params_2()
    # np.random.seed(seed=1)
    # n, k = 2, 50
    # dt = p.dt

    # db = DubinsV1(dt, params=p.system_dynamics_params.simulation_params)
    # x_dim, u_dim = db._x_dim, db._u_dim

    # x_n13 = tf.constant(np.zeros((n, 1, x_dim)), dtype=tf.float32)
    # v_1k, w_1k = np.ones((k-1, 1))*.1, np.linspace(.5, .3, k-1)[:, None]

    # u_1k2 = tf.constant(np.concatenate([v_1k, w_1k], axis=1)[None],
    #                     dtype=tf.float32)
    # u_nk2 = tf.zeros((n, k-1, 2), dtype=tf.float32)+u_1k2
    # trajectory_ref = db.simulate_T(x_n13, u_nk2, T=k)

    # x_nk3, u_nk2 = db.parse_trajectory(trajectory_ref)

    # # stack two different reference trajectories together
    # # to verify that batched LQR works across the batch dim
    # goal_x, goal_y = 18, 16.0
    # goal = np.array([goal_x, goal_y, 0.], dtype=np.float32)
    # x_ref_nk3 = tf.constant(np.tile(goal, (1, k, 1)))
    # u_ref_nk2 = tf.constant(np.zeros((1, k, u_dim), dtype=np.float32))
    # x_nk3 = tf.concat([x_ref_nk3, x_nk3[0:1]], axis=0)
    # u_nk2 = tf.concat([u_ref_nk2, u_nk2[0:1]], axis=0)
    # trajectory_ref = db.assemble_trajectory(x_nk3, u_nk2)

    # cost_fn = QuadraticRegulatorRef(trajectory_ref, db, p)

    # x_nk3 = tf.constant(np.zeros((n, k, x_dim), dtype=np.float32))
    # u_nk2 = tf.constant(np.zeros((n, k, u_dim), dtype=np.float32))
    # trajectory = db.assemble_trajectory(x_nk3, u_nk2)

    # lqr_solver = LQRSolver(T=k-1, dynamics=db, cost=cost_fn)

    # start_config = db.init_egocentric_robot_config(dt=dt, n=n)
    # lqr_res = lqr_solver.lqr(start_config, trajectory, verbose=False)
    # trajectory = lqr_res['trajectory_opt']

    # Trying with dynamics model:
    # dt = 0.1
    # x_dim, u_dim = 5, 2
    # n, k = 17, 12
    # ctrl = 1
    # db = DubinsV3(dt, create_system_dynamics_params())
    # state_n15 = tf.constant(np.zeros((n, 1, x_dim)), dtype=tf.float32)
    # ctrl_nk2 = tf.constant(np.ones((n, k, u_dim))*ctrl, dtype=tf.float32)
    # trajectory = db.simulate_T(state_n15, ctrl_nk2, T=k)
    # state_nk5 = trajectory.position_heading_speed_and_angular_speed_nk5()
    # trajectory = db.assemble_trajectory(state_nk5[:, :-1], theta_nk2)

    # print(trajectory.valid_horizons_n1)
    fig = plt.figure()
    fig, ax = plt.subplots(4,1, figsize=(5,15), squeeze=False)
    spline_traj.render(ax, freq=4)
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
