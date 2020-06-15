import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
tf.enable_eager_execution()

from obstacles.sbpd_map import SBPDMap
from objectives.obstacle_avoidance import ObstacleAvoidance
from planners.planner import Planner
from simulators.simulator import Simulator
from trajectory.trajectory import Trajectory
from trajectory.trajectory import SystemConfig
from params.planner_params import create_params as create_planner_params
from params.simulator.sbpd_simulator_params import create_params as create_sim_params
from planners.sampling_planner import SamplingPlanner
from dotmap import DotMap


def create_params():
    # DotMap is essentially a fancy dictionary
    p = DotMap()
    # Obstacle avoidance parameters
    p.avoid_obstacle_objective = DotMap(obstacle_margin0=0.3,
                                        obstacle_margin1=0.5,
                                        power=2,
                                        obstacle_cost=25.0)
    p.obstacle_map_params = DotMap(obstacle_map=SBPDMap,
                                   map_origin_2=[0., 0.],
                                   sampling_thres=2,
                                   plotting_grid_steps=100)
    p.obstacle_map_params.renderer_params = create_renderer_params()

    return p

def create_renderer_params():
    """
    Used to generate the parameters for the environment, building and traversibles
    """
    from params.renderer_params import get_traversible_dir, get_sbpd_data_dir
    p = DotMap()
    p.dataset_name = 'sbpd'   # Stanford Building Parser Dataset (SBPD)
    p.building_name = 'area3' # Name of the building (change to whatever is downloaded on your system)
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

def test_planner():
    # Create parameters
    p = create_params()

    # Create an SBPD Map
    obstacle_map = SBPDMap(p.obstacle_map_params)

    # Create planner parameters
    planner_params = create_planner_params()
    sim_params = create_sim_params()
    # Define the objective
    # objective = ObstacleAvoidance(params=p.avoid_obstacle_objective,
    #                               obstacle_map=obstacle_map)
    sim = Simulator(sim_params, obstacle_map)
    planner = Planner(sim, planner_params)
    splanner = SamplingPlanner(planner, planner_params)
    # Define a set of positions and evaluate objective (shape = [1,k,2])
    # TRaversing via End, Waypts, Start
    pos_nk2 = tf.constant([[[18., 16.5], [10., 16.5], [8., 16.], [8., 12.5]]], dtype=tf.float32)
    trajectory = Trajectory(dt=0.1, n=1, k=4, position_nk2=pos_nk2)

    # Spline trajectory params
    n = 1
    dt = .1
    k = 10
 
    # Goal states and initial speeds
    goal_pos_n11 = tf.constant([[[8., 16.5]]]) # Goal position (must be 1x1x2 array)
    goal_heading_n11 = tf.constant([[[np.pi/2.]]])
    # Start states and initial speeds
    start_pos_n11 = tf.constant([[[8., 12.]]]) # Goal position (must be 1x1x2 array)
    start_heading_n11 = tf.constant([[[0.]]])
    # start_speed_nk1 = tf.ones((2, 1, 1), dtype=tf.float32) * 0.5
    # Define start and goal configurations
    # start_config = SystemConfig(dt, n, 1, speed_nk1=start_speed_nk1, variable=False)
    start_config = SystemConfig(dt, n,
                               k=1,
                               position_nk2=start_pos_n11,
                               heading_nk1=start_heading_n11,
                               variable=False)
    goal_config = SystemConfig(dt, n,
                               k=1,
                               position_nk2=goal_pos_n11,
                               heading_nk1=goal_heading_n11,
                               variable=True)
    #  waypts, horizons, trajectories_lqr, trajectories_spline, controllers = controller.plan(start_config, goal_config)
    splanner.eval_objective(start_config, goal_config)
    splanner.optimize(start_config)
    #obj_val, [waypts, horizons, trajectories_lqr, trajectories_spline, controllers] = splanner.eval_objective(start_config, goal_config)
    opt_traj = splanner.opt_traj
    # Expected objective values
    # distance_map = obstacle_map.fmm_map.fmm_distance_map.voxel_function_mn
    # Visualization
    fig = plt.figure()
    ax = fig.add_subplot(1,2,1)
    # obstacle_map.render(ax)
    # ax.plot(pos_nk2[0, :, 0].numpy(), pos_nk2[0, :, 1].numpy(), 'r.')
    # ax.plot(trajectory._position_nk2[0, :, 0],trajectory._position_nk2[0, :, 1], 'r-')
    # ax.plot(trajectories_spline._position_nk2[0, :, 0],trajectories_spline._position_nk2[0, :, 1], 'r-')
    # ax.plot(opt_traj._position_nk2[0, :, 0],opt_traj._position_nk2[0, :, 1], 'r-')
    splanner.simulator.simulator.reset()
    splanner.simulator.simulator.simulate()
    splanner.simulator.simulator.render(ax)
    # ax.plot(objective[0, 0], objective[0, 1], 'k*')
    ax.set_title('obstacle map')
    ax = fig.add_subplot(1,2,2)
    splanner.simulator.simulator._render_trajectory(ax)

    # Plotting the "distance map"
    # ax = fig.add_subplot(1,2,2)
    # ax.imshow(distance_map, origin='lower')
    # ax.set_title('distance map')

    fig.savefig('./tests/planner/test_planner.png', bbox_inches='tight', pad_inches=0)

if __name__ == '__main__':
    test_planner()
