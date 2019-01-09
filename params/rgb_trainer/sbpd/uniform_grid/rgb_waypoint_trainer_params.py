from params.rgb_trainer.sbpd.uniform_grid.rgb_trainer_params import create_params as create_rgb_trainer_params
from dotmap import DotMap


def create_params():
    p = create_rgb_trainer_params()

    # Change the number of inputs to the model
    p.model.num_outputs = 3  # (x, y ,theta)

    # Change the learning rate and num_samples
    p.trainer.lr = 1e-4
    p.trainer.num_samples = int(50e3)

    # Change the checkpoint
    p.trainer.ckpt_path = '/home/vtolani/Documents/Projects/visual_mpc/logs/sbpd/rgb/nn_waypoint/train_full_episode_50k/session_2018-12-12_13-34-16/checkpoints/ckpt-20'

    # Seed for selecting the test scenarios and the number of such scenarios
    p.test.seed = 10
    p.test.number_tests = 100

    # Parameters for the metric curves
    p.test.metric_curves = DotMap(start_ckpt=1,
                                  end_ckpt=10,
                                  start_seed=1,
                                  end_seed=10,
                                  plot_curves=True
                                  )

    return p
