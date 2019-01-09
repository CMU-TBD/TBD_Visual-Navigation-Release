from params.rgb_trainer.sbpd.uniform_grid.resnet50.rgb_trainer_finetune_params import create_params as create_rgb_trainer_params
from dotmap import DotMap


def create_params():
    p = create_rgb_trainer_params()

    # Change the number of inputs to the model
    p.model.num_outputs = 60  # (v, omega) for 30 timesteps

    # Change the learning rate and num_samples
    p.trainer.lr = 1e-5
    p.trainer.num_samples = int(50e3)

    # Change the checkpoint
    p.trainer.ckpt_path = '/home/ext_drive/somilb/data/sessions/varun_logs/sbpd/rgb/nn_control/uniform_grid/resnet50_v0_finetune/train_full_episode_50k/session_2019-01-04_12-14-27/checkpoints/ckpt-2'

    # Seed for selecting the test scenarios and the number of such scenarios
    p.test.seed = 10
    p.test.number_tests = 200
    
    # Parameters for the metric curves
    p.test.metric_curves = DotMap(start_ckpt=1,
                                  end_ckpt=20,
                                  start_seed=1,
                                  end_seed=10,
                                  plot_curves=True
                                  )
    return p
