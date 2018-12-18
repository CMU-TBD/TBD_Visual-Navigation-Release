from training_utils.visual_navigation_trainer import VisualNavigationTrainer
from models.visual_navigation.rgb.rgb_control_model import RGBControlModel
import os


class RGBControlTrainer(VisualNavigationTrainer):
    """
    Create a trainer that regress on the optimal control using
    first person view RGB images.
    """
    simulator_name = 'RGB_NN_Control_Simulator'

    def create_model(self, params=None):
        self.model = RGBControlModel(self.p)

    def _modify_planner_params(self, p):
        """
        Modifies a DotMap parameter object
        with parameters for a NNWaypointPlanner
        """
        from planners.nn_waypoint_planner import NNWaypointPlanner

        p.planner_params.planner = NNWaypointPlanner
        p.planner_params.model = self.model

    def _summary_dir(self):
        """
        Returns the directory name for tensorboard
        summaries
        """
        return os.path.join(self.p.session_dir, 'summaries', 'nn_control')


if __name__ == '__main__':
    RGBControlTrainer().run()