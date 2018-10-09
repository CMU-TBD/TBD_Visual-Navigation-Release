import tensorflow as tf
import tensorflow.contrib.eager as tfe
import matplotlib.pyplot as plt


class Trajectory(object):
    """
    The base class for the trajectory of a ground vehicle.
    n is the batch size and k is the # of time steps in the trajectory.
    """

    def __init__(self, dt, n, k, position_nk2=None, speed_nk1=None, acceleration_nk1=None, heading_nk1=None,
                 angular_speed_nk1=None, angular_acceleration_nk1=None,
                 dtype=tf.float32, variable=True):

        # Check dimensions now to make your life easier later
        if position_nk2 is not None:
            assert(n == position_nk2.shape[0])
            assert(k == position_nk2.shape[1])

        # Discretization step
        self.dt = dt

        # Number of timesteps
        self.k = k

        # Batch Size
        self.n = n

        self.vars = []
        if variable:
            # Translational trajectories
            self._position_nk2 = tfe.Variable(tf.zeros([n, k, 2], dtype=dtype) if position_nk2 is None
                                              else position_nk2)
            self._speed_nk1 = tfe.Variable(tf.zeros([n, k, 1], dtype=dtype) if speed_nk1 is None
                                           else tf.constant(speed_nk1, dtype=dtype))
            self._acceleration_nk1 = tfe.Variable(tf.zeros([n, k, 1], dtype=dtype) if acceleration_nk1 is None
                                                  else tf.constant(acceleration_nk1, dtype=dtype))

            # Rotational trajectories
            self._heading_nk1 = tfe.Variable(tf.zeros([n, k, 1], dtype=dtype) if heading_nk1 is None
                                             else tf.constant(heading_nk1, dtype=dtype))
            self._angular_speed_nk1 = tfe.Variable(tf.zeros([n, k, 1], dtype=dtype) if angular_speed_nk1 is None
                                                   else tf.constant(angular_speed_nk1, dtype=dtype))
            self._angular_acceleration_nk1 = tfe.Variable(
                tf.zeros([n, k, 1], dtype=dtype) if angular_acceleration_nk1 is None
                else tf.constant(angular_acceleration_nk1, dtype=dtype))

            self.vars = [self._position_nk2, self._speed_nk1,
                         self._acceleration_nk1, self._heading_nk1,
                         self._angular_speed_nk1, self._angular_speed_nk1]
        else:
            # Translational trajectories
            self._position_nk2 = tf.zeros([n, k, 2], dtype=dtype) if position_nk2 is None \
                                              else tf.constant(position_nk2, dtype=dtype)
            self._speed_nk1 = tf.zeros([n, k, 1], dtype=dtype) if speed_nk1 is None \
                                              else tf.constant(speed_nk1, dtype=dtype)
            self._acceleration_nk1 = tf.zeros([n, k, 1], dtype=dtype) if acceleration_nk1 is None \
                                                  else tf.constant(acceleration_nk1, dtype=dtype)

            # Rotational trajectories
            self._heading_nk1 = tf.zeros([n, k, 1], dtype=dtype) if heading_nk1 is None \
                                         else tf.constant(heading_nk1, dtype=dtype)
            self._angular_speed_nk1 = tf.zeros([n, k, 1], dtype=dtype) if angular_speed_nk1 is None \
                                                   else tf.constant(angular_speed_nk1, dtype=dtype)
            self._angular_acceleration_nk1 = tf.zeros([n, k, 1], dtype=dtype) if angular_acceleration_nk1 is None \
                else tf.constant(angular_acceleration_nk1, dtype=dtype)

    @classmethod
    def init_from_numpy_repr(cls, dt, n, k, position_nk2, speed_nk1,
                             acceleration_nk1, heading_nk1, angular_speed_nk1,
                             angular_acceleration_nk1):
        """Utility function to initialize a trajectory object from its numpy
        representation. Useful for loading pickled trajectories"""
        return cls(dt=dt, n=n, k=k, position_nk2=position_nk2,
                   speed_nk1=speed_nk1, acceleration_nk1=acceleration_nk1,
                   heading_nk1=heading_nk1,
                   angular_speed_nk1=angular_speed_nk1,
                   angular_acceleration_nk1=angular_acceleration_nk1,
                   variable=False)

    def assign_from_trajectory_batch_idx(self, trajectory, batch_idx):
        self.assign_trajectory_from_tensors(position_nk2=trajectory.position_nk2()[batch_idx:batch_idx+1],
                                            speed_nk1=trajectory.speed_nk1()[batch_idx:batch_idx+1],
                                            acceleration_nk1=trajectory.acceleration_nk1()[batch_idx:batch_idx+1],
                                            heading_nk1=trajectory.heading_nk1()[batch_idx:batch_idx+1],
                                            angular_speed_nk1=trajectory.angular_speed_nk1()[batch_idx:batch_idx+1],
                                            angular_acceleration_nk1=trajectory.angular_acceleration_nk1()[batch_idx:batch_idx+1])

    def assign_trajectory_from_tensors(self, position_nk2, speed_nk1, acceleration_nk1,
                                       heading_nk1, angular_speed_nk1, angular_acceleration_nk1):
        tf.assign(self.position_nk2(), position_nk2)
        tf.assign(self.speed_nk1(), speed_nk1)
        tf.assign(self.acceleration_nk1(), acceleration_nk1)
        tf.assign(self.heading_nk1(), heading_nk1)
        tf.assign(self.angular_speed_nk1(), angular_speed_nk1)
        tf.assign(self.angular_acceleration_nk1(), angular_acceleration_nk1)

    def to_numpy_repr(self):
        """Utility function to return a representation of the trajectory using
        numpy arrays. Useful for pickling trajectories."""
        numpy_dict = {'dt': self.dt, 'n': self.n, 'k': self.k,
                      'position_nk2': self.position_nk2().numpy(),
                      'speed_nk1': self.speed_nk1().numpy(),
                      'acceleration_nk1': self.acceleration_nk1().numpy(),
                      'heading_nk1': self.heading_nk1().numpy(),
                      'angular_speed_nk1': self.angular_speed_nk1().numpy(),
                      'angular_acceleration_nk1':
                      self.angular_acceleration_nk1().numpy()}
        return numpy_dict

    @property
    def trainable_variables(self):
        return self.vars

    @property
    def shape(self):
        return '(%d, %d)'.format(self.n, self.k)

    def position_nk2(self):
        return self._position_nk2

    def speed_nk1(self):
        return self._speed_nk1

    def acceleration_nk1(self):
        return self._acceleration_nk1

    def heading_nk1(self):
        return self._heading_nk1

    def angular_speed_nk1(self):
        return self._angular_speed_nk1

    def angular_acceleration_nk1(self):
        return self._angular_acceleration_nk1

    def position_and_heading_nk3(self):
        return tf.concat([self.position_nk2(), self.heading_nk1()], axis=2)

    def speed_and_angular_speed(self):
        return tf.concat([self.speed_nk1(), self.angular_speed_nk1()], axis=2)

    def position_heading_speed_and_angular_speed_nk5(self):
        return tf.concat([self.position_and_heading_nk3(),
                          self.speed_and_angular_speed()], axis=2)

    def append_along_time_axis(self, trajectory):
        """ Utility function to concatenate trajectory
        over time. Useful for assembling an entire
        trajectory from multiple sub-trajectories. """
        self._position_nk2 = tf.concat([self.position_nk2(),
                                        trajectory.position_nk2()],
                                       axis=1)
        self._speed_nk1 = tf.concat([self.speed_nk1(), trajectory.speed_nk1()],
                                    axis=1)
        self._acceleration_nk1 = tf.concat([self.acceleration_nk1(),
                                            trajectory.acceleration_nk1()],
                                           axis=1)
        self._heading_nk1 = tf.concat([self.heading_nk1(),
                                       trajectory.heading_nk1()], axis=1)
        self._angular_speed_nk1 = tf.concat([self.angular_speed_nk1(),
                                             trajectory.angular_speed_nk1()],
                                            axis=1)
        self._angular_acceleration_nk1 = tf.concat([self.angular_acceleration_nk1(),
                                                    trajectory.angular_acceleration_nk1()],
                                                   axis=1)
        self.k = self.k + trajectory.k

    def clip_along_time_axis(self, horizon):
        """ Utility function for clipping a trajectory along
        the time axis. Useful for clipping a trajectory within
        a specified horizon."""
        if self.k <= horizon:
            return

        self._position_nk2 = self._position_nk2[:, :horizon]
        self._speed_nk1 = self._speed_nk1[:, :horizon]
        self._acceleration_nk1 = self._acceleration_nk1[:, :horizon]
        self._heading_nk1 = self._heading_nk1[:, :horizon]
        self._angular_speed_nk1 = self._angular_speed_nk1[:, :horizon]
        self._angular_acceleration_nk1 = self._angular_acceleration_nk1[:, :horizon]
        self.k = horizon

    def render(self, ax, batch_idx=0, freq=4):
        xs = self._position_nk2[batch_idx, :, 0]
        ys = self._position_nk2[batch_idx, :, 1]
        thetas = self._heading_nk1[batch_idx]
        ax.plot(xs, ys, 'r-')
        ax.quiver(xs[::freq], ys[::freq],
                  tf.cos(thetas[::freq]), tf.sin(thetas[::freq]))


class State(Trajectory):
    """
    A class representing robot state using a trajectory of
    time duration = 1 step.
    """

    def __init__(self, dt, n, k, position_nk2=None, speed_nk1=None, acceleration_nk1=None, heading_nk1=None,
                 angular_speed_nk1=None, angular_acceleration_nk1=None,
                 dtype=tf.float32, variable=True):
        assert(k == 1)
        super().__init__(dt, n, k, position_nk2, speed_nk1, acceleration_nk1,
                         heading_nk1, angular_speed_nk1,
                         angular_acceleration_nk1, dtype=tf.float32,
                         variable=variable)

    def assign_from_broadcasted_batch(self, state, n):
        """ Assigns a states variables by broadcasting a given state to
        batch size n """
        k = state.k
        self.assign_state_from_tensors(position_nk2=tf.broadcast_to(state.position_nk2(), (n, k, 2)),
                                       speed_nk1=tf.broadcast_to(state.speed_nk1(), (n, k, 1)),
                                       acceleration_nk1=tf.broadcast_to(state.acceleration_nk1(), (n, k, 1)),
                                       heading_nk1=tf.broadcast_to(state.heading_nk1(), (n, k, 1)),
                                       angular_speed_nk1=tf.broadcast_to(state.angular_speed_nk1(), (n, k, 1)),
                                       angular_acceleration_nk1=tf.broadcast_to(state.angular_acceleration_nk1(), (n, k, 1)))

    @classmethod
    def init_state_from_trajectory_time_index(cls, trajectory, t):
        """ A utility method to initialize a state object
        from a particular timestep of a given trajectory object"""
        position_nk2 = trajectory.position_nk2()
        speed_nk1 = trajectory.speed_nk1()
        acceleration_nk1 = trajectory.acceleration_nk1()
        heading_nk1 = trajectory.heading_nk1()
        angular_speed_nk1 = trajectory.angular_speed_nk1()
        angular_acceleration_nk1 = trajectory.angular_acceleration_nk1()

        if t == -1:
            return cls(dt=trajectory.dt, n=trajectory.n, k=1,
                       position_nk2=position_nk2[:, t:],
                       speed_nk1=speed_nk1[:, t:],
                       acceleration_nk1=acceleration_nk1[:, t:],
                       heading_nk1=heading_nk1[:, t:],
                       angular_speed_nk1=angular_speed_nk1[:, t:],
                       angular_acceleration_nk1=angular_acceleration_nk1[:, t:])

        return cls(dt=trajectory.dt, n=trajectory.n, k=1,
                   position_nk2=position_nk2[:, t:t+1],
                   speed_nk1=speed_nk1[:, t:t+1],
                   acceleration_nk1=acceleration_nk1[:, t:t+1],
                   heading_nk1=heading_nk1[:, t:t+1],
                   angular_speed_nk1=angular_speed_nk1[:, t:t+1],
                   angular_acceleration_nk1=angular_acceleration_nk1[:, t:t+1])

    def assign_from_state_batch_idx(self, state, batch_idx):
        super().assign_from_trajectory_batch_idx(state, batch_idx)

    def assign_state_from_tensors(self, position_nk2, speed_nk1, acceleration_nk1,
                                  heading_nk1, angular_speed_nk1, angular_acceleration_nk1):
        super().assign_trajectory_from_tensors(position_nk2, speed_nk1,
                                               acceleration_nk1, heading_nk1,
                                               angular_speed_nk1, angular_acceleration_nk1)

    def render(self, ax, batch_idx=0, marker='bo'):
        pos_n12 = self.position_nk2()
        pos_2 = pos_n12[batch_idx, 0]
        ax.plot(pos_2[0], pos_2[1], marker)

    def render_with_boundary(self, ax, batch_idx, marker='bo',
                             boundary_params=None):
        self.render(ax, batch_idx, marker)
        if boundary_params is not None:
            if boundary_params['norm'] == 2:
                center = self.position_nk2()[batch_idx, 0].numpy()
                radius = boundary_params['cutoff']
                c = plt.Circle(center, radius, color=boundary_params['color'])
                ax.add_artist(c)
            else:
                assert(False)
