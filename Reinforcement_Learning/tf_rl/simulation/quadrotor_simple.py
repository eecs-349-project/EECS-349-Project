import numpy as np
from random import randint


class Quadrotor(object):

    controls = np.array([[-1, -1], [-1, 0], [-1, 1],
                         [0, -1], [0, 0], [0, 1],
                         [1, -1], [1, 0], [1, 1]])

    # controls = np.array([[-1, 0], [0, -1],
                         # [0, 1], [1, 0]])

    num_of_actions = 9

    def __init__(self):
        self.state = np.array([randint(-10, 10),
                               randint(-10, 10),
                               randint(-10, 10),
                               randint(-10, 10)])

        self.control_input = np.array([0, 0])

    def observe(self):
        """Returns an observation."""
        return self.state

    def perform_action(self, action):
        """Expects action to be in range [-1, 1]"""
        self.control_input = Quadrotor.controls[action]

    def step(self, dt):
        """Advance simulation by dt seconds"""
        self.state[:2] += self.control_input
        self.state[2:] += Quadrotor.controls[randint(0,
                                                     Quadrotor.num_of_actions-1)]

    def cost(self):
        return np.fabs(self.state[0]-self.state[2]) +\
                              np.fabs(self.state[1]-self.state[3])

    def is_over(self):
        return np.fabs(self.state[0] - self.state[2]) +\
                      np.fabs(self.state[1]-self.state[3]) < 1e-2
