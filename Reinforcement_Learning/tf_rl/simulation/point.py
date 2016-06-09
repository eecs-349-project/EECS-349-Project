import numpy as np
from random import uniform 

class Point(object):

    directions = np.linspace(-1, 1, 21)

    def __init__(self):
        x = uniform(-5, 5)
        v = 0
        self.state = np.array([x, v])

        self.control_input = Point.directions[0]

        self.num_of_actions = 50

    def observe(self):
        """Returns an observation."""
        return self.state

    def perform_action(self, action):
        """Expects action to be in range [-1, 1]"""
        self.control_input = Point.directions[action]

    def step(self, dt):
        """Advance simulation by dt seconds"""
        self.state[0] += self.state[1]*dt
        self.state[1] += self.control_input*dt

    def cost(self):
        return 10*np.fabs(self.state[0])+0.1*np.fabs(self.state[1])

    # def collect_reward(self):
        # """Reward corresponds to how high is the first joint."""

    def is_over(self):
        return np.fabs(self.state[0])+np.fabs(self.state[1]) < 1e-2
