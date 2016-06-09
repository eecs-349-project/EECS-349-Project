import numpy as np


class Quadrotor(object):

    def __init__(self, params):
        th = np.pi*np.random.rand(1)-np.pi/2
        x = 20*np.random.rand(2)-10
        w = np.array([0])
        v = np.array([0, 0])
        self.state = np.append(np.append(th, x), np.append(w, v))

        self.control_input = np.array([0, 0])

        self.params = params

        self.g = self.params['g']
        self.m = self.params['m']
        self.J = self.params['J']
        self.kt = self.params['kt']
        self.d = self.params['d']
        self.Fmax = self.params['Fmax']
        self.Mmax = self.params['Mmax']

        self.Ms = self.params['us']*self.Mmax
        self.Fs = self.params['us']*self.Fmax
        self.N = len(self.Ms)
        self.num_of_actions = self.N*self.N

    def reset(self):
        th = np.pi*np.random.rand(1)-np.pi/2
        x = 20*np.random.rand(2)-10
        w = np.array([0])
        v = np.array([0, 0])
        self.state = np.append(th, x, w, v)
        self.control_input = np.array([0, 0])

    def dydt(self):
        """How state of the world changes
        naturally due to gravity and momentum

        Returns a vector of four values
        which are derivatives of different
        state in the internal state representation."""
        # code below is horrible, if somebody
        # wants to clean it up, I will gladly
        # accept pull request.
        state = self.state
        u = self.control_input

        th = state[0]
        dydx = np.zeros_like(state)
        dydx[:3] = state[-3:]

        # F = (u[0]+u[1])*self.kt
        # M = (u[0]-u[1])*self.kt*self.d
        M = u[0] 
        F = u[1]

        dydx[3] = -M/self.J
        dydx[4] = F*np.sin(th)/self.m
        dydx[5] = F*np.cos(th)/self.m-self.g

        return np.array(dydx)

    def observe(self):
        """Returns an observation."""
        return self.state

    def perform_action(self, action):
        """Expects action to be in range [-1, 1]"""
        m = np.floor(action/self.N)
        n = action % self.N
        self.control_input = np.array([self.Ms[m], self.Fs[n]])

    def step(self, dt):
        """Advance simulation by dt seconds"""
        dstate = self.dydt()
        self.state += dt * dstate
        th = self.state[0]
        th %= 2*np.pi
        if th > np.pi:
            th -= 2*np.pi
        # self.state[0] = th-np.round(th/(2*np.pi))*2*np.pi
        self.state[0] = th

    def cost(self):
        th = self.state[0]
        x = self.state[1]
        y = self.state[2]
        w = self.state[3]
        vx = self.state[4]
        vy = self.state[5]

        # return 0.4*th**2+x**2+y**2+0.005*(w**2+vx**2+vy**2)
        return 0.4*np.fabs(th)+np.fabs(x)+np.fabs(y)+0.02*(np.fabs(w)+np.fabs(vx)+np.fabs(vy))

    # def collect_reward(self):
        # """Reward corresponds to how high is the first joint."""

    def is_over(self):
        return np.fabs(self.state[1])+np.fabs(self.state[2]) < 1e-2
