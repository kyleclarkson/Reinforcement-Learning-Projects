import numpy as np

# https://math.stackexchange.com/questions/1287634/implementing-ornstein-uhlenbeck-in-matlab
# https://en.wikipedia.org/wiki/Ornstein%E2%80%93Uhlenbeck_process
class QUNoise():
    """
    A simulation of the Ornsetin-Uhlenbeck process.
    """

    def __init__(self, mu, sigma=0.15, theta=0.2, dt=0.01, x0=None):
        self.mu = mu
        self.sigma = sigma
        self.theta = theta
        self.dt = dt
        self.x0 = x0
        self.reset()

    def reset(self):
        self.x_prev = self.x0 if self.x0 is not None else np.zeros_like(self.mu)


    def __call__(self, *args, **kwargs):
        x = self.x_prev + self.theta * (self.mu - self.x_prev) * self.dt +\
            self.sigma * np.sqrt(self.dt) * np.random.normal(size=self.mu.shape)
        self.x_prev = x
        return  x