import numpy as np
import math
from math import sqrt, pi
import numpy.random as npr
import torch


class Network:
    def __init__(self, name='PFC', N=128, S=2, R=2, Rc=1, Ra=3, g=1.2, fb_type='aligned', seed=1):
        self.name = name
        npr.seed(seed)
        if self.name == 'PFC':
            # network parameters
            self.N = N  # RNN units
            self.dt = .1  # time bin (in units of tau)
            self.g = g  # initial recurrent weight scale
            self.S = S  # input
            self.R = R  # readout
            self.sig = 0.001  # initial activity noise
            self.z0 = []    # initial condition
            self.sa, self.ha, self.ua = [], [], []  # input, activity, output
            self.ws = (2 * npr.random((N, S)) - 1) / sqrt(S)  # input weights
            self.J = self.g * npr.standard_normal([N, N]) / np.sqrt(N)  # recurrent weights
            self.wr = (2 * npr.random((R, N)) - 1) / sqrt(N)  # readout weights
            self.fb_type = fb_type
            if fb_type == 'random':
                self.B = npr.standard_normal([N, R]) / sqrt(R)
            elif fb_type == 'aligned':
                self.B = self.wr.T * sqrt(N / R)
        elif self.name == 'BG':
            # network parameters
            self.N = N  # hidden units
            self.S = S  # input
            self.Rc = Rc  # readout critic
            self.Ra = Ra  # readout actor
            self.sa, self.ha, self.ca, self.aa = [], [], [], []  # input, activity, critic output, actor output
            self.ws = npr.random([N, S])  # input weights
            self.wc = (2 * npr.random((Rc, N)) - 1) / sqrt(N)  # readout weights critic
            self.wa = (2 * npr.random((Ra, N)) - 1) / sqrt(N)  # readout weights actor
            self.var = 0.05
            self.lam = .5
            self.gam = 1

    # nlin
    def f(self, x):
        return np.tanh(x) if not torch.is_tensor(x) else torch.tanh(x)

    # derivative of nlin
    def df(self, x):
        return 1 / (np.cosh(10*np.tanh(x/10)) ** 2) if not torch.is_tensor(x) else 1 / (torch.cosh(10*torch.tanh(x/10)) ** 2)


class Task:
    def __init__(self, name='beads-belief', sampleduration=10,
                 maxsamples=10, context=(.65, .85), rewards=(20, -400, -1), dt=0.1):
        self.name = name
        self.context = context
        duration = sampleduration * maxsamples
        NT = int(duration / dt)
        NT_sample = int(sampleduration / dt)
        # task parameters
        if self.name == 'beads-belief':
            self.T, self.dt, self.NT, self.NT_sample = duration, dt, NT, NT_sample
            self.s = 0.0 * np.ones((2, NT))         # input sample s, context q
            self.ustar = 0.5 * np.ones((NT, 2))     # belief state: p, 1-p
        elif self.name == 'beads-choice':
            self.maxsamples = maxsamples
            self.s = 0.0 * np.ones((1, 2))          # belief state input
            self.r = 0.0 * np.ones((1, 1))          # reward
            self.rewards = rewards                  # correct, incorrect, sample

    def loss(self, err):
        mse = (err ** 2).mean() / 2
        return mse


class Algorithm:
    def __init__(self, name='Adam', Nepochs=10000, lr=(0, 1e-1, 1e-1)):
        self.name = name
        # learning parameters
        self.Nepochs = Nepochs
        self.Nstart_anneal = 5000
        self.lr = lr  # learning rate
        self.annealed_lr = 1e-5
