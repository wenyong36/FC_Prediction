# -*- coding: utf-8 -*-
# @Time : 2022/12/24 16:26
# @Author : wy36
# @File : DMFNetwork.py

import torch
import numpy as np
import matplotlib.pyplot as mp


class DMFNetwork:
    """

    """
    def __init__(self, w_ie, c_ij, **kwargs):
        """

        Parameters
        ----------
        w_ie
        c_ij
        kwargs
        """
        assert isinstance(w_ie, torch.Tensor)
        assert isinstance(c_ij, torch.Tensor)
        self.m, self.n = w_ie.shape  # m means batch, n means number of brain regions
        print('w_ie.shape, dtype', w_ie.shape, w_ie.dtype)
        self.w_ie = w_ie  # shape = m, n
        self.c_ij = c_ij  # i to j, shape = n, n
        # The following shows equations states
        self.I_e = torch.zeros_like(w_ie)  # shape = m, n
        self.I_i = torch.zeros_like(w_ie)  # shape = m, n
        self.s_e = torch.rand(w_ie.shape, dtype=w_ie.dtype)  # shape = m, n
        self.s_i = torch.rand(w_ie.shape, dtype=w_ie.dtype)  # shape = m, n
        self.r_e = torch.zeros_like(w_ie)  # shape = m, n
        self.r_i = torch.zeros_like(w_ie)  # shape = m, n
        # The following parameters can be adjusted
        self.g = kwargs.get("g", 2.)
        self.w_ee = kwargs.get("w_ee", 1.4)
        self.w_ei = kwargs.get("w_ei", 1.)
        self.w_ii = kwargs.get("w_ii", 1.)
        self.w_e = kwargs.get("w_e", 1.)
        self.w_i = kwargs.get("w_i", 0.7)
        self.j_nmda = kwargs.get("j_nmda", 0.15)
        self.j_i = kwargs.get("j_i", 1.)
        self.I_b = kwargs.get("I_b", 0.382)
        self.a_e = kwargs.get("a_e", 310)
        self.b_e = kwargs.get("b_e", 125)
        self.d_e = kwargs.get("d_e", 0.16)
        self.a_i = kwargs.get("a_i", 615)
        self.b_i = kwargs.get("b_i", 177)
        self.d_i = kwargs.get("d_i", 0.087)
        self.tau_e = kwargs.get("tau_e", 0.1)
        self.tau_i = kwargs.get("tau_i", 0.01)
        self.gamma = kwargs.get("gamma", 0.641)
        self.sigma = kwargs.get("sigma", 0.01)

    @property
    def state(self):
        return self.I_e, self.I_i, self.s_e, self.s_i, self.r_e, self.r_i

    @staticmethod
    def phi(x, a, b, d):
        """
        activation function
        """
        return (a * x - b) / (1 - torch.exp(-d * (a * x - b)))

    @staticmethod
    def draw_state(state):
        # state.shape = T, m, n, state_number
        fig = mp.figure(figsize=(8, 6), dpi=100)
        ax1 = fig.add_subplot(6, 1, 1)
        ax2 = fig.add_subplot(6, 1, 2)
        ax3 = fig.add_subplot(6, 1, 3)
        ax4 = fig.add_subplot(6, 1, 4)
        ax5 = fig.add_subplot(6, 1, 5)
        ax6 = fig.add_subplot(6, 1, 6)
        for i in range(state.shape[2]):
            ax1.plot(state[:, :, i, 2].mean(-1), lw=1.)
            ax2.plot(state[:, :, i, 3].mean(-1), lw=1.)
            ax3.plot(state[:, :, i, 0].mean(-1), lw=1.)
            ax4.plot(state[:, :, i, 1].mean(-1), lw=1.)
            ax5.plot(state[:, :, i, 4].mean(-1), lw=1.)
            ax6.plot(state[:, :, i, 5].mean(-1), lw=1.)
        ax1.set_ylabel(r"$S_e$")
        ax2.set_ylabel(r"$S_i$")
        ax3.set_ylabel(r"$I_e$")
        ax4.set_ylabel(r"$I_i$")
        ax5.set_ylabel(r"$r_e$")
        ax6.set_ylabel(r"$r_i$")
        fig.tight_layout()
        fig.show()

    def initialize(self, steps: int = 1, dt: float = 1e-3, **kwargs):
        self.I_e = kwargs.get("I_e", self.I_e)
        self.I_i = kwargs.get("I_i", self.I_i)
        self.s_e = kwargs.get("s_e", self.s_e)
        self.s_i = kwargs.get("s_i", self.s_i)
        self.r_e = kwargs.get("r_e", self.r_e)
        self.r_i = kwargs.get("r_i", self.r_i)
        self.g = kwargs.get("g", self.g)
        self.w_ie = kwargs.get("w_ie", self.w_ie)
        return self.run(w_ie=self.w_ie, steps=steps, dt=dt)

    def run(self, w_ie=None, steps: int = 1, dt: float = 1e-3, print_info=False):
        """

        Parameters
        ----------
        w_ie: w_ie
        steps: iteration times
        dt: time resolution, 1ms: 1e-3
        print_info: for debug

        Returns
        -------
        Hidden states
        """
        if w_ie is not None:
            self.w_ie = w_ie
        for t in range(int(steps)):
            self.I_e = self.w_e * self.I_b + self.w_ee * self.j_nmda * self.s_i \
                       + self.g * self.j_nmda * torch.mm(self.s_e, self.c_ij) \
                       - self.w_ie * self.j_i * self.s_i
            self.I_i = self.w_i * self.I_b + self.w_ei * self.j_nmda * self.s_e - self.w_ii * self.j_i * self.s_i
            self.r_e = self.phi(self.I_e, self.a_e, self.b_e, self.d_e)
            self.r_i = self.phi(self.I_i, self.a_i, self.b_i, self.d_i)
            self.s_e += dt * (-self.s_e / self.tau_e + self.gamma * (1 - self.s_e) * self.r_e +
                              self.sigma * torch.randn(self.s_e.shape, dtype=self.s_e.dtype))
            self.s_i += dt * (-self.s_i / self.tau_i + self.r_i +
                              self.sigma * torch.randn(self.s_i.shape, dtype=self.s_i.dtype))
        if print_info:
            print('re,ri,se,si mean', self.r_e.mean(), self.r_i.mean(), self.s_e.mean(), self.s_i.mean())
            print('se,si range', self.s_e.max(), self.s_e.min(), self.s_i.max(), self.s_i.min())
            print('shape', self.s_e.shape, self.r_e.shape, self.I_e.shape)
        return self.I_e, self.I_i, self.s_e, self.s_i, self.r_e, self.r_i

    def update_all(self, para, state):
        """
        For updating model parameters and states

        Parameters
        ----------
        para: parameters assimilated

        state: hidden states assimilated

        Returns
        -------

        """
        self.I_e = state[0]
        self.I_i = state[1]
        self.s_e = torch.clamp(state[2], 0, 1)
        self.s_i = torch.clamp(state[3], 0, 1)
        self.r_e = torch.clamp(state[4], 0, 200)
        self.r_i = torch.clamp(state[5], 0, 200)
        self.w_ie = para[0]
        self.g = para[1].mean()

    def update_s(self, para, state):
        """
        For updating model parameters and s_e, s_i

        Parameters
        ----------
        para: parameters assimilated

        state: hidden states assimilated

        Returns
        -------

        """
        self.s_e = torch.clamp(state[0], 0, 1)
        self.s_i = torch.clamp(state[1], 0, 1)
        self.w_ie = para[0]
        self.g = para[1].mean()

    def simulation(self, observation_time=1000):
        state = torch.stack(self.initialize(), -1).unsqueeze(0)
        print(state.shape)
        for i in range(observation_time):
            state = torch.cat((state, torch.stack(self.run(), -1).unsqueeze(0)), 0)
        print(state[-1, 0].mean(0), state.shape)
        self.draw_state(state)
        return state[1:, 0, :, -2:].mean(-1)


if __name__ == '__main__':
    # cij = np.loadtxt("../data/Desikan_68/data/sc_train.csv", delimiter=",", dtype=np.float32)
    # cij = np.random.rand(68, 68).astype(np.float32)
    cij = np.array([[0, 0, 1], [1, 0, 0], [0, 1, 0]]).astype(np.float32)
    cij = torch.from_numpy(cij / np.max(cij))
    wie = torch.ones(1, cij.shape[0])
    dmf = DMFNetwork(wie, cij)
    observation = dmf.simulation()
    np.save('observation_n3.npy', observation.numpy())

