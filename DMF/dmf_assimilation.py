# -*- coding: utf-8 -*-
# @Time : 2022/12/21 15:26
# @Author : wy36
# @File : simulation.py


from DMF.DMFNetwork import DMFNetwork
from DMF.hemodynamic_cuda import hemodynamic
import numpy as np
import torch
import matplotlib.pyplot as mp


def observation_matrix(brain_num, state_num):
    h = torch.zeros(brain_num, brain_num*state_num)
    for i in range(brain_num):
        h[i, (i+1)*state_num-1] = 1
    return h


def enKF(w_hat, bold_sigma: float, bold_t):
    ensembles, brain_num, state_num = w_hat.shape  # ensemble, brain_n, w_ie+s+hemodynamic_state
    h = observation_matrix(brain_num, state_num).type_as(w_hat)  # brain_n, brain_n*state_num
    w_mean = torch.mean(w_hat, dim=0, keepdim=True)  # ensemble, brain_n, state_num
    w_diff = (w_hat - w_mean).reshape(ensembles, -1)  # ensemble, brain_n*state_num
    w_hx = (h.repeat(ensembles, 1, 1) @ w_diff.unsqueeze(-1)).reshape(ensembles, brain_num)  # ensemble, brain_n
    w_s = w_hx.T @ w_hx / (ensembles - 1) + bold_sigma * torch.eye(brain_num).type_as(w_hat)  # brain_n, brain_n
    print('det, cond:', torch.linalg.det(w_s), torch.linalg.cond(w_s))
    kalman = (w_diff.T @ w_hx/(ensembles - 1)) @ torch.linalg.inv(w_s)  # (brain_num*state_num, brain_n)
    bold_with_noise = bold_t + bold_sigma ** 0.5 * torch.randn(ensembles, brain_num).type_as(bold_t)
    w_hat += (kalman.repeat(ensembles, 1, 1) @ (bold_with_noise-w_hx).unsqueeze(-1)).reshape(w_hat.shape)
    return w_hat


def diffusion_enkf(w_hat, bold_sigma, bold_t, solo_rate, debug=False):
    ensembles, brain_num, state_num = w_hat.shape
    w = w_hat.clone()  # ensemble, brain_n, hp_num+hemodynamic_state
    w_mean = torch.mean(w_hat, dim=0, keepdim=True)
    w_diff = w_hat - w_mean
    w_cx = w_diff[:, :, -1] * w_diff[:, :, -1]
    w_cxx = torch.sum(w_cx, dim=0) / (ensembles - 1) + bold_sigma
    temp = w_diff[:, :, -1] / (w_cxx.reshape([1, brain_num])) / (ensembles - 1)  # (ensemble, brain)
    # kalman = torch.mm(temp.T, w_diff.reshape([ensembles, brain_num*state_num]))  # (brain_n, brain_num*state_num)
    bold_with_noise = bold_t + bold_sigma ** 0.5 * torch.normal(0, 1, size=(ensembles, brain_num)).type_as(bold_t)
    w += solo_rate * (bold_with_noise - w_hat[:, :, -1])[:, :, None] * torch.sum(temp[:, :, None] * w_diff, dim=0,
                                                                                 keepdim=True)
    w += (1 - solo_rate) * torch.mm(torch.mm(bold_with_noise - w_hat[:, :, -1], temp.T) / brain_num,
                                    w_diff.reshape([ensembles, -1])).reshape(w_hat.shape)
    if debug:
        w_debug = w_hat[:, :10, -1][None, :, :] + (bold_with_noise - w_hat[:, :, -1]).T[:, :, None] \
                  * torch.mm(temp.T, w_diff[:, :10, -1])[:, None, :]  # brain_num, ensemble, 10
        return w, w_debug
    else:
        # print(w_cxx.max(), w_cxx.min(), w[:, :, :-6].max(), w[:, :, :-6].min())
        return w


class DMFAssimilation:
    def __init__(self, w_ie, c_ij, **kwargs):
        self.w_ie = w_ie
        self.c_ij = c_ij
        self.dmfNetwork = DMFNetwork(self.w_ie, self.c_ij, **kwargs)
        # self.bold = hemodynamic(**kwargs)
        self.ensemble, self.n = w_ie.shape
        self.noise_p = kwargs.get("noise_p", 1)
        self.sigma_o = kwargs.get("sigma_o", 1e-2)
        self.w = None
        self.wie_lower = torch.zeros_like(self.w_ie[0])
        self.wie_upper = 3 * torch.ones_like(self.w_ie[0])

    @staticmethod
    def sigmoid_torch(val, lower, upper, scale=10):
        val_shape = val.shape
        assert len(lower.shape) == 1
        assert val_shape[-1] == lower.shape[-1]
        val = val.reshape(-1, lower.shape[-1])
        if isinstance(val, torch.Tensor):
            out = lower + (upper - lower) * torch.sigmoid(val / scale)
            return out.reshape(val_shape)
        elif isinstance(val, np.ndarray):
            out = lower + (upper - lower) * 1 / (1 + np.exp(-val.astype(np.float32)) / scale)
            return out.reshape(val_shape)
        else:
            print('torch.Tensor or np.ndarray?')

    @staticmethod
    def log_torch(val, lower, upper, scale=10):
        val_shape = val.shape
        assert len(lower.shape) == 1
        assert val_shape[-1] == lower.shape[-1]
        val = val.reshape(-1, lower.shape[-1])
        if (val >= upper).all() or (val <= lower).all():
            print('val <= upper).all() and (val >= lower).all()?')
        if isinstance(val, torch.Tensor):
            out = scale * (torch.log(val - lower) - torch.log(upper - val))
            return out.reshape(val_shape)
        elif isinstance(val, np.ndarray):
            out = scale * (np.log(val - lower) - np.log(upper - val))
            return out.reshape(val_shape)
        else:
            print('torch.Tensor or np.ndarray?')

    def initialize(self, steps=1, dt=1e-3, **kwargs):
        state = torch.stack(self.dmfNetwork.initialize(steps=steps, dt=dt, **kwargs), -1)
        wie_log = self.log_torch(self.w_ie, self.wie_lower, self.wie_upper)
        self.w = torch.dstack((wie_log.unsqueeze(-1), state[:, :, 2:4], state[:, :, 4:6].mean(-1).unsqueeze(-1)))
        return self.w

    def da_evolution(self, bold_t, steps=1, dt=1e-3, g=2):
        wie_log = self.w[:, :, 0] + self.noise_p * torch.randn(self.w_ie.shape)
        self.w_ie = self.sigmoid_torch(wie_log, self.wie_lower, self.wie_upper)
        self.dmfNetwork.update_s((self.w_ie, g * torch.ones(self.n)), (self.w[:, :, 1], self.w[:, :, 2]))
        state = torch.stack(self.dmfNetwork.run(steps=steps, dt=dt, print_info=False), -1)
        self.w = torch.dstack((wie_log.unsqueeze(-1), state[:, :, 2:4], state[:, :, 4:6].mean(-1).unsqueeze(-1)))
        self.w = enKF(self.w, self.sigma_o, bold_t)
        # self.w = diffusion_enkf(self.w, self.sigma_o, bold_t, 0.5)
        return self.w

    def data_assimilation(self, bold_o, times=None):
        self.initialize()
        times = bold_o.shape[0] if times is None else times
        assert times <= bold_o.shape[0]
        state = torch.stack(self.dmfNetwork.state, -1).unsqueeze(0)
        for t in range(times):
            print(t, self.w_ie.max(), self.w_ie.min())
            self.da_evolution(bold_o[t].reshape(1, self.n))
            state = torch.cat((state, torch.stack(self.dmfNetwork.state, -1).unsqueeze(0)), 0)
        print(state.shape)
        self.dmfNetwork.draw_state(state)
        print(self.w_ie-1)


def test():
    a = torch.rand(2, 3, 4)
    b = torch.rand(3)
    c = enKF(a, 1., b)
    print(c)


if __name__ == '__main__':
    cij = np.loadtxt("../data/Desikan_68/data/sc_train.csv", delimiter=",", dtype=np.float32)
    cij = torch.from_numpy(cij / np.max(cij))
    wie = torch.rand(100, cij.shape[0])
    dmfAssimilation = DMFAssimilation(wie, cij)
    observation = torch.from_numpy(np.load('observation.npy').astype(np.float32))
    dmfAssimilation.data_assimilation(observation, 100)
    # test()
