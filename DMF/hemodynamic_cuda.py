# -*- coding: utf-8 -*-
# @Time : 2022/9/18 15:30
# @Author : wy36
# @File : hemodynamic_cuda.py

import torch
import numpy as np


class hemodynamic:
    def __init__(self, init_f_in=None, init_s=None, init_v=None, init_q=None, is_cuda=True, **kwargs):
        self.epsilon = kwargs.get("epsilon", 200.)
        self.kappa = kwargs.get("kappa", 0.8)
        self.gamma = kwargs.get("gamma", 0.4)
        self.tao = kwargs.get("gamma", 1.)
        self.rho = kwargs.get("gamma", 0.8)
        self.V_0 = kwargs.get("gamma", 0.02)
        self.delta_t = kwargs.get("delta_t", 1e-3)
        self.div_alpha = kwargs.get("div_alpha", 1/0.2)
        self.s = init_s
        self.q = init_q
        self.v = init_v
        self.f_in = init_f_in
        self.k1 = kwargs.get("k1", 5.6)
        self.k2 = kwargs.get("k2", 2.)
        self.k3 = kwargs.get("k3", 1.4)
        self.is_cuda = is_cuda

    def update(self, f_str, df):
        f = self.__getattribute__(f_str)
        if f is None:
            self.__setattr__(f_str, df * self.delta_t)
        else:
            f += df * self.delta_t

    def numpy2torch(self, u):
        assert isinstance(u, np.ndarray)
        if self.is_cuda:
            return torch.from_numpy(u).cuda()
        else:
            return torch.from_numpy(u)

    def state_update(self, w):
        if isinstance(w, np.ndarray):
            w = self.numpy2torch(w.astype(np.float))
        self.s = w[:, :, 0].reshape(-1)
        self.q = torch.max(w[:, :, 1], torch.tensor([1e-05]).type_as(w)).reshape(-1)
        self.v = torch.max(w[:, :, 2], torch.Tensor([1e-05]).type_as(w)).reshape(-1)
        self.f_in = torch.max(w[:, :, 3], torch.Tensor([1e-05]).type_as(w)).reshape(-1)

    def run(self, u):
        if isinstance(u, np.ndarray):
            u = self.numpy2torch(u.astype(np.float))
        if self.s is None:
            self.s = torch.zeros_like(u)
        if self.q is None:
            self.q = torch.zeros_like(u)
        if self.v is None:
            self.v = torch.ones_like(u)
        if self.f_in is None:
            self.f_in = torch.ones_like(u)
        d_s = self.epsilon * u - self.s * self.kappa - (self.f_in-1) * self.gamma
        q_part = torch.where(self.f_in > 0, 1 - (1-self.rho)**(1/self.f_in), torch.ones_like(self.f_in))
        self.update('q', (self.f_in * q_part/self.rho - self.q * self.v ** (self.div_alpha - 1))/self.tao)
        self.update('v', (self.f_in - self.v ** self.div_alpha)/self.tao)
        self.update('f_in', self.s)
        self.f_in = torch.where(self.f_in > 0, self.f_in, torch.zeros_like(self.f_in))
        self.update('s', d_s)

        out = self.V_0 * (self.k1 * (1 - self.q) + self.k2 * (1 - self.q / self.v) + self.k3 * (1 - self.v))
        return out