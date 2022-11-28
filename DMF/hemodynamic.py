# -*- coding: utf-8 -*- 
# @Time : 2022/9/18 15:30 
# @Author : wy36
# @File : hemodynamic.py

import numpy as np


class BOLD:
    def __init__(self, epsilon, tao_s, tao_f, tao_0, alpha, E_0, V_0,
                 delta_t=1e-3, init_f_in=None, init_s=None, init_v=None, init_q=None):
        self.epsilon = epsilon
        self.kappa = tao_s
        self.gamma = tao_f
        self.tao = tao_0
        self.rho = E_0
        self.V_0 = V_0
        self.delta_t = delta_t
        self.div_alpha = 1 / alpha
        self.f_in = init_f_in
        self.s = init_s
        self.v = init_v
        self.q = init_q
        self.k1 = k1
        self.k2 = k2
        self.k3 = k3

    def update(self, f_str, df):
        f = self.__getattribute__(f_str)
        if f is None:
            self.__setattr__(f_str, df * self.delta_t)
        else:
            f += df * self.delta_t

    def run(self, u):
        assert isinstance(u, np.ndarray)
        if self.s is None:
            self.s = np.zeros_like(u)
        if self.q is None:
            self.q = np.zeros_like(u)
        if self.v is None:
            self.v = np.ones_like(u)
        if self.f_in is None:
            self.f_in = np.ones_like(u)
        d_s = self.epsilon * u - self.s * self.kappa - (self.f_in-1) * self.gamma
        q_part = np.where(self.f_in > 0, 1 - (1-self.rho)**(1/self.f_in), np.ones_like(self.f_in))
        self.update('q', (self.f_in * q_part/self.rho - self.q * self.v ** (self.div_alpha - 1))/self.tao)
        self.update('v', (self.f_in - self.v ** self.div_alpha)/self.tao)
        self.update('f_in', self.s)
        self.f_in = np.where(self.f_in > 0, self.f_in, np.zeros_like(self.f_in))
        self.update('s', d_s)

        out = self.V_0 * (self.k1 * (1 - self.q) + self.k2 * (1 - self.q / self.v) + self.k3 * (1 - self.v))
        return np.stack([self.s, self.q, self.v, self.f_in, out])
