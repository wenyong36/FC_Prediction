# -*- coding: utf-8 -*-
# @Time : 2022/9/18 15:26
# @Author : wy36
# @File : simulation.py
import numpy as np
from model import parameter
from DMF import dmf_network as node_network
import matplotlib.pyplot as plt


def phi(x, a, b, d):
    """
    activation function
    """
    return (a * x - b) / (1 - np.exp(-d * (a * x - b)))


def simulation_demo():
    T = 10
    pm = parameter.Parameter(time=T)
    pm.data2connection("../data/Desikan_68/data/sc_train.csv", True)
    Cij = pm.Cij
    w_ee, w_ei, w_ii, w_e, w_i, j_nmda, j_i, I_b, alpha_e, b_e, d_e, alpha_i, b_i, d_i, tau_e, tau_i, gamma, sigma = pm.args
    g = 1
    w_ie = 0.5
    y = node_network.integrate(pm, init=None, g=1., w_ie=1., console_output=True)
    fig = plt.figure(figsize=(8, 6), dpi=100)
    ax1 = fig.add_subplot(6, 1, 1)
    ax2 = fig.add_subplot(6, 1, 2)
    ax3 = fig.add_subplot(6, 1, 3)
    ax4 = fig.add_subplot(6, 1, 4)
    ax5 = fig.add_subplot(6, 1, 5)
    ax6 = fig.add_subplot(6, 1, 6)
    print("\ny.shape", y.shape)
    I_e = w_e * I_b + w_ee * j_nmda * y[:, ::2] + g * j_nmda * np.dot(y[:, ::2], Cij.T) - w_ie * j_i * y[:, 1::2]
    I_i = w_i * I_b + w_ei * j_nmda * y[:, ::2] - w_ii * j_i * y[:, 1::2]
    r_e = phi(I_e, alpha_e, b_e, d_e)
    r_i = phi(I_i, alpha_i, b_i, d_i)
    print()
    for i in range(I_e.shape[1]):
        ax1.plot(y[:, 2 * i], lw=1.)
        ax2.plot(y[:, 2 * i + 1], lw=1.)
        ax3.plot(I_e[:, i], lw=1.)
        ax4.plot(I_i[:, i], lw=1.)
        ax5.plot(r_e[:, i], lw=1.)
        ax6.plot(r_i[:, i], lw=1.)
    ax1.set_xticks([])
    ax2.set_xticks(np.linspace(0, y.shape[0], 5, dtype=np.int_, endpoint=False))
    ax2.set_xticklabels(np.linspace(0, T, 5, dtype=np.int_, endpoint=False))
    ax1.set_ylabel(r"$S_e$")
    ax2.set_ylabel(r"$S_i$")
    fig.tight_layout()
    fig.show()


simulation_demo()