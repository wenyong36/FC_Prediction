# -*- coding: utf-8 -*-
# @Time : 2022/10/5 20:00
# @Author : wy36
# @File : dmf_network.py


from time import time
import torch
import numpy as np
import scipy.integrate
from scipy import sparse
from utlis.helper import progress_bar


def phi(x, a, b, d):
    """
    activation function
    """
    return (a * x - b) / (1 - np.exp(-d * (a * x - b)))


def dynamical_equation(t, y, g, w_ie, Cij, args):
    """
    Mean field model, state variable: [s_e,  s_i].

    Parameters
    ----------
    t: float
        Time
    y: ndarray
        State variable, shape=(2*n, ) where n denotes the number of nodes.
        i.e., [s_e1, s_i1, s_e2, s_i2, ...s_en, s_in,]
    g: ndarray
        global coupling weight
    w_ie: float
        connection weight of inh. to exc. population
    args: tuple
        other specified parameter

    Returns
    -------
        Time derivative at time t.
    dy/dt : ndarray, 1D float
    """
    dydt = np.zeros_like(y)
    n = len(y)
    assert g.shape == y.shape
    w_ee, w_ei, w_ii, w_e, w_i, j_nmda, j_i, I_b, alpha_e, b_e, d_e, alpha_i, b_i, d_i, tau_e, tau_i, gamma, sigma = args
    I_e = w_e * I_b + w_ee * j_nmda * y[::2] + g * j_nmda * np.dot(Cij, y[::2]) - w_ie * j_i * y[1::2]
    I_i = w_i * I_b + w_ei * j_nmda * y[::2] - w_ii * j_i * y[1::2]
    r_e = phi(I_e, alpha_e, b_e, d_e)
    r_i = phi(I_i, alpha_i, b_i, d_i)
    dydt[::2] = - y[::2] / tau_e + (1 - y[::2]) * gamma * r_e + sigma * np.random.normal(size=(int(n / 2)))
    dydt[1::2] = - y[1::2] / tau_i + r_i + sigma * np.random.normal(size=(int(n / 2)))
    return dydt


def integrate(pm, init=None, g=1., w_ie=0.1, console_output=True):
    """
    Perform time integration of neuronal network.

    Parameters
    ----------
    pm : parameter.py
        Parameter file.
    init : optional
        Initial conditions.
    g: float
        global coupling parameter
    w_ie: float
        connection weight of inh. to exc. population
    console_output: bool
        Whether to print details to the console.

    Returns
    -------
    theta_t : ndarray, 2D float [time, neuron]
        Neuron states at respective times.
    """

    # Initialise network for t=0
    theta_t = np.zeros((len(pm.t), pm.N * 2))
    # If init is not specified, choose uniform distribution.
    if init is None:
        init = np.linspace(0., 1., pm.N * 2)
    theta_t[0] = init

    # Initialise integrator
    network = scipy.integrate.ode(dynamical_equation)
    network.set_integrator('dopri5')
    network.set_initial_value(theta_t[0], pm.t[0])
    network.set_f_params(g, w_ie, pm.Cij.astype(np.float64), pm.args)
    # if np.count_nonzero(pm.Cij) / pm.N ** 2 < 0.4:  # Check for sparsity of A
    #     network.set_f_params(g, w_ie,
    #                          sparse.csc_matrix(pm.Cij.astype(np.float64)), pm.args)
    # else:
    #     network.set_f_params(g, w_ie, pm.Cij.astype(np.float64), pm.args)

    # Time integration
    if console_output:
        print('\nNetwork with', pm.N, 'nodes | Integrating', pm.t[-1],
              'time units:')
    computation_start = time()
    step = 1
    while network.successful() and step < len(pm.t):
        network.integrate(pm.t[step])
        theta_t[step] = network.y
        if console_output:
            progress = step / (len(pm.t))
            progress_bar(progress, time() - computation_start)
        step += 1

    return theta_t