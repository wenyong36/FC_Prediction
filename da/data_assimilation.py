# -*- coding: utf-8 -*- 
# @Time : 2022/9/23 21:10 
# @Author : lepold
# @File : assimilation.py

from multiprocessing.pool import ThreadPool as Thpool
import os
import time

import matplotlib.pyplot as mp
import numpy as np


def regular_dict(**kwargs):
    return kwargs


bold_params = regular_dict(epsilon=200, tao_s=0.8, tao_f=0.4, tao_0=1, alpha=0.2, E_0=0.8, V_0=0.02)
v_th = -50


class DA(object):
    """

    Parameters
    ----------
    evolution: class
        evolution model
    observation: class
        observation model

    """

    @staticmethod
    def show_bold(W, bold, T, path, brain_num):
        iteration = [i for i in range(T)]
        for i in range(brain_num):
            print("show_bold" + str(i))
            fig = mp.figure(figsize=(5, 3), dpi=500)
            ax1 = fig.add_subplot(1, 1, 1)
            ax1.plot(iteration, bold[:T, i], 'r-')
            ax1.plot(iteration, np.mean(W[:T, :, i, -1], axis=1), 'b-')
            mp.fill_between(iteration, np.mean(W[:T, :, i, -1], axis=1) -
                            np.std(W[:T, :, i, -1], axis=1), np.mean(W[:T, :, i, -1], axis=1)
                            + np.std(W[:T, :, i, -1], axis=1), color='b', alpha=0.2)
            mp.ylim((0.0, 0.08))
            ax1.set(xlabel='observation time/800ms', ylabel='bold', title=str(i + 1))
            mp.savefig(os.path.join(path, "bold" + str(i) + ".png"), bbox_inches='tight', pad_inches=0)
            mp.close(fig)
        return None

    @staticmethod
    def show_hp(hp, T, path, brain_num, hp_num, hp_real=None):
        iteration = [i for i in range(T)]
        for i in range(brain_num):
            for j in range(hp_num):
                print("show_hp", i, 'and', j)
                fig = mp.figure(figsize=(5, 3), dpi=500)
                ax1 = fig.add_subplot(1, 1, 1)
                ax1.plot(iteration, np.mean(hp[:T, :, i, j], axis=1), 'b-')
                if hp_real is None:
                    pass
                else:
                    ax1.plot(iteration, np.tile(hp_real[j], T), 'r-')
                mp.fill_between(iteration, np.mean(hp[:T, :, i, j], axis=1) -
                                np.sqrt(np.var(hp[:T, :, i, j], axis=1)), np.mean(hp[:T, :, i, j], axis=1)
                                + np.sqrt(np.var(hp[:T, :, i, j], axis=1)), color='b', alpha=0.2)
                ax1.set(xlabel='observation time/800ms', ylabel='hyper parameter')
                mp.savefig(os.path.join(path, "hp" + str(i) + "_" + str(j) + ".png"), bbox_inches='tight', pad_inches=0)
                mp.close(fig)
        return None

    def __init__(self, evolution, observation, ensembles=100, param_noise=0.1, noise=1e-8,
                 node=1):  # real signature unknown

        self.evolution = evolution
        self.observation = observation

        self.ensembles = ensembles
        self.param_noise = param_noise
        self.noise = noise
        self.node = node

    def set_param_range(self, centered_param, param_ind):
        """

        Parameters
        ----------
        centered_param: ndarray
            initialized params in evolution model
        param_ind: tuple
            the param index which will be filtered.

        Returns
        -------
          prams: ndarray, shape is (ensembles, num_params)

        """
        self.num_params = len(param_ind)
        self.num_states = self.num_params + 1 + 5  # 1 represents fire rate, and 5 denotes (s, q, v, fin, bold)
        param_range = np.stack((centered_param / 4, centered_param * 4), axis=1)
        self.param_low = np.tile(param_range[param_ind, 0], (self.node, 1))  # shape = brain_num,hp_num
        self.param_high = np.tile(param_range[param_ind, 1], (self.node, 1))
        self.param = np.linspace(self.param_low, self.param_high, 3 * self.ensembles, dtype=np.float32)[
                     self.ensembles: -1 * self.ensembles]

        return None

    def sigmoid_map(self, vals, scale=10):
        """
        scale parameters with a sigmoid function.
        Returns
        -------
        """
        assert np.isfinite(vals).all()
        return self.param_low + (self.param_high - self.param_low) / (1 + np.exp(-vals / scale))

    def log_map(self, val, scale=10):
        """
        scale parameters with a log function.
        Returns
        -------
        """
        assert np.all(val <= self.param_high) and np.all(val >= self.param_low)
        return scale * (np.log(val - self.param_low) - np.log(self.param_high - val))

    def initialize(self, time=800):
        """
        initialize these evolution models.
        Returns
        -------
        """
        self.state_matrix = np.zeros((self.ensembles, self.node, self.num_states), dtype=np.float32)
        print("Initialize")
        for t in range(time):
            with Thpool(processes=16) as pool:
                out = pool.map(lambda x: x.run(0.007), self.evolution_models)  # out = list((sum_activate, mean_vi),.....)
                act = np.stack(out)
                bold_state = self.observation_model.run(act)
        print("ins_rate", act.squeeze())
        self.state_matrix[:, :, -5:] = np.transpose(bold_state, [1, 2, 0])
        self.state_matrix[:, :, [-6]] = act.reshape((self.ensembles, self.node, 1))
        self.state_matrix[:, :, :-6] = self.log_map(self.param)

        return None

    def filter_new(self, signal: np.ndarray, solo_rate=1.):
        w_hat = self.state_matrix.copy()  # ensemble, brain_n, hp_num+hemodynamic_state
        w_mean = np.mean(w_hat, axis=0, keepdims=True)
        w_diff = w_hat - w_mean
        w_cx = w_diff[:, :, -1] * w_diff[:, :, -1]
        w_cxx = np.sum(w_cx, axis=0) / (self.ensembles - 1) + self.noise
        temp = w_diff[:, :, -1] / (w_cxx.reshape([1, self.node])) / (self.ensembles - 1)
        bold_with_noise = signal + np.sqrt(self.noise) * np.random.randn(self.ensembles, self.node)
        self.state_matrix += solo_rate * (bold_with_noise - w_hat[:, :, -1])[:, :, None] * np.sum(
            temp[:, :, None] * w_diff, axis=0,
            keepdims=True)
        self.state_matrix += (1 - solo_rate) * np.matmul(
            np.matmul(bold_with_noise - w_hat[:, :, -1], temp.T) / self.node,
            w_diff.reshape([self.ensembles, -1])).reshape(w_hat.shape)
        return None

    def filter(self, signal: np.ndarray):
        self.state_matrix = self.state_matrix.copy()
        w_mean = np.mean(self.state_matrix, axis=0, keepdims=True)  # 1,3
        w_diff = self.state_matrix - w_mean
        w_cxx = np.sum(w_diff[:, :, -1] * w_diff[:, :, -1], axis=0) / (self.ensembles - 1) + self.noise
        temp = np.einsum('ijk,ijm->jkm', w_diff, w_diff[:, :, [-1]]) / (self.ensembles - 1)
        kalman = temp / (w_cxx.reshape([self.node, 1, 1]))
        observation_diff = signal[0, :, None] + np.sqrt(self.noise) * np.random.randn(self.ensembles, self.node, 1)
        self.state_matrix = self.state_matrix + np.einsum('ijk,mik->mij', kalman, observation_diff)
        return None

    def evolve(self, steps=800):
        param_with_noise = self.state_matrix[:, :, :self.num_params] + np.sqrt(self.param_noise) * np.random.randn(
            self.ensembles, self.node, self.num_params)
        for i in range(self.ensembles):
            self.evolution_models[i].update_param(
                self.sigmoid_map(param_with_noise[i]).squeeze())  # TODO: add method update property
        for t in range(steps):
            with Thpool(processes=16) as pool:
                out = pool.map(lambda x: x.run(0.007), self.evolution_models)  # out = list((sum_activate, mean_vi),.....)
                act = np.stack(out)
                bold_state = self.observation_model.run(act)
        print("ins_rate", act.squeeze())
        self.state_matrix[:, :, -5:] = np.transpose(bold_state, [1, 2, 0])
        self.state_matrix[:, :, [-6]] = act.reshape((self.ensembles, self.node, 1))
        self.state_matrix[:, :, :-6] = param_with_noise
        return None

    def __call__(self, *args, **kwargs):
        if not hasattr(self, "param"):
            raise NotImplementedError("before call this DA modle, you should set_param_range")
        self.evolution_models = [self.evolution(self.param[i].squeeze()) for i in range(self.ensembles)]
        self.observation_model = self.observation(**bold_params)
        self.initialize(time=800)
        assert 'bold_observation' in kwargs.keys()
        bold_observation = kwargs['bold_observation']
        if isinstance(bold_observation, str):
            bold_observation = np.load(bold_observation)
            bold_observation = np.array(bold_observation)
        bold_observation = 0.04 + 0.03 * (bold_observation - bold_observation.min()) / (
                bold_observation.max() - bold_observation.min())
        T = kwargs.get("T", 100)
        state_all = []
        for t in range(T):
            start_time = time.time()
            bold_t = bold_observation[t].reshape((1, self.node))
            self.evolve(steps=800)
            print(f"\nevolve {t} cost time: {time.time() - start_time:.1f}")
            state_all.append(self.state_matrix)
            print(self.state_matrix[:, 0, -1].mean(axis=0), bold_t[0, 0])
            self.filter_new(bold_t)
            print(f"da {t} cost time: {time.time() - start_time:.1f}")
            self.observation_model.s = self.state_matrix[:, :, self.num_params + 1].copy()
            self.observation_model.q = np.maximum(self.state_matrix[:, :, self.num_params + 2].copy(), 1e-05)
            self.observation_model.v = np.maximum(self.state_matrix[:, :, self.num_params + 3].copy(), 1e-05)
            self.observation_model.log_f_in = np.maximum(self.state_matrix[:, :, self.num_params + 5].copy(),
                                                         -15)
        state_all = np.array(state_all)
        path = kwargs.get("path", "./da_results")
        np.save(os.path.join(path, "state.npy"), state_all)
        self.show_bold(self.state_matrix, bold_observation.reshape((-1, 1)), T, path, self.node)
        hp = self.sigmoid_map(self.state_matrix[:, :, :, :self.num_params]).mean(axis=1)
        self.show_hp(hp, T, path, self.node, self.num_params, hp_real=None)


if __name__ == '__main__':
    pass
