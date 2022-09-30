# -*- coding: utf-8 -*- 
# @Time : 2022/9/18 13:54 
# @Author : lepold
# @File : mean_field.py

import numpy as np
import pandas as pd


class Parameter:
    def __init__(self, time=3):
        self.N = 90  # number of nodes
        self.t = np.linspace(0, time, time * 100)
        w_ee, w_ei, w_ii = 1.4, 1., 1.
        w_e, w_i = 1., 0.7
        j_nmda, j_i = 0.15, 1.
        I_b = 0.382
        alpha_e, b_e, d_e, alpha_i, b_i, d_i = 310., 125., 0.16, 615., 177., 0.087
        tau_e, tau_i, gamma = 0.1, 0.01, 0.641
        sigma = 0.01

        self.args = (
            w_ee, w_ei, w_ii, w_e, w_i, j_nmda, j_i, I_b, alpha_e, b_e, d_e, alpha_i, b_i, d_i, tau_e, tau_i, gamma,
            sigma,)
        self. Cij = None
        # TODO(PENG): Add the construction or data of connection matrix Cij.

    @staticmethod
    def edges_from_sequence(K_in, K_out):
        """ Generate a list containing the edges of the network from given in-degree list and out-degree list.

        Parameters
        ----------
        K_in : ndarray, 1D int
            In-degree sequence.
        K_out : ndarray, 1D int
            Out-degree sequence.

        Returns
        -------
        edges : ndarray, 2D int
            Edges of the shape: edges[:, 0] post-synaptic neurons
                                edges[:, 1] pre-synaptic neurons
        """

        edges = np.empty((K_in.sum(), 2), dtype=int)  # list of edges holding
        # neurons labels
        i_in = 0
        i_out = 0
        for i in range(len(K_in)):
            edges[i_in:i_in + K_in[i], 0] = [i] * K_in[i]
            edges[i_out:i_out + K_out[i], 1] = [i] * K_out[i]
            i_in += K_in[i]
            i_out += K_out[i]
        np.random.shuffle(edges[:, 1])  # randomise who is connected to whom

        return edges

    @staticmethod
    def matrix_from_edges(edges):
        """ Build an adjacency matrix A from edge list, where
        A[i, j] != 0 if neuron j's output is connected to neuron i's input or
                = 0 if not.
        Edges may be repeated in 'edges', which leads to A[i,j] > 1.

        Parameters
        ----------
        edges : ndarray, 2D int
            Directed edges from edges[:, 1] to edges[:, 0].

        Returns
        -------
        A : ndarray, 2D int
            Adjacency matrix.
        """

        N = edges.max() + 1
        A = np.zeros((N, N), dtype=int)  # adjacency matrix A
        unique_edges, counts = np.unique(edges, return_counts=True, axis=0)
        A[(unique_edges[:, 0], unique_edges[:, 1])] = counts

        return A

    @staticmethod
    def remove_self_edges(A):
        """ If adjacency matrix has non-zero entries on its diagonal (self-edges)
        they will be removed by swapping connections. This will not affect degree
        distribution and degree correlation.

        Parameters
        ----------
        A : ndarray, 2D int
            Adjacency matrix.

        Returns
        -------
        A will be modified.
        """

        edges = np.empty((0, 2), dtype=int)
        for i in range(1, A.max() + 1):
            edges = np.append(edges, np.argwhere(A >= i), axis=0)
        num_edges = edges.shape[0]
        self_edge_indices = np.flatnonzero(edges[:, 0] == edges[:, 1])

        for self_edge_index in self_edge_indices:
            I = np.copy(edges[self_edge_index])
            A_I_init = np.copy(A[I[0], I[1]])
            while A[I[0], I[1]] == A_I_init:
                rand_index = np.random.choice(num_edges)
                J = np.copy(edges[rand_index])
                # avoid multi-edges
                if A[J[0], I[1]] == 0 and A[I[0], J[1]] == 0:
                    Parameter.reconnect_edge_pair(A, I, J)
                    # update edges
                    edges[rand_index] = [J[0], I[1]]
                    edges[self_edge_index] = [I[0], J[1]]

        return

    @staticmethod
    def remove_multi_edges(A, console_output=False):
        """  If adjacency matrix has entries higher than 1 (multi-edge) they will
        be removed by swapping connections. This will not affect degree distribution
        and node correlation.

        Parameters
        ----------
        A : ndarray, 2D int
            Adjacency matrix.
        console_output: bool
            Whether or not to print details to the console.

        Returns
        -------
        A will be modified.
        """

        if console_output:
            print('\nRemoving multi-edges')
            print('|......................................')
            print('| N =', A.shape[0], 'and N_edges =', A.sum())
            print('| N_multi_edges =', A[A > 1].sum())
            print('|......................................')



        edges = np.empty((0, 2), dtype=int)
        for i in range(1, A.max() + 1):
            edges = np.append(edges, np.argwhere(A >= i), axis=0)
        num_edges = edges.shape[0]
        multi_indices = np.argwhere(A[edges[:, 0], edges[:, 1]] > 1).squeeze()

        for multi_index in multi_indices:
            I = np.copy(edges[multi_index])
            A_I_init = np.copy(A[I[0], I[1]])
            if A_I_init > 1:
                while A[I[0], I[1]] == A_I_init:
                    rand_index = np.random.choice(num_edges)
                    J = np.copy(edges[rand_index])
                    # avoid self-edges
                    if I[0] != J[1] and J[0] != I[1]:
                        # avoid additional multi-edges
                        if A[J[0], I[1]] == 0 and A[I[0], J[1]] == 0:
                            Parameter.reconnect_edge_pair(A, I, J)
                            # update edges
                            edges[multi_index] = [I[0], J[1]]
                            edges[rand_index] = [J[0], I[1]]

        if console_output:
            print('| N_multi_edges =', A[A > 1].sum())
            print('|......................................')
        return

    @staticmethod
    def reconnect_edge_pair(A, I, J):
        """ Reconnect a pair of edges (I and J) in the adjacency matrix A. I is an
        edge with a connection from I[1] to I[0] and the same holds for J.

        Parameters
        ----------
        A : ndarray, 2D int
            Adjacency matrix.
        I : ndarray, 1D int
            An edge from A. I[0] is post-synaptic and I[1] is pre-synaptic.
        J : ndarray, 1D int
            An edge from A. J[0] is post-synaptic and J[1] is pre-synaptic.

        Returns
        -------
        A will be modified.
        """
        if len(I.shape) > 1:
            if A[A > 1].sum() > 0:
                I0I1, I0I1_counts = np.unique(np.asarray([I[0], I[1]]), axis=1,
                                              return_counts=True)
                J0J1, J0J1_counts = np.unique(np.asarray([J[0], J[1]]), axis=1,
                                              return_counts=True)
                I0J1, I0J1_counts = np.unique(np.asarray([I[0], J[1]]), axis=1,
                                              return_counts=True)
                J0I1, J0I1_counts = np.unique(np.asarray([J[0], I[1]]), axis=1,
                                              return_counts=True)
                A[I0I1[0], I0I1[1]] -= I0I1_counts
                A[J0J1[0], J0J1[1]] -= J0J1_counts
                A[I0J1[0], I0J1[1]] += I0J1_counts
                A[J0I1[0], J0I1[1]] += J0I1_counts
            else:
                I0J1, I0J1_counts = np.unique(np.asarray([I[0], J[1]]), axis=1,
                                              return_counts=True)
                J0I1, J0I1_counts = np.unique(np.asarray([J[0], I[1]]), axis=1,
                                              return_counts=True)
                A[I[0], I[1]] -= 1
                A[J[0], J[1]] -= 1
                A[I0J1[0], I0J1[1]] += I0J1_counts
                A[J0I1[0], J0I1[1]] += J0I1_counts
        else:
            A[I[0], I[1]] -= 1
            A[J[0], J[1]] -= 1
            A[I[0], J[1]] += 1
            A[J[0], I[1]] += 1

        return

    def sparse_interconnected_populations(self, num1, num2, deg1, deg2, inter_edges):
        """
        generate a list containing edges for two interconnected populations based on given information.

        Parameters
        ----------
        num1: int
            number of population 1
        num2: int
            number of population 2
        deg1: int
            the in-degree among neurons in population 1
        deg2: int
            the in-degree among neurons in population 2
        inter_edges: int
            the inter-connected edges between the two populations for both 2 directions. connection is random sampling.

        Returns
        -------
        edges : ndarray, 2D int
            Edges of the shape: edges[:, 0] post-synaptic neurons
                                edges[:, 1] pre-synaptic neurons
        """
        pass

    def all_to_all_interconnected_populations(self, num1, num2, kappa):
        """

        Parameters
        ----------
        num1: int
            number of the population 1
        num2: int
            number of the population 2, usually set to be the same as populaiton 1
        Returns
        -------
        A: ndarray, 2D int
            adjacent matrix (connection matrix)
        """
        pass

    def data2connection(self, path="../data/Desikan_68/data/sc_train.csv", console_output=False):
        data = pd.read_csv(path, header=None)
        self.Cij = data.values
        self.Cij /= self.Cij.max()
        self.N = data.shape[0]
        if console_output:
            print("read csv file finish")
            print(f"set attr for Cij, shape is {self.Cij.shape}")