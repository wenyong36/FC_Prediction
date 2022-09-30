# -*- coding: utf-8 -*- 
# @Time : 2022/9/26 12:04 
# @Author : lepold
# @File : test_da.py
import os

import h5py
import numpy as np
import torch

from da.block_model import Block
from da.bold_model import BOLD
from da.data_assimilation import DA
from generation.read_block import connect_for_block

# block_path = "../data/normal_1ms_blocks"
block_path = "./block"
prop, wuij = connect_for_block(block_path)
print(prop[0, 10:14])


class block(Block):
    def __init__(self, initialized_param=None, param_ind=(0, 2)):
        super(block, self).__init__(prop.cuda(), wuij.cuda(), delta_t=1.)
        self.param_ind = param_ind
        if initialized_param is None:
            self.initialized_param = prop[0, self.param_ind]
        else:
            self.initialized_param = initialized_param

        self.g_ui[:, self.param_ind] = torch.from_numpy(initialized_param.astype(np.float32)).cuda()

    def update_param(self, params):
        self.g_ui[:, self.param_ind] = torch.from_numpy(params.astype(np.float32)).cuda()


model = DA(block, BOLD, ensembles=32, param_noise=0.1, noise=1e-8,
           node=1)
centred_param = prop[0, 10:14]
model.set_param_range(centred_param, param_ind=(0, 2))
# bold_path = r"E:\PycharmProjects\spliking_nn_for_brain_simulation\DTI_voxel_network_mat_0719.mat"
# bold_y = h5py.File(bold_path, 'r')[
bold_y = np.load("block/bold.npy")
bold_y = bold_y[0, :]
bold_y = 0.04 + 0.03 * (bold_y - bold_y.min()) / (bold_y.max() - bold_y.min())
write_path = "./da_results/1ms_iter_normal_da"
os.makedirs(write_path, exist_ok=True)
model(bold_observation=bold_y, T=40, path=write_path)
