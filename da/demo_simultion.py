# -*- coding: utf-8 -*- 
# @Time : 2022/9/29 18:48 
# @Author : lepold
# @File : demo_simultion.py

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import gridspec
from scipy.ndimage import uniform_filter1d

from da.block_model import Block
from da.bold_model import BOLD
from generation.read_block import connect_for_block


def regular_dict(**kwargs):
    return kwargs


bold_params = regular_dict(epsilon=200, tao_s=0.8, tao_f=0.4, tao_0=1, alpha=0.2, E_0=0.8, V_0=0.02)
bold = BOLD(**bold_params)
# block_path = "../data/normal_1ms_blocks"
block_path = "./block"
prop, wuij = connect_for_block(block_path)
print(prop[0, 10:14])
block = Block(prop.cuda(), wuij.cuda())

fr = []
bold_signal = []
log = []
for i in range(3000):
    x = block.run(noise_rate=0.007)
    log.append(block.active.cpu().numpy())
    y = bold.run(x)
    fr.append(x)
    bold_signal.append(y)
fr = np.array(fr, dtype=np.float32)
log = np.stack(log, axis=0)
bold_signal = np.array(bold_signal)[:, -1]
fig = plt.figure(figsize=(8, 6))
ax = dict()
ax[0] = fig.add_axes([0.09, 0.68, 0.9, 0.27])
gs1 = gridspec.GridSpec(2, 1)
gs1.update(left=0.09, right=0.96, top=0.6, bottom=0.07, hspace=0.1)
for i in range(2):
    ax[i + 1] = plt.subplot(gs1[i, 0], frameon=True)
ax[0].scatter(*log[-1000:, 100:200].nonzero(), marker='.', s=1., c="b")
ax[0].set_xlim([0, 1000])
ax[0].set_ylim([0, 100])
ax[0].set_xlabel("Neuron")
ax[0].set_ylabel("Time(ms)")

# ax[1].plot(fr, color='0.8')
smooth_fr = uniform_filter1d(fr, size=5)
ax[1].plot(smooth_fr, color="k")
ax[1].set_ylabel("Fr")
ax[2].plot(bold_signal, color="g")
ax[2].set_xlabel("Time(ms)")
ax[2].set_ylabel("Bold")
fig.show()
