# -*- coding: utf-8 -*- 
# @Time : 2022/9/18 15:26 
# @Author : lepold
# @File : simulation_demo.py
import numpy as np
from model import parameter
from model import node_network
import matplotlib.pyplot as plt


T = 100
pm = parameter.Parameter(time=T)
pm.data2connection("../data/Desikan_68/data/sc_train.csv", True)
y = node_network.integrate(pm, init=None, g=1., w_ie=1., console_output=True)
fig = plt.figure(figsize=(8, 6), dpi=100)
ax1 = fig.add_subplot(2, 1, 1)
ax2 = fig.add_subplot(2, 1, 2)
print("y.shape", y.shape)
for i in range(3):
    ax1.plot(y[:, 2 * i], lw=1.,)
    ax2.plot(y[:, 2 * i+1], lw=1.)
ax1.set_xticks([])
ax2.set_xticks(np.linspace(0, y.shape[0], 5, dtype=np.int_, endpoint=False))
ax2.set_xticklabels(np.linspace(0, T, 5, dtype=np.int_, endpoint=False))
ax1.set_ylabel(r"$S_e$")
ax2.set_ylabel(r"$S_i$")
fig.tight_layout()
fig.show()
