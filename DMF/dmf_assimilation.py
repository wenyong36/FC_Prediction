# -*- coding: utf-8 -*-
# @Time : 2022/12/21 15:26
# @Author : wy36
# @File : simulation.py


from DMF.DMFNetwork import DMFNetwork
from DMF.hemodynamic_cuda import hemodynamic
import numpy as np
import torch
from model import parameter
import matplotlib.pyplot as plt


class DMFAssimilation:
    def __init__(self, w_ie, c_ij, **kwargs):
        self.dmfNetwork = DMFNetwork(w_ie, c_ij, **kwargs)
        self.bold = hemodynamic(**kwargs)

    def initialize(self):
        pass