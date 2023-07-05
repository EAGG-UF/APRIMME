#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul  3 10:36:57 2023

@author: joseph.melville
"""


import functions as fs
import h5py 
import numpy as np
import matplotlib.pyplot as plt
import scipy.io
import pandas
# from scipy.stats import multivariate_normal
# import scipy as sp
import shutil
import torch
# import scipy.io
from tqdm import tqdm

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


import time


aaa = torch.rand(10000,10000).to(device)

t0 = time.time()
# bbb, ccc = torch.mode(aaa, dim=1)
ddd = fs.find_mode(aaa, miso_matrix=None, cut=0)
t1 = time.time()
print(t1-t0)


