#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 29 16:26:49 2023

@author: joseph.melville
"""



import PRIMME as fsp
import functions as fs
import h5py 
import numpy as np
import matplotlib.pyplot as plt
import scipy.io
# import pandas
# from scipy.stats import multivariate_normal
# import scipy as sp
# import shutil
import torch
# import scipy.io
# from tqdm import tqdm
from matplotlib.patches import Rectangle
from tqdm import tqdm


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


trainset = './data/trainset_spparks_sz(64x64x64)_ng(1024-1024)_nsets(200)_future(4)_max(10)_kt(0.66)_freq(0.1)_cut(0).h5'
modelname = fsp.train_primme(trainset, num_eps=1000, obs_dim=17, act_dim=17, lr=5e-5, reg=1, if_miso=False, plot_freq=1)
   

ic, ea, _ = fs.voronoi2image(size=[256,256,256], ngrain=2**12)
ma = fs.find_misorientation(ea, mem_max=1) 

modelname = './data/model_dim(3)_sz(17_17)_lr(5e-05)_reg(1)_ep(1000)_kt(0.66)_freq(0.1)_cut(0).h5'
fp = fsp.run_primme(ic, ea, nsteps=1000, modelname=modelname, miso_array=ma, pad_mode='circular', if_miso=False, plot_freq=1)
fs.compute_grain_stats(fp)





