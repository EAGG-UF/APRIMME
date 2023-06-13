#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 13 11:18:08 2023

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




ic, ea, _ = fs.voronoi2image(size=[512, 512], ngrain=512) #nsteps=500, pad_mode='circular'


fs.run_spparks(ic, ea, nsteps=5, kt=0.66, cut=0.0, freq=(1,1), rseed=None, miso_array=None, which_sim='eng', num_processors=1, bcs=['p','p','p'], save_sim=True, del_sim=False, path_sim=None)





device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

fp = './data/trainset_spparks_sz(257x257)_ng(256-256)_nsets(200)_future(4)_max(100)_kt(0.66)_cut(0).h5'
fp = './data/trainset_spparks_sz(257x257)_ng(256-256)_nsets(200)_future(4)_max(100)_kt(0.66)_cut(25).h5'
fp = './data/trainset_spparks_sz(257x257)_ng(256-256)_nsets(200)_future(4)_max(100)_kt(0.66)_cut(360).h5'
with h5py.File(fp, 'r') as f:
    ims_id = torch.from_numpy(f['ims_id'][:]).to(device)
    miso_array = torch.from_numpy(f['miso_array'][:]).to(device)
    euler_angles = torch.from_numpy(f['euler_angles'][:]).to(device)
               

miso_mats = fs.miso_array_to_matrix(miso_array)


log = []
for i in tqdm(range(ims_id.shape[0])):
    log0 = []
    for j in range(ims_id.shape[1]):
        ims = ims_id[i,j][None]
        miso_matrices = miso_mats[i][None]
        im_miso = fs.neighborhood_miso_spparks(ims, miso_matrices, cut=360)
        log0.append(im_miso[0])
    log.append(torch.stack(log0))
    
ims_miso0 = torch.stack(log)
ims_miso25 = torch.stack(log)
ims_miso360 = torch.stack(log)





plt.imshow(ims_id[0,0,0].cpu(), interpolation=None)
plt.imshow(ims_miso0[0,0,0].cpu(), interpolation=None)
plt.imshow(ims_miso25[0,0,0].cpu(), interpolation=None)
plt.imshow(ims_miso360[0,0,0].cpu(), interpolation=None)
plt.imshow(ims_miso25[0,0,0].cpu()-ims_miso0[0,0,0].cpu(), interpolation=None)


ims = ims_id.reshape(1000,1,257,257)
log = []
for im in tqdm(ims): 
    da_std = fs.find_dihedral_stats(im[None], if_plot=False)
    log.append(da_std)

da_stds0 = torch.stack(log)
da_stds25 = torch.stack(log)

plt.plot(da_stds0.cpu())
plt.plot(da_stds25.cpu())


# training set statistics



# neighborhood miso


# dihedral std










