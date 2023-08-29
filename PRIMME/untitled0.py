#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 20 15:19:50 2023

@author: joseph.melville
"""

# import PRIMME as fsp
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


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")





import numpy as np



tms = [1/100, 2.09, 5.38, 9.87, 15.98, 24.91, 50, 100, 150] #[90, 95, 100, 105, 110, 115, 120, 120, 120]
log = []
for tm in tms:
    tc = 25
    tr = tm/tc/2
    if tr>1: a = 1
    else: a = tr*(1-np.log(tr))
    th = np.arccos(-a/2)/np.pi*180
    print(th)
    log.append(th)

print(a)

.9164
.4765


50, 
71.1


#Train the network with different normalizations, check drifting and stats
#No normalization (definitely biased, drifting)
#Full standardization (stops drift, but biases against small grains)
#Full standardize just ouput (I don't see drifting, probably because accuracy dropped when it started to learn to drift, so none of those models were kept)
#Full standardize just label (no drift, and would probably get better with training, lots of noise though)
#sigmoid of output, not relu (no drift, lots of noise, probably better with more training)
#Center means (no scale)
#Normalize 0 to 1
#Set min to 0 (no scale)
#Change from max likelyhood to weighted distribution sample
#Rotate before finding indicies

#Train network
trainset_location = "./data/trainset_spparks_sz(257x257)_ng(256-256)_nsets(200)_future(4)_max(100)_kt(0.66)_cut(0).h5"
model_location = fsp.train_primme(trainset_location, num_eps=10000, obs_dim=17, act_dim=17, lr=5e-5, reg=1, pad_mode="circular", plot_freq=50, if_miso=False)

#Test drift
# model_location = "./data/model_dim(2)_sz(17_17)_lr(5e-05)_reg(1)_ep(3)_kt(0.66)_cut(25).h5"
ic, ea, _ = fs.voronoi2image(size=[512, 512], ngrain=512) #nsteps=500, pad_mode='circular'
fp_primme = fsp.run_primme(ic, ea, nsteps=100, modelname=model_location, pad_mode='circular', plot_freq=1, if_miso=False)
fs.make_videos(fp_primme)

#Test stats
# model_location = "./data/model_dim(2)_sz(17_17)_lr(5e-05)_reg(1)_ep(3)_kt(0.66)_cut(25).h5"
ic, ea, _ = fs.voronoi2image(size=[512, 512], ngrain=512) #nsteps=500, pad_mode='circular'
fp_primme = fs.run_primme(ic, ea, nsteps=3, modelname=model_location, pad_mode='circular', if_plot=False)
fs.compute_grain_stats(fp_primme)



# fp_primme = "./data/primme_sz(512x512)_ng(512)_nsteps(3)_freq(1)_kt(0.66)_cut(25).h5"
fs.compute_grain_stats(fp_primme)











#Check how many of the energies are actually reduced by a cutoff of 25

fp = './data/trainset_spparks_sz(257x257)_ng(256-256)_nsets(200)_future(4)_max(100)_kt(0.66)_cut(0).h5'
with h5py.File(fp, 'r') as f:
    print(f.keys())
    print(f['ims_id'].shape)
    im = torch.from_numpy(f['ims_id'][1,0,0])
    ma = torch.from_numpy(f['miso_array'][0])
mm = fs.miso_array_to_matrix(ma[None])

fn = fs.num_diff_neighbors(im[None,None], window_size=7)
fm = fs.neighborhood_miso(im[None,None], mm, window_size=7)

log = []
f0 = fs.neighborhood_miso_spparks(im[None,None], mm, cut=0, window_size=7) 
cuts = np.arange(0,365,5)
for i, c in enumerate(cuts): 
    f = fs.neighborhood_miso_spparks(im[None,None], mm, cut=c, window_size=7) 
    imd = f-f0
    # log.append(imd[imd!=0].abs().mean())
    log.append(imd.mean())
    # plt.imshow(imd[0,0], interpolation=None)
    # plt.axis('off')

plt.figure(figsize=[4,2], dpi=600)
plt.rcParams['font.size'] = 8
plt.plot(cuts, log)
plt.title('')
plt.xlabel('Cut off angle')
plt.ylabel('Mean $\Delta H_i$')
plt.show()


