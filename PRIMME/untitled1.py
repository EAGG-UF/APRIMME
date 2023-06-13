#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 12 11:59:47 2023

@author: joseph.melville
"""

#train an isotropic and anisotropic model
#run and calculate statistics for for both


import PRIMME as fsp
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






# TRAINING

trainset = './data/trainset_spparks_sz(257x257)_ng(256-256)_nsets(200)_future(4)_max(100)_kt(0.66)_cut(0).h5'
model_name = fsp.train_primme(trainset, num_eps=1000, obs_dim=17, act_dim=17, lr=5e-5, reg=0, if_miso=False, plot_freq=10)
    
trainset = './data/trainset_spparks_sz(257x257)_ng(256-256)_nsets(200)_future(4)_max(100)_kt(0.66)_cut(25).h5'
model_name = fsp.train_primme(trainset, num_eps=1000, obs_dim=17, act_dim=17, lr=5e-5, reg=1, if_miso=True, plot_freq=10)
    

# SIMULATIONS
# ic, ea, _ = fs.voronoi2image([1024, 1024], 4096)
# ma = fs.find_misorientation(ea, mem_max=10)

fp = './data/primme_sz(1024x1024)_ng(4096)_nsteps(500)_freq(1)_kt(0.66)_cut(0).h5'
with h5py.File('./data/primme_sz(1024x1024)_ng(4096)_nsteps(500)_freq(1)_kt(0.66)_cut(0).h5', 'r') as f:
    ic = torch.from_numpy(f['sim0/ims_id'][0,0].astype(float))
    ea = torch.from_numpy(f['sim0/euler_angles'][:].astype(float))
    ma = f['sim0/miso_array'][:].astype(float)

modelname = './data/model_dim(2)_sz(17_17)_lr(5e-05)_reg(1)_ep(1000)_kt(0.66)_cut(0).h5'
fp = fsp.run_primme(ic, ea, nsteps=500, modelname=modelname, miso_array=ma, if_miso=False, plot_freq=20)
fp = './data/primme_sz(1024x1024)_ng(4096)_nsteps(500)_freq(1)_kt(0.66)_cut(0).h5'
fs.compute_grain_stats(fp, n=50)

modelname = './data/model_dim(2)_sz(17_17)_lr(5e-05)_reg(1)_ep(1000)_kt(0.66)_cut(25).h5'
fp = fsp.run_primme(ic, ea, nsteps=500, modelname=modelname, miso_array=ma, if_miso=True, plot_freq=20)
fs.compute_grain_stats(fp, n=50)





# PLOTS
fp0 = './data/primme_sz(1024x1024)_ng(4096)_nsteps(500)_freq(1)_kt(0.66)_cut(0).h5'
fp1 = './data/primme_sz(1024x1024)_ng(4096)_nsteps(500)_freq(1)_kt(0.66)_cut(25).h5'


fs.make_videos([fp0, fp1], gps='last')


#ra2 plot and dist
for fp in [fp0, fp1]:
    with h5py.File(fp, 'r') as f:
        tmp = list(f.keys())
        # tmp = ['sim0']
        a = f['%s/grain_areas'%tmp[-1]][:]
    
    #normalized r through time
    plt.figure()
    n = (a!=0).sum(1)
    i = np.argmin(np.abs(n-200))
    r = np.sqrt(a/np.pi)
    ra = r.sum(1)/n #find mean without zeros
    ra2 = (ra**2)[:i] #square after the mean
    p = np.polyfit(np.arange(i), ra2, deg=1)[0]
    t = np.arange(i)
    plt.plot(t, ra2)
    plt.show()
    
    #mr2 distribution
    plt.figure()
    n = (a!=0).sum(1)
    j = np.argmin(np.abs(n-2000))
    a_single = a[j]
    r = np.sqrt(a_single/np.pi)[a_single!=0]
    rn = r/np.mean(r)
    h, x_edges = np.histogram(rn, bins='auto', density=True)
    x = x_edges[:-1]+np.diff(x_edges)/2
    plt.plot(x, h)
    plt.show()


#num sides, plot and dist
for fp in [fp0, fp1]:
    with h5py.File(fp, 'r') as f:
        tmp = list(f.keys())
        # tmp = ['sim0']
        s = f['%s/grain_sides'%tmp[-1]][:]
        sa = f['%s/grain_sides_avg'%tmp[-1]][:]
    
    #avg number of sides through time
    plt.figure()
    plt.plot(t, sa[:i])
    plt.show()
    
    #number of sides distribution
    plt.figure()
    s = s[j][s[j]!=0]
    bins = np.arange(1,20)-0.5
    h, x_edges = np.histogram(s, bins=bins, density=True)
    x = x_edges[:-1]+np.diff(x_edges)/2
    plt.plot(x, h)
    plt.show()



#miso and dihedral
with h5py.File(fp0, 'r') as f:
    tmp = list(f.keys())
    # tmp = ['sim0']
    msa0 = f['%s/ims_miso_avg'%tmp[-1]][:]
    das0 = f['%s/dihedral_std'%tmp[-1]][:]

with h5py.File(fp1, 'r') as f:
    tmp = list(f.keys())
    # tmp = ['sim0']
    msa25=  f['%s/ims_miso_avg'%tmp[-1]][:]
    da025 = f['%s/dihedral_std'%tmp[-1]][:]

#mean miso
plt.figure()
plt.plot(msa0)
plt.plot(msa25)
plt.show()

#dihedral std
plt.figure()
plt.plot(das0)
plt.plot(da025)
plt.show()



#get spparks in here!!!
#actually, just analyze the data we trained with!!!!



#Show changes in all the statistics (show all plots for now)
#What changes for spparks, do the same changes happend for primme?

#meant that we updated the code, not changing overall approach

#possible - show likelyhood for iso and ani
#configuration study






