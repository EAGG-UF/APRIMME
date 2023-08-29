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




ic, ea, _ = fs.voronoi2image(size=[514,514], ngrain=512)
modelname = './data/model_dim(2)_sz(17_17)_lr(5e-05)_reg(1)_ep(1000)_kt(0.66)_cut(0).h5'
modelname = './data/model_dim(2)_sz(17_17)_lr(5e-05)_reg(1)_ep(1000)_kt(0.66)_cut(0)_norm.h5'

fp = fsp.run_primme(ic, ea, nsteps=1000, modelname=modelname,  if_miso=False, plot_freq=20)
fs.make_videos([fp])



fp = './data/primme_sz(514x514)_ng(512)_nsteps(1000)_freq(1)_kt(0.66)_cut(0)_norm.h5'
fp = './data/primme_sz(512x512)_ng(512)_nsteps(1000)_freq(1)_kt(0.66)_cut(0)_keep.h5'


import imageio

with h5py.File(fp, 'r') as f:
    print(f.keys())
    ims = f['sim0/ims_id'][:50, 0]
    

plt.imshow(ims[1])

ims = (255/np.max(ims)*ims).astype(np.uint8)
imageio.mimsave('./plots/ims_id.gif', ims)




# TRAINING

trainset = './data/trainset_spparks_sz(257x257)_ng(256-256)_nsets(200)_future(4)_max(100)_kt(0.66)_cut(0).h5'
model_name = fsp.train_primme(trainset, num_eps=1000, obs_dim=17, act_dim=17, lr=5e-5, reg=1, if_miso=False, plot_freq=20)
    

trainset = './data/trainset_spparks_sz(257x257)_ng(256-256)_nsets(200)_future(4)_max(100)_kt(0.66)_cut(25).h5'
model_name = fsp.train_primme(trainset, num_eps=1000, obs_dim=17, act_dim=17, lr=5e-5, reg=1, if_miso=True, plot_freq=20)
    

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
fp2 = './data/spparks_sz(1024x1024)_ng(4096)_nsteps(1000)_freq(1.0)_kt(0.66)_cut(0).h5'
fp3 = './data/spparks_sz(1024x1024)_ng(4096)_nsteps(1000)_freq(1.0)_kt(0.66)_cut(25).h5'
fp4 = './data/spparks_sz(1024x1024)_ng(4096)_nsteps(1000)_freq(1.0)_kt(0.66)_cut(360).h5'

fpp = [fp0, fp1, fp2, fp3]

# fs.make_videos([fp0, fp1, fp2], gps='last')


#normalized r through time
plt.figure()
for fp in fpp:
    with h5py.File(fp, 'r') as f:
        tmp = list(f.keys())
        a = f['%s/grain_areas'%tmp[-1]][:]
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
for fp in fpp:
    with h5py.File(fp, 'r') as f:
        tmp = list(f.keys())
        a = f['%s/grain_areas'%tmp[-1]][:]

    n = (a!=0).sum(1)
    j = np.argmin(np.abs(n-2000))
    a_single = a[j]
    r = np.sqrt(a_single/np.pi)[a_single!=0]
    rn = r/np.mean(r)
    h, x_edges = np.histogram(rn, bins='auto', density=True)
    x = x_edges[:-1]+np.diff(x_edges)/2
    plt.plot(x, h)
plt.legend(['PRIMME (cut=0)','PRIMME (cut=25)','SPPARKS (cut=0)','SPPARKS (cut=25)'])
plt.show()


#avg number of sides through time
plt.figure()
for fp in fpp:
    with h5py.File(fp, 'r') as f:
        tmp = list(f.keys())
        s = f['%s/grain_sides'%tmp[-1]][:]
        sa = f['%s/grain_sides_avg'%tmp[-1]][:]
    n = (s!=0).sum(1)
    i = np.argmin(np.abs(n-200))
    t = np.arange(i)
    plt.plot(t, sa[:i])
plt.show()
    

#number of sides distribution
plt.figure()
for fp in fpp:
    with h5py.File(fp, 'r') as f:
        tmp = list(f.keys())
        s = f['%s/grain_sides'%tmp[-1]][:]
        sa = f['%s/grain_sides_avg'%tmp[-1]][:]
        
    s = s[j][s[j]!=0]
    bins = np.arange(1,20)-0.5
    h, x_edges = np.histogram(s, bins=bins, density=True)
    x = x_edges[:-1]+np.diff(x_edges)/2
    plt.plot(x, h)
plt.legend(['PRIMME (cut=0)','PRIMME (cut=25)','SPPARKS (cut=0)','SPPARKS (cut=25)'])
plt.show()
    

#mean miso
plt.figure()
for fp in fpp:
    with h5py.File(fp, 'r') as f:
        tmp = list(f.keys())
        msa = f['%s/ims_miso_avg'%tmp[-1]][:]
    plt.plot(msa)
plt.show()


#dihedral std
plt.figure()
for fp in fpp:
    with h5py.File(fp, 'r') as f:
        tmp = list(f.keys())
        da = f['%s/dihedral_std'%tmp[-1]][:]
    plt.plot(da)
plt.show()






with h5py.File(fpp[0], 'r') as f:
    tmp = list(f.keys())
    da0 = f['%s/dihedral_std'%tmp[-1]][:]
    
with h5py.File(fpp[1], 'r') as f:
    tmp = list(f.keys())
    da25 = f['%s/dihedral_std'%tmp[-1]][:]

(da25-da0).mean()



with h5py.File(fpp[0], 'r') as f:
    tmp = list(f.keys())
    msa0 = f['%s/ims_miso_avg'%tmp[-1]][:]
    
with h5py.File(fpp[1], 'r') as f:
    tmp = list(f.keys())
    msa25 = f['%s/ims_miso_avg'%tmp[-1]][:]

(msa25-msa0).mean()






#get spparks in here!!!
#actually, just analyze the data we trained with!!!!



#Show changes in all the statistics (show all plots for now)
#What changes for spparks, do the same changes happend for primme?

#meant that we updated the code, not changing overall approach

#possible - show likelyhood for iso and ani
#configuration study






