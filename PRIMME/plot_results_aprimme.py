#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 13 11:18:08 2023

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



# model_dim(2)_sz(17_17)_lr(5e-05)_reg(1)_ep(1000)_kt(0.66)_cut(50)_4.h5 #True miso 

trainset = './data/trainset_spparks_sz(257x257)_ng(256-256)_nsets(200)_future(4)_max(100)_kt(0.66)_cut(0).h5'
modelname = fsp.train_primme(trainset, num_eps=1000, obs_dim=17, act_dim=17, lr=5e-5, reg=1, if_miso=False, plot_freq=20)
     
trainset = './data/trainset_spparks_sz(257x257)_ng(256-256)_nsets(200)_future(4)_max(100)_kt(0.66)_cut(25).h5'
modelname = fsp.train_primme(trainset, num_eps=1000, obs_dim=17, act_dim=17, lr=5e-5, reg=1, if_miso=True, plot_freq=20)




fp = '../../MF/data/spparks_sz(2400x2400)_ng(20000)_nsteps(1600)_freq(1.0)_kt(0.66)_cut(0).h5'
with h5py.File(fp, 'r') as f: 
    ic = f['sim0/ims_id'][0,0].astype(float)
    ea = f['sim0/euler_angles'][:]
    ma = f['sim0/miso_array'][:]

modelname = './data/model_dim(2)_sz(17_17)_lr(5e-05)_reg(1)_ep(1000)_kt(0.66)_cut(0).h5'
fp = fsp.run_primme(ic, ea, nsteps=1000, modelname=modelname, miso_array=ma, pad_mode='circular', if_miso=True, plot_freq=50)
fp = './data/primme_sz(2400x2400)_ng(20000)_nsteps(1000)_freq(1)_kt(0.66)_cut(0).h5'
# fs.compute_grain_stats(fp)

modelname = './data/model_dim(2)_sz(17_17)_lr(5e-05)_reg(1)_ep(1000)_kt(0.66)_cut(25).h5'
fp = fsp.run_primme(ic, ea, nsteps=1000, modelname=modelname, miso_array=ma, pad_mode='circular', if_miso=True, plot_freq=50)
fp = './data/primme_sz(2400x2400)_ng(20000)_nsteps(1000)_freq(1)_kt(0.66)_cut(25).h5'
fs.compute_grain_stats(fp)

modelname = './data/model_dim(2)_sz(17_17)_lr(5e-05)_reg(1)_ep(1000)_kt(0.66)_cut(50).h5'
fp = fsp.run_primme(ic, ea, nsteps=1000, modelname=modelname, miso_array=ma, pad_mode='circular', if_miso=True, plot_freq=50)
fp = './data/primme_sz(2400x2400)_ng(20000)_nsteps(1000)_freq(1)_kt(0.66)_cut(50).h5'
fs.compute_grain_stats(fp)





#I want to choose the model that has the best statistical performance
#I am particularly concerned about:
#   Mean radius square curve - linear even from the beginning
#   Normalized radius distribution - no dip for small grains
#   Triple grain dihedral angles that go up with misorientiton - ideally matching theory
#   Dihedral angles drop slower for anisotropic

#The models I have
#   Reulstd
#   Sig
#   Signoise
#   Signoisestd




fp = './data/spparks_sz(1024x1024)_ng(4096)_nsteps(1000)_freq(1.0)_kt(0.66)_cut(0).h5'
with h5py.File(fp, 'r') as f:
    grain_areas = f['sim0/grain_areas'][:]
ng = (grain_areas!=0).sum(1)
s0 = 0#np.argmin(np.abs(ng-5000))
si = np.argmin(np.abs(ng-1500))
grain_radii = np.sqrt(grain_areas/np.pi)
grain_radii_avg = grain_radii.sum(1)/ng #find mean without zeros
r2 = (grain_radii_avg**2)[s0:si] #square after the mean
p_m = np.polyfit(np.arange(si-s0), r2, deg=1)[0]
# p_mcp = np.sum(np.linalg.pinv(np.arange(si-s0)[:,None])*(r2-r2[0]))
t_m = np.arange(si-s0)
ng_m = ng[s0:si]
r2_m = r2




fps = [
       './data/primme_sz(1024x1024)_ng(4096)_nsteps(500)_freq(1)_kt(0.66)_cut(0)_relustd.h5',
       './data/primme_sz(1024x1024)_ng(4096)_nsteps(500)_freq(1)_kt(0.66)_cut(50)_relustd.h5',
       './data/primme_sz(1024x1024)_ng(4096)_nsteps(500)_freq(1)_kt(0.66)_cut(0)_sig.h5',
       './data/primme_sz(1024x1024)_ng(4096)_nsteps(500)_freq(1)_kt(0.66)_cut(50)_sig.h5',
       './data/primme_sz(1024x1024)_ng(4096)_nsteps(500)_freq(1)_kt(0.66)_cut(0)_signoise.h5',
       './data/primme_sz(1024x1024)_ng(4096)_nsteps(500)_freq(1)_kt(0.66)_cut(50)_signoise.h5',
       './data/primme_sz(1024x1024)_ng(4096)_nsteps(500)_freq(1)_kt(0.66)_cut(0)_signoisestd.h5',
       './data/primme_sz(1024x1024)_ng(4096)_nsteps(500)_freq(1)_kt(0.66)_cut(50)_signoisestd.h5'
       ]

log = []

for i, fp in enumerate(fps):
    
    
    with h5py.File(fp, 'r') as f:
        print(f.keys())
        grain_areas = f['sim0/grain_areas'][:]
    ng = (grain_areas!=0).sum(1)
    s0 = 0#np.argmin(np.abs(ng-5000))
    si = np.argmin(np.abs(ng-1500))
    grain_radii = np.sqrt(grain_areas/np.pi)
    grain_radii_avg = grain_radii.sum(1)/ng #find mean without zeros
    r2 = (grain_radii_avg**2)[s0:si] #square after the mean
    p = np.polyfit(np.arange(si-s0), r2, deg=1)[0]
    # p = np.sum(np.linalg.pinv(np.arange(si-s0)[:,None])*(r2-r2[0]))
    scale = p/p_m
    t = np.arange(si-s0)*scale
    ng_p = ng[s0:si]
    r2_p = r2
    
    log.append([t, r2])
    
    if i%2==0: plt.plot(t, r2*1e-12, 'C%d'%np.floor(i/2))
    elif i%2==1: plt.plot(t, r2*1e-12, '--C%d'%np.floor(i/2))

plt.legend(['relustd (cut=0)', 'relustd (cut=50)', 
            'sig (cut=0)', 'sig (cut=50)',
            'signoise (cut=0)', 'signoise (cut=50)',
            'signoisestd (cut=0)', 'signoisestd (cut=50)'])
plt.ylabel('$<R>^2$ ($m^2$)')
plt.xlabel('Time (unitless')














trainset = './data/trainset_spparks_sz(257x257)_ng(256-256)_nsets(200)_future(4)_max(100)_kt(0.66)_cut(0).h5'
modelname = fsp.train_primme(trainset, num_eps=1000, obs_dim=17, act_dim=17, lr=5e-5, reg=1, if_miso=False, plot_freq=20)


fp = './data/spparks_sz(1024x1024)_ng(4096)_nsteps(1000)_freq(1.0)_kt(0.66)_cut(0).h5'
with h5py.File(fp, 'r') as f: 
    ic = f['sim0/ims_id'][0,0].astype(float)
    ea = f['sim0/euler_angles'][:]
    ma = f['sim0/miso_array'][:]
   
modelname = './data/model_dim(2)_sz(17_17)_lr(5e-05)_reg(1)_ep(1000)_kt(0.66)_cut(100)_signoise.h5'
fp = fsp.run_primme(ic, ea, nsteps=500, modelname=modelname, miso_array=ma, pad_mode='circular', if_miso=True, plot_freq=50)
fs.compute_grain_stats(fp)



trainset = './data/trainset_spparks_sz(257x257)_ng(256-256)_nsets(200)_future(4)_max(100)_kt(0.66)_cut(100).h5'
modelname = fsp.train_primme(trainset, num_eps=1000, obs_dim=17, act_dim=17, lr=5e-5, reg=1, if_miso=True, plot_freq=20)


fp = './data/spparks_sz(1024x1024)_ng(4096)_nsteps(1000)_freq(1.0)_kt(0.66)_cut(0).h5'
with h5py.File(fp, 'r') as f: 
    ic = f['sim0/ims_id'][0,0].astype(float)
    ea = f['sim0/euler_angles'][:]
    ma = f['sim0/miso_array'][:]
    
modelname = './data/model_dim(2)_sz(17_17)_lr(5e-05)_reg(1)_ep(1000)_kt(0.66)_cut(100).h5'
fp = fsp.run_primme(ic, ea, nsteps=500, modelname=modelname, miso_array=ma, pad_mode='circular', if_miso=True, plot_freq=50)
fs.compute_grain_stats(fp)










#FIGURE OUT HOW TO RUN DIHEDRALS TOMORROW






# modelname = './data/model_dim(2)_sz(17_17)_lr(5e-05)_reg(1)_ep(1000)_kt(0.66)_cut(50)_4.h5'
modelname = './data/model_dim(2)_sz(17_17)_lr(5e-05)_reg(1)_ep(1000)_kt(0.66)_cut(50)_tmpname.h5'

ic, ea = fs.generate_3grainIC(size=[256,256], h=150)


tms = [1/100, 2.09, 5.38, 9.87, 15.98, 24.91, 50, 100, 150, 200] 
theory = [90.05 , 95, 100, 105, 110, 115, 120, 120, 120]

log2 = []

for tm in tms: 
    ma = np.array([50, tm ,50])*np.pi/180/2
    fp = fsp.run_primme(ic, ea, nsteps=1000, modelname=modelname, miso_array=ma, pad_mode='clip', if_miso=True, plot_freq=999)
    with h5py.File(fp, 'r') as f: 
        k = len(f.keys())-1
        ims = f['sim%d/ims_id'%k][:]
    
    log = []
    i = np.linspace(0,len(ims)-1,100).astype(int)
    for im in tqdm(torch.from_numpy(ims[i]).to(device)): 
        aaa = fs.find_dihedral_angles(im[None], if_plot=False, num_plot_jct=10, pad_mode='reflect')
        if aaa is None: th = np.nan
        else: th = (aaa[-2:,-1]).mean().cpu().numpy()
        log.append(th)
    
    tmp = np.array(log)
    log2.append(tmp)
    plt.plot(np.stack(log2).T)
    plt.legend([str(t) for t in theory[:len(log2)]])
    plt.title('Three grain predicted and actuall angles')
    plt.xlabel('Steps/10')
    plt.ylabel('Angle (degrees)')
    plt.show()
    
    np.save('./data/three_grain_dihedrals_sigmoid50', np.stack(log2).T)

#RECREATE the relu, norm model (without added noise) and see if it gives the same results as before


fp = './data/spparks_sz(256x256)_ng(3)_nsteps(1000)_freq(1.0)_kt(0.66)_cut(50).h5'
with h5py.File(fp, 'r') as f: 
    k = len(f.keys())

log2 = []
for i in range(k):
    with h5py.File(fp, 'r') as f: 
        ims = f['sim%d/ims_id'%i][:]
    
    log = []
    i = np.linspace(0,len(ims)-1,100).astype(int)
    for im in tqdm(torch.from_numpy(ims[i]).to(device)): 
        aaa = fs.find_dihedral_angles(im[None], if_plot=False, num_plot_jct=10, pad_mode='reflect')
        if aaa is None: th = np.nan
        else: th = (aaa[-2:,-1]).mean().cpu().numpy()
        log.append(th)
    
    tmp = np.array(log)
    log2.append(tmp)
    plt.plot(np.stack(log2).T)
    plt.legend([str(t) for t in theory[:len(log2)]])
    plt.title('Three grain predicted and actual angles')
    plt.xlabel('Steps/10')
    plt.ylabel('Angle (degrees)')
    plt.show()



ims = np.load('./data/spparks_expected.npy')[:,None,:,:]

log = []
for im in tqdm(torch.from_numpy(ims).to(device)): 
    aaa = fs.find_dihedral_angles(im[None], if_plot=False, num_plot_jct=10, pad_mode='reflect')
    if aaa is None: th = np.nan
    else: th = (aaa[-2:,-1]).mean().cpu().numpy()
    log.append(th)

tmp = np.array(log)
plt.plot(tmp)
plt.legend(['105'])
plt.title('Three grain predicted and actual angles')
plt.xlabel('Steps')
plt.ylabel('Angle (degrees)')
plt.show()







modelname = './data/model_dim(2)_sz(17_17)_lr(5e-05)_reg(1)_ep(1000)_kt(0.66)_cut(200)_3.h5'
ic, ea = fs.generate_3grainIC(size=[256,256], h=200)
ic = np.flipud(ic).copy()

ma = np.array([1,1/1000,1])*200/180*np.pi
fp = fsp.run_primme(ic, ea, nsteps=1000, modelname=modelname, miso_array=ma, pad_mode='clip', if_miso=True, plot_freq=1)
with h5py.File(fp, 'r') as f: ims0 = f['sim0/ims_id'][:]





### Train PRIMME to not drift
# Find a training method doesn't drift or have noise
#1 - normal (isotropic)
#2 - add noise to output ~ U[0,1e-9] (isotropic)
#3 - #2 but Trained with spparks cut=200 (anisotropic), cut=50 also (two different ones)
#4 - #3 but sigmoid on output instead of relu

trainset = './data/trainset_spparks_sz(257x257)_ng(256-256)_nsets(200)_future(4)_max(100)_kt(0.66)_cut(50).h5'
modelname = fsp.train_primme(trainset, num_eps=1000, obs_dim=17, act_dim=17, lr=5e-5, reg=1, if_miso=True, plot_freq=20)
    
modelname = './data/model_dim(2)_sz(17_17)_lr(5e-05)_reg(1)_ep(1000)_kt(0.66)_cut(200)_4.h5'

ic, ea = fs.generate_circleIC(size=[258,258], r=100)
fp = fsp.run_primme(ic, ea, nsteps=500, modelname=modelname, pad_mode='circular', if_miso=False, plot_freq=20)

ic, ea = fs.generate_3grainIC(size=[258,258], h=150)
fp = fsp.run_primme(ic, ea, nsteps=500, modelname=modelname, pad_mode='clip', if_miso=False, plot_freq=20)

ic, ea, _ = fs.voronoi2image(size=[258,258], ngrain=512)
fp = fsp.run_primme(ic, ea, nsteps=500, modelname=modelname, pad_mode='circular', if_miso=False, plot_freq=20)


fp = '../../MF/data/spparks_sz(2400x2400)_ng(20000)_nsteps(1600)_freq(1.0)_kt(0.66)_cut(0).h5'
with h5py.File(fp, 'r') as f:
    ic = f['sim0/ims_id'][0,0].astype(float)
    ea = f['sim0/euler_angles'][:]
    ma = f['sim0/miso_array'][:]


fp = fsp.run_primme(ic, ea, nsteps=1000, modelname=modelname, miso_array=ma, pad_mode='circular', if_miso=False, plot_freq=20)
fs.compute_grain_stats(fp)


fp = './data/primme_sz(2400x2400)_ng(20000)_nsteps(1000)_freq(1)_kt(0.66)_cut(0)_1.h5'






fp = './data/spparks_sz(256x256)_ng(3)_nsteps(1000)_freq(1.0)_kt(0.66)_cut(50).h5'
with h5py.File(fp, 'r') as f: 
    ims0 = f['sim0/ims_id'][:]
    ims05 = f['sim1/ims_id'][:]
    ims1 = f['sim2/ims_id'][:]


plt.imshow(ims1[-1,0])



### Three grain dihedral angle

modelname = './data/model_dim(2)_sz(17_17)_lr(5e-05)_reg(1)_ep(1000)_kt(0.66)_cut(50)_4.h5'
ic, ea = fs.generate_3grainIC(size=[256,256], h=150)

ma = np.array([1,1/1000,1])*50/180*np.pi
fp = fsp.run_primme(ic, ea, nsteps=1000, modelname=modelname, miso_array=ma, pad_mode='clip', if_miso=True, plot_freq=50)
with h5py.File(fp, 'r') as f: print(f.keys()); ims0 = f['sim3/ims_id'][:]

# ma = np.array([1,1/2,1])*50/180*np.pi
tms = [1/100, 2.09, 5.38, 9.87, 15.98, 24.91, 50, 100, 150] 
#theory - [90,05 , 95, 100, 105, 110, 115, 120, 120, 120]
#experiment - [90.2, 90.6, 90.9, 91, 91, 111.9]
ma = np.array([50, 50 ,50])*np.pi/180
fp = fsp.run_primme(ic, ea, nsteps=1000, modelname=modelname, miso_array=ma, pad_mode='clip', if_miso=True, plot_freq=50)
with h5py.File(fp, 'r') as f: print(f.keys()); ims05 = f['sim16/ims_id'][:]

ma = np.array([1,1,1])*50/180*np.pi
fp = fsp.run_primme(ic, ea, nsteps=1000, modelname=modelname, miso_array=ma, pad_mode='clip', if_miso=True, plot_freq=50)
with h5py.File(fp, 'r') as f: ims1 = f['sim5/ims_id'][:]


#Show last image and find dihedral angles 
ims = ims05
plt.imshow(ims[-1,0]); plt.show()
log = []
i = np.linspace(0,len(ims)-1,100).astype(int)
for im in tqdm(torch.from_numpy(ims[i]).to(device)): 
    aaa = fs.find_dihedral_angles(im[None], if_plot=False, num_plot_jct=10, pad_mode='reflect')
    th = (aaa[-2:,-1]).mean()
    log.append(th)
plt.plot(torch.stack(log).cpu())
plt.show()


aaa = torch.stack(log).cpu()
plt.plot(aaa[aaa<140])
torch.mean(aaa[aaa<140])

#Plot
alphas = ['1',r'$\dfrac{1}{2}$',r'$\dfrac{1}{1000}$']
ims_list = [ims1, ims05, ims0]
frames = [1,200,400,800]

plt.figure(figsize=[6.5,3], dpi=600)
plt.rcParams['font.size'] = 8
for i in range(len(ims_list)):
    for j in range(len(frames)):
    
        plt.subplot(3,4,1+j+i*4)
        plt.imshow(ims_list[i][frames[j],0], interpolation='none')
        plt.tick_params(bottom=False, left=False,labelleft=False, labelbottom=False)
        
        if i==0: plt.title('Time step %d'%frames[j], fontsize=8)
        if j==0: plt.ylabel(r'MF: $\alpha$ = %s'%alphas[i], fontsize=8)
    
# plt.savefig('/blue/joel.harley/joseph.melville/tmp/mode_case2.png', bbox_inches='tight', dpi=600)
plt.show()







### Anisotropic metrics vs cut off angle

cs = [0,25,50,75,100,125,150,175,200]
log_ds = []
log_ms = []
legend = []
for c in cs: 
    fp = './data/primme_sz(1024x1024)_ng(4096)_nsteps(1000)_freq(1)_kt(0.66)_cut(%d).h5'%c
    with h5py.File(fp, 'r') as f:
        ds = f['sim0/dihedral_std'][:]; log_ds.append(ds[:,0])
        ms = f['sim0/ims_miso_avg'][:]; log_ms.append(ms)
        legend.append('Cut off = %d'%c)
        plt.plot(ms)

plt.xlabel('Frame')
# plt.title('Dihedral Angles STD')
plt.title('Mean Misorientation (without zeros)')
plt.legend(legend)
plt.show()

log = []
for i in range(len(log_ds)):
    log.append((log_ds[i]-log_ds[0]).mean())
plt.plot(cs,log)
plt.xlabel('Cut off angle (degrees)')
plt.ylabel('Mean difference from cut off of zero')
plt.title('Dihedral Angles STD')
plt.show()

log = []
for i in range(len(log_ms)):
    log.append((log_ms[i]-log_ms[0]).mean())
plt.plot(cs,log)
plt.xlabel('Cut off angle (degrees)')
plt.ylabel('Mean difference from cut off of zero')
plt.title('Mean Misorientation (without zeros)')
plt.show()




### Create a figure comparing number of neighbors and misorientation

fp = './data/trainset_spparks_sz(257x257)_ng(256-256)_nsets(200)_future(4)_max(100)_kt(0.66)_cut(0).h5'
with h5py.File(fp, 'r') as f:
    print(f.keys())
    print(f['miso_array'].shape)
    im = torch.from_numpy(f['ims_id'][0,0,0])
    ma = torch.from_numpy(f['miso_array'][0])
mm = fs.miso_array_to_matrix(ma[None])

f0 = fs.num_diff_neighbors(im[None,None], window_size=7)
f1 = fs.neighborhood_miso(im[None,None], mm, window_size=7)

plt.figure(figsize=[5,2], dpi=600)
plt.rcParams['font.size'] = 8

plt.subplot(1,3,1)
plt.imshow(im, interpolation=None)
plt.tick_params(bottom=False, left=False,labelleft=False, labelbottom=False)
plt.title('Microstructure')

plt.subplot(1,3,2)
plt.imshow(f0[0,0], interpolation=None)
plt.tick_params(bottom=False, left=False,labelleft=False, labelbottom=False)
plt.title('Number of\nDifferent Neighbors')

plt.subplot(1,3,3)
plt.imshow(f1[0,0], interpolation=None)
plt.tick_params(bottom=False, left=False,labelleft=False, labelbottom=False)
plt.title('Neighborhood\nMisorientation')

plt.tight_layout()
plt.savefig('/blue/joel.harley/joseph.melville/tmp_APRIMME/feature_comp.png', bbox_inches='tight', dpi=600)
plt.show()



### Create a figure that shows sample dihedral angles 

_ = fs.find_dihedral_angles_pretty_plot(im[None,None], if_plot=True, num_plot_jct=4)



















# tms = [1/100, 2.09, 5.38, 9.87, 15.98, 24.91, 50, 100, 150, 200] 
# theory = [90.05 , 95, 100, 105, 110, 115, 120, 120, 120]


w=2; h=2
sw=3; sh=2
if_leg = True

def export_legend(legend, filename="legend.png", expand=[-5,-5,5,5]):
    fig  = legend.figure
    fig.canvas.draw()
    bbox  = legend.get_window_extent()
    bbox = bbox.from_extents(*(bbox.extents + np.array(expand)))
    bbox = bbox.transformed(fig.dpi_scale_trans.inverted())
    fig.savefig(filename, dpi="figure", bbox_inches=bbox)
    
    


### Circle simulations #!!!

#Run - 
# ic, ea = fs.generate_circleIC(size=[512,512], r=200)
# ma = np.array([5])*np.pi/180

# modelname = './data/model_dim(2)_sz(17_17)_lr(5e-05)_reg(1)_ep(1000)_kt(0.66)_cut(50).h5'
# fp = fsp.run_primme(ic, ea, nsteps=3000, modelname=modelname, miso_array=ma, if_miso=True, plot_freq=50)

plt.figure(figsize=[6,2], dpi=600)
plt.rcParams['font.size'] = 8
fp = './data/primme_sz(512x512)_ng(2)_nsteps(3000)_freq(1)_kt(0.66)_cut(0)_reg(1).h5'
for i, nf in enumerate([0,1500,3000]):
    with h5py.File(fp, 'r') as f:
        im = f['sim0/ims_id'][nf,0]
        plt.subplot(1,3,i+1)
        plt.imshow(im, interpolation='none')
        plt.tick_params(bottom=False, left=False,labelleft=False, labelbottom=False)
        plt.xlabel('Frame: %d'%nf)
plt.tight_layout()
plt.savefig('/blue/joel.harley/joseph.melville/tmp_APRIMME/circles.png', bbox_inches='tight', dpi=600)
plt.show()

plt.figure(figsize=[6,2], dpi=600)
plt.rcParams['font.size'] = 8

plt.subplot(1,2,1)
for reg in [0,1,10]:
    fp = './data/primme_sz(512x512)_ng(2)_nsteps(3000)_freq(1)_kt(0.66)_cut(0)_reg(%d).h5'%reg
    with h5py.File(fp, 'r') as f:
        print(f.keys())
        ims = f['sim0/ims_id'][:]
    
    a = (ims==1).sum(1).sum(1).sum(1)
    plt.plot(a)  
plt.legend(['Reg=0', 'Reg=1', 'Reg=10'])
plt.ylabel('Circle Pixels Remaining')
plt.xlabel('Number of frames')

plt.subplot(1,2,2)
for ma in np.array([5, 20, 35, 50]): #[0.25, 0.5, 0.75, 1]:
    fp = './data/primme_sz(512x512)_ng(2)_nsteps(3000)_freq(1)_kt(0.66)_cut(50)_ma(%d).h5'%ma
    with h5py.File(fp, 'r') as f:
        print(f.keys())
        ims = f['sim0/ims_id'][:]
    
    a = (ims==1).sum(1).sum(1).sum(1)
    plt.plot(a)   
plt.legend(['Miso=5', 'Miso=20', 'Miso=35', 'Miso=50'])
plt.ylabel('Circle Pixels Remaining')
plt.xlabel('Number of frames')

plt.tight_layout()
plt.savefig('/blue/joel.harley/joseph.melville/tmp_APRIMME/circle_stats.png', bbox_inches='tight', dpi=600)
plt.show()
    
# plt.figure(figsize=[sw,sh], dpi=600)
# plt.rcParams['font.size'] = 8
# for reg in [0,1,10]:
#     fp = './data/primme_sz(512x512)_ng(2)_nsteps(3000)_freq(1)_kt(0.66)_cut(0)_reg(%d).h5'%reg
#     with h5py.File(fp, 'r') as f:
#         print(f.keys())
#         ims = f['sim0/ims_id'][:]
    
#     a = (ims==1).sum(1).sum(1).sum(1)
#     plt.plot(a)  
# plt.legend(['Reg=0', 'Reg=1', 'Reg=10'])
# plt.ylabel('Circle Pixels Remaining')
# plt.xlabel('Number of frames')
# plt.savefig('/blue/joel.harley/joseph.melville/tmp_APRIMME/circle_regs.png', bbox_inches='tight', dpi=600)
# plt.show()

# plt.figure(figsize=[sw,sh], dpi=600)
# plt.rcParams['font.size'] = 8
# for ma in [0.25, 0.5, 0.75, 1]:
#     fp = './data/primme_sz(512x512)_ng(2)_nsteps(3000)_freq(1)_kt(0.66)_cut(25)_ma(%1.2f).h5'%ma
#     with h5py.File(fp, 'r') as f:
#         print(f.keys())
#         ims = f['sim0/ims_id'][:]
    
#     a = (ims==1).sum(1).sum(1).sum(1)
#     plt.plot(a)   
# plt.legend(['Miso=0.25', 'Miso=0.5', 'Miso=0.75', 'Miso=1'])
# plt.ylabel('Circle Pixels Remaining')
# plt.xlabel('Number of frames')
# plt.savefig('/blue/joel.harley/joseph.melville/tmp_APRIMME/circle_mas.png', bbox_inches='tight', dpi=600)
# plt.show()


    


### Triple grain #!!!

# # Run
# ic, ea = fs.generate_3grainIC(size=[256,256], h=150)
# ma = np.array([50,50,10])/180*np.pi
# modelname = './data/model_dim(2)_sz(17_17)_lr(5e-05)_reg(1)_ep(1000)_kt(0.66)_cut(50).h5'
# fp = fsp.run_primme(ic, ea, nsteps=1000, modelname=modelname, miso_array=ma, if_miso=True, pad_mode=['clip','circular'], plot_freq=50)


# ic, ea = fs.generate_3grainIC(size=[512,512], h=350)

# #grain 0 (right) shrink
# ma = np.array([1,1,0.1]) #[0/1, 0/2, 1/2]
# modelname = './data/model_dim(2)_sz(17_17)_lr(5e-05)_reg(1)_ep(1000)_kt(0.66)_cut(25)_norm.h5'
# fp = fsp.run_primme(ic, ea, nsteps=3000, modelname=modelname, miso_array=ma, if_miso=True, pad_mode=['clip','circular'], plot_freq=50)

# #grain 2 (left) shrink
# ma = np.array([0.1,1,1]) #[0/1, 0/2, 1/2]
# modelname = './data/model_dim(2)_sz(17_17)_lr(5e-05)_reg(1)_ep(1000)_kt(0.66)_cut(25)_norm.h5'
# fp = fsp.run_primme(ic, ea, nsteps=3000, modelname=modelname, miso_array=ma, if_miso=True, pad_mode=['clip','circular'], plot_freq=50)

# #More configurations 

# #Train 25, Run 0 is isotropic (grain 0 (right) shrinks same rate)
# ma = np.array([1,1,0.1]) #[0/1, 0/2, 1/2]
# modelname = './data/model_dim(2)_sz(17_17)_lr(5e-05)_reg(1)_ep(1000)_kt(0.66)_cut(25)_norm.h5'
# fp = fsp.run_primme(ic, ea, nsteps=3000, modelname=modelname, miso_array=ma, if_miso=False, pad_mode=['clip','circular'], plot_freq=50)

# #Train 0, Run 0 is isotropic (grain 0 (right) shrinks same rate)
# ma = np.array([1,1,0.1]) #[0/1, 0/2, 1/2]
# modelname = './data/model_dim(2)_sz(17_17)_lr(5e-05)_reg(1)_ep(1000)_kt(0.66)_cut(0)_norm.h5'
# fp = fsp.run_primme(ic, ea, nsteps=3000, modelname=modelname, miso_array=ma, if_miso=False, pad_mode=['clip','circular'], plot_freq=50)

# #Train 0, Run 25 is anisotropic (grain 0 (right) shrink)
# ma = np.array([1,1,0.1]) #[0/1, 0/2, 1/2]
# modelname = './data/model_dim(2)_sz(17_17)_lr(5e-05)_reg(1)_ep(1000)_kt(0.66)_cut(0)_norm.h5'
# fp = fsp.run_primme(ic, ea, nsteps=3000, modelname=modelname, miso_array=ma, if_miso=True, pad_mode=['clip','circular'], plot_freq=50)

fp0 = './data/primme_sz(256x256)_ng(3)_nsteps(1000)_freq(1)_kt(0.66)_cut(50)_iso.h5'
fpr = './data/primme_sz(256x256)_ng(3)_nsteps(1000)_freq(1)_kt(0.66)_cut(50)_right.h5'
fpl = './data/primme_sz(256x256)_ng(3)_nsteps(1000)_freq(1)_kt(0.66)_cut(50)_left.h5'

plt.figure(figsize=[5,3.5], dpi=600)
plt.rcParams['font.size'] = 8
for i, fp in enumerate([fpl, fp0, fpr]):
    for j, nf in enumerate([500, 1000]):
        with h5py.File(fp, 'r') as f:
            im = f['sim0/ims_id'][nf,0]
            
            plt.subplot(2,3,i+1+j*3)
            plt.imshow(im, interpolation='none')
            plt.tick_params(bottom=False, left=False,labelleft=False, labelbottom=False)
            if j==1:
                if i==0: 
                    plt.xlabel('Left Shrink')
                    plt.ylabel('Frame: %d'%nf)
                elif i==1: plt.xlabel('Isotropic')
                elif i==2: plt.xlabel('Right Shrink')
            elif j==0: 
                if i==0: plt.ylabel('Frame: %d'%nf)
plt.tight_layout()
plt.savefig('/blue/joel.harley/joseph.melville/tmp_APRIMME/triple_config.png', bbox_inches='tight', dpi=600)
plt.show()



### Hex grains #!!!


# # Run
# ic, ea = fs.generate_hexIC()

# mm = torch.ones(64,64)*50
# mm[63,:] = 50
# mm[:,63] = 50
# mm[np.arange(64), np.arange(64)] = 0
# ma = fs.miso_matrix_to_array(mm).numpy()/180*np.pi

# modelname = './data/model_dim(2)_sz(17_17)_lr(5e-05)_reg(1)_ep(1000)_kt(0.66)_cut(50).h5'
# fp = fsp.run_primme(ic, ea, nsteps=1000, modelname=modelname, miso_array=ma, if_miso=True, pad_mode=['circular','circular'], plot_freq=50)


# ic, ea = fs.generate_hexIC()

# #isotropic
# ma = np.ones(2016)
# modelname = './data/model_dim(2)_sz(17_17)_lr(5e-05)_reg(1)_ep(1000)_kt(0.66)_cut(25)_keep.h5'
# fp = fsp.run_primme(ic, ea, nsteps=3000, modelname=modelname, miso_array=ma, if_miso=True, pad_mode='circular', plot_freq=50)

# #anisoptric
# ma = fs.find_misorientation(ea, mem_max=1) 
# modelname = './data/model_dim(2)_sz(17_17)_lr(5e-05)_reg(1)_ep(1000)_kt(0.66)_cut(25)_keep.h5'
# fp = fsp.run_primme(ic, ea, nsteps=3000, modelname=modelname, miso_array=ma, if_miso=True, pad_mode='circular', plot_freq=50)

# #grain grow
# mm = torch.ones(64,64)
# mm[19,:] = 0.1
# mm[:,19] = 0.1
# mm[np.arange(64), np.arange(64)] = 0
# ma = fs.miso_matrix_to_array(mm).numpy()
# modelname = './data/model_dim(2)_sz(17_17)_lr(5e-05)_reg(1)_ep(1000)_kt(0.66)_cut(25)_keep.h5'
# fp = fsp.run_primme(ic, ea, nsteps=3000, modelname=modelname, miso_array=ma, if_miso=True, pad_mode='circular', plot_freq=50)

# #grain shrink
# mm = torch.ones(64,64)*0.1
# mm[19,:] = 1
# mm[:,19] = 1
# mm[np.arange(64), np.arange(64)] = 0
# ma = fs.miso_matrix_to_array(mm).numpy()
# modelname = './data/model_dim(2)_sz(17_17)_lr(5e-05)_reg(1)_ep(1000)_kt(0.66)_cut(25)_keep.h5'
# fp = fsp.run_primme(ic, ea, nsteps=3000, modelname=modelname, miso_array=ma, if_miso=True, pad_mode='circular', plot_freq=50)

fp0 = './data/primme_sz(443x512)_ng(64)_nsteps(1000)_freq(1)_kt(0.66)_cut(50)_iso.h5'
fp1 = './data/primme_sz(443x512)_ng(64)_nsteps(1000)_freq(1)_kt(0.66)_cut(50)_shrink.h5'
fp2 = './data/primme_sz(443x512)_ng(64)_nsteps(1000)_freq(1)_kt(0.66)_cut(50)_grow.h5'

plt.figure(figsize=[6.5,2], dpi=600)
plt.rcParams['font.size'] = 8

plt.subplot(1,3,1)
with h5py.File(fp0, 'r') as f: im = f['sim0/ims_id'][1000,0]
plt.imshow(im, interpolation='none')
plt.tick_params(bottom=False, left=False,labelleft=False, labelbottom=False)
plt.xlabel('Isotropic')

# plt.subplot(1,3,2)
# im = f['sim1/ims_id'][3000,0]
# plt.imshow(im, interpolation='none')
# plt.tick_params(bottom=False, left=False,labelleft=False, labelbottom=False)
# plt.xlabel('Anisotropic')

plt.subplot(1,3,2)
with h5py.File(fp1, 'r') as f: im = f['sim0/ims_id'][120,0]
plt.imshow(im, interpolation='none')
plt.tick_params(bottom=False, left=False,labelleft=False, labelbottom=False)
plt.xlabel('One Grain\nLow Misorientation')
plt.gca().add_patch(Rectangle((115,90),150,150,linewidth=2,edgecolor='r',facecolor='none'))

plt.subplot(1,3,3)
with h5py.File(fp2, 'r') as f: im = f['sim0/ims_id'][120,0]
plt.imshow(im, interpolation='none')
plt.tick_params(bottom=False, left=False,labelleft=False, labelbottom=False)
plt.xlabel('One Grain\nHigh Misorientation')
plt.gca().add_patch(Rectangle((115,90),150,150,linewidth=2,edgecolor='r',facecolor='none'))

plt.savefig('/blue/joel.harley/joseph.melville/tmp_APRIMME/hex.png', bbox_inches='tight', dpi=600)
plt.show()
    
    

### 2D isotropic <R>^2 vs time, number of grains vs time (AP0, AP25, MCP) #!!!

fp = '../../MF/data/spparks_sz(2400x2400)_ng(20000)_nsteps(1600)_freq(1.0)_kt(0.66)_cut(0).h5'
with h5py.File(fp, 'r') as f:
    grain_areas = f['sim0/grain_areas'][:]
ng = (grain_areas!=0).sum(1)
s0 = 0#np.argmin(np.abs(ng-5000))
si = np.argmin(np.abs(ng-1500))
grain_radii = np.sqrt(grain_areas/np.pi)
grain_radii_avg = grain_radii.sum(1)/ng #find mean without zeros
r2 = (grain_radii_avg**2)[s0:si] #square after the mean
p_m = np.polyfit(np.arange(si-s0), r2, deg=1)[0]
# p_mcp = np.sum(np.linalg.pinv(np.arange(si-s0)[:,None])*(r2-r2[0]))
t_m = np.arange(si-s0)
ng_m = ng[s0:si]
r2_m = r2

fp = './data/spparks_sz(2400x2400)_ng(20000)_nsteps(2001)_freq(1.0)_kt(0.66)_cut(25).h5'
with h5py.File(fp, 'r') as f:
    grain_areas = f['sim0/grain_areas'][:]
ng = (grain_areas!=0).sum(1)
s0 = 0#np.argmin(np.abs(ng-5000))
si = np.argmin(np.abs(ng-1500))
grain_radii = np.sqrt(grain_areas/np.pi)
grain_radii_avg = grain_radii.sum(1)/ng #find mean without zeros
r2 = (grain_radii_avg**2)[s0:si] #square after the mean
p_ma = np.polyfit(np.arange(si-s0), r2, deg=1)[0]
# p_mcp = np.sum(np.linalg.pinv(np.arange(si-s0)[:,None])*(r2-r2[0]))
t_ma = np.arange(si-s0)
ng_ma = ng[s0:si]
r2_ma = r2

fp = './data/primme_sz(2400x2400)_ng(20000)_nsteps(1000)_freq(1)_kt(0.66)_cut(0)_signoise_miso.h5'
with h5py.File(fp, 'r') as f:
    print(f.keys())
    grain_areas = f['sim0/grain_areas'][:]
ng = (grain_areas!=0).sum(1)
s0 = 0#np.argmin(np.abs(ng-5000))
si = np.argmin(np.abs(ng-1500))
grain_radii = np.sqrt(grain_areas/np.pi)
grain_radii_avg = grain_radii.sum(1)/ng #find mean without zeros
r2 = (grain_radii_avg**2)[s0:si] #square after the mean
p = np.polyfit(np.arange(si-s0), r2, deg=1)[0]
# p = np.sum(np.linalg.pinv(np.arange(si-s0)[:,None])*(r2-r2[0]))
scale = p/p_m
t_p = np.arange(si-s0)*scale
ng_p = ng[s0:si]
r2_p = r2

fp = './data/primme_sz(2400x2400)_ng(20000)_nsteps(1000)_freq(1)_kt(0.66)_cut(25)_signoise.h5'
with h5py.File(fp, 'r') as f:
    print(f.keys())
    grain_areas = f['sim1/grain_areas'][:]
ng = (grain_areas!=0).sum(1)
s0 = 0#np.argmin(np.abs(ng-5000))
si = np.argmin(np.abs(ng-1500))
grain_radii = np.sqrt(grain_areas/np.pi)
grain_radii_avg = grain_radii.sum(1)/ng #find mean without zeros
r2 = (grain_radii_avg**2)[s0:si] #square after the mean
p = np.polyfit(np.arange(si-s0), r2, deg=1)[0]
# p = np.sum(np.linalg.pinv(np.arange(si-s0)[:,None])*(r2-r2[0]))
scale = p/p_m
t_pa = np.arange(si-s0)*scale
ng_pa = ng[s0:si]
r2_pa = r2

# plt.figure(figsize=[sw,sh], dpi=600)
# plt.rcParams['font.size'] = 8
# plt.plot(t_m, r2_m*1e-12, '-')
# plt.plot(t_ma, r2_ma*1e-12, '-')
# plt.plot(t_p, r2_p*1e-12, '--')
# plt.plot(t_pa, r2_pa*1e-12, '-.')
# plt.xlabel('Time (unitless)')
# plt.ylabel('$<R>^2$ ($m^2$)')
# if if_leg: plt.legend(['MCP (cut=0)','MCP (cut=25)','APRIMME (cut=0)','APRIMME (cut=25)'])
# plt.savefig('/blue/joel.harley/joseph.melville/tmp_APRIMME/2d_r2_vs_time.png', bbox_inches='tight', dpi=600)
# plt.show()

# plt.figure(figsize=[3,2], dpi=600)
# plt.rcParams['font.size'] = 8
# plt.plot(t_m, r2_m*1e-12, '-')
# plt.plot(t_ma, r2_ma*1e-12, '-')
# plt.plot(t_p*.93, r2_p*1e-12-.12e-9, '--')
# plt.plot(t_pa*.89 ,r2_pa*1e-12-.19e-9, '-.')
# plt.xlabel('Time (unitless)')
# plt.ylabel('$<R>^2$ ($m^2$)')
# if if_leg: plt.legend(['MCP (cut=0)','MCP (cut=25)','APRIMME (cut=0)','APRIMME (cut=25)'])
# plt.savefig('/blue/joel.harley/joseph.melville/tmp_APRIMME/2d_r2_vs_time2.png', bbox_inches='tight', dpi=600)
# plt.show()

plt.figure(figsize=[6,2], dpi=600)
plt.rcParams['font.size'] = 8
plt.subplot(1,3,1)
plt.plot(t_m, r2_m*1e-12, '-')
plt.plot(t_ma, r2_ma*1e-12, '-')
plt.plot(t_p, r2_p*1e-12, '--')
plt.plot(t_pa, r2_pa*1e-12, '-.')
plt.xlabel('Time (unitless)')
plt.ylabel('$<R>^2$ ($m^2$)')
plt.ylim([-1.08e-10, 1.06e-09])

plt.subplot(1,3,2)
# plt.plot(t_m, r2_m*1e-12, '-')
# plt.plot(t_ma, r2_ma*1e-12, '-')
# plt.plot(t_p*.93, r2_p*1e-12-.12e-9, '--')
# plt.plot(t_pa*.89 ,r2_pa*1e-12-.19e-9, '-.')
plt.plot(t_m, r2_m*1e-12, '-')
plt.plot(t_ma, r2_ma*1e-12, '-')
plt.plot(t_p, r2_p*1e-12-.10e-9, '--')
plt.plot(t_pa,r2_pa*1e-12-.10e-9, '-.')
plt.xlabel('Time (unitless)')
# plt.ylabel('$<R>^2$ ($m^2$)')
plt.ylim([-1.08e-10, 1.06e-09])
plt.tick_params(bottom=True, left=True,labelleft=False, labelbottom=True)

plt.subplot(1,3,3)
plt.axis('off')
plt.plot(0,0, '-'); plt.plot(0,0, '-'); plt.plot(0,0, '--'); plt.plot(0,0, '-.')
plt.legend(['MCP (cut=0)','MCP (cut=25)','APRIMME (cut=0)','APRIMME (cut=25)'], loc=10, fontsize=8)
plt.tight_layout()
plt.savefig('/blue/joel.harley/joseph.melville/tmp_APRIMME/r2_vs_time.png', bbox_inches='tight', dpi=600)
plt.show()


### Average number of sides through time (AP0, AP25, SPPARKS, PF) #!!!

# fp = './data/spparks_sz(1024x1024)_ng(4096)_nsteps(1000)_freq(1.0)_kt(0.66)_cut(0).h5'
fp = '../../MF/data/spparks_sz(2400x2400)_ng(20000)_nsteps(1600)_freq(1.0)_kt(0.66)_cut(0).h5'
with h5py.File(fp, 'r') as f:
    gsa_m = f['sim0/grain_sides_avg'][:500]

# fp = './data/spparks_sz(1024x1024)_ng(4096)_nsteps(1000)_freq(1.0)_kt(0.66)_cut(25).h5'
fp = './data/spparks_sz(2400x2400)_ng(20000)_nsteps(2001)_freq(1.0)_kt(0.66)_cut(25).h5'
with h5py.File(fp, 'r') as f:
    gsa_ma = f['sim0/grain_sides_avg'][:500]

# fp = './data/primme_sz(1024x1024)_ng(4096)_nsteps(500)_freq(1)_kt(0.66)_cut(0).h5'
fp = './data/primme_sz(2400x2400)_ng(20000)_nsteps(1000)_freq(1)_kt(0.66)_cut(0).h5'
# fp = fp = './data/primme_sz(2400x2400)_ng(20000)_nsteps(1000)_freq(1)_kt(0.66)_cut(0).h5'
with h5py.File(fp, 'r') as f:
    gsa_p = f['sim0/grain_sides_avg'][:500]

# fp = './data/primme_sz(1024x1024)_ng(4096)_nsteps(500)_freq(1)_kt(0.66)_cut(25).h5'
fp = './data/primme_sz(2400x2400)_ng(20000)_nsteps(1000)_freq(1)_kt(0.66)_cut(25).h5'
# fp = './data/primme_sz(2400x2400)_ng(20000)_nsteps(1000)_freq(1)_kt(0.66)_cut(25).h5'
with h5py.File(fp, 'r') as f:
    gsa_pa = f['sim1/grain_sides_avg'][:500]

plt.figure(figsize=[4,3], dpi=600)
plt.rcParams['font.size'] = 8
plt.plot(gsa_m, 'C0-', linewidth=2, zorder=10)
plt.plot(gsa_ma, 'C1-', linewidth=1, zorder=11)
plt.plot(gsa_p, 'C2--', linewidth=1, zorder=0)
plt.plot(gsa_pa, 'C3-.', linewidth=1, zorder=0)
plt.xlabel('Number of Frames')
plt.ylabel('Avg Number \nof Sides')
plt.legend(['MCP (cut=0)','MCP (cut=25)','APRIMME (cut=0)','APRIMME (cut=25)'])
plt.tight_layout()
plt.savefig('/blue/joel.harley/joseph.melville/tmp_APRIMME/sides_vs_time.png', bbox_inches='tight', dpi=600)
plt.show()

# plt.figure(figsize=[sw,sh], dpi=600)
# plt.rcParams['font.size'] = 8
# plt.plot(t_m, gsa_m[:len(t_m)], '-', linewidth=1, alpha=1)
# plt.plot(t_ma, gsa_ma[:len(t_ma)], '-', linewidth=1, alpha=1)
# plt.plot(t_p, gsa_p[:len(t_p)], '--', linewidth=1, alpha=1)
# plt.plot(t_pa, gsa_pa[:len(t_pa)], '-.', linewidth=1, alpha=1)
# plt.xlabel('Time (s)')
# plt.ylabel('Avg Number \nof Sides')
# if if_leg: plt.legend(['MCP (cut=0)','MCP (cut=25)','APRIMME (cut=0)','APRIMME (cut=25)'])
# plt.savefig('/blue/joel.harley/joseph.melville/tmp_APRIMME/2d_num_sides_vs_time.png', bbox_inches='tight', dpi=600)
# plt.show()



### 2D isotropic normalized radius distribution (AP0, AP25, MCP, PF, Yadav, Zollinger) #!!!

plt.figure(figsize=[6.5,6], dpi=600)
plt.rcParams['font.size'] = 8
for ii, num_grains in enumerate([4000, 3000, 2000]): #4000, 3000, 2000

    fp = '../../MF/data/spparks_sz(2400x2400)_ng(20000)_nsteps(1600)_freq(1.0)_kt(0.66)_cut(0).h5'
    with h5py.File(fp, 'r') as f:
        grain_areas = f['sim0/grain_areas'][:]
    n = (grain_areas!=0).sum(1)
    j = np.argmin(np.abs(n-num_grains))
    a = grain_areas[j]
    r = np.sqrt(a[a!=0]/np.pi)
    rn = r/np.mean(r)
    h_m, x_edges = np.histogram(rn, bins='auto', density=True)
    x_m = x_edges[:-1]+np.diff(x_edges)/2
    n_m = len(rn)
    
    fp = './data/spparks_sz(2400x2400)_ng(20000)_nsteps(2001)_freq(1.0)_kt(0.66)_cut(25).h5'
    with h5py.File(fp, 'r') as f:
        grain_areas = f['sim0/grain_areas'][:]
    n = (grain_areas!=0).sum(1)
    j = np.argmin(np.abs(n-num_grains))
    a = grain_areas[j]
    r = np.sqrt(a[a!=0]/np.pi)
    rn = r/np.mean(r)
    h_ma, x_edges = np.histogram(rn, bins='auto', density=True)
    x_ma = x_edges[:-1]+np.diff(x_edges)/2
    n_ma = len(rn)
    
    fp = './data/primme_sz(2400x2400)_ng(20000)_nsteps(1000)_freq(1)_kt(0.66)_cut(0).h5'
    # fp = './data/primme_sz(2400x2400)_ng(20000)_nsteps(1000)_freq(1)_kt(0.66)_cut(0).h5'
    with h5py.File(fp, 'r') as f:
        grain_areas = f['sim0/grain_areas'][:]
    n = (grain_areas!=0).sum(1)
    j = np.argmin(np.abs(n-num_grains))
    a = grain_areas[j]
    r = np.sqrt(a[a!=0]/np.pi)
    rn = r/np.mean(r)
    h_p, x_edges = np.histogram(rn, bins='auto', density=True)
    x_p = x_edges[:-1]+np.diff(x_edges)/2
    n_p = len(rn)
    
    fp = './data/primme_sz(2400x2400)_ng(20000)_nsteps(1000)_freq(1)_kt(0.66)_cut(25).h5'
    # fp = './data/primme_sz(2400x2400)_ng(20000)_nsteps(1000)_freq(1)_kt(0.66)_cut(25).h5'
    with h5py.File(fp, 'r') as f:
        grain_areas = f['sim1/grain_areas'][:]
    n = (grain_areas!=0).sum(1)
    j = np.argmin(np.abs(n-num_grains))
    a = grain_areas[j]
    r = np.sqrt(a[a!=0]/np.pi)
    rn = r/np.mean(r)
    h_pa, x_edges = np.histogram(rn, bins='auto', density=True)
    x_pa = x_edges[:-1]+np.diff(x_edges)/2
    n_pa = len(rn)
    
    mat = scipy.io.loadmat('../../MF/data/previous_figures/RadiusDistPureTP2DNew.mat')
    x_yad = mat['y1'][0]
    h_yad = mat['RtotalFHD5'][0]
    
    x_zol, h_zol = np.loadtxt('../../MF/data/previous_figures/Results.csv', delimiter=',',skiprows=1).T
    
    plt.subplot(4,3,ii+1)
    plt.plot(x_m, h_m, 'C0')
    plt.plot(x_ma, h_ma, 'C0--')
    plt.plot(x_p, h_p, 'C1')
    plt.plot(x_pa, h_pa, 'C1--')
    plt.plot(x_yad, h_yad, 'C2*', ms = 4)
    plt.plot(x_zol, h_zol, 'C3d', ms = 4)
    if ii==1: plt.xlabel('$R/<R>$ - Normalized Radius')
    if ii==0: plt.ylabel('Frequency')
    else: plt.tick_params(bottom=True, left=True,labelleft=False, labelbottom=True)
    plt.xlim([0,3])
    plt.ylim([0,1.2])
    plt.title('Grains: %d'%num_grains)

# plt.subplot(1,4,4)
# plt.axis('off')
# plt.plot(0,0, 'C0'); plt.plot(0,0, 'C0--'); plt.plot(0,0, 'C1'); plt.plot(0,0, 'C1--'); plt.plot(0,0, '*'); plt.plot(0,0, 'd')
# plt.legend(['MCP (cut=0)','MCP (cut=25)','APRIMME (cut=0)','APRIMME (cut=25)','Yadav 2018', 'Zollner 2016'], loc=10, framealpha=1)

# plt.savefig('/blue/joel.harley/joseph.melville/tmp_APRIMME/r2_dists.png', bbox_inches='tight', dpi=600)
# plt.show()



### 2D isotropic number of sides distribution (AP0, AP25, MCP, PF, Yadav, Mason) #!!!


plt.figure(figsize=[6.5,5], dpi=600)
plt.rcParams['font.size'] = 8
for ii, num_grains in enumerate([4000, 3000, 2000]): #4000, 3000, 2000
    
    fp = '../../MF/data/spparks_sz(2400x2400)_ng(20000)_nsteps(1600)_freq(1.0)_kt(0.66)_cut(0).h5'
    with h5py.File(fp, 'r') as f:
        grain_sides = f['sim0/grain_sides'][:]
    n = (grain_sides!=0).sum(1)
    i = np.argmin(np.abs(n-num_grains))
    s = grain_sides[i][grain_sides[i]!=0]
    bins = np.arange(1,20)-0.5
    h_m, x_edges = np.histogram(s, bins=bins, density=True)
    x_m = x_edges[:-1]+np.diff(x_edges)/2
    n_m = len(s)
    
    fp = './data/spparks_sz(2400x2400)_ng(20000)_nsteps(2001)_freq(1.0)_kt(0.66)_cut(25).h5'
    with h5py.File(fp, 'r') as f:
        grain_sides = f['sim0/grain_sides'][:]
    n = (grain_sides!=0).sum(1)
    i = np.argmin(np.abs(n-num_grains))
    s = grain_sides[i][grain_sides[i]!=0]
    bins = np.arange(1,20)-0.5
    h_ma, x_edges = np.histogram(s, bins=bins, density=True)
    x_ma = x_edges[:-1]+np.diff(x_edges)/2
    n_ma = len(s)
    
    fp = './data/primme_sz(2400x2400)_ng(20000)_nsteps(1000)_freq(1)_kt(0.66)_cut(0).h5'
    # fp = './data/primme_sz(2400x2400)_ng(20000)_nsteps(1000)_freq(1)_kt(0.66)_cut(0).h5'
    with h5py.File(fp, 'r') as f:
        grain_sides = f['sim0/grain_sides'][:]
    n = (grain_sides!=0).sum(1)
    i = np.argmin(np.abs(n-num_grains))
    s = grain_sides[i][grain_sides[i]!=0]
    bins = np.arange(1,20)-0.5
    h_p, x_edges = np.histogram(s, bins=bins, density=True)
    x_p = x_edges[:-1]+np.diff(x_edges)/2
    n_p = len(s)
    
    fp = './data/primme_sz(2400x2400)_ng(20000)_nsteps(1000)_freq(1)_kt(0.66)_cut(25).h5'
    # fp = './data/primme_sz(2400x2400)_ng(20000)_nsteps(1000)_freq(1)_kt(0.66)_cut(25).h5'
    with h5py.File(fp, 'r') as f:
        grain_sides = f['sim1/grain_sides'][:]
    n = (grain_sides!=0).sum(1)
    i = np.argmin(np.abs(n-num_grains))
    s = grain_sides[i][grain_sides[i]!=0]
    bins = np.arange(1,20)-0.5
    h_pa, x_edges = np.histogram(s, bins=bins, density=True)
    x_pa = x_edges[:-1]+np.diff(x_edges)/2
    n_pa = len(s)
    
    mat = scipy.io.loadmat('../../MF/data/previous_figures_sides/FaceDistPureTP2DNew.mat')
    x_yad = mat['y1'][0]
    h_yad = mat['FtotalFHD5'][0]
    
    x_mas, h_mas = np.loadtxt('../../MF/data/previous_figures_sides/ResultsMasonLazar2DGTD.txt').T
    
    plt.subplot(4,3,ii+4)
    plt.plot(x_m, h_m, 'C0')
    plt.plot(x_ma, h_ma, 'C0--')
    plt.plot(x_p, h_p, 'C1')
    plt.plot(x_pa, h_pa, 'C1--')
    plt.plot(x_yad, h_yad, 'C2*', ms = 4)
    plt.plot(x_mas, h_mas, 'C4^', ms = 4)
    if ii==1: plt.xlabel('Number of Sides')
    if ii==0: plt.ylabel('Frequency')
    else: plt.tick_params(bottom=True, left=True,labelleft=False, labelbottom=True)
    plt.xlim([0,15])
    plt.ylim([0,0.35])
    # plt.title('Grains: %d'%num_grains)
    
# plt.subplot(1,4,4)
# plt.axis('off')
# plt.plot(0,0, 'C0'); plt.plot(0,0, 'C0--'); plt.plot(0,0, 'C1'); plt.plot(0,0, 'C1--'); plt.plot(0,0, '*'); plt.plot(0,0, 'd')
# plt.legend(['MCP (cut=0)','MCP (cut=25)','APRIMME (cut=0)','APRIMME (cut=25)','Yadav 2018', 'Masson 2015'], loc=10, framealpha=1)

# plt.savefig('/blue/joel.harley/joseph.melville/tmp_APRIMME/num_sides_dists.png', bbox_inches='tight', dpi=600)
# plt.show()



### Compare dihedral angle distributions at different numbers of grains #!!!

bins = np.linspace(40,200,50)
x = bins[:-1]+np.diff(bins)/2

# plt.figure(figsize=[6.5,1.5], dpi=600)
# plt.rcParams['font.size'] = 8
for ii, num_grains in enumerate([4000, 3000, 2000]): #4000, 3000, 2000

    fp = '../../MF/data/spparks_sz(2400x2400)_ng(20000)_nsteps(1600)_freq(1.0)_kt(0.66)_cut(0).h5'
    with h5py.File(fp, 'r') as f:
        grain_areas = f['sim0/grain_areas'][:]
        n = (grain_areas!=0).sum(1)
        j = np.argmin(np.abs(n-num_grains))
        im = torch.from_numpy(f['sim0/ims_id'][j, 0].astype(float)).to(device)[None,]
    _, da = fs.find_dihedral_stats2(im[:, None], if_plot=False)
    h_m, _ = np.histogram(da[0].flatten().cpu(), bins=bins, density=True)

    fp = './data/spparks_sz(2400x2400)_ng(20000)_nsteps(2001)_freq(1.0)_kt(0.66)_cut(25).h5'
    with h5py.File(fp, 'r') as f:
        grain_areas = f['sim0/grain_areas'][:]
        n = (grain_areas!=0).sum(1)
        j = np.argmin(np.abs(n-num_grains))
        im = torch.from_numpy(f['sim0/ims_id'][j, 0].astype(float)).to(device)[None,]
    _, da = fs.find_dihedral_stats2(im[:, None], if_plot=False)
    h_ma, _ = np.histogram(da[0].flatten().cpu(), bins=bins, density=True)
    
    fp = './data/primme_sz(2400x2400)_ng(20000)_nsteps(1000)_freq(1)_kt(0.66)_cut(0).h5'
    # fp = './data/primme_sz(2400x2400)_ng(20000)_nsteps(1000)_freq(1)_kt(0.66)_cut(0).h5'
    with h5py.File(fp, 'r') as f:
        grain_areas = f['sim0/grain_areas'][:]
        n = (grain_areas!=0).sum(1)
        j = np.argmin(np.abs(n-num_grains))
        im = torch.from_numpy(f['sim0/ims_id'][j, 0].astype(float)).to(device)[None,]
    _, da = fs.find_dihedral_stats2(im[:, None], if_plot=False)
    h_p, _ = np.histogram(da[0].flatten().cpu(), bins=bins, density=True)
    
    fp = './data/primme_sz(2400x2400)_ng(20000)_nsteps(1000)_freq(1)_kt(0.66)_cut(25).h5'
    # fp = './data/primme_sz(2400x2400)_ng(20000)_nsteps(1000)_freq(1)_kt(0.66)_cut(25).h5'
    with h5py.File(fp, 'r') as f:
        grain_areas = f['sim1/grain_areas'][:]
        n = (grain_areas!=0).sum(1)
        j = np.argmin(np.abs(n-num_grains))
        im = torch.from_numpy(f['sim1/ims_id'][j, 0].astype(float)).to(device)[None,]
    _, da = fs.find_dihedral_stats2(im[:, None], if_plot=False)
    h_pa, _ = np.histogram(da[0].flatten().cpu(), bins=bins, density=True)
    
    
    plt.subplot(4,3,ii+7)
    plt.plot(x, h_m, 'C0')
    plt.plot(x, h_ma, 'C0--')
    plt.plot(x, h_p, 'C1')
    plt.plot(x, h_pa, 'C1--')
    if ii==1: plt.xlabel('Dihedral Angle Distribution')
    if ii==0: plt.ylabel('Frequency')
    else: plt.tick_params(bottom=True, left=True,labelleft=False, labelbottom=True)
    plt.ylim([0,0.03])
    # plt.title('Grains: %d'%num_grains)
    
# plt.subplot(1,4,4)
# plt.axis('off')
# plt.plot(0,0, 'C0'); plt.plot(0,0, 'C0--'); plt.plot(0,0, 'C1'); plt.plot(0,0, 'C1--'); plt.plot(0,0, '*'); plt.plot(0,0, 'd')
# plt.legend(['MCP (cut=0)','MCP (cut=25)','APRIMME (cut=0)','APRIMME (cut=25)'], loc=10, framealpha=1)

# plt.savefig('/blue/joel.harley/joseph.melville/tmp_APRIMME/da_dists.png', bbox_inches='tight', dpi=600)
# plt.show()

# plt.figure(figsize=[6.5,2], dpi=600)
# plt.rcParams['font.size'] = 8
# plt.subplot(4,3,11)
# plt.axis('off')
# plt.plot(0,0, 'C0'); plt.plot(0,0, 'C0--'); plt.plot(0,0, 'C1'); 
# plt.plot(0,0, 'C1--'); plt.plot(0,0, 'C2*'); plt.plot(0,0, 'C3d'); plt.plot(0,0, 'C4^')
# plt.legend(['MCP (cut=0)','MCP (cut=25)','APRIMME (cut=0)','APRIMME (cut=25)', 'Yadav 2018', 'Zollner 2016', 'Masson 2015'], loc=10, framealpha=1)

plt.tight_layout()
plt.savefig('/blue/joel.harley/joseph.melville/tmp_APRIMME/dists.png', bbox_inches='tight', dpi=600)
plt.show()
    


### Microstructure comparisons (MF, MCP, PF) #!!!

# # RUN
# fp = '../../MF/data/mf_sz(512x512)_ng(512)_nsteps(1000)_cov(25)_numnei(64)_cut(0).h5'
# with h5py.File(fp, 'r') as f:
#     ic = torch.from_numpy(f['sim0/ims_id'][0,0].astype(float))-1
#     ea = torch.from_numpy(f['sim0/euler_angles'][:].astype(float))
#     ma = f['sim0/miso_array'][:].astype(float)

# modelname = './data/model_dim(2)_sz(17_17)_lr(5e-05)_reg(1)_ep(1000)_kt(0.66)_cut(0).h5'
# fp = fsp.run_primme(ic, ea, nsteps=1000, modelname=modelname, miso_array=ma, if_miso=True, plot_freq=50)

# modelname = './data/model_dim(2)_sz(17_17)_lr(5e-05)_reg(1)_ep(1000)_kt(0.66)_cut(25).h5'
# fp = fsp.run_primme(ic, ea, nsteps=1000, modelname=modelname, miso_array=ma, if_miso=True, plot_freq=50)


num_grains = [512, 300, 150, 50]

with h5py.File('../../MF/data/32c512grs512stsPkT066_img.hdf5', 'r') as f:
    ims_m = f['images'][:]
ng_m = np.array([len(np.unique(im))for im in ims_m])

with h5py.File('./data/spparks_sz(512x512)_ng(512)_nsteps(2000)_freq(1.0)_kt(0.66)_cut(25).h5', 'r') as f:
    ims_ma = f['sim0/ims_id'][:]
ng_ma = np.array([len(np.unique(im))for im in ims_ma])

with h5py.File('./data/primme_sz(512x512)_ng(512)_nsteps(1000)_freq(1)_kt(0.66)_cut(0).h5', 'r') as f:
    ims_p = f['sim0/ims_id'][:]
ng_p = np.array([len(np.unique(im))for im in ims_p])

with h5py.File('./data/primme_sz(512x512)_ng(512)_nsteps(1000)_freq(1)_kt(0.66)_cut(25).h5', 'r') as f:
    ims_pa = f['sim0/ims_id'][:]
ng_pa = np.array([len(np.unique(im))for im in ims_pa])


plt.figure(figsize=[6,6], dpi=600)
plt.rcParams['font.size'] = 8
for i in range(len(num_grains)):
    
    j = np.argmin(np.abs(ng_m-num_grains[i]))
    im_m = ims_m[j,]
    
    j = np.argmin(np.abs(ng_ma-num_grains[i]))
    im_ma = ims_ma[j,0]
    
    j = np.argmin(np.abs(ng_p-num_grains[i]))
    im_p = ims_p[j,0]
    
    j = np.argmin(np.abs(ng_pa-num_grains[i]))
    im_pa = ims_pa[j,0]
    
    plt.subplot(4,4,1+i)
    plt.title('$N_G$=%d'%num_grains[i], fontsize=8)
    plt.imshow(im_m, interpolation='none')
    plt.tick_params(bottom=False, left=False,labelleft=False, labelbottom=False)
    if i==0: plt.ylabel('MCP (cut=0)', fontsize=8)
    
    plt.subplot(4,4,5+i)
    plt.imshow(im_ma, interpolation='none')
    plt.tick_params(bottom=False, left=False,labelleft=False, labelbottom=False)
    if i==0: plt.ylabel('MCP (cut=25)', fontsize=8)
    
    plt.subplot(4,4,9+i)
    plt.imshow(im_p, interpolation='none')
    plt.tick_params(bottom=False, left=False,labelleft=False, labelbottom=False)
    if i==0: plt.ylabel('APRIMME (cut=0)', fontsize=8)
    
    plt.subplot(4,4,13+i)
    plt.imshow(im_pa, interpolation='none')
    plt.tick_params(bottom=False, left=False,labelleft=False, labelbottom=False)
    if i==0: plt.ylabel('APRIMME (cut=25)', fontsize=8)
    
plt.tight_layout()
plt.savefig('/blue/joel.harley/joseph.melville/tmp_APRIMME/2d_comp.png', bbox_inches='tight', dpi=600)
plt.show()



### Dihedral STD difference through time  #!!!

fp = '../../MF/data/spparks_sz(2400x2400)_ng(20000)_nsteps(1600)_freq(1.0)_kt(0.66)_cut(0).h5'
with h5py.File(fp, 'r') as f:
    print(f['sim0'].keys())
    das_m = f['sim0/dihedral_std'][:1000,0]
    msa_m = f['sim0/ims_miso_avg'][:1000]
    ng_m = (f['sim0/grain_areas'][:1000]!=0).sum(1)
    
fp = './data/spparks_sz(2400x2400)_ng(20000)_nsteps(2001)_freq(1.0)_kt(0.66)_cut(25).h5'
with h5py.File(fp, 'r') as f:
    das_ma = f['sim0/dihedral_std'][:1000,0]
    msa_ma = f['sim0/ims_miso_avg'][:1000]
    ng_ma = (f['sim0/grain_areas'][:1000]!=0).sum(1)

fp = './data/primme_sz(2400x2400)_ng(20000)_nsteps(1000)_freq(1)_kt(0.66)_(T0R0)_cut(0).h5'
# fp = './data/primme_sz(2400x2400)_ng(20000)_nsteps(1000)_freq(1)_kt(0.66)_cut(0).h5'
with h5py.File(fp, 'r') as f:
    das_p = f['sim0/dihedral_std'][:1000,0]
    msa_p = f['sim0/ims_miso_avg'][:1000]
    ng_p = (f['sim0/grain_areas'][:1000]!=0).sum(1)

fp = './data/primme_sz(2400x2400)_ng(20000)_nsteps(1000)_freq(1)_kt(0.66)_(T25R25)_cut(25).h5'
# fp = './data/primme_sz(2400x2400)_ng(20000)_nsteps(1000)_freq(1)_kt(0.66)_cut(25).h5'
with h5py.File(fp, 'r') as f:  
    das_pa = f['sim0/dihedral_std'][:1000,0]
    msa_pa = f['sim0/ims_miso_avg'][:1000]
    ng_pa = (f['sim0/grain_areas'][:1000]!=0).sum(1)





### Average miso difference through time  #!!!


# # Run
fp = './data/spparks_sz(1024x1024)_ng(4096)_nsteps(1000)_freq(1.0)_kt(0.66)_cut(0).h5'
with h5py.File(fp, 'r') as f: 
    ic = f['sim0/ims_id'][0,0].astype(float)
    ea = f['sim0/euler_angles'][:]
    ma = f['sim0/miso_array'][:]

# modelname = './data/model_dim(2)_sz(17_17)_lr(5e-05)_reg(1)_ep(1000)_kt(0.66)_cut(0).h5'
# fp = fsp.run_primme(ic, ea, nsteps=500, modelname=modelname, miso_array=ma, pad_mode='circular', if_miso=True, plot_freq=50)
# fs.compute_grain_stats(fp)

modelname = './data/model_dim(2)_sz(17_17)_lr(5e-05)_reg(1)_ep(1000)_kt(0.66)_cut(0)_relu.h5'
fp = fsp.run_primme(ic, ea, nsteps=500, modelname=modelname, miso_array=ma, pad_mode='circular', if_miso=False, plot_freq=50)
fs.compute_grain_stats(fp)

modelname = './data/model_dim(2)_sz(17_17)_lr(5e-05)_reg(1)_ep(1000)_kt(0.66)_cut(25)_relu.h5'
fp = fsp.run_primme(ic, ea, nsteps=500, modelname=modelname, miso_array=ma, pad_mode='circular', if_miso=True, plot_freq=50)
fs.compute_grain_stats(fp)

#RENAME models and sims to "relu", then see if they are back to normal
#TRY a new model with sigmoid, but no random choices, or with standardization








fp = './data/spparks_sz(1024x1024)_ng(4096)_nsteps(1000)_freq(1.0)_kt(0.66)_cut(0).h5'
with h5py.File(fp, 'r') as f:
    das_m = f['sim0/dihedral_std'][:500]
    msa_m = f['sim0/ims_miso_avg'][:500]
    ng_m = (f['sim0/grain_areas'][:]!=0).sum(1)

fp = './data/spparks_sz(1024x1024)_ng(4096)_nsteps(1000)_freq(1.0)_kt(0.66)_cut(25).h5'
with h5py.File(fp, 'r') as f:
    das_ma = f['sim0/dihedral_std'][:500]
    msa_ma = f['sim0/ims_miso_avg'][:500]
    ng_ma = (f['sim0/grain_areas'][:]!=0).sum(1)

fp = './data/primme_sz(1024x1024)_ng(4096)_nsteps(500)_freq(1)_kt(0.66)_cut(0)_relu.h5'
with h5py.File(fp, 'r') as f:
    das_p = f['sim0/dihedral_std'][:500]
    msa_p = f['sim0/ims_miso_avg'][:500]
    ng_p = (f['sim0/grain_areas'][:]!=0).sum(1)

fp = './data/primme_sz(1024x1024)_ng(4096)_nsteps(500)_freq(1)_kt(0.66)_cut(25)_relu.h5'
with h5py.File(fp, 'r') as f:  
    das_pa = f['sim0/dihedral_std'][:500]
    msa_pa = f['sim0/ims_miso_avg'][:500]
    ng_pa = (f['sim0/grain_areas'][:]!=0).sum(1)




plt.figure(figsize=[6.5,4], dpi=600)
plt.rcParams['font.size'] = 8

plt.subplot(2,2,1)
plt.plot(das_m, 'C0-')
plt.plot(das_ma, 'C1--')
plt.plot(das_p, 'C2-')
plt.plot(das_pa, 'C3--')
plt.title('Dihedral Angles')
plt.ylabel('STD')
plt.tick_params(bottom=True, left=False,labelleft=True, labelbottom=False)
plt.legend(['MCP (cut=0)','MCP (cut=25)','APRIMME (cut=0)','APRIMME (cut=25)'], loc='best')

plt.subplot(2,2,3)
plt.plot(das_ma-das_m, 'C0.', ms=3)
plt.plot(das_pa-das_p, 'C1^', ms=3)
plt.plot([0,500],[(das_ma-das_m).mean(),]*2, 'k--',linewidth=2)
plt.plot([0,500],[(das_pa-das_p).mean(),]*2, 'k--',linewidth=2)
plt.ylabel('Difference in STD')
plt.xlabel('Number of Frames')
# plt.ylim([-5,8])
plt.legend(['MCP - Mean: %1.2f'%((das_ma-das_m).mean()), 'APRIMME - Mean: %1.2f'%((das_pa-das_p).mean())], fontsize=7, loc='best')

# np.min([np.min(msa_m), np.min(msa_ma), np.min(msa_p), np.min(msa_pa)])
# np.max([np.max(msa_m), np.max(msa_ma), np.max(msa_p), np.max(msa_pa)])

plt.subplot(2,2,2)
plt.plot(msa_m, 'C0-')
plt.plot(msa_ma, 'C1--')
plt.plot(msa_p, 'C2-')
plt.plot(msa_pa, 'C3--')
plt.title('Neighborhood Misorientation')
plt.ylabel('Mean without Zeros')
# plt.ylim([1.55, 1.90])
plt.tick_params(bottom=True, left=False,labelleft=True, labelbottom=False)
plt.legend(['MCP (cut=0)','MCP (cut=25)','APRIMME (cut=0)','APRIMME (cut=25)'], loc='best')

plt.subplot(2,2,4)
plt.plot(msa_ma-msa_m, 'C0.', ms=3)
plt.plot(msa_pa-msa_p, 'C1^', ms=3)
plt.plot([0,500],[(msa_ma-msa_m).mean(),]*2, 'k--',linewidth=2)
plt.plot([0,500],[(msa_pa-msa_p).mean(),]*2, 'k--',linewidth=2)
plt.ylabel('Difference in Mean')
plt.xlabel('Number of Frames')
plt.legend(['MCP - Mean: %1.2f'%((msa_ma-msa_m).mean()), 'APRIMME - Mean: %1.2f'%((msa_pa-msa_p).mean())], fontsize=7, loc='best')

plt.tight_layout()
plt.savefig('/blue/joel.harley/joseph.melville/tmp_APRIMME/da_and_miso_differences.png', bbox_inches='tight', dpi=600)
plt.show()






fp = './data/primme_sz(1024x1024)_ng(4096)_nsteps(500)_freq(1)_kt(0.66)_cut(0).h5'
with h5py.File(fp, 'r') as f:
    das0 = f['sim0/dihedral_std'][:500]
    msa0 = f['sim0/ims_miso_avg'][:500]
    ng0 = (f['sim0/grain_areas'][:]!=0).sum(1)

fp = './data/primme_sz(1024x1024)_ng(4096)_nsteps(500)_freq(1)_kt(0.66)_cut(0)_miso.h5'
with h5py.File(fp, 'r') as f:
    das1 = f['sim0/dihedral_std'][:500]
    msa1 = f['sim0/ims_miso_avg'][:500]
    ng1 = (f['sim0/grain_areas'][:]!=0).sum(1)

fp = './data/primme_sz(1024x1024)_ng(4096)_nsteps(500)_freq(1)_kt(0.66)_cut(25)_miso.h5'
with h5py.File(fp, 'r') as f:
    das2 = f['sim0/dihedral_std'][:500]
    msa2 = f['sim0/ims_miso_avg'][:500]
    ng2 = (f['sim0/grain_areas'][:]!=0).sum(1)

fp = './data/primme_sz(1024x1024)_ng(4096)_nsteps(500)_freq(1)_kt(0.66)_cut(50)_miso.h5'
with h5py.File(fp, 'r') as f:
    das3 = f['sim0/dihedral_std'][:500]
    msa3 = f['sim0/ims_miso_avg'][:500]
    ng3 = (f['sim0/grain_areas'][:]!=0).sum(1)


plt.plot(das0)
# plt.plot(das1)
plt.plot(das2)
plt.plot(das3)

np.mean(das0)
np.mean(das1)
np.mean(das2)
np.mean(das3)



### 2D Dihedral STD through time (iso vs ani) 

# n0 = 5000
# n1 = 2000

# fp = './data/primme_sz(2400x2400)_ng(20000)_nsteps(1000)_freq(1)_kt(0.66)_(T0R0)_cut(0).h5'
# with h5py.File(fp, 'r') as f:
#     grain_areas = f['sim0/grain_areas'][:]
#     n = (grain_areas!=0).sum(1)
#     j0 = np.argmin(np.abs(n-n0))
#     j1 = np.argmin(np.abs(n-n1))
#     log = []
#     for i in tqdm(range(j0,j1+1)): 
#         ims = torch.from_numpy(f['sim0/ims_id'][i, 0].astype(float)).to(device)[None,]
#         da_std, _ = fs.find_dihedral_stats(ims[:, None], if_plot=False)
#         log.append(da_std)
# da_stds = torch.stack(log)[:,0]
# np.save('./data/aprimme0_da_std_ng5000_2000', da_stds.cpu().numpy())
# da_std_p = np.load('./data/aprimme0_da_std_ng5000_2000.npy')

# fp = './data/primme_sz(2400x2400)_ng(20000)_nsteps(1000)_freq(1)_kt(0.66)_(T25R25)_cut(25).h5'
# with h5py.File(fp, 'r') as f:
#     grain_areas = f['sim0/grain_areas'][:]
#     n = (grain_areas!=0).sum(1)
#     j0 = np.argmin(np.abs(n-n0))
#     j1 = np.argmin(np.abs(n-n1))
#     log = []
#     for i in tqdm(range(j0,j1+1)): 
#         ims = torch.from_numpy(f['sim0/ims_id'][i, 0].astype(float)).to(device)[None,]
#         da_std, _ = fs.find_dihedral_stats(ims[:, None], if_plot=False)
#         log.append(da_std)
# da_stds = torch.stack(log)[:,0]
# np.save('./data/aprimme25_da_std_ng5000_2000', da_stds.cpu().numpy())
# da_std_pa = np.load('./data/aprimme25_da_std_ng5000_2000.npy')

# fp = '../../MF/data/spparks_sz(2400x2400)_ng(20000)_nsteps(1600)_freq(1.0)_kt(0.66)_cut(0).h5'
# with h5py.File(fp, 'r') as f:
#     grain_areas = f['sim0/grain_areas'][:]
#     n = (grain_areas!=0).sum(1)
#     j0 = np.argmin(np.abs(n-n0))
#     j1 = np.argmin(np.abs(n-n1))
#     log = []
#     for i in tqdm(range(j0,j1+1)): 
#         ims = torch.from_numpy(f['sim0/ims_id'][i, 0].astype(float)).to(device)[None,]
#         da_std, _ = fs.find_dihedral_stats(ims[:, None], if_plot=False)
#         log.append(da_std)
# da_stds = torch.stack(log)[:,0]
# np.save('./data/mcp0_da_std_ng5000_2000', da_stds.cpu().numpy())
# da_std_m = np.load('./data/mcp0_da_std_ng5000_2000.npy')



# plt.plot(da_std_p)
# plt.plot(da_std_pa)
# plt.plot(da_std_m)



# ### Average miso through time  

# with h5py.File('./data/primme_sz(1024x1024)_ng(4096)_nsteps(500)_freq(1)_kt(0.66)_cut(0).h5', 'r') as f:
#     print(f['sim0'].keys())
#     das_p = f['sim0/dihedral_std'][:]
#     msa_p = f['sim0/ims_miso_avg'][:]
#     ng_p = (f['sim0/grain_areas'][:]!=0).sum(1)

# with h5py.File('./data/primme_sz(1024x1024)_ng(4096)_nsteps(500)_freq(1)_kt(0.66)_cut(25).h5', 'r') as f:
#     print(f['sim0'].keys())
#     das_pa = f['sim0/dihedral_std'][:]
#     msa_pa = f['sim0/ims_miso_avg'][:]
#     ng_pa = (f['sim0/grain_areas'][:]!=0).sum(1)

# fp = './data/spparks_sz(1024x1024)_ng(4096)_nsteps(1000)_freq(1.0)_kt(0.66)_cut(0).h5'
# with h5py.File(fp, 'r') as f:
#     print(f['sim0'].keys())
#     das_m = f['sim0/dihedral_std'][:]
#     msa_m = f['sim0/ims_miso_avg'][:]
#     ng_m = (f['sim0/grain_areas'][:]!=0).sum(1)
    
# fp = './data/spparks_sz(1024x1024)_ng(4096)_nsteps(1000)_freq(1.0)_kt(0.66)_cut(25).h5'
# with h5py.File(fp, 'r') as f:
#     print(f['sim0'].keys())
#     das_ma = f['sim0/dihedral_std'][:]
#     msa_ma = f['sim0/ims_miso_avg'][:]
#     ng_ma = (f['sim0/grain_areas'][:]!=0).sum(1)

# # plt.figure(figsize=[sw,sh], dpi=600)
# plt.figure(dpi=600)
# plt.rcParams['font.size'] = 8
# plt.plot(ng_p, msa_p)
# plt.plot(ng_pa, msa_pa,'--',linewidth=1)
# plt.plot(ng_m, msa_m)
# plt.plot(ng_ma, msa_ma,'--',linewidth=1)
# plt.xlim([4096, 0])
# plt.title('')
# plt.xlabel('Number of grains')
# plt.ylabel('Avg Boundary \nMisorientation')
# plt.legend(['APRIMME (Cut=0)','APRIMME (Cut=25)', 'MCP (Cut=0)','MCP (Cut=25)'])
# plt.savefig('/blue/joel.harley/joseph.melville/tmp/2d_avg_miso.png', bbox_inches='tight', dpi=600)
# plt.show()


# # plt.figure(figsize=[sw,sh], dpi=600)
# plt.figure(dpi=600)
# plt.rcParams['font.size'] = 8
# plt.plot(das_p)
# plt.plot(das_pa,'--',linewidth=1)
# plt.plot(ng_m, das_m)
# plt.plot(ng_ma, das_ma,'--',linewidth=1)
# plt.xlim([4096, 0])
# plt.title('')
# plt.xlabel('Number of grains')
# plt.ylabel('Avg Boundary \nMisorientation')
# plt.legend(['APRIMME (Cut=0)','APRIMME (Cut=25)', 'MCP (Cut=0)','MCP (Cut=25)'])
# plt.savefig('/blue/joel.harley/joseph.melville/tmp/2d_avg_miso.png', bbox_inches='tight', dpi=600)
# plt.show()






### Compare training data statistics to PRIMME simuations


#Find all stats for training data (r2, s, da, ms - through time and distributions)
#Find a good way to compare it to the APRIMME simulations













# Heres the plan
# I'll rerun each version, without dihedrals, should be done by end of night
# Find the one with best stats and write about that one


# model_dim(2)_sz(17_17)_lr(5e-05)_reg(1)_ep(1000)_kt(0.66)_cut(0)_nonorm.h5
# model_dim(2)_sz(17_17)_lr(5e-05)_reg(1)_ep(1000)_kt(0.66)_cut(25)_nonorm.h5
#Trained with no normalization 


# model_dim(2)_sz(17_17)_lr(5e-05)_reg(1)_ep(1000)_kt(0.66)_cut(0).h5
# model_dim(2)_sz(17_17)_lr(5e-05)_reg(1)_ep(1000)_kt(0.66)_cut(25).h5
#Trained with normalization 


# fp = './data/primme_sz(2400x2400)_ng(20000)_nsteps(1000)_freq(1)_kt(0.66)_cut(0).h5'
#model_dim(2)_sz(17_17)_lr(5e-05)_reg(1)_ep(1000)_kt(0.66)_cut(0)_nonorm.h5
#9 - rotate for argmax (max likelyhood)

# fp = './data/primme_sz(2400x2400)_ng(20000)_nsteps(1000)_freq(1)_kt(0.66)_cut(25).h5'
#model_dim(2)_sz(17_17)_lr(5e-05)_reg(1)_ep(1000)_kt(0.66)_cut(25)_nonorm.h5
#1 - rotate for argmax (max likelyhood) 






#Run dihedral angles for MCP cut=0
#Not now - train zero mean, run, stats (bot iso and ani)
#Run 1024x1024 for miso comparisons



# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# num_grains = 4000

# bins = np.linspace(40,200,50)
# x = bins[:-1]+np.diff(bins)/2

# fp = './data/primme_sz(2400x2400)_ng(20000)_nsteps(1000)_freq(1)_kt(0.66)_cut(0).h5'
# with h5py.File(fp, 'r') as f:
#     grain_areas = f['sim9/grain_areas'][:]
#     n = (grain_areas!=0).sum(1)
#     j = np.argmin(np.abs(n-num_grains))
    
    
#     # im = f['sim9/ims_id'][170,0]
#     # plt.imshow(im)
    
    
#     im = torch.from_numpy(f['sim9/ims_id'][j, 0].astype(float)).to(device)[None,]
# _, da = fs.find_dihedral_stats2(im[:, None], if_plot=False)


# plt.imshow(im[0].cpu())



# torch.unique(im)
# plt.plot(n)



# ic, ea, _ = fs.voronoi2image(size=[512,512], ngrain=512)

# modelname = './data/model_dim(2)_sz(17_17)_lr(5e-05)_reg(1)_ep(1000)_kt(0.66)_cut(0)_keep.h5'
# fp = fsp.run_primme(ic, ea, nsteps=200, modelname=modelname, if_miso=False, plot_freq=20)


# fp = '../../MF/data/spparks_sz(2400x2400)_ng(20000)_nsteps(1600)_freq(1.0)_kt(0.66)_cut(0).h5'

# fs.compute_grain_stats(fp)

# #train new models for norm tomorrow (iso and ani)
# #finalize plots
# #write paper





# fp = './data/primme_sz(2400x2400)_ng(20000)_nsteps(1000)_freq(1)_kt(0.66)_cut(0).h5'
# fs.make_videos([fp], gps='sim9')




# fp = './data/spparks_sz(2400x2400)_ng(20000)_nsteps(2001)_freq(1.0)_kt(0.66)_cut(25).h5'
# fp = '../../MF/data/spparks_sz(2400x2400)_ng(20000)_nsteps(1600)_freq(1.0)_kt(0.66)_cut(0).h5'



# with h5py.File(fp, 'r') as f:
#     print(f.keys())
#     print(f['sim0'].keys())
#     print(f['sim0/ims_id'].shape)
#     # im = f['sim0/ims_id'][2000,0]
#     ds = f['sim0/dihedral_std'][:]
# plt.imshow(im)

# plt.plot(ds[:,0])








# fp = '../../MF/data/mf_sz(2400x2400)_ng(20000)_nsteps(1000)_cov(16)_numnei(64)_cut(0).h5'
# fs.compute_grain_stats(fp)

# fp = './data/primme_sz(2400x2400)_ng(20000)_nsteps(1000)_freq(1)_kt(0.66)_(T0R0)_cut(0).h5'
# fs.compute_grain_stats(fp)

# fp = './data/primme_sz(2400x2400)_ng(20000)_nsteps(1000)_freq(1)_kt(0.66)_(T25R25)_cut(25).h5'
# fs.compute_grain_stats(fp)


# with h5py.File(fp, 'r') as f:
#     print(f['sim0'].keys())


# fp = '../../MF/data/mf_sz(2400x2400)_ng(20000)_nsteps(1000)_cov(16)_numnei(64)_cut(0).h5'
# with h5py.File(fp, 'r') as f:
#     print(f.keys())
#     print(f['sim0'].keys())
#     ea = f['sim0/euler_angles'][:]
#     ma = f['sim0/miso_array'][:]
#     mm = f['sim0/miso_matrix'][:]
#     im0 = f['sim0/ims_id'][0,0]


# fp = './data/32c20000grs2400stskT066_cut25_img.hdf5'
# with h5py.File(fp, 'r') as f:
#     print(f.keys())
#     print(f['images'].shape)
#     im1 = f['images'][0,]-1
#     ims_id = f['images'][:][:,None]-1
 

# fp = './data/spparks_sz(2400x2400)_ng(20000)_nsteps(2001)_freq(1.0)_kt(0.66)_cut(25).h5'
# with h5py.File(fp, 'r') as f:
#     print(f['sim0'].keys())
    
#     f['sim0/euler_angles'] = ea
#     f['sim0/miso_array'] = ma
#     f['sim0/miso_matrix'] = mm
#     f['sim0/ims_id'] = ims_id







# fp = '../../MF/data/mf_sz(2400x2400)_ng(20000)_nsteps(1000)_cov(16)_numnei(64)_cut(0).h5'
# with h5py.File(fp, 'r') as f:
#     ic = torch.from_numpy(f['sim0/ims_id'][0,0].astype(float))
#     ea = torch.from_numpy(f['sim0/euler_angles'][:].astype(float))
#     ma = f['sim0/miso_array'][:].astype(float)

# modelname = './data/model_dim(2)_sz(17_17)_lr(5e-05)_reg(1)_ep(1000)_kt(0.66)_cut(0).h5'
# fp = fsp.run_primme(ic, ea, nsteps=1000, modelname=modelname, miso_array=ma, if_miso=False, plot_freq=20)
# # fp = './data/primme_sz(2400x2400)_ng(20000)_nsteps(1000)_freq(1)_kt(0.66)_(T0R25)_cut(0).h5'
# fs.compute_grain_stats(fp)


# modelname = './data/model_dim(2)_sz(17_17)_lr(5e-05)_reg(1)_ep(1000)_kt(0.66)_cut(25).h5'
# fp = fsp.run_primme(ic, ea, nsteps=1000, modelname=modelname, miso_array=ma, if_miso=True, plot_freq=20)
# # fp = './data/primme_sz(2400x2400)_ng(20000)_nsteps(1000)_freq(1)_kt(0.66)_(T25R0)_cut(25).h5'
# fs.compute_grain_stats(fp)


# fp = './data/primme_sz(2400x2400)_ng(20000)_nsteps(1000)_freq(1)_kt(0.66)_cut(0).h5'
# fs.compute_grain_stats(fp)


# fp = './data/spparks_sz(2400x2400)_ng(20000)_nsteps(2001)_freq(1.0)_kt(0.66)_cut(25).h5'
# fs.compute_grain_stats(fp)





# #The first nonorm cut0 is trained without rotations, run is with rotations (net and arg) - most linear growth
# #The first nonorm cut25 is trained and run with rotations (net and arg)


# #The second nonorm cut0 is trained without rotations, run is with rotations (only on arg)







# fp = '../../MF/data/mf_sz(512x512)_ng(512)_nsteps(1000)_cov(25)_numnei(64)_cut(0).h5'
# with h5py.File(fp, 'r') as f:
#     ic = torch.from_numpy(f['sim0/ims_id'][0,0].astype(float))-1
#     ea = torch.from_numpy(f['sim0/euler_angles'][:].astype(float))
#     ma = f['sim0/miso_array'][:].astype(float)

# modelname = './data/model_dim(2)_sz(17_17)_lr(5e-05)_reg(1)_ep(1000)_kt(0.66)_cut(0)_keep.h5'
# fp = fsp.run_primme(ic, ea, nsteps=1000, modelname=modelname, miso_array=ma, if_miso=False, plot_freq=20)


# modelname = './data/model_dim(2)_sz(17_17)_lr(5e-05)_reg(1)_ep(1000)_kt(0.66)_cut(25)_keep.h5'
# fp = fsp.run_primme(ic, ea, nsteps=1000, modelname=modelname, miso_array=ma, if_miso=True, plot_freq=20)










# ### 2D isotropic normalized radius distribution (AP0, AP25, MCP, PF, Yadav, Zollinger) 

# for num_grains in [4000, 3000, 2000]: #4000, 3000, 2000
    
#     # fp = './data/primme_sz(2400x2400)_ng(20000)_nsteps(1000)_freq(1)_kt(0.66)_(T0R0)_cut(0).h5'
#     fp = './data/primme_sz(2400x2400)_ng(20000)_nsteps(1000)_freq(1)_kt(0.66)_cut(0).h5'
#     with h5py.File(fp, 'r') as f:
#         grain_areas = f['sim9/grain_areas'][:]
#     n = (grain_areas!=0).sum(1)
#     j = np.argmin(np.abs(n-num_grains))
#     a = grain_areas[j]
#     r = np.sqrt(a[a!=0]/np.pi)
#     rn = r/np.mean(r)
#     h_p, x_edges = np.histogram(rn, bins='auto', density=True)
#     x_p = x_edges[:-1]+np.diff(x_edges)/2
#     n_p = len(rn)
    
#     # fp = './data/primme_sz(2400x2400)_ng(20000)_nsteps(1000)_freq(1)_kt(0.66)_(T25R25)_cut(25).h5'
#     fp = './data/primme_sz(2400x2400)_ng(20000)_nsteps(1000)_freq(1)_kt(0.66)_cut(25).h5'
#     with h5py.File(fp, 'r') as f:
#         grain_areas = f['sim1/grain_areas'][:]
#     n = (grain_areas!=0).sum(1)
#     j = np.argmin(np.abs(n-num_grains))
#     a = grain_areas[j]
#     r = np.sqrt(a[a!=0]/np.pi)
#     rn = r/np.mean(r)
#     h_pa, x_edges = np.histogram(rn, bins='auto', density=True)
#     x_pa = x_edges[:-1]+np.diff(x_edges)/2
#     n_pa = len(rn)
    
#     mat = scipy.io.loadmat('../../MF/data/previous_figures/Case4GSizeMCPG%d.mat'%num_grains)
#     rn = mat['rnorm'][:,0]
#     h_mcp, x_edges = np.histogram(rn, bins='auto', density=True)
#     x_mcp = x_edges[:-1]+np.diff(x_edges)/2
#     n_mcp = len(rn)
    
#     mat = scipy.io.loadmat('../../MF/data/previous_figures/Case4GSizePFG%d.mat'%num_grains)
#     rn = mat['rnorm'][0]
#     h_pf, x_edges = np.histogram(rn, bins='auto', density=True)
#     x_pf = x_edges[:-1]+np.diff(x_edges)/2
#     n_pf = len(rn)
    
#     mat = scipy.io.loadmat('../../MF/data/previous_figures/RadiusDistPureTP2DNew.mat')
#     x_yad = mat['y1'][0]
#     h_yad = mat['RtotalFHD5'][0]
    
#     x_zol, h_zol = np.loadtxt('../../MF/data/previous_figures/Results.csv', delimiter=',',skiprows=1).T
    
#     # plt.figure(figsize=[sw,sh], dpi=600)
#     plt.figure(dpi=600)
#     plt.rcParams['font.size'] = 8
#     plt.plot(x_p, h_p, '-')
#     plt.plot(x_pa, h_pa, '-')
#     plt.plot(x_mcp, h_mcp, '--')
#     plt.plot(x_pf, h_pf, '-.')
#     plt.plot(x_yad, h_yad, '*', ms = 3)
#     plt.plot(x_zol, h_zol, 'd', ms = 3)
#     plt.xlabel('$R/<R>$ - Normalized Radius')
#     plt.ylabel('Frequency')
#     plt.xlim([0,3])
#     plt.ylim([0,1.2])
#     if if_leg: plt.legend(['APRIMME (cut=0), $N_G$ - %d'%n_p, 'APRIMME (cut=25), $N_G$ - %d'%n_pa, 'MCP, $N_G$ - %d'%n_mcp, 'PF, $N_G$ - %d'%n_pf, 'Yadav 2018', 'Zollner 2016'], fontsize=7)
#     plt.savefig('/blue/joel.harley/joseph.melville/tmp_APRIMME/2d_r_dist%d.png'%num_grains, bbox_inches='tight', dpi=600)
#     plt.show()



# ### 2D isotropic number of sides distribution (AP0, AP25, MCP, PF, Yadav, Mason) 

# for num_grains in [4000, 3000, 2000]: #4000, 3000, 2000
    
#     # fp = './data/primme_sz(2400x2400)_ng(20000)_nsteps(1000)_freq(1)_kt(0.66)_(T0R0)_cut(0).h5'
#     fp = './data/primme_sz(2400x2400)_ng(20000)_nsteps(1000)_freq(1)_kt(0.66)_cut(0).h5'
#     with h5py.File(fp, 'r') as f:
#         grain_sides = f['sim9/grain_sides'][:]
#     n = (grain_sides!=0).sum(1)
#     i = np.argmin(np.abs(n-num_grains))
#     s = grain_sides[i][grain_sides[i]!=0]
#     bins = np.arange(1,20)-0.5
#     h_p, x_edges = np.histogram(s, bins=bins, density=True)
#     x_p = x_edges[:-1]+np.diff(x_edges)/2
#     n_p = len(s)
    
#     # fp = './data/primme_sz(2400x2400)_ng(20000)_nsteps(1000)_freq(1)_kt(0.66)_(T25R25)_cut(25).h5'
#     fp = './data/primme_sz(2400x2400)_ng(20000)_nsteps(1000)_freq(1)_kt(0.66)_cut(25).h5'
#     with h5py.File(fp, 'r') as f:
#         grain_sides = f['sim1/grain_sides'][:]
#     n = (grain_sides!=0).sum(1)
#     i = np.argmin(np.abs(n-num_grains))
#     s = grain_sides[i][grain_sides[i]!=0]
#     bins = np.arange(1,20)-0.5
#     h_pa, x_edges = np.histogram(s, bins=bins, density=True)
#     x_pa = x_edges[:-1]+np.diff(x_edges)/2
#     n_pa = len(s)
    
#     mat = scipy.io.loadmat('../../MF/data/previous_figures_sides/Case4SidesMCPG%d.mat'%num_grains)
#     s = mat['the_sides'][0]
#     h_mcp, x_edges = np.histogram(s, bins=bins, density=True)
#     x_mcp = x_edges[:-1]+np.diff(x_edges)/2
#     n_mcp = len(s)
    
#     mat = scipy.io.loadmat('../../MF/data/previous_figures_sides/Case4SidesPFG%d.mat'%num_grains)
#     s = mat['the_sides'][0]
#     h_pf, x_edges = np.histogram(s, bins=bins, density=True)
#     x_pf = x_edges[:-1]+np.diff(x_edges)/2
#     n_pf = len(s)
    
#     mat = scipy.io.loadmat('../../MF/data/previous_figures_sides/FaceDistPureTP2DNew.mat')
#     x_yad = mat['y1'][0]
#     h_yad = mat['FtotalFHD5'][0]
    
#     x_mas, h_mas = np.loadtxt('../../MF/data/previous_figures_sides/ResultsMasonLazar2DGTD.txt').T
    
#     # plt.figure(figsize=[sw,sh], dpi=600)
#     plt.figure(dpi=600)
#     plt.rcParams['font.size'] = 8
#     plt.plot(x_p, h_p, '-')
#     plt.plot(x_pa, h_pa, '-')
#     plt.plot(x_mcp, h_mcp, '--')
#     plt.plot(x_pf, h_pf, '-.')
#     plt.plot(x_yad, h_yad, '*', ms=4)
#     plt.plot(x_mas, h_mas, '^C6', ms=3)
#     plt.xlabel('Number of Sides')
#     plt.ylabel('Frequency')
#     plt.xlim([0,15])
#     plt.ylim([0,0.4])
#     if if_leg: plt.legend(['APRIMME (cut=0), $N_G$ - %d'%n_p, 'APRIMME (cut=25), $N_G$ - %d'%n_pa, 'MCP, $N_G$ - %d'%n_mcp, 'PF, $N_G$ - %d'%n_pf, 'Yadav 2018', 'Masson 2015'], fontsize=7)
#     plt.savefig('/blue/joel.harley/joseph.melville/tmp_APRIMME/2d_num_sides_dist%d.png'%num_grains, bbox_inches='tight', dpi=600)
#     plt.show()

# plt.figure(figsize=[2,1], dpi=600)
# plt.rcParams['font.size'] = 8
# plt.plot(x_p, h_p, '-')
# plt.plot(x_pa, h_pa, '-')
# plt.plot(x_mcp, h_mcp, '--')
# plt.plot(x_pf, h_pf, '-.')
# plt.plot(x_yad, h_yad, '*', ms = 5)
# plt.plot(x_zol, h_zol, 'd', ms = 5)
# plt.plot(x_mas, h_mas, '^C6', ms = 5)
# legend = plt.legend(['MF', 'MCP', 'PF', 'Yadav 2018', 'Zollner 2016', 'Masson 2015'], bbox_to_anchor=[1,1,1,1])
# export_legend(legend, filename='../tmp_APRIMME/2d_stats_legend.png')
# plt.show()



# c = np.array([0,400,600,800])

# fp = './data/primme_sz(512x512)_ng(3)_nsteps(3000)_freq(1)_kt(0.66)_cut(25)_keep.h5'
# with h5py.File(fp, 'r') as f:
#     print(f.keys())
#     im = torch.from_numpy(f['sim0/ims_id'][c, 0].astype(float)).to(device)

# da_std, da = fs.find_dihedral_stats(im[:, None], if_plot=False)
# bins = np.linspace(40,200,50)
# x = bins[:-1]+np.diff(bins)/2

# for d in da: 
#     h, _ = np.histogram(d.flatten().cpu(), bins=bins, density=True)
#     plt.plot(x, h)
#     plt.legend(((''.join(['Frames = %d\\',]*len(c))[:-1])%tuple(c)).split('\\'))



# plt.imshow(im[0,].cpu())


# ncombo = fs.find_ncombo(im[:, None], n=3)
















# fp = './data/spparks_sz(1024x1024)_ng(4096)_nsteps(999)_freq(1.0)_kt(0.66)_cut(25).h5'
# with h5py.File(fp, 'r') as f:
#     gps = list(fp.keys())


# fs.compute_grain_stats(fp, n=50)





# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# c = np.array([200,400,600,800])

# fp = '../../MF/data/spparks_sz(2400x2400)_ng(20000)_nsteps(1600)_freq(1.0)_kt(0.66)_cut(0).h5'
# with h5py.File(fp, 'r') as f:
#     im = torch.from_numpy(f['sim0/ims_id'][c, 0].astype(float)).to(device)

# fp = '../../MF/data/mf_sz(2400x2400)_ng(20000)_nsteps(1000)_cov(16)_numnei(64)_cut(0).h5'
# with h5py.File(fp, 'r') as f:
#     im = torch.from_numpy(f['sim0/ims_id'][c, 0].astype(float)).to(device)

# fp = './data/Case4_2400p.hdf5'
# with h5py.File(fp, 'r') as f:
#     im = torch.from_numpy(f['images'][c].astype(float)).to(device)
   
# fp = './data/primme_sz(2400x2400)_ng(20000)_nsteps(1000)_reg(1)_freq(1)_kt(0.66)_cut(0).h5'
# with h5py.File(fp, 'r') as f:   
#     im = torch.from_numpy(f['sim1/ims_id'][c, 0].astype(float)).to(device)




# fs.find_dihedral_angles(im[-1][None,None], if_plot=True, num_plot_jct=10)




# plt.imshow(im[-1].cpu(), interpolation=None)
# plt.savefig('../../spparks.png', dpi=600)


# plt.imshow(im[-1].cpu(), interpolation=None)
# plt.savefig('../../mf.png', dpi=600)


# plt.imshow(im[-1].cpu(), interpolation=None)
# plt.savefig('../../primme_old.png', dpi=600)

# plt.imshow(im[-1].cpu(), interpolation=None)
# plt.savefig('../../primme_new.png', dpi=600)




    



# c = np.array([200,400,600,800])

# fp = '../../MF/data/spparks_sz(2400x2400)_ng(20000)_nsteps(1600)_freq(1.0)_kt(0.66)_cut(0).h5'
# fp = './data/primme_sz(2400x2400)_ng(20000)_nsteps(1000)_freq(1)_kt(0.66)_cut(0).h5'
# fp = './data/primme_sz(2400x2400)_ng(20000)_nsteps(1000)_freq(1)_kt(0.66)_cut(25).h5'
# with h5py.File(fp, 'r') as f:
#     im = torch.from_numpy(f['sim0/ims_id'][c, 0].astype(float)).to(device)

# da_std, da = fs.find_dihedral_stats(im[:, None], if_plot=False)
# bins = np.linspace(40,200,50)
# x = bins[:-1]+np.diff(bins)/2

# for d in da: 
#     h, _ = np.histogram(d.flatten().cpu(), bins=bins, density=True)
#     plt.plot(x, h)
#     plt.legend(((''.join(['Frames = %d\\',]*len(c))[:-1])%tuple(c)).split('\\'))


# plt.plot(da_std.cpu())


# # ds_mcp = da_std
# # ds_p = da_std
# ds_pa = da_std




    
    

































# fp0 = './data/trainset_spparks_sz(257x257)_ng(256-256)_nsets(200)_future(4)_max(100)_kt(0.66)_cut(0).h5'
# fp1 = './data/trainset_spparks_sz(257x257)_ng(256-256)_nsets(200)_future(4)_max(100)_kt(0.66)_cut(25).h5'
# fp2 = './data/trainset_spparks_sz(257x257)_ng(256-256)_nsets(200)_future(4)_max(100)_kt(0.66)_cut(360).h5'

# fpp = [fp0, fp1, fp2]

# for fp in fpp:
#     with h5py.File(fp, 'r') as f:
#         ims_id = torch.from_numpy(f['ims_id'][:]).to(device)
#         miso_array = torch.from_numpy(f['miso_array'][:]).to(device)
#     miso_mats = fs.miso_array_to_matrix(miso_array)

#     log = []
#     for i in tqdm(range(ims_id.shape[0])):
#         log0 = []
#         for j in range(ims_id.shape[1]):
#             ims = ims_id[i,j][None]
#             miso_matrices = miso_mats[i][None]
#             im_miso = fs.neighborhood_miso_spparks(ims, miso_matrices, cut=360)
#             log0.append(im_miso[0])
#         log.append(torch.stack(log0))

#     with h5py.File(fp, 'a') as f:
#         ims_miso = torch.stack(log)
#         if 'ims_miso' not in f.keys(): f['ims_miso'] = ims_miso.cpu().numpy()
#         if 'ims_miso_avg' not in f.keys(): f['ims_miso_avg'] = fs.iterate_function(ims_miso.cpu().numpy(), fs.mean_wo_zeros, args=[])




# for fp in fpp:
#     with h5py.File(fp, 'r') as f:
#         ims_id = torch.from_numpy(f['ims_id'][:]).to(device)

#     ims = ims_id.reshape(1000,1,257,257)
#     log = []
#     for im in tqdm(ims): 
#         da_std = fs.find_dihedral_stats(im[None], if_plot=False)
#         log.append(da_std)
#     dihedral_std = torch.stack(log)[:,0].reshape(200,5)
    
#     with h5py.File(fp, 'a') as f:
#         if 'dihedral_std' not in f.keys(): f['dihedral_std'] = dihedral_std.cpu().numpy()


# aaa = torch.stack(log)[:,0].cpu()


# plt.plot(aaa)
    

# with h5py.File(fpp[0], 'r') as f:
#     msa0 = f['ims_miso_avg'][:]
# with h5py.File(fpp[1], 'r') as f:
#     msa25 = f['ims_miso_avg'][:]
# (msa25-msa0).mean()



# with h5py.File(fpp[0], 'r') as f:
#     da0 = f['dihedral_std'][:]
# with h5py.File(fpp[1], 'r') as f:
#     da25 = f['dihedral_std'][:]
# (da25-da0).mean()



        







# ims = ims_id.reshape(1000,1,257,257)
# log_da = []
# log_std = []
# for im in tqdm(ims): 
#     da_std, da = fs.find_dihedral_stats(im[None], if_plot=False)
#     log_da.append(da)
#     log_std.append(da_std)

# da0 = torch.cat(log_da, dim=2)
# da_stds0 = torch.stack(log_std)

# da25 = torch.cat(log_da, dim=2)
# da_stds25 = torch.stack(log_std)


# bins = np.linspace(0,360,50)
# h0, _ = np.histogram(da0.flatten().cpu(), bins=bins)
# h25, _ = np.histogram(da25.flatten().cpu(), bins=bins)
# plt.plot(h0)
# plt.plot(h25)









# fp = './data/primme_sz(1024x1024)_ng(4096)_nsteps(500)_freq(1)_kt(0.66)_cut(0).h5'
# fp = './data/primme_sz(1024x1024)_ng(4096)_nsteps(500)_freq(1)_kt(0.66)_cut(25).h5'
# fp = './data/spparks_sz(1024x1024)_ng(4096)_nsteps(1000)_freq(1.0)_kt(0.66)_cut(0).h5'
# fp = './data/spparks_sz(1024x1024)_ng(4096)_nsteps(1000)_freq(1.0)_kt(0.66)_cut(25).h5'

# with h5py.File(fp, 'r') as f:
#     tmp = list(f.keys())
#     ims = torch.from_numpy(f['%s/ims_id'%tmp[-1]][:].astype(float)).to(device)
#     ims = ims[np.linspace(0,499,50).astype(int),]



# log_da = []
# log_std = []
# for im in tqdm(ims): 
#     da_std, da = fs.find_dihedral_stats(im[None], if_plot=False)
#     log_da.append(da)
#     log_std.append(da_std)


# da_p0 = log_da
# da_stds_p0 = torch.stack(log_std)


# da_p25 = log_da
# da_stds_p25 = torch.stack(log_std)

# da_s0 = log_da
# da_stds_s0 = torch.stack(log_std)

# da_s25 = log_da
# da_stds_s25 = torch.stack(log_std)


# plt.plot(da_stds_s0.cpu())  
# plt.plot(da_stds_s25.cpu())  

# (da_stds_s25.cpu()-da_stds_s0.cpu()).mean()


# plt.plot(da_stds_s25.cpu()-da_stds_s0.cpu())



# bins = np.linspace(40,200,50)
# x = bins[:-1]+np.diff(bins)/2
# h0, _ = np.histogram(da_s0[0].flatten().cpu(), bins=bins, density=True)
# log = []
# for da in da_s0:
#     h, _ = np.histogram(da.flatten().cpu(), bins=bins, density=True)
#     log.append(((h0-h)**2).sum())
#     plt.plot(x,h)
#     plt.ylim([0,0.030])
    
# plt.plot(log)  
# plt.ylim([0,0.0010])
    
    
# h25, _ = np.histogram(da25.flatten().cpu(), bins=bins)
# plt.plot(h0)
# plt.plot(h25)



# training set statistics



# neighborhood miso


# dihedral std




