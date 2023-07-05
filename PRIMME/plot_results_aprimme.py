#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 13 11:18:08 2023

@author: joseph.melville
"""


# import PRIMME as fsp
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

# ic, ea = fs.generate_circleIC(size=[512,512], r=200)
# ma = np.array([1])

# modelname = './data/model_dim(2)_sz(17_17)_lr(5e-05)_reg(1)_ep(1000)_kt(0.66)_cut(25)_norm.h5'
# fp = fsp.run_primme(ic, ea, nsteps=3000, modelname=modelname, miso_array=ma, if_miso=True, plot_freq=50)

plt.figure(figsize=[sw,sh], dpi=600)
plt.rcParams['font.size'] = 8
fp = './data/primme_sz(512x512)_ng(2)_nsteps(3000)_freq(1)_kt(0.66)_cut(0)_reg(1).h5'
for i, nf in enumerate([0,1500,3000]):
    with h5py.File(fp, 'r') as f:
        im = f['sim0/ims_id'][nf,0]
        plt.subplot(1,3,i+1)
        plt.imshow(im, interpolation='none')
        plt.tick_params(bottom=False, left=False,labelleft=False, labelbottom=False)
        plt.xlabel('Frame: %d'%nf)
plt.savefig('/blue/joel.harley/joseph.melville/tmp_APRIMME/circles.png', bbox_inches='tight', dpi=600)
plt.show()


plt.figure(figsize=[sw,sh], dpi=600)
plt.rcParams['font.size'] = 8
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
plt.savefig('/blue/joel.harley/joseph.melville/tmp_APRIMME/circle_regs.png', bbox_inches='tight', dpi=600)
plt.show()
    

plt.figure(figsize=[sw,sh], dpi=600)
plt.rcParams['font.size'] = 8
for ma in [0.25, 0.5, 0.75, 1]:
    fp = './data/primme_sz(512x512)_ng(2)_nsteps(3000)_freq(1)_kt(0.66)_cut(25)_ma(%1.2f).h5'%ma
    with h5py.File(fp, 'r') as f:
        print(f.keys())
        ims = f['sim0/ims_id'][:]
    
    a = (ims==1).sum(1).sum(1).sum(1)
    plt.plot(a)   
plt.legend(['Miso=0.25', 'Miso=0.5', 'Miso=0.75', 'Miso=1'])
plt.ylabel('Circle Pixels Remaining')
plt.xlabel('Number of frames')
plt.savefig('/blue/joel.harley/joseph.melville/tmp_APRIMME/circle_mas.png', bbox_inches='tight', dpi=600)
plt.show()
    


### Triple grain #!!!

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

fp0 = './data/primme_sz(512x512)_ng(3)_nsteps(3000)_freq(1)_kt(0.66)_(T25R0)_cut(25).h5'
fpr = './data/primme_sz(512x512)_ng(3)_nsteps(3000)_freq(1)_kt(0.66)_(T25R25)_cut(25)_right.h5'
fpl = './data/primme_sz(512x512)_ng(3)_nsteps(3000)_freq(1)_kt(0.66)_(T25R25)_cut(25)_left.h5'

plt.figure(figsize=[3,2], dpi=600)
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
    plt.savefig('/blue/joel.harley/joseph.melville/tmp_APRIMME/triple_config.png', bbox_inches='tight', dpi=600)
plt.show()



### Hex grains #!!!

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

plt.figure(figsize=[6.5,2], dpi=600)
plt.rcParams['font.size'] = 8
fp = './data/primme_sz(443x512)_ng(64)_nsteps(3000)_freq(1)_kt(0.66)_cut(25)_keep.h5'
with h5py.File(fp, 'r') as f:
    
    plt.subplot(1,3,1)
    im = f['sim0/ims_id'][1000,0]
    plt.imshow(im, interpolation='none')
    plt.tick_params(bottom=False, left=False,labelleft=False, labelbottom=False)
    plt.xlabel('Isotropic')
    
    # plt.subplot(1,3,2)
    # im = f['sim1/ims_id'][3000,0]
    # plt.imshow(im, interpolation='none')
    # plt.tick_params(bottom=False, left=False,labelleft=False, labelbottom=False)
    # plt.xlabel('Anisotropic')
    
    plt.subplot(1,3,2)
    im = f['sim2/ims_id'][350,0]
    plt.imshow(im, interpolation='none')
    plt.tick_params(bottom=False, left=False,labelleft=False, labelbottom=False)
    plt.xlabel('One Grain\nLow Misorientation')
    plt.gca().add_patch(Rectangle((345,255),150,150,linewidth=2,edgecolor='r',facecolor='none'))
    
    plt.subplot(1,3,3)
    im = f['sim3/ims_id'][350,0]
    plt.imshow(im, interpolation='none')
    plt.tick_params(bottom=False, left=False,labelleft=False, labelbottom=False)
    plt.xlabel('One Grain\nHigh Misorientation')
    plt.gca().add_patch(Rectangle((345,255),150,150,linewidth=2,edgecolor='r',facecolor='none'))
    
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

fp = './data/primme_sz(2400x2400)_ng(20000)_nsteps(1000)_freq(1)_kt(0.66)_(T0R0)_cut(0).h5'
# fp = './data/primme_sz(2400x2400)_ng(20000)_nsteps(1000)_freq(1)_kt(0.66)_cut(0).h5'
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

fp = './data/primme_sz(2400x2400)_ng(20000)_nsteps(1000)_freq(1)_kt(0.66)_(T25R25)_cut(25).h5'
# fp = './data/primme_sz(2400x2400)_ng(20000)_nsteps(1000)_freq(1)_kt(0.66)_cut(25).h5'
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
t_pa = np.arange(si-s0)*scale
ng_pa = ng[s0:si]
r2_pa = r2

plt.figure(figsize=[sw,sh], dpi=600)
plt.rcParams['font.size'] = 8
plt.plot(t_m, r2_m*1e-12, '-')
plt.plot(t_ma, r2_ma*1e-12, '-')
plt.plot(t_p, r2_p*1e-12, '--')
plt.plot(t_pa, r2_pa*1e-12, '-.')
plt.xlabel('Time (unitless)')
plt.ylabel('$<R>^2$ ($m^2$)')
if if_leg: plt.legend(['MCP (cut=0)','MCP (cut=25)','APRIMME (cut=0)','APRIMME (cut=25)'])
plt.savefig('/blue/joel.harley/joseph.melville/tmp_APRIMME/2d_r2_vs_time.png', bbox_inches='tight', dpi=600)
plt.show()

plt.figure(figsize=[3,2], dpi=600)
plt.rcParams['font.size'] = 8
plt.plot(t_m, r2_m*1e-12, '-')
plt.plot(t_ma, r2_ma*1e-12, '-')
plt.plot(t_p*.93, r2_p*1e-12-.12e-9, '--')
plt.plot(t_pa*.89 ,r2_pa*1e-12-.19e-9, '-.')
plt.xlabel('Time (unitless)')
plt.ylabel('$<R>^2$ ($m^2$)')
if if_leg: plt.legend(['MCP (cut=0)','MCP (cut=25)','APRIMME (cut=0)','APRIMME (cut=25)'])
plt.savefig('/blue/joel.harley/joseph.melville/tmp_APRIMME/2d_r2_vs_time2.png', bbox_inches='tight', dpi=600)
plt.show()



### Average number of sides through time (AP0, AP25, SPPARKS, PF) #!!!

fp = './data/spparks_sz(1024x1024)_ng(4096)_nsteps(1000)_freq(1.0)_kt(0.66)_cut(0).h5'
# fp = '../../MF/data/spparks_sz(2400x2400)_ng(20000)_nsteps(1600)_freq(1.0)_kt(0.66)_cut(0).h5'
with h5py.File(fp, 'r') as f:
    gsa_m = f['sim0/grain_sides_avg'][:500]

fp = './data/spparks_sz(1024x1024)_ng(4096)_nsteps(1000)_freq(1.0)_kt(0.66)_cut(25).h5'
# fp = './data/spparks_sz(2400x2400)_ng(20000)_nsteps(2001)_freq(1.0)_kt(0.66)_cut(25).h5'
with h5py.File(fp, 'r') as f:
    gsa_ma = f['sim0/grain_sides_avg'][:500]

fp = './data/primme_sz(1024x1024)_ng(4096)_nsteps(500)_freq(1)_kt(0.66)_cut(0).h5'
# fp = './data/primme_sz(2400x2400)_ng(20000)_nsteps(1000)_freq(1)_kt(0.66)_(T0R0)_cut(0).h5'
# fp = fp = './data/primme_sz(2400x2400)_ng(20000)_nsteps(1000)_freq(1)_kt(0.66)_cut(0).h5'
with h5py.File(fp, 'r') as f:
    gsa_p = f['sim5/grain_sides_avg'][:500]

fp = './data/primme_sz(1024x1024)_ng(4096)_nsteps(500)_freq(1)_kt(0.66)_cut(25).h5'
# fp = './data/primme_sz(2400x2400)_ng(20000)_nsteps(1000)_freq(1)_kt(0.66)_(T25R25)_cut(25).h5'
# fp = './data/primme_sz(2400x2400)_ng(20000)_nsteps(1000)_freq(1)_kt(0.66)_cut(25).h5'
with h5py.File(fp, 'r') as f:
    gsa_pa = f['sim3/grain_sides_avg'][:500]

plt.figure(figsize=[sw,sh], dpi=600)
plt.rcParams['font.size'] = 8
plt.plot(gsa_m, '-')
plt.plot(gsa_ma, '-')
plt.plot(gsa_p, '--')
plt.plot(gsa_pa, '-.')
plt.xlabel('Number of Frames')
plt.ylabel('Avg Number \nof Sides')
if if_leg: plt.legend(['MCP (cut=0)','MCP (cut=25)','APRIMME (cut=0)','APRIMME (cut=25)'])
plt.savefig('/blue/joel.harley/joseph.melville/tmp_APRIMME/2d_num_sides_vs_time.png', bbox_inches='tight', dpi=600)
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

for num_grains in [4000, 3000, 2000]: #4000, 3000, 2000

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
    
    fp = './data/primme_sz(2400x2400)_ng(20000)_nsteps(1000)_freq(1)_kt(0.66)_(T0R0)_cut(0).h5'
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
    
    fp = './data/primme_sz(2400x2400)_ng(20000)_nsteps(1000)_freq(1)_kt(0.66)_(T25R25)_cut(25).h5'
    # fp = './data/primme_sz(2400x2400)_ng(20000)_nsteps(1000)_freq(1)_kt(0.66)_cut(25).h5'
    with h5py.File(fp, 'r') as f:
        grain_areas = f['sim0/grain_areas'][:]
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
    

    # plt.figure(figsize=[sw,sh], dpi=600)
    plt.figure(dpi=600)
    plt.rcParams['font.size'] = 8
    plt.plot(x_m, h_m, 'C0')
    plt.plot(x_ma, h_ma, 'C0--')
    plt.plot(x_p, h_p, 'C1')
    plt.plot(x_pa, h_pa, 'C1--')
    plt.plot(x_yad, h_yad, '*', ms = 10)
    plt.plot(x_zol, h_zol, 'd', ms = 10)
    plt.xlabel('$R/<R>$ - Normalized Radius')
    plt.ylabel('Frequency')
    plt.xlim([0,3])
    plt.ylim([0,1.2])
    if if_leg: plt.legend(['MCP (cut=0)','MCP (cut=25)','APRIMME (cut=0)','APRIMME (cut=25)','Yadav 2018', 'Zollner 2016'])
    plt.savefig('/blue/joel.harley/joseph.melville/tmp_APRIMME/2d_r_dist%d.png'%num_grains, bbox_inches='tight', dpi=600)
    plt.show()



### 2D isotropic number of sides distribution (AP0, AP25, MCP, PF, Yadav, Mason) #!!!

for num_grains in [4000, 3000, 2000]: #4000, 3000, 2000
    
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
    
    fp = './data/primme_sz(2400x2400)_ng(20000)_nsteps(1000)_freq(1)_kt(0.66)_(T0R0)_cut(0).h5'
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
    
    fp = './data/primme_sz(2400x2400)_ng(20000)_nsteps(1000)_freq(1)_kt(0.66)_(T25R25)_cut(25).h5'
    # fp = './data/primme_sz(2400x2400)_ng(20000)_nsteps(1000)_freq(1)_kt(0.66)_cut(25).h5'
    with h5py.File(fp, 'r') as f:
        grain_sides = f['sim0/grain_sides'][:]
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
    
    # plt.figure(figsize=[sw,sh], dpi=600)
    plt.figure(dpi=600)
    plt.rcParams['font.size'] = 8
    plt.plot(x_m, h_m, 'C0')
    plt.plot(x_ma, h_ma, 'C0--')
    plt.plot(x_p, h_p, 'C1')
    plt.plot(x_pa, h_pa, 'C1--')
    plt.plot(x_yad, h_yad, '*', ms = 10)
    plt.plot(x_mas, h_mas, 'd', ms = 10)
    plt.xlabel('Number of Sides')
    plt.ylabel('Frequency')
    plt.xlim([0,15])
    plt.ylim([0,0.4])
    if if_leg: plt.legend(['MCP (cut=0)','MCP (cut=25)','APRIMME (cut=0)','APRIMME (cut=25)','Yadav 2018', 'Masson 2015'])
    plt.savefig('/blue/joel.harley/joseph.melville/tmp_APRIMME/2d_num_sides_dist%d.png'%num_grains, bbox_inches='tight', dpi=600)
    plt.show()



### 2D isotropic microstructure comparisons (MF, MCP, PF) #!!!

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
    
    
    # plt.imshow(np.fliplr(im_pf), interpolation='none')
    plt.savefig('/blue/joel.harley/joseph.melville/tmp_APRIMME/2d_comp.png', bbox_inches='tight', dpi=600)
plt.show()



### Compare dihedral angle distributions at different numbers of grains #!!!

bins = np.linspace(40,200,50)
x = bins[:-1]+np.diff(bins)/2

for num_grains in [4000, 3000, 2000]: #4000, 3000, 2000

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
    
    fp = './data/primme_sz(2400x2400)_ng(20000)_nsteps(1000)_freq(1)_kt(0.66)_(T0R0)_cut(0).h5'
    # fp = './data/primme_sz(2400x2400)_ng(20000)_nsteps(1000)_freq(1)_kt(0.66)_cut(0).h5'
    with h5py.File(fp, 'r') as f:
        grain_areas = f['sim0/grain_areas'][:]
        n = (grain_areas!=0).sum(1)
        j = np.argmin(np.abs(n-num_grains))
        im = torch.from_numpy(f['sim0/ims_id'][j, 0].astype(float)).to(device)[None,]
    _, da = fs.find_dihedral_stats2(im[:, None], if_plot=False)
    h_p, _ = np.histogram(da[0].flatten().cpu(), bins=bins, density=True)
    
    fp = './data/primme_sz(2400x2400)_ng(20000)_nsteps(1000)_freq(1)_kt(0.66)_(T25R25)_cut(25).h5'
    # fp = './data/primme_sz(2400x2400)_ng(20000)_nsteps(1000)_freq(1)_kt(0.66)_cut(25).h5'
    with h5py.File(fp, 'r') as f:
        grain_areas = f['sim0/grain_areas'][:]
        n = (grain_areas!=0).sum(1)
        j = np.argmin(np.abs(n-num_grains))
        im = torch.from_numpy(f['sim0/ims_id'][j, 0].astype(float)).to(device)[None,]
    _, da = fs.find_dihedral_stats2(im[:, None], if_plot=False)
    h_pa, _ = np.histogram(da[0].flatten().cpu(), bins=bins, density=True)
    
    # plt.figure(figsize=[sw,sh], dpi=600)
    plt.figure(dpi=600)
    plt.rcParams['font.size'] = 8
    plt.plot(x, h_m, 'C0')
    plt.plot(x, h_ma, 'C0--')
    plt.plot(x, h_p, 'C1')
    plt.plot(x, h_pa, 'C1--')
    plt.xlabel('Dihedral Angle Distribution')
    plt.ylabel('Frequency')
    plt.legend(['MCP (cut=0)', 'MCP (cut=25)','APRIMME (cut=0)','APRIMME (cut=25)'], fontsize=7)
    plt.savefig('/blue/joel.harley/joseph.melville/tmp_APRIMME/2d_da_dist%d.png'%num_grains, bbox_inches='tight', dpi=600)
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

plt.figure(figsize=[3,2], dpi=600)
plt.rcParams['font.size'] = 8
plt.plot(das_ma-das_m, 'C0.', ms=3)
plt.plot(das_pa-das_p, 'C1^', ms=3)
plt.plot([0,1000],[(das_ma-das_m).mean(),]*2, 'k--',linewidth=2)
plt.plot([0,1000],[(das_pa-das_p).mean(),]*2, 'k--',linewidth=2)
plt.ylabel('Dihehedral Angle\nSTD Difference')
plt.xlabel('Number of Frames')
plt.legend(['MCP - Mean: %1.2f'%((das_ma-das_m).mean()), 'APRIMME - Mean: %1.2f'%((das_pa-das_p).mean())], fontsize=7, loc='best')
plt.savefig('/blue/joel.harley/joseph.melville/tmp_APRIMME/da_std_time.png', bbox_inches='tight', dpi=600)
plt.show()



### Average miso difference through time  #!!!

fp = './data/spparks_sz(1024x1024)_ng(4096)_nsteps(1000)_freq(1.0)_kt(0.66)_cut(0).h5'
with h5py.File(fp, 'r') as f:
    # das_m = f['sim0/dihedral_std'][:]
    msa_m = f['sim0/ims_miso_avg'][:500]
    ng_m = (f['sim0/grain_areas'][:]!=0).sum(1)

fp = './data/spparks_sz(1024x1024)_ng(4096)_nsteps(1000)_freq(1.0)_kt(0.66)_cut(25).h5'
with h5py.File(fp, 'r') as f:
    # das_ma = f['sim0/dihedral_std'][:]
    msa_ma = f['sim0/ims_miso_avg'][:500]
    ng_ma = (f['sim0/grain_areas'][:]!=0).sum(1)

fp = './data/primme_sz(1024x1024)_ng(4096)_nsteps(500)_freq(1)_kt(0.66)_cut(0).h5'
with h5py.File(fp, 'r') as f:
    # das_p = f['sim5/dihedral_std'][:]
    msa_p = f['sim5/ims_miso_avg'][:500]
    ng_p = (f['sim5/grain_areas'][:]!=0).sum(1)

fp = './data/primme_sz(1024x1024)_ng(4096)_nsteps(500)_freq(1)_kt(0.66)_cut(25).h5'
with h5py.File(fp, 'r') as f:  
    # das_pa = f['sim3/dihedral_std'][:]
    msa_pa = f['sim3/ims_miso_avg'][:500]
    ng_pa = (f['sim3/grain_areas'][:]!=0).sum(1)


plt.figure(figsize=[3,2], dpi=600)
plt.rcParams['font.size'] = 8
plt.plot(msa_ma-msa_m, 'C0.', ms=3)
plt.plot(msa_pa-msa_p, 'C1^', ms=3)
plt.plot([0,500],[(msa_ma-msa_m).mean(),]*2, 'k--',linewidth=2)
plt.plot([0,500],[(msa_pa-msa_p).mean(),]*2, 'k--',linewidth=2)
plt.ylabel('Mean Misorientation\nDifference')
plt.xlabel('Number of Frames')
plt.legend(['MCP - Mean: %1.2f'%((msa_ma-msa_m).mean()), 'APRIMME - Mean: %1.2f'%((msa_pa-msa_p).mean())], fontsize=7, loc='best')
plt.savefig('/blue/joel.harley/joseph.melville/tmp_APRIMME/miso_mean_time.png', bbox_inches='tight', dpi=600)
plt.show()





### 2D Dihedral STD through time (iso vs ani) #!!!

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



# ### Average miso through time  #!!!

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




