#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr  5 09:24:36 2022
This script loads dispersion data and plots the quality control factors of
the dispersion measurements.
@author: hbwoo
"""
import sys
lib_path = '/Users/hbwoo/Documents/GitHub/xcorr'
sys.path.insert(1, lib_path) # Add xcorr path to library
import scipy.io
import numpy as np
import quality_control_aux as qcaux
from matplotlib import pyplot as plt
from matplotlib import cm
from matplotlib.axes._axes import _log as matplotlib_axes_logger
matplotlib_axes_logger.setLevel('ERROR')
plt.rcParams.update({'figure.max_open_warning': 0})

#%% Load dispersion data

# Phase2-PWS method results
dat_dir = '/Dispersion_Data/'
title_str = '(PCC-PWS)'
# Load group wave dispersion measurements
disp = scipy.io.loadmat(dat_dir+'/xcor.synthetic_Phase2_PWS_final.mat')
# Load group wave dispersion measurements for each progressive stack
p_disp = scipy.io.loadmat(dat_dir+'/xcor.synthetic_Phase2_PWS_progressive.mat')

dist = disp['dist_arr'][0]
periods = disp['per'][0]; freqs = disp['FRQ'][0]
snrs = disp['SNR']; nwvs = disp['NWV']
gvs = disp['GV']; gv_intp = disp['GV_INTP']

p_snrs = p_disp['SNR_prog']; p_nwvs = p_disp['NWV_prog']
p_gvs = p_disp['GV_prog']

# Time-Lin results
dat_dir2 = '/Dispersion_Data/'
title_str2 = '(TCC-Lin)'
# Load group wave dispersion measurements
disp2 = scipy.io.loadmat(dat_dir2+'/xcor.synthetic_Time_Lin_final.mat')
# Load group wave dispersion measurements for each progressive stack
p_disp2 = scipy.io.loadmat(dat_dir2+'/xcor.synthetic_Time_Lin_progressive.mat')

dist2 = disp2['dist_arr'][0]
periods2 = disp2['per'][0]; freqs2 = disp2['FRQ'][0]
snrs2 = disp2['SNR']; nwvs2 = disp2['NWV']
gvs2 = disp2['GV']; gv_intp2 = disp2['GV_INTP']

p_snrs2 = p_disp2['SNR_prog']; p_nwvs2 = p_disp2['NWV_prog']
p_gvs2 = p_disp2['GV_prog']

#%% Set Quality control parameters

# Vary only SNR
# snrlim = [20, 25, 30, 35, 40, 45, 50]; # For Phase2-PWS
# snrlim2 = [5, 7, 9, 11, 13, 15, 20]; # For Time-Lin
# nwvl = [0.5]
# nwvh = [70];

# Vary only NWVL
# snrlim = [20]; # For Phase2-PWS
# snrlim2 = [7]; # For Time-Lin
# nwvl = [0.5, 1, 1.5, 2, 2.5, 3, 3.5];
# nwvh = [70];

# Vary only NWVH
# snrlim = [20]; # For Phase2-PWS
# snrlim2 = [7]; # For Time-Lin
# nwvl = [3];
# nwvh = [5, 6, 7, 8, 9, 10];

# Vary All
snrlim = [20, 25, 30, 35, 40, 45, 50]; # For Phase2-PWS
snrlim2 = [5, 7, 9, 11, 13, 15, 20]; # For Time-Lin
nwvl = [1, 1.5, 2, 2.5, 3];
nwvh = [10, 20, 30, 40, 50, 70];

# Single values
# snrlim = [20]; # For Phase2-PWS
# snrlim2 = [7]; # For Time-Lin
# nwvl = [3];
# nwvh = [10];

# Skip periods every # when plotting
evry = 1

#%% Plot Center Periods VS SNR

qcaux.plot_cp_snr(dist,snrs,freqs,title_str,dat_dir,evry,logsnr=True,savefig=False)
qcaux.plot_cp_snr(dist2,snrs2,freqs2,title_str2,dat_dir2,evry,logsnr=True,savefig=False)

#%% Plot Center Periods VS NWL

qcaux.plot_cp_nwl(dist,nwvs,freqs,title_str,dat_dir,evry,lognwl=False,savefig=False)
qcaux.plot_cp_nwl(dist2,nwvs2,freqs2,title_str2,dat_dir2,evry,lognwl=False,savefig=False)

#%% Plot NWL VS SNR based on Distance

qcaux.plot_nwl_snr(dist,nwvs,snrs,freqs,title_str,dat_dir,evry,savefig=False)
qcaux.plot_nwl_snr(dist2,nwvs2,snrs2,freqs2,title_str2,dat_dir2,evry,savefig=False)

#%% Plot Interstation Distance VS Maximum T based on NWL

qcaux.plot_dist_maxT(dist,periods,nwvl,nwvs,title_str,dat_dir,savefig=False)
qcaux.plot_dist_maxT(dist2,periods,nwvl,nwvs2,title_str2,dat_dir2,savefig=False)

#%% Plot raw group velocity curve

qcaux.plot_raw_gv(dist,periods,gvs,title_str,dat_dir,evry,savefig=False)
qcaux.plot_raw_gv(dist,periods2,gvs2,title_str2,dat_dir2,evry,savefig=False)

#%% Plot each quality controlled gv curves and network averaged gv curve
# number of cleaned gv curves with smallest mean error
n_plots = 5

# Phase-PWS
gv_clean,gv_err,err_mean,qc_comb,gv_clean_ind = qcaux.plot_qc_gv(dist,periods,gvs,
                                              nwvs,nwvl,nwvh,snrs,snrlim,title_str,
                                              dat_dir,evry,plotfig=False,savefig=False)
min_err_ind = qcaux.plot_net_gv(periods,gv_clean,gv_err,err_mean,qc_comb,title_str,
                                    dat_dir,DIR,n_plots,savefig=False,savegv=False)
# Time-Lin
gv_clean2,gv_err2,err_mean2,qc_comb2,gv_clean_ind2 = qcaux.plot_qc_gv(dist2,periods2,
                                              gvs2,nwvs2,nwvl,nwvh,snrs2,snrlim2,
                                              title_str2,dat_dir2,evry,
                                              plotfig=False,savefig=False)
min_err_ind2 = qcaux.plot_net_gv(periods2,gv_clean2,gv_err2,err_mean2,qc_comb2,title_str2,
                                    dat_dir2,DIR,n_plots,savefig=False,savegv=False)

#%% Plot PCC-PWS against TCC-Lin with single parameters

qcaux.plot_clean_gv_pcc_tcc(periods,gv_clean,gv_err,err_mean,qc_comb,gv_clean2,
                            gv_err2,err_mean2,qc_comb2,dat_dir,DIR,savefig=False)

#%% Plot all clean gv's and error for each periods

qcaux.plot_all_gv(periods,gv_clean,gv_err,title_str,dat_dir,DIR,savefig=False)
qcaux.plot_all_gv(periods2,gv_clean2,gv_err2,title_str2,dat_dir2,DIR,savefig=False)

#%% Plot all clean gv's and error for each periods

qcaux.plot_clean_av_gv(periods,gv_clean,title_str,dat_dir,DIR,savefig=False)
qcaux.plot_clean_av_gv(periods2,gv_clean2,title_str2,dat_dir2,DIR,savefig=False)

#%% Work on Progressive stacks
# Plot change in SNR of EGF vs number of stacks
evry = 1
qcaux.plot_stcks_snr(dist,p_snrs,title_str,dat_dir,evry,logsnr=True,savefig=True)
qcaux.plot_stcks_snr(dist2,p_snrs2,title_str2,dat_dir2,evry,logsnr=True,savefig=True)

#%% Compute nstcks to meet reasonalbe quality of GV
# Phase2-PWS

# Choose index from mean error of cleaned network averaged group velocity curves
# comb_ind = 1 uses the network averaged grup velocity curve with the lowest error
comb_ind = 1
rmse_cut = 0.05
c_nwvl = qc_comb[comb_ind][0]
c_nwvh= qc_comb[comb_ind][1]
c_snrlim = qc_comb[comb_ind][2]
gv_nstcks_full,gvs_nstcks = qcaux.nstkcs_for_gv2(periods,gvs,rmse_cut,p_gvs,snrs,
                                 p_snrs,nwvs,p_nwvs,c_snrlim,c_nwvl,c_nwvh)
# Time-Lin
c_nwvl2 = qc_comb2[comb_ind][0]
c_nwvh2 = qc_comb2[comb_ind][1]
c_snrlim2 = qc_comb2[comb_ind][2]
gv_nstcks_full2,gvs_nstcks2 = qcaux.nstkcs_for_gv2(periods2,gvs2,rmse_cut,p_gvs2,
                                    snrs2,p_snrs2,nwvs2,p_nwvs2,c_snrlim2,c_nwvl2,c_nwvh2)

qcaux.plot_nstcks_for_gv(periods,periods2,gv_nstcks_full,gv_nstcks_full2,
                         rmse_cut,xvar='Period',yvar='Stacks')






