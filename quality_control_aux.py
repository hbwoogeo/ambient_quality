#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 18 10:03:51 2022

@author: hbwoo
"""

import scipy.io
import csv
import numpy as np
import matplotlib as mpl
from itertools import cycle
from sklearn.metrics import mean_squared_error as mse
from matplotlib import pyplot as plt
from matplotlib import cm
from matplotlib.axes._axes import _log as matplotlib_axes_logger
matplotlib_axes_logger.setLevel('ERROR')
plt.rcParams.update({'figure.max_open_warning': 0})

#%%
def get_colors(inp, colormap, vmin=None, vmax=None):
    norm = plt.Normalize(vmin, vmax)
    return colormap(norm(inp))

#%% Plot Center Periods VS SNR

def plot_cp_snr(dist,snrs,freqs,title_str,dat_dir,evry,logsnr=True,savefig=False):
    
    bounds_tick = []
    bounds = np.linspace(0,5,len(dist[::evry]))
    bounds2 = np.linspace(0,5,11)
    for b2 in bounds2:
        bounds_tick.append(str(b2))
    # cmap = cm.inferno
    cmap = cm.get_cmap('inferno', len(bounds))
    norm = mpl.colors.BoundaryNorm(bounds, cmap.N)
    colors = get_colors(dist,cmap,vmin=0,vmax=5)[::evry]
    
    fig, ax = plt.subplots(figsize=(4,3),dpi=400)
    cbaxes = fig.add_axes([0.92, 0.125, 0.02, 0.75])
    for i, csnr in enumerate(snrs[::evry]):
        if (logsnr==True):
            ax.scatter(1./freqs, np.log10(csnr[1:]), s=10, c=colors[i], alpha=0.8,
                       edgecolors='k',linewidths=0.2)
        else:
            ax.scatter(1./freqs, csnr[1:], s=10, c=colors[i], alpha=0.8,
                       edgecolors='k',linewidths=0.2)
    if (logsnr==True):
        ax.plot(1./freqs,np.log10(7)*np.ones(len(freqs)),'r--')
        ax.text(0.45,np.log10(3.5),'SNR = 7',fontsize=12, color="red")
        ax.set_ylabel('Logarithmic SNR')
        ax.set_ylim([0.2,5]);
    else:
        ax.plot(1./freqs,7*np.ones(len(freqs)),'r--')
        ax.text(0.45,3.5,'SNR = 7',fontsize=12, color="red")
        ax.set_ylabel('SNR')
        ax.set_ylim([0.2,5*10**2]);
    ax.set_xlim([0,np.max(1./freqs)]);
    ax.set_xlabel('Period (s)')
    ax.set_title('SNR ' + title_str)
    cbar = fig.colorbar(cm.ScalarMappable(norm=norm, cmap=cmap), cax=cbaxes, norm=norm,ticks=bounds,
                  boundaries=bounds,spacing='uniform',orientation='vertical')
    cbar.set_label('Interstation Distance (km)')
    cbar.set_ticks(bounds2)
    cbar.set_ticklabels(bounds_tick)
    # save plot to file
    if (savefig == True):
        if (logsnr==True):
            fig.savefig(dat_dir+'_PLOTS/CT_LogSNR.png',dpi=400,bbox_inches = "tight")
        else:
            fig.savefig(dat_dir+'_PLOTS/CT_SNR.png',dpi=400,bbox_inches = "tight")
            
#%%
def plot_cp_snr2(dist,snrs,freqs,title_str,dat_dir,evry,logsnr=True,savefig=False):
    
    bounds_tick = []
    bounds = np.linspace(0,5,len(dist[::evry]))
    bounds2 = np.linspace(0,5,11)
    for b2 in bounds2:
        bounds_tick.append(str(b2))
    # cmap = cm.inferno
    cmap = cm.get_cmap('inferno', len(bounds))
    norm = mpl.colors.BoundaryNorm(bounds, cmap.N)
    colors = get_colors(dist,cmap,vmin=0,vmax=5)[::evry]
    
    fig, ax = plt.subplots(figsize=(4,3),dpi=400)
    cbaxes = fig.add_axes([0.92, 0.125, 0.02, 0.75])
    for i, csnr in enumerate(snrs[::evry]):
        ax.scatter(1./freqs, csnr[1:], s=10, c=colors[i], alpha=0.8,
                       edgecolors='k',linewidths=0.2)
        ax.plot(1./freqs,7*np.ones(len(freqs)),'r--')
        ax.text(0.45,2.5,'SNR = 7',fontsize=12, color="red")
    if (logsnr==True):
        ax.set_yscale('log')
        ax.set_ylabel('Logarithmic SNR')
        ax.set_ylim([1,2*10**5]);
    else:
        pass
        ax.set_ylabel('SNR')
        ax.set_ylim([0.2,5*10**2]);
    ax.set_xlim([0,np.max(1./freqs)]);
    ax.set_xlabel('Period (s)')
    ax.set_title('SNR ' + title_str)
    cbar = fig.colorbar(cm.ScalarMappable(norm=norm, cmap=cmap), cax=cbaxes, norm=norm,ticks=bounds,
                  boundaries=bounds,spacing='uniform',orientation='vertical')
    cbar.set_label('Interstation Distance (km)')
    cbar.set_ticks(bounds2)
    cbar.set_ticklabels(bounds_tick)
    # save plot to file
    if (savefig == True):
        if (logsnr==True):
            fig.savefig(dat_dir+'_PLOTS/CT_LogSNR.png',dpi=400,bbox_inches = "tight")
        else:
            fig.savefig(dat_dir+'_PLOTS/CT_SNR.png',dpi=400,bbox_inches = "tight")


#%% Plot Center Periods VS NWL

def plot_cp_nwl(dist,nwvs,freqs,title_str,dat_dir,evry,lognwl=False,savefig=False):
        
    bounds_tick = []
    bounds = np.linspace(0,5,len(dist[::evry]))
    bounds2 = np.linspace(0,5,11)
    for b2 in bounds2:
        bounds_tick.append(str(b2))
    # cmap = cm.inferno
    cmap = cm.get_cmap('inferno', len(bounds))
    norm = mpl.colors.BoundaryNorm(bounds, len(bounds))
    colors = get_colors(dist,cmap,vmin=0,vmax=5)[::evry]
        
    fig, ax = plt.subplots(figsize=(4,3),dpi=400)
    cbaxes = fig.add_axes([0.92, 0.125, 0.02, 0.75])
    for i, cnwv in enumerate(nwvs[::evry]):
        if (lognwl==True):
            ax.scatter(1./freqs, np.log10(cnwv), s=10, c=colors[i], cmap=cmap, alpha=0.8,
                       edgecolors='k',linewidths=0.2)
        else:
            ax.scatter(1./freqs, cnwv, s=10, c=colors[i], cmap=cmap, alpha=0.8,
                       edgecolors='k',linewidths=0.2)
    if (lognwl==True):
        ax.plot(1./freqs,np.log10(3*np.ones(len(freqs))),'r--')
        ax.text(0.8,np.log10(5),'NWL = 3',fontsize=12, color="red")
        ax.set_ylim([-1,np.log10(50)]);
        ax.set_ylabel('Logarithmic NWL')
    else:
        ax.plot(1./freqs,3*np.ones(len(freqs)),'r--')
        ax.text(0.8,7,'NWL = 3',fontsize=12, color="red")
        ax.set_ylim([0,50]);
        ax.set_ylabel('NWL')
    ax.set_xlim([0,np.max(1./freqs)]);
    ax.set_xlabel('Period (s)')
    
    ax.set_title('NWL ' + title_str)
    cbar = fig.colorbar(cm.ScalarMappable(norm=norm, cmap=cmap), cax=cbaxes, norm=norm,ticks=bounds,
                 boundaries=bounds,spacing='uniform',orientation='vertical')
    cbar.set_label('Interstation Distance (km)')
    cbar.set_ticks(bounds2)
    cbar.set_ticklabels(bounds_tick)
    # save plot to file
    if (savefig == True):
        if (lognwl==True):
            fig.savefig(dat_dir+'_PLOTS/CT_LogNWL.png',dpi=400,bbox_inches = "tight")
        else:
            fig.savefig(dat_dir+'_PLOTS/CT_NWL.png',dpi=400,bbox_inches = "tight")

#%% Plot NWL VS SNR based on Distance

def plot_nwl_snr(dist,nwvs,snrs,freqs,title_str,dat_dir,evry,savefig=False):
    
    bounds_tick = []
    bounds = np.linspace(0,5,len(dist[::evry]))
    bounds2 = np.linspace(0,5,11)
    for b2 in bounds2:
        bounds_tick.append(str(b2))
    # cmap = cm.inferno
    cmap = cm.get_cmap('inferno', len(bounds))
    norm = mpl.colors.BoundaryNorm(bounds, len(bounds))
    colors = get_colors(dist,cmap,vmin=0,vmax=5)[::evry]
    
    fig, ax = plt.subplots(figsize=(4,3),dpi=400)
    cbaxes = fig.add_axes([0.92, 0.125, 0.02, 0.75])
    # ax2 = ax.twinx()
    for i, cnwv in enumerate(nwvs[::evry]):
        ax.scatter(cnwv,np.log10(snrs[i][1:]), s=10, c=colors[i], cmap=cmap, alpha=0.8,
                   edgecolors='k',linewidths=0.2)
    l1=ax.plot(np.linspace(0,125,len(freqs)),np.log10(7)*np.ones(len(freqs)),'b--',label='SNR = 7')
    l2=ax.plot(3*np.ones(len(snrs[0])),np.linspace(0,5.2,len(snrs[0])),'r--',label='NWL = 3')
    ax.legend()
    ax.set_xlim([0,60]); ax.set_ylim([0,5.2]);
    ax.set_xlabel('NWL')
    ax.set_ylabel('Logarithmic SNR')
    ax.set_title('SNR VS NWL ' + title_str)
    cbar = fig.colorbar(cm.ScalarMappable(norm=norm, cmap=cmap), cax=cbaxes, norm=norm,ticks=bounds,
                 boundaries=bounds,spacing='uniform',orientation='vertical')
    cbar.set_label('Interstation Distance (km)')
    cbar.set_ticks(bounds2)
    cbar.set_ticklabels(bounds_tick)
    # save plot to file
    if (savefig == True):
        fig.savefig(dat_dir+'_PLOTS/NWL_LogSNR.png',dpi=400,bbox_inches = "tight")

#%% Plot Interstation Distance VS Maximum T based on NWL

def plot_dist_maxT(dist,periods,nwvl,nwvs,title_str,dat_dir,savefig=False):
    
    # cmap = cm.hot
    cmap = cm.get_cmap('hot', len(nwvl))
    colors2 = cm.hot(np.linspace(0,1,len(nwvl)))
    fig, ax = plt.subplots(figsize=(4,3),dpi=400)
    for k, nwv in enumerate(nwvl):
        max_T = []
        for i, cnwv in enumerate(nwvs):
            try:
                max_T.append(periods[np.max(np.where(cnwv>=nwv)[0])])
            except:
                max_T.append(np.nan)
        ax.scatter(dist, max_T, s=20, c=colors2[k], cmap=cmap, edgecolors='black',
                   linewidths=0.5, alpha=0.7)
    ax.set_xlim([0,np.max(dist)])
    ax.set_ylim([0,np.max(periods[:-1])])
    ax.set_xlabel('Interstation Distance (km)')
    ax.set_ylabel('Maximum Period (s)')
    ax.set_title('Maximum Period ' + title_str)
    ax.legend(nwvl,title="(LNWL)",fontsize='small',prop={'size': 8})
    # save plot to file
    if (savefig == True):
        fig.savefig(dat_dir+'_PLOTS/DIST_MAXT.png',dpi=400,bbox_inches = "tight")

#%% Plot raw group velocity curve

def plot_raw_gv(dist,periods,gvs,title_str,dat_dir,evry,savefig=False):
    
    bounds_tick = []
    bounds = np.linspace(0,5,len(dist[::evry]))
    bounds2 = np.linspace(0,5,11)
    for b2 in bounds2:
        bounds_tick.append(str(b2))
    # cmap = cm.inferno
    cmap = cm.get_cmap('inferno', len(bounds))
    norm = mpl.colors.BoundaryNorm(bounds, len(bounds))
    colors = get_colors(dist,cmap,vmin=0,vmax=5)[::evry]
    
    fig, ax = plt.subplots(figsize=(4,3),dpi=400)
    cbaxes = fig.add_axes([0.92, 0.125, 0.02, 0.75])
    lines=[]
    for i, gv in enumerate(gvs):
        ax.plot(periods, gv,linewidth=3,c=colors[i])
    ax.set_xlim([0,np.max(periods)]); plt.ylim([0.4,3]);
    ax.set_xlabel('Period (s)')
    ax.set_ylabel('Group Velocity (km/s)')
    ax.set_title('Raw GV ' + title_str)
    cbar = fig.colorbar(cm.ScalarMappable(norm=norm, cmap=cmap), cax=cbaxes, norm=norm,ticks=bounds,
                 boundaries=bounds,spacing='uniform',orientation='vertical')
    cbar.set_label('Interstation Distance (km)')
    cbar.set_ticks(bounds2)
    cbar.set_ticklabels(bounds_tick)
    # save plot to file
    if (savefig == True):
        fig.savefig(dat_dir+'_PLOTS/Raw_GV.png',dpi=400,bbox_inches = "tight")

#%% Plot each quality controlled gv curves

def plot_qc_gv(dist,periods,gvs,nwvs,nwvl,nwvh,snrs,snrlim,title_str,dat_dir,evry,plotfig=True,savefig=False):
    
    bounds_tick = []
    bounds = np.linspace(0,5,len(dist[::evry]))
    bounds2 = np.linspace(0,5,11)
    for b2 in bounds2:
        bounds_tick.append(str(b2))
    # cmap = cm.inferno
    cmap = cm.get_cmap('inferno', len(bounds))
    norm = mpl.colors.BoundaryNorm(bounds, len(bounds))
    colors = get_colors(dist,cmap,vmin=0,vmax=5)[::evry]
    
    gv_clean = []; gv_err = []
    err_mean = []; qc_comb = []
    for k, cnwvl in enumerate(nwvl):
        for l, cnwvh in enumerate(nwvh):
            for j, csnrlim in enumerate(snrlim):
                if (plotfig==True):
                    fig, ax = plt.subplots(figsize=(4,3),dpi=400)
                    cbaxes = fig.add_axes([0.92, 0.125, 0.02, 0.75])
                gv_all = [];
                gv_clean_ind = []; gv_clean_count = np.zeros(len(gvs[0]))
                for i, cdist in enumerate(dist):
                    gv = np.empty((len(gvs[i]),)); gv[:] = np.nan
                    snr_ind = np.where(snrs[i] >= csnrlim)
                    nwl_ind = np.where((nwvs[i] >= cnwvl) & (nwvs[i] <= cnwvh))
                    snr_nwl_ind=np.intersect1d(snr_ind,nwl_ind, return_indices=True)[0]
                    gv[snr_nwl_ind] = gvs[i][snr_nwl_ind]
                    if (plotfig==True):
                        ax.plot(periods, gv,linewidth=3,c=colors[i])
                    gv_cl = np.empty((len(gvs[i]),)); gv_cl[:] = np.nan
                    try:
                        gv_clean_fix_ind = np.add(np.where(~np.isnan(gv[snr_nwl_ind]))[0],snr_nwl_ind[0])
                        gv_cl[gv_clean_fix_ind] = gvs[i][gv_clean_fix_ind]
                        gv_clean_count[gv_clean_fix_ind] += int(1)
                    except:
                        gv_clean_fix_ind = np.array([])
                    gv_all.append(gv_cl)
                    gv_clean_ind.append(gv_clean_fix_ind)
                    
                if (plotfig==True):
                    ax.text(0.05,2.35,'SNR: '+ str(csnrlim)+' LNWL: '+ str(cnwvl)+' UNWL: '+ str(cnwvh))
                    ax.set_title('Clean GV ' + title_str,fontsize=10)
                    # ax.set_xlim([0,np.max(periods)]); ax.set_ylim([0.6,2.6]);
                    ax.set_xlim([0,1]); ax.set_ylim([0.6,2.6]);
                    ax.set_xlabel('Period (s)')
                    ax.set_ylabel('Group Velocity (km/s)')
                    # Colorbar for interstation distance
                    cbar = fig.colorbar(cm.ScalarMappable(norm=norm, cmap=cmap), cax=cbaxes, norm=norm,ticks=bounds,
                                 boundaries=bounds,spacing='uniform',orientation='vertical')
                    cbar.set_label('Interstation Distance (km)')
                    cbar.set_ticks(bounds2)
                    cbar.set_ticklabels(bounds_tick)
                    #save plot to file
                    if (savefig == True):
                        fig.savefig(dat_dir+'_PLOTS/GV_QC_'+str(csnrlim)+'_'+str(cnwvl)+'_'+
                                    str(cnwvh)+'.png',dpi=400,bbox_inches = "tight")
                
                # Get network average group velocity curve
                gv_c = np.empty(len(gv_all[0])); gv_c[:] = np.nan
                for m in range(len(gv_all)):
                    gv_sum = np.nansum(np.dstack((gv_c,gv_all[m])),2)[0]
                    gv_sum[np.where(gv_sum == 0)] = np.nan
                    gv_c = gv_sum
                gv_c /= gv_clean_count
                gv_clean.append(gv_c)
                
                # Get std of each periods
                err_list = []
                for l in range(len(gv_c)):
                    err_c = np.array(list(zip(*gv_all))[l])
                    err_c = err_c[~np.isnan(err_c)]
                    err_list.append(np.std(err_c))
                    # try:
                    #     weights = np.ones_like(err_c)*(1/len(err_c))
                    # except:
                    #     weights = np.ones_like(err_c)
                    # weights = np.ones_like(err_c)*0.05
                    # weighted_stats = DescrStatsW(err_c, weights=weights, ddof=0)
                    # err_list.append(weighted_stats.std_mean)
                gv_err.append(err_list)
                err_mean.append(np.round(np.nanmean(err_list),5))
                qc_comb.append([cnwvl,cnwvh,csnrlim])
    
    # Sort mean error array and get indices
    sort_indx = np.argsort(err_mean)
    err_mean = [err_mean[i] for i in sort_indx]
    qc_comb = [qc_comb[i] for i in sort_indx]
    gv_err = [gv_err[i] for i in sort_indx]
    gv_clean = [gv_clean[i] for i in sort_indx]
                
    return gv_clean,gv_err,err_mean,qc_comb,gv_clean_ind

#%% Plot network averaged clean gv curves and save output

def plot_net_gv(periods,gv_clean,gv_err,err_mean,qc_comb,title_str,dat_dir,DIR,\
                n_plots,savefig=False,savegv=False):
    
    lstyle = ['-','--','-.',':']
    linecycler = cycle(lstyle)
    
    min_err_ind = np.where(np.asarray(err_mean) < 0.15)[0]
    min_err_ind = min_err_ind[:n_plots]
    colors2 = cm.jet(np.linspace(0,1,len(min_err_ind)))
    
    leg_list = []
    plt.figure(figsize=(4,3),dpi=400)
    for i,gv_ind in enumerate(min_err_ind):
        linestyle = {"linestyle":next(linecycler), "linewidth":0.5, "elinewidth":5, "capsize":10}
        leg_list.append('LNWL: '+str(qc_comb[gv_ind][0])+', UNWL: '+str(qc_comb[gv_ind][1])+
                        ', SNR: '+str(qc_comb[gv_ind][2])+', ERR = '+str(np.round(err_mean[gv_ind],3)))
        plt.errorbar(periods, gv_clean[gv_ind], yerr=gv_err[gv_ind],
                     elinewidth=2/(i+1),linestyle=next(linecycler),c=colors2[i])
    plt.legend(leg_list,fontsize='small',prop={'size': 6},loc = 'upper left')   
    plt.xlim([min(periods),1]); plt.ylim([0.4,1.8]);
    # plt.xlim([0.15,0.7]); plt.ylim([0.75,1.8]);
    # plt.xlim([0.6,1]); plt.ylim([1,1.7]);
    plt.locator_params(axis="y", nbins=7)
    plt.xlabel('Period (s)')
    plt.ylabel('Group Velocity (km/s)')
    plt.title('Network Averaged GV Curve ' + title_str)
    
    # Save clean gv plot to file
    if (savefig == True):
        plt.savefig(dat_dir+'_PLOTS/Clean_GV.png',dpi=400,bbox_inches = "tight")
    if (savegv == True):
        np.savez(DIR+'kernels/'+'xcor_disp_clean_pcc.dat',periods=periods,
             gv_clean=gv_clean[min_err_ind[0]],gv_err=gv_err[min_err_ind[0]])
    
    return min_err_ind

#%%
def plot_clean_gv_pcc_tcc(periods,gv_clean,gv_err,err_mean,qc_comb,gv_clean2,\
                          gv_err2,err_mean2,qc_comb2,dat_dir,DIR,savefig=False):
    
    lstyle = ['--','-.']
    linecycler = cycle(lstyle)
    colors2 = cm.jet(np.linspace(0,1,2))
    leg_list = []
    plt.figure(figsize=(4,3),dpi=400)
    linestyle = {"linestyle":next(linecycler), "linewidth":0.3, "elinewidth":0.5, "capsize":7}
    leg_list.append('PCC-PWS, '+'LNWL: '+str(qc_comb[0][0])+', UNWL: '+str(qc_comb[0][1])+
                    ', SNR: '+str(qc_comb[0][2])+', ERR = '+str(np.round(err_mean[0],3)))
    leg_list.append('TCC-Lin, '+'LNWL: '+str(qc_comb2[0][0])+', UNWL: '+str(qc_comb2[0][1])+
                    ', SNR: '+str(qc_comb2[0][2])+', ERR = '+str(np.round(err_mean2[0],3)))
    plt.errorbar(periods, gv_clean[0], yerr=gv_err[0],linestyle=next(linecycler),c=colors2[0])
    plt.errorbar(periods, gv_clean2[0], yerr=gv_err2[0],linestyle=next(linecycler),c=colors2[1])
    plt.legend(leg_list,fontsize='small',prop={'size': 7},loc = 'upper left')   
    plt.xlim([min(periods),1]); plt.ylim([0.4,1.6]);
    plt.locator_params(axis="y", nbins=7)
    plt.xlabel('Period (s)')
    plt.ylabel('Group Velocity (km/s)')
    plt.title('Network Averaged GV Curve')
    
    # Save clean gv plot to file
    if (savefig == True):
        plt.savefig(dat_dir+'_PLOTS/Clean_GV_PCC_TCC.png',dpi=400,bbox_inches = "tight")

#%% Plot all clean gv's and error for each periods

def plot_all_gv(periods,gv_clean,gv_err,title_str,dat_dir,DIR,savefig=False):
    
    lstyle = ['-','--','-.',':']
    linecycler = cycle(lstyle)
    colors2 = cm.jet(np.linspace(0,1,len(gv_clean)))
    
    plt.figure(figsize=(4,3),dpi=400)
    for i,gv_ind in enumerate(gv_clean):
        plt.errorbar(periods, gv_clean[i], yerr=gv_err[i],
                     elinewidth=2/(i+1),linestyle=next(linecycler),c=colors2[i]) 
    # plt.xlim([0,np.max(periods)]); plt.ylim([0.6,1.7]);
    plt.xlim([0,1]); plt.ylim([0.6,1.7]);
    plt.locator_params(axis="y", nbins=7) 
    plt.xlabel('Period (s)')
    plt.ylabel('Group Velocity (km/s)')
    plt.title('All Clean GV Curves ' + title_str)
    
    # Save clean gv plot to file
    if (savefig == True):
        plt.savefig(dat_dir+'_PLOTS/All_Clean_GV.png',dpi=400,bbox_inches = "tight")

#%% Plot all clean gv's and error for each periods

def plot_clean_av_gv(periods,gv_clean,title_str,dat_dir,DIR,savefig=False):
    
    lstyle = ['-','--','-.',':']
    linecycler = cycle(lstyle)
    colors2 = cm.jet(np.linspace(0,1,len(gv_clean)))
    
    # Get mean and std of each periods
    gv_all_mean = np.nanmean(gv_clean,axis=0)
    gv_all_err = np.nanstd(gv_clean,axis=0)
    
    plt.figure(figsize=(4,3),dpi=400)
    plt.errorbar(periods, gv_all_mean, yerr=gv_all_err,lw=2,ls='-',c='black',
                 ecolor='gray', capsize=5, capthick=2, solid_capstyle='projecting')
    # plt.xlim([0,np.max(periods)]); plt.ylim([0.6,1.7]);
    plt.xlim([0,1]); plt.ylim([0.5,1.7]);
    plt.locator_params(axis="y", nbins=7) 
    plt.xlabel('Period (s)')
    plt.ylabel('Group Velocity (km/s)')
    plt.title('Mean Clean GV Curves ' + title_str)
    
    # Save clean gv plot to file
    if (savefig == True):
        plt.savefig(dat_dir+'_PLOTS/Mean_Clean_GV.png',dpi=400,bbox_inches = "tight")
        
#%% Plot change in SNR of EGFs vs number of stacks

def plot_stcks_snr(dist,p_snrs,title_str,dat_dir,evry,logsnr=True,savefig=False):
    
    bounds_tick = []
    bounds = np.linspace(0,5,len(dist[::evry]))
    bounds2 = np.linspace(0,5,11)
    for b2 in bounds2:
        bounds_tick.append(str(b2))
    cmap = cm.inferno
    norm = mpl.colors.BoundaryNorm(bounds, cmap.N)
    colors = get_colors(dist,cmap,vmin=0,vmax=5)[::evry]
    stck_arr = np.linspace(1,len(p_snrs[0,:,0]),len(p_snrs[0,:,0]))
    
    fig, ax = plt.subplots(figsize=(4,3),dpi=400)
    cbaxes = fig.add_axes([0.92, 0.125, 0.02, 0.75])
    for i, pcsnr in enumerate(p_snrs[::evry]):
        if (logsnr == True):
            ax.scatter(stck_arr, np.log10(pcsnr[:,0]), s=10, c=colors[i], alpha=0.8,
                       edgecolors='k',linewidths=0.2)
            ax.plot(stck_arr,np.log10(7*np.ones(len(stck_arr))),'r--')
            ax.text(100,np.log10(3),'SNR = 7',fontsize=10, color="red")
            ax.set_ylabel('Logarithmic SNR')
        if (logsnr == False):
            ax.scatter(stck_arr, pcsnr[:,0], s=10, c=colors[i], alpha=0.8,
                       edgecolors='k',linewidths=0.2)
            ax.plot(stck_arr,7*np.ones(len(stck_arr)),'r--')
            ax.text(100,-8,'SNR = 7',fontsize=10, color="red")
            ax.set_ylabel('SNR')
    ax.set_xlabel('# of Stacks')
    ax.set_xlim([0,np.max(stck_arr)]);
    ax.set_title('Change in SNR from # of stacks ' + title_str)
    cbar = fig.colorbar(cm.ScalarMappable(norm=norm, cmap=cmap), cax=cbaxes, norm=norm,ticks=bounds,
                  boundaries=bounds,spacing='uniform',orientation='vertical')
    cbar.set_label('Interstation Distance (km)')
    cbar.set_ticks(bounds2)
    cbar.set_ticklabels(bounds_tick)
    
    # Save plot to file
    if (savefig == True):
        if (logsnr==True):
            fig.savefig(dat_dir+'_PLOTS/CNstacks_LogSNR.png',dpi=400,bbox_inches = "tight")
        else:
            fig.savefig(dat_dir+'_PLOTS/Nstacks_SNR.png',dpi=400,bbox_inches = "tight")

#%% Quality control for specific parameters

def qcontrol_single(gvc,snr_s,nwv_s,c_snrlim,c_nwvl,c_nwvh):
    
    gv = np.empty((len(gvc),)); gv[:] = np.nan
    snr_ind = np.where(snr_s >= c_snrlim)
    nwl_ind = np.where((nwv_s >= c_nwvl) | (nwv_s >= c_nwvh))
    snr_nwl_ind = np.intersect1d(snr_ind,nwl_ind, return_indices=True)[0]
    gv[snr_nwl_ind] = gvc[snr_nwl_ind]
    gv_cl = np.empty((len(gvc),)); gv_cl[:] = np.nan
    
    try:
        gv_clean_fix_ind = np.add(np.where(~np.isnan(gv[snr_nwl_ind]))[0],snr_nwl_ind[0])
        gv_cl[gv_clean_fix_ind] = gvc[gv_clean_fix_ind]
    except:
        gv_clean_fix_ind = snr_nwl_ind
    
    return gv_cl, gv_clean_fix_ind

#%% Compute number of stacks to reach reliable dispersion

def nstkcs_for_gv(periods,gvs,rmse_cut,p_gvs,snrs,p_snrs,nwvs,p_nwvs,c_snrlim,c_nwvl,c_nwvh):
    
    gvs_nstcks = []
    gvs_nstcks_count = np.empty((len(periods),)); gvs_nstcks_count[:]= 0
    gvs_nstcks_mcount = np.empty((len(periods),)); gvs_nstcks_mcount[:]= 0
    
    for i,cegf in enumerate(p_gvs):
        gv_cl,gv_clean_fix_ind = qcontrol_single(gvs[i],snrs[i][1:],nwvs[i],
                                                 c_snrlim,c_nwvl,c_nwvh)
        gv_nstcks = np.empty((len(periods),)); gv_nstcks[:]= np.nan
        gv_nstcks_count = np.empty((len(periods),)); gv_nstcks_count[:]= 0
        for k, gvc_stck in enumerate(cegf):
            gv_cl2,gv_clean_fix_ind2 = qcontrol_single(gvc_stck,p_snrs[i,k][1:],
                                                       p_nwvs[i,k],c_snrlim,c_nwvl,c_nwvh)
            gvcl_int_ind = np.intersect1d(gv_clean_fix_ind,gv_clean_fix_ind2, return_indices=True)[0]
            try:
                gv_rmse = np.sqrt(mse(gv_cl[gvcl_int_ind],gv_cl2[gvcl_int_ind]))
                if (gv_rmse <= rmse_cut):
                    gv_nan_ind = np.where(np.isnan(gv_nstcks[gvcl_int_ind]))[0]
                    nan_int_ind = gvcl_int_ind[gv_nan_ind]
                    gv_nstcks[nan_int_ind] = int(k)
            except:
                pass
        gvs_nstcks_count[np.where(~np.isnan(gv_nstcks))[0]] += int(1)
        gvs_nstcks_mcount = np.nansum(np.dstack((gvs_nstcks_mcount,gv_nstcks)),2)
        gvs_nstcks.append(gv_nstcks)
    gv_nstcks_full = np.round(np.divide(gvs_nstcks_mcount[0],gvs_nstcks_count))
    
    return gv_nstcks_full, gvs_nstcks

#%% Compute number of stacks to reach reliable dispersion

def nstkcs_for_gv2(periods,gvs,rmse_cut,p_gvs,snrs,p_snrs,nwvs,p_nwvs,c_snrlim,c_nwvl,c_nwvh):
    
    gvs_nstcks = []
    gvs_nstcks_count = np.empty((len(periods),)); gvs_nstcks_count[:]= 0
    gvs_nstcks_mcount = np.empty((len(periods),)); gvs_nstcks_mcount[:]= 0
    
    for i,cegf in enumerate(p_gvs):
        gv_cl,gv_clean_fix_ind = qcontrol_single(gvs[i],snrs[i][1:],nwvs[i],
                                                 c_snrlim,c_nwvl,c_nwvh)
        gv_nstcks = np.empty((len(periods),)); gv_nstcks[:]= np.nan
        gv_nstcks_count = np.empty((len(periods),)); gv_nstcks_count[:]= 0
        for k, gvc_stck in enumerate(cegf):
            gv_cl2,gv_clean_fix_ind2 = qcontrol_single(gvc_stck,p_snrs[i,k][1:],
                                                       p_nwvs[i,k],c_snrlim,c_nwvl,c_nwvh)
            gvcl_int_ind = np.intersect1d(gv_clean_fix_ind,gv_clean_fix_ind2, return_indices=True)[0]
            for l,int_idx in enumerate(gvcl_int_ind):
                try:
                    gv_rmse = np.sqrt(mse(np.asarray([gv_cl[int_idx]]),
                                          np.asarray([gv_cl2[int_idx]])))
                    if (gv_rmse <= rmse_cut) and (np.isnan(gv_nstcks[int_idx])):
                        gv_nstcks[int_idx] = int(k)
                except:
                    pass
        gvs_nstcks_count[np.where(~np.isnan(gv_nstcks))[0]] += int(1)
        gvs_nstcks_mcount = np.nansum(np.dstack((gvs_nstcks_mcount,gv_nstcks)),2)
        gvs_nstcks.append(gv_nstcks)
    gv_nstcks_full = np.round(np.divide(gvs_nstcks_mcount[0],gvs_nstcks_count))
    
    return gv_nstcks_full, gvs_nstcks

#%% Plot number of stacks for each period to meet reasonable quality of gv

def plot_nstcks_for_gv(periods,periods2,gv_nstcks_full,gv_nstcks_full2,rmse_cut,xvar='Period',yvar='Stacks'):
    
    fig, ax = plt.subplots(figsize=(4,3),dpi=400)
    if (xvar == 'Period'):
        if (yvar == 'Stacks'):
            ax.scatter(periods, gv_nstcks_full, s=10, c='black', alpha=0.8, edgecolors='k',linewidths=0.2)
            ax.scatter(periods2, gv_nstcks_full2, s=10, c='red', alpha=0.8, edgecolors='k',linewidths=0.2)
            ax.set_xlim([0.05,1])
            ax.set_xlabel('Period (s)')
            ax.set_ylabel('# of Stacks')
            ax.set_title('# of Stacks required, RMSE cut: '+str(rmse_cut),fontsize=10)
        else:
            ax.scatter(periods, gv_nstcks_full*20/60, s=10, c='black', alpha=0.8, edgecolors='k',linewidths=0.2)
            ax.scatter(periods2, gv_nstcks_full2*20/60, s=10, c='red', alpha=0.8, edgecolors='k',linewidths=0.2)
            ax.set_xlim([0.05,1])
            ax.set_xlabel('Period (s)')
            ax.set_ylabel('# of Hours')
            ax.set_title('# of Hours required, RMSE cut: '+str(rmse_cut),fontsize=10)
    else:
        if (yvar == 'Stacks'):
            ax.scatter(1/periods, gv_nstcks_full, s=10, c='black', alpha=0.8, edgecolors='k',linewidths=0.2)
            ax.scatter(1/periods2, gv_nstcks_full2, s=10, c='red', alpha=0.8, edgecolors='k',linewidths=0.2)
            ax.set_xlim([1,20])
            ax.set_xlabel('Frequency (Hz)')
            ax.set_ylabel('# of Stacks')
        else:
            ax.scatter(1/periods, gv_nstcks_full*20/60, s=10, c='black', alpha=0.8, edgecolors='k',linewidths=0.2)
            ax.scatter(1/periods2, gv_nstcks_full2*20/60, s=10, c='red', alpha=0.8, edgecolors='k',linewidths=0.2)
            ax.set_xlim([1,20])
            ax.set_xlabel('Frequency (Hz)')
            ax.set_ylabel('# of Hours')
    ax.legend(['PCC-PWS','Time-Lin'])
    

#%% Plot Cosine-Similarity between two methods

def plot_similarity(dist,periods,sim1,gv_clean_ind,sim_type,savefig=False):
    
    bounds_tick = []
    bounds = np.linspace(0,5,len(dist))
    bounds2 = np.linspace(0,5,11)
    for b2 in bounds2:
        bounds_tick.append(str(b2))
    cmap = cm.inferno
    norm = mpl.colors.BoundaryNorm(bounds, cmap.N)
    colors = get_colors(dist,cmap,vmin=0,vmax=5)
    
    fig, ax = plt.subplots(figsize=(4,3),dpi=400)
    cbaxes = fig.add_axes([0.92, 0.125, 0.02, 0.75])
    for i, csim in enumerate(sim1):
        try:
            sim_arr =  np.empty((len(periods),)); sim_arr[:] = np.nan
            sim_arr[gv_clean_ind[i]] = csim[gv_clean_ind[i]]
            ax.scatter(periods, sim_arr, s=10, c=colors[i],
                       cmap=cmap, alpha=0.8, edgecolors='k',linewidths=0.2)
        except:
            pass
    ax.set_xlim([0,1]); ax.set_ylim([0.1,1])
    ax.set_ylabel('Correlation Coefficient')
    ax.set_xlabel('Period (s)')
    ax.set_title(sim_type.split('_')[0]+' '+sim_type.split('_')[1]+' between methods')
    cbar = fig.colorbar(cm.ScalarMappable(norm=norm, cmap=cmap), cax=cbaxes, norm=norm,ticks=bounds,
                 boundaries=bounds,spacing='uniform',orientation='vertical')
    cbar.set_label('Interstation Distance (km)')
    cbar.set_ticks(bounds2)
    cbar.set_ticklabels(bounds_tick)
    # save plot to file
    if (savefig == True):
        fig.savefig(dat_dir+'_PLOTS/'+sim_type+'.png',dpi=400,bbox_inches = "tight")
