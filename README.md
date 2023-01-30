PYTHON code for measuring seismic travel-time changes with the wavelet method

Contact: Han Byul Woo (hanbyulwoo@gmail.com)
This PYTHON script package contains codes and test data for plotting quality of dispersion measurements obtained from two ambient-noise data processing methods.
1. Phase cross-correlation and phase-weighted stacking
2. Time cross-correlation and linear stacking

PYTHON version 3.9 was used to run the script and following packages are required to run the scripts.
1.numpy 2.scipy 3.csv 4.sklearn 5.matplotlib

Table of contents:

—— quality_control_aux.py: Core functions to plot quality parameters and quality controlled group velocity curves. The script also includes estimation of reproducibility and find the number of progressive stacks required to have a root-mean-squared error value away from a selected reference network-averaged group velocity curve.

—— plot_quality_control.py: Loads dispersion measurements and quality (signal-to-noise ratio and number of wavelengths) to plot the quality and quality controlled group velocity curves. 

—— 

—— synthetic_dvov_0.05percent.mat: Two synthetic waveforms for testing the codes. The synthetic seismograms are generated using velocity models of a homogeneous background superimposed by random heterogeneities. The perturbation between the current and reference velocity models is a homogeneous increase of 0.05% dv/v throughout the medium. (If interested, see Section 3.1 in the following reference for more details.)
