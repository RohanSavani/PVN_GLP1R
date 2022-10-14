"""#TODO: 
Script containing all functions used to fit, scale, and baseline correct fiber photometry data
[link to github]
[contact info]
"""

import numpy as np
from scipy.sparse import csc_matrix, eye, diags
from scipy.sparse.linalg import spsolve
import scipy.stats as stats
from sklearn.linear_model import Lasso 
import pandas as pd 
from scipy.optimize import curve_fit
from photometry_smoothing import * 

def normalize_channel(data, time=[], method='mean', baseline_start=0, baseline_end=100):
    #normalize a datastream with various methods
    #time array only necessary if normalizing stream with a baseline window 

    if method == 'baselineperiod': 
        idx = np.where((time > baseline_start) & (time < baseline_end))[0]
        if idx.shape[0]==0:
            raise Exception('Baseline params for calculation not correct.')
        else:
            baseline_mean = np.nanmean(data[idx]) 
            baseline_std = np.nanstd(data[idx])
            numerator = np.subtract(data, baseline_mean)
            normalized_channel = np.divide(numerator, baseline_std)

    if method == 'baselineperiodmedian':
        idx = np.where((time > baseline_start) & (time < baseline_end))[0]
        if idx.shape[0]==0:
            raise Exception('Baseline params for calculation not correct.')
        else:
            median = np.nanmedian(data[idx]) 
            mad = np.median(np.absolute(data[idx]-median))
            numerator = np.subtract(data, median)
            normalized_channel = np.divide(numerator, mad)

    if method == 'mean': 
        numerator = np.subtract(data, np.nanmean(data))
        normalized_channel = np.divide(numerator, np.nanstd(data))

    if method == 'median': #Uses median and median absolute deviation to normalize 
        median = np.median(data)
        mad = np.median(np.absolute(data-median))
        numerator = data-median
        normalized_channel = np.divide(numerator, mad)

    return normalized_channel 


def linear_fitting(isos, signal, method): 
    #Fit a linear model of isosbestic channel to signal channel, returning scaled isosbestic channel 
    if method == 'simple': #least-squares polynomial fit of degree 1 
        p = np.polyfit(isos, signal, 1)
        scaled_channel = np.multiply(p[0], isos)+p[1]

    if method == 'Lasso': #non-negative Lasso linear regression
        lin = Lasso(alpha=0.0001,precompute=True,max_iter=1000,
            positive=True, random_state=9999, selection='random')
        n = len(isos)
        lin.fit(isos.reshape(n,1), signal.reshape(n,1))
        scaled_channel = lin.predict(isos.reshape(n,1)).reshape(n,)

    return scaled_channel 

def delta_FF(isos, signal, method='unstandardized'):
    #Calculate dFF from scaled isosbestic channel and signal channel
    if method == 'unstandardized': #for unstandardized isos and signal
        num = np.subtract(signal, isos)
        dFF = np.divide(num, isos)
        dFF = dFF*100 

    if method == 'standardized': #Data already normalized in airPLS process 
        dFF = signal - isos

    return dFF 


# curve fit exponential function
def curveFitFn(x,a,b,c):
    return a+(b*np.exp(-(1/c)*x))

# create control channel by fitting signal channel to exponential 
def onetermexpfit(signal, time, window, p0=[5, 50, 60]):
	# check if window is greater than signal shape
	if window>signal.shape[0]:
		window = ((window+1)/2)+1
	filtered_signal = ss.savgol_filter(signal, window_length=window, polyorder=3)
	popt, pcov = curve_fit(curveFitFn, time, filtered_signal, p0)
	control = curveFitFn(time,*popt)

	return control

def twotermexpfit(xinc, yinc, p0=(0.8, 3e-4, 0.2 , 3e-4, 1)):
    popt, pcov = curve_fit(twotermexpfunc, xinc, yinc, p0=p0)
    new_y = twotermexpfunc(xinc, *popt)
    return new_y, popt

def twotermexpfunc(x,a,b,c,d,e):
    return (a*np.exp(-b*x)) + (c*np.exp(-d*x)) +e
    

'''
Below are functions for airPLS baseline correction
Obtained from https://github.com/zmzhang/airPLS
'''

def WhittakerSmooth(x,w,lambda_,differences=1):
    '''
    Penalized least squares algorithm for background fitting
    
    input
        x: input data (i.e. chromatogram of spectrum)
        w: binary masks (value of the mask is zero if a point belongs to peaks and one otherwise)
        lambda_: parameter that can be adjusted by user. The larger lambda is,  the smoother the resulting background
        differences: integer indicating the order of the difference of penalties
    
    output
        the fitted background vector
    '''
    X=np.matrix(x)
    m=X.size
    E=eye(m,format='csc')
    for i in range(differences):
        E=E[1:]-E[:-1] # numpy.diff() does not work with sparse matrix. This is a workaround.
    W=diags(w,0,shape=(m,m))
    A=csc_matrix(W+(lambda_*E.T*E))
    B=csc_matrix(W*X.T)
    background=spsolve(A,B)
    return np.array(background)

def airPLS(x, lambda_=100, porder=1, itermax=15):
    '''
    Adaptive iteratively reweighted penalized least squares for baseline fitting
    
    input
        x: input data (i.e. chromatogram of spectrum)
        lambda_: parameter that can be adjusted by user. The larger lambda is,  the smoother the resulting background, z
        porder: adaptive iteratively reweighted penalized least squares for baseline fitting
    
    output
        the fitted background vector
    '''
    m=x.shape[0]
    w=np.ones(m)
    for i in range(1,itermax+1):
        z=WhittakerSmooth(x,w,lambda_, porder)
        d=x-z
        dssn=np.abs(d[d<0].sum())
        if(dssn<0.001*(abs(x)).sum() or i==itermax):
            if(i==itermax): print('WARING max iteration reached!')
            break
        w[d>=0]=0 # d>0 means that this point is part of a peak, so its weight is set to 0 in order to ignore it
        w[d<0]=np.exp(i*np.abs(d[d<0])/dssn)
        w[0]=np.exp(i*(d[d<0]).max()/dssn) 
        w[-1]=w[0]
    return z









