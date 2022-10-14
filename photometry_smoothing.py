"""#TODO: 
Script containing all functions used to smooth out fiber photometry data
[link to github]
[contact info]
"""

import numpy as np
from scipy.ndimage import uniform_filter1d
from scipy import signal as ss 


def simple_average(data, time, average_over=100): 
    #Averages every 10 data points together and decimates time array to match 
    #Input already offset data stream, offset time array, and number of data points to average over
    #Returns np.array of smoothed data, with truncated time array to match 

    smoothed_data = []

    for i in range(0, len(data), average_over): # iterate through data stream, averaging every 10 points
        smoothed_data.append(np.mean(data[i:i+average_over])) # append averaged data to smoothed_data

    time = time[::average_over] # decimate time array to match smoothed_data
    time = time[:len(smoothed_data)] # truncate time array to match smoothed_data

    return np.array(smoothed_data), time 

def moving_average(data, window_size=100, method='scipy'):
    #Returns smoothed data using a moving average window of size window_size
    #Method can be 'scipy' or 'numpy', to test which is faster/more accurate -- scipy is default
    if method == 'scipy':
        smoothed_data = uniform_filter1d(data, size=window_size)
    
    elif method == 'numpy': 
        s = np.r_[data[window_size-1:0:-1],data,data[-2:-window_size-1:-1]] #pad data with reflected copies of signal at both ends 
        w = np.ones(window_size,'d') #create window of ones
        smoothed_data = np.convolve(w/w.sum(),s,mode='valid') 
        smoothed_data = smoothed_data[(int(window_size/2)):(int(window_size/2)+len(data))] #remove reflected copies of signal at both ends

    return smoothed_data

def zerophase_filter(data, filter_window=100): 
    #Filter high frequency noise from channel data with a window of size filter_window
    b = np.divide(np.ones((filter_window,)), filter_window)
    a = 1
    smoothed_data = ss.filtfilt(b, a, data)
    
    return smoothed_data



