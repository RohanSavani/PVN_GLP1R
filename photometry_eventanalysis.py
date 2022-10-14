"""#TODO: 
Script containing all functions used to analyze a behavioral event with fiber photometry data.
[link to github]
[contact info]
"""

import numpy as np
import yaml 
import pandas as pd 
from scipy.integrate import simps
from scipy.signal import find_peaks

def event_analysis(time=[], data=[], epochs=[], shade='sem', saveresults=False, savepath='', load_data=False, datapath='', datachannel='SlowDFF',load_behaviors=False, behaviorpath='', behavior='', baseline_start=-10, baseline_end=0, view_window=[-100, 100], method='no_baseline', checklength=False):
    """
    Calculate mean fluoerescence response to a behavioral event recorded around a view window (epoc at t=0)

    Parameters
    ----------
    time : array
        Array of time values to match data 
    data : array
        Array of data values (typically normalized dF/F) to match time
    epochs : array
        Array of behavior times to match data
    shade : str
        'sem' or 'std' to calculate standard error of the mean (sem) or standard deviation (std) of the response
    saveresults : bool
        Whether to save the results to an h5 file
    savepath : str
        Path to save the results to
    load_data : bool
        Whether to load data from an h5 file, which should contain a df with a time column and a data column
    datapath : str
        Path to load the data from
    datachannel : str
        Name of the data channel column to load from the h5 file's dataframe 
    load_behaviors : bool
        Whether to load behavior data from an h5 file
    behaviorpath : str
        Path to load the behavior data from
    behavior : str
        Name of the behavior column to load from the h5 file's dataframe
    baseline_start : int
        Time (in seconds) to start the baseline period, relative to t=0 as the event 
    baseline_end : int
        Time (in seconds) to end the baseline period, relative to t=0 as the event
    view_window : list
        List of two integers, the start and end time (in seconds) of the view window, relative to t=0 as the event
    method : str
        Whether to perform baseline subtraction using a baseline period or not (if not, baseline start/end are not used)

    Returns
    -------
    df : DataFrame containing trace data for each behavioral event (each event is a column, with each row being a timepoint of data collection)
    mean_peth : array
        Array of mean fluorescence response to behavioral event
    std_peth : array
        Array of standard deviation of fluorescence response to behavioral event
    sem_peth : array   
        Array of standard error of the mean fluorescence response to behavioral event
    if saveresults: 
        Saves dataframe to h5 file in savepath
    """

    if load_data == True: 
        dfd = pd.read_hdf(datapath)
        time = dfd['Time']
        data = dfd[datachannel]
    
    if load_behaviors == True:
        with open(behaviorpath, 'r') as file:
            behaviors=yaml.load(file, Loader=yaml.FullLoader)
        epochs = behaviors[behavior]

    if method == 'baseline_subtraction': #Assumes input data is already normalized 
    #Assumes events are already timestamp corrected and filtered; that is, they are at least min_time apart and view_window away from recording beginning and end   
        new_columns = {} 
        for epoc in epochs: 
            baseline_idx = np.where((time>epoc+baseline_start) & (time<epoc+baseline_end))[0]
            baseline_average = np.nanmean(data[baseline_idx])
            epoc_data = data[np.where((time>epoc+view_window[0]) & (time<epoc+view_window[1]))[0]] #Center data around epoch
            if checklength:
                while len(epoc_data) > (view_window[1] - view_window[0]):
                    print(f'shortening data (len: {len(epoc_data)})to match view window (len: {(view_window[1] - view_window[0])})...')
                    epoc_data = epoc_data[:-1]
            epoc_data = np.subtract(epoc_data, baseline_average) #Subtract baseline from all data 
            new_columns[epoc] = pd.Series(epoc_data)

        df = pd.DataFrame.from_dict(new_columns).dropna()

    elif method == 'no_baseline':
        #Similar to GuPPy, except no baseline subtraction 
        new_columns = {} 
        for epoc in epochs: 
            epoc_data = data[np.where((time>epoc+view_window[0]) & (time<epoc+view_window[1]))[0]]
            new_columns[epoc] = pd.Series(epoc_data)
        df = pd.DataFrame.from_dict(new_columns).dropna()
    
    if saveresults == True:
        print('saving...')
        df.to_hdf(savepath, key='df', mode='w')
    mean_peth = df.mean(axis=1).to_numpy()
    if shade == 'std':
        std_peth = df.std(axis=1).to_numpy()
        return df, mean_peth, std_peth

    elif shade == 'sem':
        sem_peth = df.sem(axis=1).to_numpy()
        return df, mean_peth, sem_peth


def group_analysis(paths, behaviorfilename, shade='sem'):
    '''
    Generate group mean and sem/std of fluorescence response to behavioral event across multiple recordings 

    Parameters
    ----------
    paths : list
        List of paths to folder containing h5 files with behavior-centered signal data
    behaviorfilename : str
        Name of the behavior file to load from each folder 
    shade : str
        'sem' or 'std' to calculate standard error of the mean (sem) or standard deviation (std) of the response
    
    Returns
    -------
    df : DataFrame containing trace data for each behavioral event (
    mean_peth : array
        Array of mean fluorescence response to behavioral event
    std_peth : array
        Array of standard deviation of fluorescence response to behavioral event
    '''

    df = pd.DataFrame()
    verify_columns = []
    for path in paths:
        df2 = pd.read_hdf(path + behaviorfilename)
        print(len(df2))
        verify_columns.append(len(df2.columns))
        df = pd.concat([df, df2], axis=1, ignore_index=True)

    if len(df.columns) != sum(verify_columns):
        print(f'Length of sum df != summed dfs. {len(df.columns)} vs {sum(verify_columns)}')

    mean_peth = df.mean(axis=1).to_numpy()

    if shade == 'std':
        std_peth = df.std(axis=1).to_numpy()
        return df, mean_peth, std_peth

    elif shade == 'sem':
        sem_peth = df.sem(axis=1).to_numpy()
        return df, mean_peth, sem_peth

def get_AUC(time, data, start, end):
    #Calculate area under the curve marked by start and end using Simpson's rule for integration 
    idx = np.where((time>start) & (time<end))[0]
    auc = simps(data[idx], time[idx])
    return auc

def find_tracepeaks(data, threshold=[0.15, 5], distance=10):
    #Input dF/F data, find peaks 
    peaks, = find_peaks(data, threshold=threshold, distance=distance)
    #for peaks in peaks:
        #break
    

    return peaks 













