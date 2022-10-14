"""#TODO: 
Collection of functions to visualize data from fiber photometry and behavioral experiments in different ways 
[link to github]
[contact info]
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns 
import pandas as pd 
import math
import yaml
from photometry_correcting import *
import matplotlib.patches as patches
from matplotlib.lines import Line2D
import pickle
from matplotlib.widgets import RadioButtons
import warnings 
from matplotlib.ticker import FormatStrFormatter

def visualize_behaviordistribution(ax, behavioronsets=[], behavioroffsets=[], behavior_name='', label1='', binsize=200, both=False, behavioronsets2=[], behavioroffsets2=[], label2='', loadbehaviors=False, paths=[], behavior_dictname='', paths2=[], behavior_dictname2=''): 
    #Visualize distribution of a behavioral event's duration
    #Flatten list of behavioronsets and behavioroffsets into one 
    #Can plot both manual and DLC together 
    sumonset = []
    sumoffset = []
    if loadbehaviors: #load behavior onsets and offsets from yaml file if True 
        for path in paths: 
                with open(path, 'r') as file:
                    behaviors=yaml.load(file, Loader=yaml.Loader)
                sumonset.append(behaviors[f'{behavior_dictname}_onsets'])
                sumoffset.append(behaviors[f'{behavior_dictname}_offsets']) 
    
        behavioronsets = sumonset 
        behavioroffsets = sumoffset
        if both: #if plotting two behaviors side by side 
            sumonset2 = []
            sumoffset2 = []
            for path in paths2: 
                with open(path, 'r') as file:
                    behaviors=yaml.load(file, Loader=yaml.Loader)
                sumonset2.append(behaviors[f'{behavior_dictname2}_onsets'])
                sumoffset2.append(behaviors[f'{behavior_dictname2}_offsets'])
            behavioronsets2 = sumonset2
            behavioroffsets2 = sumoffset2

    #Flatten onsets/offsets into one list if nested, and check if both are same length
    if all(isinstance(x, list) for x in behavioronsets):
        behavioronsets = np.asarray([item for sublist in behavioronsets for item in sublist])
        behavioroffsets = np.asarray([item for sublist in behavioroffsets for item in sublist])
    if len(behavioronsets) != len(behavioroffsets):
        print('behavioronsets and behavioroffsets are not the same length. ')

    if both == True: 
        if all(isinstance(x, list) for x in behavioronsets2):
            behavioronsets2 = np.asarray([item for sublist in behavioronsets2 for item in sublist])
            behavioroffsets2 = np.asarray([item for sublist in behavioroffsets2 for item in sublist])
        if len(behavioronsets2) != len(behavioroffsets2):
            print('behavioronsets2 and behavioroffsets2 are not the same length. ')

    #Duration of behavior (offset - onset)
    behavior_duration = behavioroffsets - behavioronsets
    mean = np.mean(behavior_duration)
    std = np.std(behavior_duration)
    n = len(behavior_duration)
    
    if both == True:
        behavior_duration2 = behavioroffsets2 - behavioronsets2
        mean2 = np.mean(behavior_duration2)
        std2 = np.std(behavior_duration2)
        n2 = len(behavior_duration2)
        if max(behavior_duration) > max(behavior_duration2): #calculate list of bins for plotting based on desired bin size and range of behavior durations 
           bins = np.arange(0, (math.ceil(max(behavior_duration)/binsize)*binsize)+binsize, binsize) #if first behavior has highest duration 
           ax.set_xlim(0, (math.ceil(max(behavior_duration)/binsize)*binsize)+binsize, binsize)
        else:
            bins = np.arange(0, (math.ceil(max(behavior_duration2)/binsize)*binsize)+binsize, binsize) #if second behavior has highest duration 
            ax.set_xlim(0,(math.ceil(max(behavior_duration2)/binsize)*binsize)+binsize, binsize)

        g = sns.distplot(behavior_duration2, bins=bins, hist=True, kde=True, color='red', hist_kws={'edgecolor':'black', 'alpha': 0.5}, kde_kws={'linewidth': 1}, label=label2, ax=ax)
        ax.set_title(f'Distribution of {behavior_name} Duration, n = {n} Events, n2 = {n2} Events')
        ax.text(0.95, 0.95, f'{label1} Mean: {mean:.2f} \n{label1} STD: {std:.2f} \n{label2} Mean: {mean2:.2f} \n{label2} STD: {std2:.2f}', horizontalalignment='right', verticalalignment='top', transform=ax.transAxes, fontsize=12)
    
    else:
        bins = np.arange(0, (math.ceil(max(behavior_duration)/binsize)*binsize)+binsize, binsize)
        ax.set_title(f'Distribution of {behavior_name} Duration, n = {n} Events')
        ax.text(0.95, 0.95, f'Mean: {mean:.2f} s\nSD: {std:.2f} s', transform=ax.transAxes, horizontalalignment='right', verticalalignment='top')
        ax.set_xlim(0, (math.ceil(max(behavior_duration)/binsize)*binsize)+binsize, binsize)
    g = sns.distplot(behavior_duration, bins=bins, hist=True, kde=True, color='blue', hist_kws={'edgecolor':'black', 'alpha': 0.5}, kde_kws={'linewidth': 1}, label=label1, ax=ax)
    ax.set_xlabel('Seconds')    
    ax.set_ylabel('Probability Density') 
    ax.set_xticks(bins)
    ax.set_xticklabels(bins)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
    ax.legend(loc='center right')

    return ax 


def visualize_raw(time, isos, signal): 
    #Plot raw isosbestic and signal traces 

    #If channels do not have overlapping signal intensities, zoom them in a little bit to make them more comparable by cropping y axis 
    if max(isos) + 20 < min(signal):

        fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(8,5))
        fig.subplots_adjust(hspace=0.05)
        # Plotting the traces
        for ax in [ax1, ax2]:
            ax.plot(time, signal, linewidth=1, color='green', label='Signal')
            ax.plot(time, isos, linewidth=1, color='blueviolet', label='ISOS')

        ax2.set_ylim(min(isos)-10, max(isos)+10)
        ax1.set_ylim(min(signal)-10,max(signal)+10)

        ax1.spines.bottom.set_visible(False)
        ax2.spines.top.set_visible(False)
        ax1.xaxis.tick_top()
        ax1.tick_params(labeltop=False)  # don't put tick labels at the top
        ax2.xaxis.tick_bottom()

        d = .5  # proportion of vertical to horizontal extent of the slanted line
        kwargs = dict(marker=[(-1, -d), (1, d)], markersize=12,
                    linestyle="none", color='k', mec='k', mew=1, clip_on=False)
        ax1.plot([0, 1], [0, 0], transform=ax1.transAxes, **kwargs)
        ax2.plot([0, 1], [1, 1], transform=ax2.transAxes, **kwargs)

        ax1.set_ylabel('mV')
        ax2.set_xlabel('Seconds')
        ax1.set_title('Raw Trace with Buffer Time Removed')
        ax1.legend(loc='upper right')
        fig.tight_layout()
    #If not, just plot the raw traces without cropping the y axis 
    else: 
        fig = plt.figure(figsize=(10,3))
        ax1 = fig.add_subplot(111)
        # Plotting the traces
        ax1.plot(time, signal, linewidth=1, color='green', label='Signal')
        ax1.plot(time, isos, linewidth=1, color='blueviolet', label='ISOS')

        ax1.set_ylabel('mV')
        ax1.set_xlabel('Seconds')
        ax1.set_title('Raw Trace with Buffer Time Removed')
        ax1.legend(loc='upper right')
        fig.tight_layout()
        

def visualize_event(ax, df, mean_peth, std_peth, event_name, view_window): 
    #Visualize the event triggered average fluorescence and its standard deviation or standard error 
    #Requires dataframe from event_analysis in order to calculate amount of behavioral events 
    time = np.linspace(view_window[0], view_window[1], num=len(mean_peth))

    try:
        #fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8,5))
        ax.plot(time, mean_peth, linewidth=0.5, color='green', label='Mean Response')
        ax.fill_between(time, mean_peth-std_peth, mean_peth+std_peth, color='green', alpha=0.2)
        eventcount = len(df.columns)
        ax.axvline(0, color='black', linewidth=0.5, linestyle='--', alpha=0.5)
        ax.axhline(0, color='black', linewidth=0.5, linestyle='--', alpha=0.5)
        ax.set_ylabel('z-score')
        ax.set_xlabel('Seconds')
        ax.set_title(f'PETH for {event_name}, total of {eventcount} events')
        ax.legend(loc='upper right')
    except Exception as e:
        print(f'Error. Time: {len(time)}. MeanPETH: {len(mean_peth)}. STD_PETH: {len(std_peth)}')
        print(e)
    return ax 

def align_yaxis(ax1, ax2):
    '''Align y axis of subplots to give same origin
    Args:
        ax1 (matplotlib.axes._subplots.AxesSubplot): first subplot
        ax2 (matplotlib.axes._subplots.AxesSubplot): second subplot
    '''
    y_lims = np.array([ax.get_ylim() for ax in [ax1, ax2]])

    # force 0 to appear on both axes, comment if don't need
    y_lims[:, 0] = y_lims[:, 0].clip(None, 0)
    y_lims[:, 1] = y_lims[:, 1].clip(0, None)

    # normalize both axes
    y_mags = (y_lims[:,1] - y_lims[:,0]).reshape(len(y_lims),1)
    y_lims_normalized = y_lims / y_mags

    # find combined range
    y_new_lims_normalized = np.array([np.min(y_lims_normalized), np.max(y_lims_normalized)])

    # denormalize combined range to get new axes
    new_lim1, new_lim2 = y_new_lims_normalized * y_mags
    ax1.set_ylim(new_lim1)
    ax2.set_ylim(new_lim2)


def visualize_slow(fig, ax1, time, signal, isos, streamrate, fed_epocs, behaviors, lambda_=8e5, window=100, dff=True): 
    #Visualize the variaton, signal/isos channels, and behaviors (feeding and immobility) of a single recording 
    time = time[::int(streamrate)]/3600 #Convert to hours
    
    #Calculate variation of signal and isos channels, variation shown is max within 10 second window 
    arr = pd.Series(signal)
    mean = arr.rolling(window=window).mean()
    std = arr.rolling(window=window).std()
    variation = std/mean
    variation = variation.rolling(window=int(10*streamrate)).max()
    variation = variation[::int(streamrate)]
    ax1.plot(time, variation, color='green', linewidth=0.25, label='465 Variation')

    arr = pd.Series(isos)
    mean = arr.rolling(window=window).mean()
    std = arr.rolling(window=window).std()
    variation = std/mean
    variation = variation.rolling(window=int(10*streamrate)).max()
    variation = variation[::int(streamrate)]
    ax1.plot(time, variation, color='blue', linewidth=0.25, alpha=0.5, label='405 Variation')

    ax1.set_xlabel('Hours')
    ax1.set_ylabel('variation')


    ax2 = ax1.twinx()
    ax2.set_ylabel('Z-Score')

    #Calculate dF/F of signal and isos channels and display on overlapped axis
    signal = signal[::int(streamrate)]
    isos = isos[::int(streamrate)]

    signal_baseline = airPLS(signal, lambda_=lambda_, itermax=28)
    isos_baseline = airPLS(isos, lambda_=lambda_, itermax=28)


    if dff: 
        signal = normalize_channel(delta_FF(signal_baseline, signal))
        isos = normalize_channel(delta_FF(isos_baseline, isos))
    else:
        signal = normalize_channel(signal-signal_baseline)
        isos = normalize_channel(isos-isos_baseline)

    signal = moving_average(signal)
    isos = moving_average(isos)

    ax2.plot(time, signal, color='red', alpha=0.5, label='465 z-score')
    ax2.plot(time, isos, color='purple', alpha=0.5, label='405 z-score')

    for i in fed_epocs:
        plt.axvline(i/3600,linewidth=0.1, color='black',alpha=.3)

    align_yaxis(ax1, ax2)

    #Plot behaviors on ax as well 
    for idx, val in enumerate(behaviors['immobility_onsets']):
        rect = patches.Rectangle((val/3600, -9), width=(behaviors['immobility_offsets'][idx]-val)/3600, height=1, color='violet', alpha=0.4)
        ax2.add_patch(rect)

    for idx, val in enumerate(behaviors['meal_onsets']):
        rect = patches.Rectangle((val/3600, -10), width=(behaviors['meal_offsets'][idx]-val)/3600, height=1, color='coral', alpha=0.4)
        ax2.add_patch(rect)

    plt.ylim(bottom=-10)

    #Create custom legend shown to the side of the figure 
    legend_elements = [Line2D([0], [0], color = 'green', lw=3, label=f'465 Variation'),
                    Line2D([0], [0], color = 'blue', lw=3, label=f'405 Variation'),
                    Line2D([0], [0], color = 'red', lw=3, label=f'465 dF/F (z)'),
                    Line2D([0], [0], color = 'purple', lw=3, label=f'405 dF/F (z)'),
                    Line2D([0], [0], color = 'coral', lw=3, label=f'Meal')]
    ax1.legend(handles=legend_elements, loc='center left', bbox_to_anchor=(1.15, 0.5), frameon=False, handlelength=0.75, handleheight=1, edgecolor=None)
    fig.subplots_adjust(right=0.7) #Adjust figure space to allow for legend 

    return ax1, ax2  


def visualize_all(fig, ax1, time, signal, isos, streamrate, fed_epocs, behaviors, signal_lambda, isos_lambda, window=100, plot_variations=True, plot_seconds=True, plot_behaviors=True, limit_z=False, normmethod='median'):
    #Visualize the variaton, signal/isos channels, and behaviors (feeding and immobility) of a single recording 
    warnings.simplefilter(action='ignore', category=UserWarning)
    time = time[::int(streamrate)]/3600 #Convert to hours

    if plot_variations:
        #Calculate variation of signal and isos channels, variation shown is max within 10 second window 
        arr = pd.Series(signal)
        mean = arr.rolling(window=window).mean()
        std = arr.rolling(window=window).std()
        variation = std/mean
        variation = variation.rolling(window=int(10*streamrate)).max()
        variation = variation[::int(streamrate)]
        ax1.plot(time, variation, color='green', linewidth=0.25, label='465 Variation')

        arr = pd.Series(isos)
        mean = arr.rolling(window=window).mean()
        std = arr.rolling(window=window).std()
        variation = std/mean
        variation = variation.rolling(window=int(10*streamrate)).max()
        variation = variation[::int(streamrate)]
        ax1.plot(time, variation, color='blue', linewidth=0.25, alpha=0.5, label='405 Variation')

        ax1.set_ylabel('variation')

    ax2 = ax1.twinx()
    ax2.set_ylabel('Z-Score')

    #Calculate dF/F of signal and isos channels and display on overlapped axis
    smoothsignal = moving_average(signal, window_size=1000)#, window_size=500)
    baseline = airPLS(smoothsignal[::int(streamrate)], lambda_=signal_lambda, itermax=28)
    #dFF1 = delta_FF(baseline, smoothsignal[::int(streamrate)])
    #signal_dFF = normalize_channel(dFF1, method=normmethod)

    signal_dFF = smoothsignal[::int(streamrate)]-baseline
    signal_dFF = normalize_channel(signal_dFF, method=normmethod)


    smoothisos = moving_average(isos, window_size=1000)#, window_size=500)
    baseline = airPLS(smoothisos[::int(streamrate)], lambda_=isos_lambda, itermax=28)
    dFF1 = delta_FF(baseline, smoothisos[::int(streamrate)])
    isos_dFF = normalize_channel(dFF1, method=normmethod)

    ax2.plot(time, signal_dFF, color='red', alpha=0.75, label='465 z-score', linewidth=0.25)
    ax2.plot(time, isos_dFF, color='purple', alpha=0.5, label='405 z-score', linewidth=0.25)

    for i in fed_epocs:
        ax2.axvline(i/3600, color='black',linewidth=0.1, alpha=.3)

    align_yaxis(ax1, ax2)

    #Plot behaviors on ax as well 
    if plot_behaviors: 
        for idx, val in enumerate(behaviors['immobility_onsets']):
            rect = patches.Rectangle((val/3600, -9), width=(behaviors['immobility_offsets'][idx]-val)/3600, height=1, color='violet', alpha=0.4)
            ax2.add_patch(rect)

        for idx, val in enumerate(behaviors['meal_onsets']):
            rect = patches.Rectangle((val/3600, -10), width=(behaviors['meal_offsets'][idx]-val)/3600, height=1, color='orange', alpha=0.4)
            ax2.add_patch(rect)

        for idx, val in enumerate(behaviors['wiggle_onsets']):
            rect = patches.Rectangle((val/3600, -8), width=(behaviors['wiggle_offsets'][idx]-val)/3600, height=1, color='red', alpha=0.4)
            ax2.add_patch(rect)

    plt.ylim(bottom=-10)
    if limit_z: 
        if ax2.get_ylim()[1] > 25:
            plt.ylim(top=25)

    #Create custom legend shown to the side of the figure 
    legend_elements = [Line2D([0], [0], color = 'green', lw=3, label=f'465 Variation'),
                    Line2D([0], [0], color = 'blue', lw=3, label=f'405 Variation'),
                    Line2D([0], [0], color = 'red', lw=3, label=f'465 dF/F (z)'),
                    Line2D([0], [0], color = 'purple', lw=3, label=f'405 dF/F (z)'),
                    Line2D([0], [0], color = 'violet', lw=3, label=f'Immobility'),
                    Line2D([0], [0], color = 'orange', lw=3, label=f'Meal'),
                    Line2D([0], [0], color = 'red', lw=3, label=f'Wiggle')]

    ax1.legend(handles=legend_elements, loc='center left', bbox_to_anchor=(1.15, 0.75), frameon=False, handlelength=0.75, handleheight=1, edgecolor=None)
    fig.subplots_adjust(right=0.7) #Adjust figure space to allow for legend

    ax1.set_xlabel('Hours', fontsize=10)
    ax2.set_xlabel('Hours', fontsize=10)

    if plot_seconds:
        secax = ax1.secondary_xaxis('top', functions=(to_seconds, to_hours))
        secax.set_xlabel('Seconds', fontsize=10)
    ax1.xaxis.set_tick_params(labelsize=10)
    secax.xaxis.set_tick_params(labelsize=10)
    ax2.xaxis.set_tick_params(labelsize=10)
    #fig.tight_layout()
    #plt.show()
    return ax1, ax2  

def save_fig(path, fig): 
    #Save interactive figure as pickle file that can be opened at any time 
    with open(path + '/PhotometryAnnotations.pkl','wb') as file:
        pickle.dump(fig, file)

def load_fig(path):
    #Load interactive figure from pickle file
    with open(path + '/PhotometryAnnotations.pkl','rb') as file:
        fig = pickle.load(file)
    return fig 

def to_seconds(x):
    return x*3600
def to_hours(x):
    return x/3600














