"""#TODO: 
Script containing functions to analyze variation in fiber photometry data. 
[link to github]
[contact info]
"""

import numpy as np 
import matplotlib.pyplot as plt 
import pandas as pd 
import pickle 
import scipy.stats as stats
import seaborn as sns


def calculate_OnOffVariation(channel, time, behavioronset, behavioroffset, save=False, savepath='', behaviorname='', channelname='', narrow_time=0): 
    #Calculate the coefficient of variation of each onset/offset interval of a behavior 
    #Specify narrow time if want to offset time used to calculate variation to be a different length 
    variation = [] 
    
    if len(behavioronset) != len(behavioroffset):
        print('Error: Onset and offset arrays are not the same length')
        return 1
    if len(channel) != len(time):
        print('Error: Channel and time arrays are not the same length')
        return 1

    for i in range(len(behavioronset)):
        idx = np.where((time >= (behavioronset[i] + narrow_time)) & (time <= (behavioroffset[i] - narrow_time))) #Obtain index of time array that is within the onset/offset window
        behavior_data = channel[idx] #Obtain data from time array that is within the onset/offset window
        behavior_variation = stats.variation(behavior_data) #Calculate coefficient of variation of channel data corresponding to the onset/offset windows
        mean = np.mean(behavior_data)
        std = np.std(behavior_data)
        if behavior_variation != std/mean:
            print(f'Scipy variation: {behavior_variation}. Manual variation: {std/mean}') #Double check that scipy and manual calculations are the same

        #variation.append(std)
        variation.append(behavior_variation)

    if save == True:
        with open(f'{savepath}/{behaviorname}_{channelname}_OnOffVariation.pkl', 'wb') as file:
            pickle.dump(variation, file)
        return 

    return variation


def calculate_WindowVariation(channel, time, behavior, window, save=False, savepath='', behaviorname=''):
    #Calculate the coefficient of variation in a window of time around a behavior, e.g. [-300, 300] for 300 seconds before and after a behavior time
    #For behaviors without an onset/offset window 

    variation = []
    if len(channel) != len(time):
        print('Error: Channel and time arrays are not the same length')
        return 1
    behavior = [i for i in behavior if ((i+window[0]) > 0)]

    for i in range(len(behavior)):
        idx = np.where((time >= behavior[i]+window[0]) & (time <= behavior[i]+window[1]))
        behavior_data = channel[idx]
        behavior_variation = stats.variation(behavior_data, nan_policy='raise')
        mean = np.mean(behavior_data)
        std = np.std(behavior_data)
        if behavior_variation != std/mean:
            print(f'Scipy variation: {behavior_variation}. Manual variation {std/mean}')

        variation.append(behavior_variation)

    if save == True:
        with open(f'{savepath}/{behaviorname}_WindowVariation.pkl', 'wb') as file:
            pickle.dump(variation, file)
        return 

    return variation


def group_variations(variations=[], load_data=False, data_paths=[], behaviorname=''): 
    #Combine variations from different trials into one array to summarize 

    if load_data == True:
        for path in data_paths:
            with open(path, 'rb') as file:
                variations.append(pickle.load(file))

    variations = [x for list in variations for x in list] #Flatten nasted list of variations 
    df = pd.DataFrame()
    df[behaviorname] = variations
    print(f'{behaviorname} Variation: {np.mean(variations)}')

    return variations, df

