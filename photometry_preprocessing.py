"""#TODO: 
Script containing functions to preprocess TDT and behavioral data for use in fiber photometry analysis. 
Supports manual annotations (as created by video_view and video_annotation) as well as DeepLabCut annotations (as created by DLC and deeplabcut_returnanalysis)
[link to github]
[contact info]
"""

import numpy as np 
import yaml 
import tdt 
from photometry_smoothing import moving_average

def preprocess_data(tdt_path, behaviors_path=None, savebehaviors=False, savetime=False, savesignal=False, saveisos=False, savestreamrate=False, saveepocs=False, annotationstyle='both', ISOS = '_405G', SIGNAL = '_465G', offsettime=5, min_time_apart=20, min_time=2700, immobility_time=300, base_meals='FED', roi=False): 
    """ Preprocesses data from TDT into a usable format, as well as offset it by buffer time 
    Filter behavioral data, subtracting buffer time from it, and filtering wiggles to every min_time_apart seconds

    Parameters
    -----------
    tdt_path : str
        Path to folder containing TDT data, without '/' (e.g. /Users/rohan/Desktop/TDT_data)
    behaviors_path : str
        Path to folder containing behavioral data, with '/' (e.g. /Users/rohan/Desktop/TDT_data/) (Default value = None)
    savebehaviors : bool
        Whether to save the processed behavioral data to a yaml file (Default value = False)
    savetime : bool
        Whether to save the processed time data to a npy file (Default value = False)
    savesignal : bool
        Whether to save the processed signal channel data to a npy file (Default value = False)
    saveisos : bool
        Whether to save the processed isosbestic channel data to a npy file (Default value = False)
    savestreamrate : bool
        Whether to save the streamrate of the channels to a npy file (Default value = False)
    saveepocs : bool
        Whether to save the FED device epocs to a npy file (Default value = False)
    annotationstyle : str
        Whether to use manual annotations or DeepLabCut annotations (Default value = None)
        Can be 'manual' or 'DLC' or 'both' 
    ISOS : str
        Name of isosbestic channel in TDT block
    SIGNAL : str
        Name of signal channel in TDT block
    offsettime : int
        Time to offset data by (in seconds) (Default value = 2)
    min_time_apart : int
        Minimum time apart to filter wiggles by in seconds (if two events < min_time_apart seconds, first one is recorded only) (Default value = 20)
    min_time : int
        Behavioral events must be at least min_time seconds from the recording's start and end (Default value = 1800)
    immobility_time : int
        Minimum time (seconds) for each immobility event to be (Default value = 300)
    base_meals : str
        Whether to use meals based on manual annotations ('annotations') or FED device data ('FED') (Default value = 'FED') 

    Returns 
    -----------
    newtime : np.array
        Array of time values corresponding to each data point in the TDT block, offset by offsettime
    signal_channel: np.array
        Array of signal channel values offset by offsettime
    isos_channel: np.array
        Array of isosbestic channel values offset by offsettime
    streamrate: float
        Streamrate of TDT block (Hz) 
    *behaviors: variable 
        Dependent on annotationstyle, can either return arrays of each behavior's timestamps or dicts (if == 'both') containing each behavior and its timestamps
    """

    data = tdt.read_block(tdt_path)
    
    streamrate = data.streams[SIGNAL].fs 
    if data.streams[ISOS].fs != data.streams[SIGNAL].fs: 
        raise Exception(f'Streamrates different for each. Isosbestic: {data.streams[ISOS].fs}. Signal: {data.streams[SIGNAL].fs}')
    duration = data.info.duration.total_seconds()
    #time = np.linspace(0, duration, num=len(data.streams[SIGNAL].data))
    time = np.arange(len(data.streams[SIGNAL].data))/streamrate #create time array in seconds to match experiment length
    index = np.where(time>offsettime)[0][0] #find index of first data point after offset time

    signal_channel = data.streams[SIGNAL].data[index:] #offset signal by buffer time
    isos_channel = data.streams[ISOS].data[index:] #offset isosbestic by buffer time
    newtime = time[np.where(time>offsettime)[0]] #update time array
    newtime = newtime - newtime[0] #subtract offsettime 

    if savetime:
        np.save(f'{tdt_path}/time.npy', newtime)
    if savesignal:
        np.save(f'{tdt_path}/signal.npy', signal_channel)
    if saveisos:
        np.save(f'{tdt_path}/isos.npy', isos_channel)
    if savestreamrate:
        np.save(f'{tdt_path}/streamrate.npy', streamrate)
    if saveepocs:
        epocs = data.epocs.Plet.onset - offsettime
        np.save(f'{tdt_path}/epocs.npy', epocs)


    if annotationstyle == 'manual': #Open file containing manual annotations 
        with open(behaviors_path + 'behaviors.yml', 'r') as file:
            behaviors=yaml.load(file, Loader=yaml.FullLoader)
        wiggles = behaviors['wiggling']
        drinking = behaviors['drinking']
        eating = behaviors['eating']
        immobility_onset = behaviors['sleeping_onset']
        immobility_offset = behaviors['sleeping_offset']
        rearing = behaviors['rearing']
        #Filter wiggling behaviors to be min_time into the recording and min_time_apart apart, as well as subtracted by offsettime
        wiggles = [epoc-offsettime for epoc in wiggles if (epoc > min_time) and (epoc < (time[-1]-min_time))]
        fixed_wiggles = [] 
        for i in range(len(wiggles)-1):
            if (wiggles[i+1]-wiggles[i]) > min_time_apart:
                fixed_wiggles.append(wiggles[i])
        #Filter all other behaviors to be min_time into the recording and subtracted by offsettime
        drinking = [epoc-offsettime for epoc in drinking if (epoc > min_time) and (epoc < (time[-1]-min_time))]
        eating = [epoc-offsettime for epoc in eating if (epoc > min_time) and (epoc < (time[-1]-min_time))]
        #Also ensure len(immobility onsets) equal len(immobility offsets)
        fixed_immobility_onset = [immobility_onset[i]-offsettime for i in range(len(immobility_onset)) if (immobility_onset[i]>min_time) and (immobility_offset[i]>min_time) and (immobility_onset[i]<(time[-1]-min_time)) and (immobility_offset[i]<(time[-1]-min_time)) and ((immobility_offset[i]-immobility_onset[i])>immobility_time)]
        fixed_immobility_offset = [immobility_offset[i]-offsettime for i in range(len(immobility_offset)) if (immobility_onset[i]>min_time) and (immobility_offset[i]>min_time) and (immobility_onset[i]<(time[-1]-min_time)) and (immobility_offset[i]<(time[-1]-min_time)) and ((immobility_offset[i]-immobility_onset[i])>immobility_time)]
        rearing = [epoc-offsettime for epoc in rearing if (epoc > min_time) and (epoc < (time[-1]-min_time))]
         
        if base_meals == 'annotations':
            meal_onset, meal_offset = calculate_meals(eating) #based on eating annotations 
        else:
            try:
                epocs = data.epocs.Plet.onset - offsettime
                meal_onset, meal_offset = calculate_meals2(newtime[::int(streamrate)], epocs, 600) #based on FED data
            except:
                print('calculating meals based on manual annotations...')
                meal_onset, meal_offset = calculate_meals(eating)

        fixed_meal_onset = [meal_onset[i] for i in range(len(meal_onset)) if (meal_onset[i] > min_time) and (meal_offset[i] > min_time) and (meal_onset[i] < (time[-1]-min_time)) and ((meal_offset[i] < (time[-1])-min_time))]
        fixed_meal_offset = [meal_offset[i] for i in range(len(meal_offset)) if (meal_onset[i] > min_time) and (meal_offset[i] > min_time) and (meal_onset[i] < (time[-1]-min_time)) and ((meal_offset[i] < (time[-1])-min_time))]


        #Remove onsets/offsets that are too close to each other (within 2 seconds) - Removes bugs with time arrays
        final_meal_onsets = fixed_meal_onset.copy()
        final_meal_offsets = fixed_meal_offset.copy()

        for i in range(len(fixed_meal_onset)-1):
            if fixed_meal_onset[i+1] - fixed_meal_offset[i] < 2:
                final_meal_onsets.pop(i+1)
                final_meal_offsets.pop(i)

        behaviors = { #Create dict of behaviors and their timestamps
            'wiggles': wiggles,
            'drinking': drinking,
            'eating': eating,
            'immobility_onsets': fixed_immobility_onset,
            'immobility_offsets': fixed_immobility_offset,
            'rearing': rearing,
            'meal_onsets': final_meal_onsets,
            'meal_offsets': final_meal_offsets
        }

        if savebehaviors == True: #save to yaml file 
            with open(behaviors_path + 'ProcessedManualBehaviors.yml', 'w+') as file:
                yaml.dump(behaviors, file, default_flow_style=False, sort_keys=False)

        return newtime, signal_channel, isos_channel, streamrate, fixed_wiggles, drinking, eating, meal_onset, meal_offset, fixed_immobility_onset, fixed_immobility_offset, rearing

    elif annotationstyle == 'DLC': #Open file containing DeepLabCut annotations
        with open(behaviors_path + 'DLCbehaviors.yml', 'r') as file:
            behaviors=yaml.load(file, Loader=yaml.FullLoader)
        #TODO: Add eating and drinking when refined 
        immobility_onset = behaviors['immobility_onsets']
        immobility_offset = behaviors['immobility_offsets']
        wiggle_onset = behaviors['wiggle_onsets']
        wiggle_offset = behaviors['wiggle_offsets']
        if roi:
            drink_onset = behaviors['drink_onsets']
            drink_offset = behaviors['drink_offsets']
            feed_enter = behaviors['feed_onsets']
            feed_exit = behaviors['feed_offsets']
            drink_onset = [epoc-offsettime for epoc in drink_onset if (epoc > min_time) and (epoc < (time[-1]-min_time))]
            drink_offset = [epoc-offsettime for epoc in drink_offset if (epoc > min_time) and (epoc < (time[-1]-min_time))]

        fixed_immobility_onset = [immobility_onset[i]-offsettime for i in range(len(immobility_onset)) if (immobility_onset[i]>min_time) and (immobility_offset[i]>min_time) and (immobility_onset[i]<(time[-1]-min_time)) and (immobility_offset[i]<(time[-1]-min_time)) and ((immobility_offset[i]-immobility_onset[i])>immobility_time)]
        fixed_immobility_offset = [immobility_offset[i]-offsettime for i in range(len(immobility_offset)) if (immobility_onset[i]>min_time) and (immobility_offset[i]>min_time) and (immobility_onset[i]<(time[-1]-min_time)) and (immobility_offset[i]<(time[-1]-min_time)) and ((immobility_offset[i]-immobility_onset[i])>immobility_time)]

        filt_wiggle_onset = [wiggle_onset[i]-offsettime for i in range(len(wiggle_onset)) if (wiggle_onset[i] > min_time) and (wiggle_offset[i] > min_time) and (wiggle_onset[i] < (time[-1]-min_time)) and ((wiggle_offset[i] < (time[-1])-min_time))]
        filt_wiggle_offset = [wiggle_offset[i]-offsettime for i in range(len(wiggle_offset)) if (wiggle_onset[i] > min_time) and (wiggle_offset[i] > min_time) and (wiggle_onset[i] < (time[-1]-min_time)) and ((wiggle_offset[i] < (time[-1])-min_time))]

        fixed_wiggle_onset = []
        fixed_wiggle_offset = []
        for i in range(len(fixed_immobility_offset)):
           for j in range(len(filt_wiggle_onset)):
                if (filt_wiggle_onset[j] > fixed_immobility_onset[i]) and (filt_wiggle_offset[j] < fixed_immobility_offset[i]):
                    fixed_wiggle_onset.append(filt_wiggle_onset[j]) 
                if (filt_wiggle_onset[j] > fixed_immobility_onset[i]) and (filt_wiggle_offset[j] < fixed_immobility_offset[i]): 
                    fixed_wiggle_offset.append(filt_wiggle_offset[j])
        
        try:
            epocs = data.epocs.Plet.onset - offsettime
            meal_onset, meal_offset = calculate_meals2(newtime[::int(streamrate)], epocs, 600) #based on FED data
        except Exception as e:
            print('Error in calculating meals based on epocs.' + e)

        fixed_meal_onset = [meal_onset[i] for i in range(len(meal_onset)) if (meal_onset[i] > min_time) and (meal_offset[i] > min_time) and (meal_onset[i] < (time[-1]-min_time)) and ((meal_offset[i] < (time[-1])-min_time))]
        fixed_meal_offset = [meal_offset[i] for i in range(len(meal_offset)) if (meal_onset[i] > min_time) and (meal_offset[i] > min_time) and (meal_onset[i] < (time[-1]-min_time)) and ((meal_offset[i] < (time[-1])-min_time))]

        #Remove onset and offset of meals that are too close to each other (within 2 seconds) - Removes bugs with time arrays
        final_meal_onsets = fixed_meal_onset.copy()
        final_meal_offsets = fixed_meal_offset.copy()

        for i in range(len(fixed_meal_onset)-1):
            if fixed_meal_onset[i+1] - fixed_meal_offset[i] < 2:
                final_meal_onsets.pop(i+1)
                final_meal_offsets.pop(i)

        behaviors = { #Create dict of behaviors and their timestamps
            'immobility_onsets': fixed_immobility_onset,
            'immobility_offsets': fixed_immobility_offset,
            'wiggle_onsets': fixed_wiggle_onset,
            'wiggle_offsets': fixed_wiggle_offset,
            'meal_onsets': final_meal_onsets,
            'meal_offsets': final_meal_offsets
            }
        if roi:
            behaviors['feed_onsets']: feed_enter
            behaviors['feed_offsets']: feed_exit
            behaviors['drink_onsets']: drink_onset
            behaviors['drink_offset']: drink_offset
        if savebehaviors == True:
            with open(behaviors_path + 'ProcessedDLCBehaviors.yml', 'w+') as file:
                yaml.dump(behaviors, file, default_flow_style=False, sort_keys=False)

        return newtime, signal_channel, isos_channel, streamrate, behaviors


    elif annotationstyle == 'both':
        with open(behaviors_path + 'behaviors.yml', 'r') as file:
            behaviors=yaml.load(file, Loader=yaml.FullLoader)
        wiggles = behaviors['wiggling']
        drinking = behaviors['drinking']
        eating = behaviors['eating']
        immobility_onset = behaviors['sleeping_onset']
        immobility_offset = behaviors['sleeping_offset']
        rearing = behaviors['rearing']

        wiggles = [epoc-offsettime for epoc in wiggles if (epoc > min_time) and (epoc < (time[-1]-min_time))]
        fixed_wiggles = [] 
        for i in range(len(wiggles)-1):
            if (wiggles[i+1]-wiggles[i]) > min_time_apart:
                fixed_wiggles.append(wiggles[i])

        drinking = [epoc-offsettime for epoc in drinking if (epoc > min_time) and (epoc < (time[-1]-min_time))]
        eating = [epoc-offsettime for epoc in eating if (epoc > min_time) and (epoc < (time[-1]-min_time))]
        fixed_immobility_onset = [immobility_onset[i]-offsettime for i in range(len(immobility_onset)) if (immobility_onset[i]>min_time) and (immobility_offset[i]>min_time) and (immobility_onset[i]<(time[-1]-min_time)) and (immobility_offset[i]<(time[-1]-min_time)) and ((immobility_offset[i]-immobility_onset[i])>immobility_time)]
        fixed_immobility_offset = [immobility_offset[i]-offsettime for i in range(len(immobility_offset)) if (immobility_onset[i]>min_time) and (immobility_offset[i]>min_time) and (immobility_onset[i]<(time[-1]-min_time)) and (immobility_offset[i]<(time[-1]-min_time)) and ((immobility_offset[i]-immobility_onset[i])>immobility_time)]
        rearing = [epoc-offsettime for epoc in rearing if (epoc > min_time) and (epoc < (time[-1]-min_time))]
        
        if base_meals == 'annotations':
            meal_onset, meal_offset = calculate_meals(eating) #based on eating annotations 
        else:
            try:
                epocs = data.epocs.Plet.onset - offsettime
                meal_onset, meal_offset = calculate_meals2(newtime[::int(streamrate)], epocs, 600) #based on FED data
            except:
                print('calculating meals based on manual annotations...')
                meal_onset, meal_offset = calculate_meals(eating)

        fixed_meal_onset = [meal_onset[i] for i in range(len(meal_onset)) if (meal_onset[i] > min_time) and (meal_offset[i] > min_time) and (meal_onset[i] < (time[-1]-min_time)) and ((meal_offset[i] < (time[-1])-min_time))]
        fixed_meal_offset = [meal_offset[i] for i in range(len(meal_offset)) if (meal_onset[i] > min_time) and (meal_offset[i] > min_time) and (meal_onset[i] < (time[-1]-min_time)) and ((meal_offset[i] < (time[-1])-min_time))]

        #Remove onset and offset of meals that are too close to each other (within 2 seconds) - Removes bugs with time arrays
        final_meal_onsets = fixed_meal_onset.copy()
        final_meal_offsets = fixed_meal_offset.copy()

        for i in range(len(fixed_meal_onset)-1):
            if fixed_meal_onset[i+1] - fixed_meal_offset[i] < 2:
                final_meal_onsets.pop(i+1)
                final_meal_offsets.pop(i)

        manualbehaviors = {
            'wiggles': fixed_wiggles,
            'drinking': drinking,
            'eating': eating,
            'immobility_onsets': fixed_immobility_onset,
            'immobility_offsets': fixed_immobility_offset,
            'rearing': rearing,
            'meal_onsets': final_meal_onsets,
            'meal_offsets': final_meal_offsets
        }
        if savebehaviors == True:
            with open(behaviors_path + 'ProcessedManualBehaviors.yml', 'w+') as file:
                yaml.dump(manualbehaviors, file, default_flow_style=False, sort_keys=False)

        with open(behaviors_path + 'DLCbehaviors.yml', 'r') as file:
            behaviors=yaml.load(file, Loader=yaml.FullLoader)
        #TODO: Add eating and drinking when refined 
        immobility_onset = behaviors['immobility_onsets']
        immobility_offset = behaviors['immobility_offsets']
        wiggle_onset = behaviors['wiggle_onsets']
        wiggle_offset = behaviors['wiggle_offsets']
        if roi:
            drink_onset = behaviors['drink_onsets']
            drink_offset = behaviors['drink_offsets']
            feed_enter = behaviors['feed_onsets']
            feed_exit = behaviors['feed_offsets']
            drink_onset = [epoc-offsettime for epoc in drink_onset if (epoc > min_time) and (epoc < (time[-1]-min_time))]
            drink_offset = [epoc-offsettime for epoc in drink_offset if (epoc > min_time) and (epoc < (time[-1]-min_time))]

        fixed_immobility_onset = [immobility_onset[i]-offsettime for i in range(len(immobility_onset)) if (immobility_onset[i]>min_time) and (immobility_offset[i]>min_time) and (immobility_onset[i]<(time[-1]-min_time)) and (immobility_offset[i]<(time[-1]-min_time)) and ((immobility_offset[i]-immobility_onset[i])>immobility_time)]
        fixed_immobility_offset = [immobility_offset[i]-offsettime for i in range(len(immobility_offset)) if (immobility_onset[i]>min_time) and (immobility_offset[i]>min_time) and (immobility_onset[i]<(time[-1]-min_time)) and (immobility_offset[i]<(time[-1]-min_time)) and ((immobility_offset[i]-immobility_onset[i])>immobility_time)]

        filt_wiggle_onset = [wiggle_onset[i]-offsettime for i in range(len(wiggle_onset)) if (wiggle_onset[i] > min_time) and (wiggle_offset[i] > min_time) and (wiggle_onset[i] < (time[-1]-min_time)) and ((wiggle_offset[i] < (time[-1])-min_time))]
        filt_wiggle_offset = [wiggle_offset[i]-offsettime for i in range(len(wiggle_offset)) if (wiggle_onset[i] > min_time) and (wiggle_offset[i] > min_time) and (wiggle_onset[i] < (time[-1]-min_time)) and ((wiggle_offset[i] < (time[-1])-min_time))]

        fixed_wiggle_onset = []
        fixed_wiggle_offset = []
        for i in range(len(fixed_immobility_offset)):
           for j in range(len(filt_wiggle_onset)):
                if (filt_wiggle_onset[j] > fixed_immobility_onset[i]) and (filt_wiggle_offset[j] < fixed_immobility_offset[i]):
                    fixed_wiggle_onset.append(filt_wiggle_onset[j]) 
                if (filt_wiggle_onset[j] > fixed_immobility_onset[i]) and (filt_wiggle_offset[j] < fixed_immobility_offset[i]): 
                    fixed_wiggle_offset.append(filt_wiggle_offset[j])
        
        DLCbehaviors = {
            'immobility_onsets': fixed_immobility_onset,
            'immobility_offsets': fixed_immobility_offset,
            'wiggle_onsets': fixed_wiggle_onset,
            'wiggle_offsets': fixed_wiggle_offset,
            'meal_onsets': final_meal_onsets,
            'meal_offsets': final_meal_offsets
            }
        if roi:
            DLCbehaviors['feed_onsets']: feed_enter
            DLCbehaviors['feed_offsets']: feed_exit
            DLCbehaviors['drink_onsets']: drink_onset
            DLCbehaviors['drink_offset']: drink_offset
        if savebehaviors == True:
            with open(behaviors_path + 'ProcessedDLCBehaviors.yml', 'w+') as file:
                yaml.dump(DLCbehaviors, file, default_flow_style=False, sort_keys=False)

        return newtime, signal_channel, isos_channel, streamrate, manualbehaviors, DLCbehaviors

    else: 
        return newtime, signal_channel, isos_channel, streamrate


def calculate_meals(eating, max_time_apart=180, min_events=3):
    #Calculate meal onsets and offsets from eating events
    #Max time apart (int): maximum time in seconds between each individual eating event
    #Min events (int): minimum number of eating events to be considered a meal
    meal_onset = [] 
    meal_offset = []
    for i in range(len(eating)-1):
        if not any(eating[i] <= meal for meal in meal_offset):
            if eating[i+1] - eating[i] < max_time_apart:
                counter = 0
                for j in range(len(eating)-i):
                    try: 
                        if eating[i+j+1] - eating[i+j] < max_time_apart:
                            counter += 1
                        else:
                            break 
                    except IndexError:
                        break

                if counter > min_events: 
                    meal_onset.append(eating[i])
                    meal_offset.append(eating[i+counter])

    return meal_onset, meal_offset


def calculate_meals2(newtime, eatingevents, window_size):
    #Base meals off the rate of pellet consumption (preferred method)
    #Smooth eating events over a window and calculate the mean pellet rate over time 
    rate = np.zeros(len(newtime))
    for epoc in eatingevents:
        rate[(int(epoc))] = 1 
    rate = moving_average(rate, window_size) * 600 

    event_onset_idx = [] 
    event_offset_idx = []

    for idx, val in enumerate(rate):
        if val > 3 and rate[idx-1] <= 3:
            event_onset_idx.append(idx)
        if val < 4 and rate[idx-1] >=4:
            event_offset_idx.append(idx)
        if idx == len(rate)-1:
            if (rate[idx-1] > 3) and (rate[idx] > 3):
                event_offset_idx.append(idx)
            event_offset_idx = [idx for idx in event_offset_idx if idx != 0]
    meal_onsets = [newtime[idx] for idx in event_onset_idx]
    meal_offsets = [newtime[idx] for idx in event_offset_idx if newtime[idx] != 0]

    return meal_onsets, meal_offsets


def load_data(data_path):
    #load .npy file data (time, signal, isos)
    with open(data_path, 'rb') as file:
        data=np.load(file)
    return data

def load_processed_behaviors(folder_path, manual=True, DLC=True):
    #load processed behavior .yaml files 
    manualbehaviors = {}
    DLCbehaviors = {}
    if manual:
        with open(folder_path + 'ProcessedManualBehaviors.yml', 'r') as file:
            manualbehaviors=yaml.load(file, Loader=yaml.Loader)
    if DLC:
        with open(folder_path + 'ProcessedDLCBehaviors.yml', 'r') as file:
            DLCbehaviors=yaml.load(file, Loader=yaml.Loader)
    return manualbehaviors, DLCbehaviors


def process_all_behaviors(behaviors_path, annotationstyle='both', base_meals='FED', savebehaviors=False, offsettime=5, min_time_apart=20, min_time=1800, immobility_time=300, roi=False):
    #Shortened version of pre_process data that only loads all behaviors (both manual and DLC)
    
    newtime = load_data(f'{behaviors_path}time.npy') 
    time = newtime + offsettime
    epocs = load_data(f'{behaviors_path}epocs.npy') + offsettime
    streamrate = load_data(f'{behaviors_path}streamrate.npy')

    if annotationstyle == 'manual': #Open file containing manual annotations 
        with open(behaviors_path + 'behaviors.yml', 'r') as file:
            behaviors=yaml.load(file, Loader=yaml.FullLoader)
        wiggles = behaviors['wiggling']
        drinking = behaviors['drinking']
        eating = behaviors['eating']
        immobility_onset = behaviors['sleeping_onset']
        immobility_offset = behaviors['sleeping_offset']
        rearing = behaviors['rearing']
        #Filter wiggling behaviors to be min_time into the recording and min_time_apart apart, as well as subtracted by offsettime
        wiggles = [epoc-offsettime for epoc in wiggles if (epoc > min_time) and (epoc < (time[-1]-min_time))]
        fixed_wiggles = [] 
        for i in range(len(wiggles)-1):
            if (wiggles[i+1]-wiggles[i]) > min_time_apart:
                fixed_wiggles.append(wiggles[i])
        #Filter all other behaviors to be min_time into the recording and subtracted by offsettime
        drinking = [epoc-offsettime for epoc in drinking if (epoc > min_time) and (epoc < (time[-1]-min_time))]
        eating = [epoc-offsettime for epoc in eating if (epoc > min_time) and (epoc < (time[-1]-min_time))]
        #Also ensure len(immobility onsets) equal len(immobility offsets)
        fixed_immobility_onset = [immobility_onset[i]-offsettime for i in range(len(immobility_onset)) if (immobility_onset[i]>min_time) and (immobility_offset[i]>min_time) and (immobility_onset[i]<(time[-1]-min_time)) and (immobility_offset[i]<(time[-1]-min_time)) and ((immobility_offset[i]-immobility_onset[i])>immobility_time)]
        fixed_immobility_offset = [immobility_offset[i]-offsettime for i in range(len(immobility_offset)) if (immobility_onset[i]>min_time) and (immobility_offset[i]>min_time) and (immobility_onset[i]<(time[-1]-min_time)) and (immobility_offset[i]<(time[-1]-min_time)) and ((immobility_offset[i]-immobility_onset[i])>immobility_time)]
        rearing = [epoc-offsettime for epoc in rearing if (epoc > min_time) and (epoc < (time[-1]-min_time))]
         
        if base_meals == 'annotations':
            meal_onset, meal_offset = calculate_meals(eating) #based on eating annotations 
        else:
            try:
                meal_onset, meal_offset = calculate_meals2(newtime[::int(streamrate)], epocs, 600) #based on FED data
            except:
                print('calculating meals based on manual annotations...')
                meal_onset, meal_offset = calculate_meals(eating)

        fixed_meal_onset = [meal_onset[i] for i in range(len(meal_onset)) if (meal_onset[i] > min_time) and (meal_offset[i] > min_time) and (meal_onset[i] < (time[-1]-min_time)) and ((meal_offset[i] < (time[-1])-min_time))]
        fixed_meal_offset = [meal_offset[i] for i in range(len(meal_offset)) if (meal_onset[i] > min_time) and (meal_offset[i] > min_time) and (meal_onset[i] < (time[-1]-min_time)) and ((meal_offset[i] < (time[-1])-min_time))]

        final_meal_onsets = fixed_meal_onset.copy()
        final_meal_offsets = fixed_meal_offset.copy()

        for i in range(len(fixed_meal_onset)-1):
            if fixed_meal_onset[i+1] - fixed_meal_offset[i] < 2:
                final_meal_onsets.pop(i+1)
                final_meal_offsets.pop(i)

        behaviors = { #Create dict of behaviors and their timestamps
            'wiggles': wiggles,
            'drinking': drinking,
            'eating': eating,
            'immobility_onsets': fixed_immobility_onset,
            'immobility_offsets': fixed_immobility_offset,
            'rearing': rearing,
            'meal_onsets': final_meal_onsets,
            'meal_offsets': final_meal_offsets
        }

        if savebehaviors == True: #save to yaml file 
            with open(behaviors_path + 'ProcessedManualBehaviors.yml', 'w+') as file:
                yaml.dump(behaviors, file, default_flow_style=False, sort_keys=False)

        return newtime, streamrate, fixed_wiggles, behaviors

    elif annotationstyle == 'DLC': #Open file containing DeepLabCut annotations
        with open(behaviors_path + 'DLCbehaviors.yml', 'r') as file:
            behaviors=yaml.load(file, Loader=yaml.FullLoader)
        immobility_onset = behaviors['immobility_onsets']
        immobility_offset = behaviors['immobility_offsets']
        wiggle_onset = behaviors['wiggle_onsets']
        wiggle_offset = behaviors['wiggle_offsets']
        if roi: 
            drink_onset = behaviors['drink_onsets']
            drink_offset = behaviors['drink_offsets']
            feed_enter = behaviors['feed_onsets']
            feed_exit = behaviors['feed_offsets']
            drink_onset = [epoc-offsettime for epoc in drink_onset if (epoc > min_time) and (epoc < (time[-1]-min_time))]
            drink_offset = [epoc-offsettime for epoc in drink_offset if (epoc > min_time) and (epoc < (time[-1]-min_time))]

        fixed_immobility_onset = [immobility_onset[i]-offsettime for i in range(len(immobility_onset)) if (immobility_onset[i]>min_time) and (immobility_offset[i]>min_time) and (immobility_onset[i]<(time[-1]-min_time)) and (immobility_offset[i]<(time[-1]-min_time)) and ((immobility_offset[i]-immobility_onset[i])>immobility_time)]
        fixed_immobility_offset = [immobility_offset[i]-offsettime for i in range(len(immobility_offset)) if (immobility_onset[i]>min_time) and (immobility_offset[i]>min_time) and (immobility_onset[i]<(time[-1]-min_time)) and (immobility_offset[i]<(time[-1]-min_time)) and ((immobility_offset[i]-immobility_onset[i])>immobility_time)]

        filt_wiggle_onset = [wiggle_onset[i]-offsettime for i in range(len(wiggle_onset)) if (wiggle_onset[i] > min_time) and (wiggle_offset[i] > min_time) and (wiggle_onset[i] < (time[-1]-min_time)) and ((wiggle_offset[i] < (time[-1])-min_time))]
        filt_wiggle_offset = [wiggle_offset[i]-offsettime for i in range(len(wiggle_offset)) if (wiggle_onset[i] > min_time) and (wiggle_offset[i] > min_time) and (wiggle_onset[i] < (time[-1]-min_time)) and ((wiggle_offset[i] < (time[-1])-min_time))]

        fixed_wiggle_onset = []
        fixed_wiggle_offset = []
        for i in range(len(fixed_immobility_offset)):
           for j in range(len(filt_wiggle_onset)):
                if (filt_wiggle_onset[j] > fixed_immobility_onset[i]) and (filt_wiggle_offset[j] < fixed_immobility_offset[i]):
                    fixed_wiggle_onset.append(filt_wiggle_onset[j]) 
                if (filt_wiggle_onset[j] > fixed_immobility_onset[i]) and (filt_wiggle_offset[j] < fixed_immobility_offset[i]): 
                    fixed_wiggle_offset.append(filt_wiggle_offset[j])

        try:
            meal_onset, meal_offset = calculate_meals2(newtime[::int(streamrate)], epocs, 600) #based on FED data
        except Exception as e:
            print('Error in calculating meals based on epocs.' + e)

        fixed_meal_onset = [meal_onset[i] for i in range(len(meal_onset)) if (meal_onset[i] > min_time) and (meal_offset[i] > min_time) and (meal_onset[i] < (time[-1]-min_time)) and ((meal_offset[i] < (time[-1])-min_time))]
        fixed_meal_offset = [meal_offset[i] for i in range(len(meal_offset)) if (meal_onset[i] > min_time) and (meal_offset[i] > min_time) and (meal_onset[i] < (time[-1]-min_time)) and ((meal_offset[i] < (time[-1])-min_time))]

        final_meal_onsets = fixed_meal_onset.copy()
        final_meal_offsets = fixed_meal_offset.copy()

        for i in range(len(fixed_meal_onset)-1):
            if fixed_meal_onset[i+1] - fixed_meal_offset[i] < 2:
                final_meal_onsets.pop(i+1)
                final_meal_offsets.pop(i)

        behaviors = { #Create dict of behaviors and their timestamps
            'immobility_onsets': fixed_immobility_onset,
            'immobility_offsets': fixed_immobility_offset,
            'wiggle_onsets': fixed_wiggle_onset,
            'wiggle_offsets': fixed_wiggle_offset,
            'meal_onsets': final_meal_onsets,
            'meal_offsets': final_meal_offsets
            }
        if roi:
            DLCbehaviors['feed_onsets']: feed_enter
            DLCbehaviors['feed_offsets']: feed_exit
            DLCbehaviors['drink_onsets']: drink_onset
            DLCbehaviors['drink_offset']: drink_offset
        if savebehaviors == True:

            with open(behaviors_path + 'ProcessedDLCBehaviors.yml', 'w+') as file:
                yaml.dump(behaviors, file, default_flow_style=False, sort_keys=False)

        return newtime, streamrate, behaviors


    elif annotationstyle == 'both':
        with open(behaviors_path + 'behaviors.yml', 'r') as file:
            behaviors=yaml.load(file, Loader=yaml.FullLoader)
        wiggles = behaviors['wiggling']
        drinking = behaviors['drinking']
        eating = behaviors['eating']
        immobility_onset = behaviors['sleeping_onset']
        immobility_offset = behaviors['sleeping_offset']
        rearing = behaviors['rearing']

        wiggles = [epoc-offsettime for epoc in wiggles if (epoc > min_time) and (epoc < (time[-1]-min_time))]
        fixed_wiggles = [] 
        for i in range(len(wiggles)-1):
            if (wiggles[i+1]-wiggles[i]) > min_time_apart:
                fixed_wiggles.append(wiggles[i])

        drinking = [epoc-offsettime for epoc in drinking if (epoc > min_time) and (epoc < (time[-1]-min_time))]
        eating = [epoc-offsettime for epoc in eating if (epoc > min_time) and (epoc < (time[-1]-min_time))]
        fixed_immobility_onset = [immobility_onset[i]-offsettime for i in range(len(immobility_onset)) if (immobility_onset[i]>min_time) and (immobility_offset[i]>min_time) and (immobility_onset[i]<(time[-1]-min_time)) and (immobility_offset[i]<(time[-1]-min_time)) and ((immobility_offset[i]-immobility_onset[i])>immobility_time)]
        fixed_immobility_offset = [immobility_offset[i]-offsettime for i in range(len(immobility_offset)) if (immobility_onset[i]>min_time) and (immobility_offset[i]>min_time) and (immobility_onset[i]<(time[-1]-min_time)) and (immobility_offset[i]<(time[-1]-min_time)) and ((immobility_offset[i]-immobility_onset[i])>immobility_time)]
        rearing = [epoc-offsettime for epoc in rearing if (epoc > min_time) and (epoc < (time[-1]-min_time))]
        
        if base_meals == 'annotations':
            meal_onset, meal_offset = calculate_meals(eating) #based on eating annotations 
        else:
            try:
                meal_onset, meal_offset = calculate_meals2(newtime[::int(streamrate)], epocs, 600) #based on FED data
            except:
                print('calculating meals based on manual annotations...')
                meal_onset, meal_offset = calculate_meals(eating)

        fixed_meal_onset = [meal_onset[i] for i in range(len(meal_onset)) if (meal_onset[i] > min_time) and (meal_offset[i] > min_time) and (meal_onset[i] < (time[-1]-min_time)) and ((meal_offset[i] < (time[-1])-min_time))]
        fixed_meal_offset = [meal_offset[i] for i in range(len(meal_offset)) if (meal_onset[i] > min_time) and (meal_offset[i] > min_time) and (meal_onset[i] < (time[-1]-min_time)) and ((meal_offset[i] < (time[-1])-min_time))]

        final_meal_onsets = fixed_meal_onset.copy()
        final_meal_offsets = fixed_meal_offset.copy()

        for i in range(len(fixed_meal_onset)-1):
            if fixed_meal_onset[i+1] - fixed_meal_offset[i] < 2:
                final_meal_onsets.pop(i+1)
                final_meal_offsets.pop(i)

        manualbehaviors = {
            'wiggles': fixed_wiggles,
            'drinking': drinking,
            'eating': eating,
            'immobility_onsets': fixed_immobility_onset,
            'immobility_offsets': fixed_immobility_offset,
            'rearing': rearing,
            'meal_onsets': final_meal_onsets,
            'meal_offsets': final_meal_offsets
        }
        if savebehaviors == True:
            with open(behaviors_path + 'ProcessedManualBehaviors.yml', 'w+') as file:
                yaml.dump(manualbehaviors, file, default_flow_style=False, sort_keys=False)

        with open(behaviors_path + 'DLCbehaviors.yml', 'r') as file:
            behaviors=yaml.load(file, Loader=yaml.FullLoader)

        immobility_onset = behaviors['immobility_onsets']
        immobility_offset = behaviors['immobility_offsets']
        wiggle_onset = behaviors['wiggle_onsets']
        wiggle_offset = behaviors['wiggle_offsets']
        drink_onset = behaviors['drink_onsets']
        drink_offset = behaviors['drink_offsets']
        feed_enter = behaviors['feed_onsets']
        feed_exit = behaviors['feed_offsets']

        fixed_immobility_onset = [immobility_onset[i]-offsettime for i in range(len(immobility_onset)) if (immobility_onset[i]>min_time) and (immobility_offset[i]>min_time) and (immobility_onset[i]<(time[-1]-min_time)) and (immobility_offset[i]<(time[-1]-min_time)) and ((immobility_offset[i]-immobility_onset[i])>immobility_time)]
        fixed_immobility_offset = [immobility_offset[i]-offsettime for i in range(len(immobility_offset)) if (immobility_onset[i]>min_time) and (immobility_offset[i]>min_time) and (immobility_onset[i]<(time[-1]-min_time)) and (immobility_offset[i]<(time[-1]-min_time)) and ((immobility_offset[i]-immobility_onset[i])>immobility_time)]

        filt_wiggle_onset = [wiggle_onset[i]-offsettime for i in range(len(wiggle_onset)) if (wiggle_onset[i] > min_time) and (wiggle_offset[i] > min_time) and (wiggle_onset[i] < (time[-1]-min_time)) and ((wiggle_offset[i] < (time[-1])-min_time))]
        filt_wiggle_offset = [wiggle_offset[i]-offsettime for i in range(len(wiggle_offset)) if (wiggle_onset[i] > min_time) and (wiggle_offset[i] > min_time) and (wiggle_onset[i] < (time[-1]-min_time)) and ((wiggle_offset[i] < (time[-1])-min_time))]

        fixed_wiggle_onset = []
        fixed_wiggle_offset = []
        for i in range(len(fixed_immobility_offset)):
           for j in range(len(filt_wiggle_onset)):
                if (filt_wiggle_onset[j] > fixed_immobility_onset[i]) and (filt_wiggle_offset[j] < fixed_immobility_offset[i]):
                    fixed_wiggle_onset.append(filt_wiggle_onset[j]) 
                if (filt_wiggle_onset[j] > fixed_immobility_onset[i]) and (filt_wiggle_offset[j] < fixed_immobility_offset[i]): 
                    fixed_wiggle_offset.append(filt_wiggle_offset[j])
        
        drink_onset = [epoc-offsettime for epoc in drink_onset if (epoc > min_time) and (epoc < (time[-1]-min_time))]
        drink_offset = [epoc-offsettime for epoc in drink_offset if (epoc > min_time) and (epoc < (time[-1]-min_time))]
        DLCbehaviors = {
            'feed_onsets': feed_enter,
            'feed_offsets': feed_exit,
            'drink_onsets': drink_onset,
            'drink_offsets': drink_offset,
            'immobility_onsets': fixed_immobility_onset,
            'immobility_offsets': fixed_immobility_offset,
            'wiggle_onsets': fixed_wiggle_onset,
            'wiggle_offsets': fixed_wiggle_offset
            }
        if savebehaviors == True:
            with open(behaviors_path + 'ProcessedDLCBehaviors.yml', 'w+') as file:
                yaml.dump(DLCbehaviors, file, default_flow_style=False, sort_keys=False)

        return newtime, streamrate, manualbehaviors, DLCbehaviors

    else: 
        return newtime, streamrate







