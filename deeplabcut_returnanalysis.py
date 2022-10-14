"""#TODO: 
Script containing all functions used to extract behavioral information from DeepLabCut analysis. 
Our DeepLabCut model was trained on roughly 1100 frames of video, extracted from 30 video recordings, for 850,000 iterations.
- The model with the lowest error on evaluation, at 550,000 iterations, was used for inference on all videos. 
Currently only supports ROI CSV file format created by https://github.com/PolarBean/DLC_ROI_tool 

[link to github]
[contact info]
"""


import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import dlc2kinematics
import yaml
from calculate_velocity import calc_vel 

def main(): 
    ##PATH OF ANALYZED VIDEO H5 FILE 
    path = []
    dlcfile = []
    roi_output = []
    
    for i in range(len(path)):
        print('main')
        dlcanalysis(path[i], dlcfile[i], roi_output[i], compare_manualannotations=False)


def dlcanalysis(path, dlch5file, roi_output, pixel_per_mm=1, plot_velocities=False, compare_manualannotations=False, roi=False, plot=False, threshold_speed=0.6, immobilitytime=120, wiggletime=10):     
    '''
    Returns h5 file of DeepLabCut dataframe with velocity, average velocity, motion behavior, and eating/drinking behavior appended 
    Returns yaml file with onsets/offsets of eating, drinking, and immobility 
    Requires path to folder with both DLC H5 file and DLC-ROI CSV file 
    If velocity plots not needed, can turn off (only used to compare smoothed velocities vs frame-by-frame velocities)
    If manual annotations already done, can compare DLC annotations to manual ones by turning on (requires behaviors.yml file in same folder)
    '''
    
    #Read h5
    df, bodyparts, scorer = dlc2kinematics.load_data(dlch5file)
    print(f'read h5 for {path}')

    #Calculate error in predictions; velocities; plots; motion behaviors; eating/drinking behaviors 
    percent_errors = calculate_error(df, bodyparts,scorer)
    print('errors calculated')

    df, avgvels = calc_vel(df, bodyparts, scorer, pixel_per_mm=pixel_per_mm)
    print('velocities obtained')

    if plot_velocities == True:
        plotvelocity(df, avgvels, path)
    print('plotted velocities')
    
    df, immobility_onsets, immobility_offsets, wiggle_onsets, wiggle_offsets, mobility_onsets, mobility_offsets, unsmoothed_immobility_onsets, unsmoothed_immobility_offsets = calculate_motion_behaviors(df, bodyparts, threshold_speed=threshold_speed, time_for_immobility=immobilitytime, time_for_wiggles=wiggletime) 
    print('motion behaviors calculated')
    
    if roi == True: 
        df, drink_onsets, drink_offsets, feed_onsets, feed_offsets = calculate_eatdrink_behaviors(df, roi_output, bodyparts)
        print('eat/drink behaviors calculated')

    #Save behaviors and percent errors as .yaml file 
    behaviors = {
        'immobility_onsets': immobility_onsets,
        'immobility_offsets': immobility_offsets,
        'wiggle_onsets': wiggle_onsets,
        'wiggle_offsets': wiggle_offsets,
        'mobility_onsets': mobility_onsets,
        'mobility_offsets': mobility_offsets,
        'unsmoothed_immobility_onsets': unsmoothed_immobility_onsets,
        'unsmoothed_immobility_offsets': unsmoothed_immobility_offsets
    }
    if roi == True:
        behaviors['feed_onsets'] = feed_onsets
        behaviors['feed_offsets'] = feed_offsets
        behaviors['drink_onsets'] = drink_onsets
        behaviors['drink_offsets'] = drink_offsets
    print('saved to dict')
    for bp in percent_errors:
        behaviors[f'{bp} percent error'] = percent_errors[bp]
    
    print('dumping')
    with open(path + 'DLCbehaviors.yml', 'w+') as file:
        yaml.dump(behaviors, file, default_flow_style=False, sort_keys=False)

    #Save df as h5 again
    df.to_hdf(path + 'dlcanalysis.h5', key='df', mode='w')
    print('saving to h5')
    #Plot behaviors
    plot_title = path.split('/')[-1]
    if compare_manualannotations:
        with open(path + 'behaviors.yml', 'r') as file:
            manualbehaviors=yaml.load(file, Loader=yaml.FullLoader)
        behaviorplotting(
            path, 
            plot_title,
            manualdrinks=manualbehaviors['drinking'], 
            manualfeeds=manualbehaviors['eating'], 
            manualimmobilityonsets=manualbehaviors['sleeping_onset'], 
            manualimmobilityoffsets=manualbehaviors['sleeping_offset'],
            manualwiggles=manualbehaviors['wiggling'])
    elif plot == True:
        behaviorplotting(path, plot_title, roi=roi)
    print('plotted')

def calculate_error(df, bodyparts, scorer): 
    ''' 
    Calculate frames with predictions below 99% likelihood for each bodypart.
    note: bodyparts must be in list format if interested in only specific bodyparts 
    '''
    error_percents = {}
    #For each bodypart, obtain frames with low likelihood and calculate % error frames out of total frames 
    for bodypart in bodyparts:
        frames_of_error = len(df.loc[df[scorer, bodypart, 'likelihood'] < 0.99])
        error_percent = round(frames_of_error / len(df) * 100, 4)
        error_percents[bodypart] = error_percent
    return error_percents


def plotvelocity(df, avgvel, path, bodypart_of_choice='tip_of_head'):
    '''
    Plots both individual velocities (from df) and average velocities (from array) over time;
    Uses default bodypart (tip of head)—if changed, ensure avg speed is of same bodypart.
    Saves figures as PDF 
    '''
    frames = len(df.index)
    time_for_frames = np.arange(start=0, stop=frames/10, step=0.1) #Array of every 0.1 seconds to match each frame 
    time_in_s = np.arange(frames/10) #Array of every 1 second to match each average speed 
    vel = df[bodypart_of_choice, 'speed'].tolist()
 
    fig, (ax1, ax2) = plt.subplots(1, 2, sharey=True, figsize=(12,5))
    ax1.plot(time_for_frames, vel)
    ax1.set_xlabel('Time in Seconds (Individual frames)')
    ax1.set_ylabel('Frame Speed (AU)')
    ax2.plot(time_in_s, avgvel)
    ax2.set_xlabel('Time in seconds')
    ax2.set_ylabel('Averaged Speed (AU)')
    ax2.yaxis.set_tick_params(labelbottom=True)
    plt.style.use('seaborn-dark')
    fig.savefig(path + 'velocityplots.pdf')

def calculate_motion_behaviors(df, bodyparts, bp='tip_of_head', second_bp='tail_base', threshold_speed=0.6, time_for_immobility=120, time_for_speed=600, time_for_wiggles=10):
    '''
    Filter out poor-prediction frames and classify motion based as potentially mobile or immobile based on threshold speed;
    Then classify periods of "immobility" as >10 seconds of potentially immobile frames;
    Requires average velocities of desired bodypart (bp) in dataframe 
    Returns input dataframe with a 'motion' column under bodypart of choice (default: tip_of_head)
    '''

    df[bp, 'motion'] = ''
    df[second_bp, 'motion'] = ''

    #Label bad frames as BadPredict 
    for frame in df.loc[df[bp, 'likelihood'] < 0.95].index:
        df.at[frame, (bp, 'motion')] = 'BadPredict'
    for frame in df.loc[df[second_bp, 'likelihood'] < 0.95].index:
        df.at[frame, (second_bp, 'motion')] = 'BadPredict'

    #Label frames with higher speed as moving, and lower speed as immobile 
    #If main body part is badpredict, use second bodypart and check if that's also badpredict
    for frame in df.loc[df[bp, 'avg'] > threshold_speed].index:
        if df.at[frame, (bp, 'motion')] != 'BadPredict':
            df.at[frame, (bp, 'motion')] = 'Moving?'
        elif df.at[frame, (second_bp, 'motion')] != 'BadPredict' and df.loc[frame][second_bp, 'avg'] > threshold_speed: 
            #Checks if second bp is good to use for prediction, then use its speed instead 
            #Labels primary bodypart as moving or not, second bp is only used as backup reference 
            df.at[frame, (bp, 'motion')] = 'Moving?'
    for frame in df.loc[df[bp, 'avg'] <= threshold_speed].index:
        if df.at[frame, (bp, 'motion')] != 'BadPredict':
            df.at[frame, (bp, 'motion')] = 'Immobile'
        elif df.at[frame, (second_bp, 'motion')] != 'BadPredict' and df.loc[frame][second_bp, 'avg'] <= threshold_speed:
            df.at[frame, (bp, 'motion')] = 'Immobile'

    ## Full array of motion 
    fullarray = df[bp, 'motion'].to_list()
    fullarrayrange = range(len(fullarray))
    begin = None

    for frame in fullarrayrange:
        if frame == 0:
            df.at[frame, (bp, 'motion')] = '' #First frame cannot have a velocity/behavior 
            continue

        #For each frame, find first potentially moving frame (onset) and if at least time_for_speed (10) frames are potentially moving, mark as mobile 
        #Note: Not a necessary calculation for now, but I left this in for now in case the distinction needs to be made in the future 
        #E.g., in experiments where you want the animal fully moving 
        #For now, the distinction between potentially moving and mobile for at least time_for_speed frames is not included in the return
        #At the end of function, I mark potentially moving and immobile frames as just regular mobile frames, since only immobility frames are desired
        elif df.at[frame, (bp, 'motion')] == 'Moving?' and df.at[frame, (bp, 'motion')] != df.at[frame-1, (bp, 'motion')]:
            begin = frame
            if all(df.loc[begin:begin+time_for_speed][bp, 'motion'] == 'Moving?'):
                try:
                    end = begin + next(i for i in fullarrayrange if begin+i < len(fullarray) and fullarray[begin+i] != 'Moving?') -1
                    df.loc[begin:end, (bp, 'motion')] = 'Mobile'
                except StopIteration: #If video ends with potentially moving frames, iterator will stop
                    end = None
                    df.loc[begin:end, (bp, 'motion')] = 'Mobile'
                    continue 

        #For each frame, find first potentially immobile frame (onset) and if at least time_for_immobility (100) frames are potentially immobile, mark as immobility 
        elif df.at[frame, (bp, 'motion')] == 'Immobile' and df.at[frame, (bp, 'motion')] != df.at[frame-1, (bp, 'motion')]:
            begin = frame
            if all(df.loc[begin:begin+time_for_immobility][bp, 'motion'] == 'Immobile'):
                try:   
                    end = begin + next(i for i in fullarrayrange if begin+i < len(fullarray) and fullarray[begin+i] != 'Immobile') -1
                    df.loc[begin:end, (bp, 'motion')] = 'Immobility'
                except StopIteration: #If video ends with immobilitying frames, iterator will stop 
                    end = None
                    df.loc[begin:end, (bp, 'motion')] = 'Immobility'
                    continue


    
    #Mark any potentially mobile or immobile frames as just mobile, as only the immobility frames need to be marked for now 
    #df.loc[(df[bp, 'motion'] == 'Moving?') | (df[bp, 'motion'] == 'Immobile'), (bp,'motion')] = 'Mobile'


    #Once immobility frames have been marked on dataframe, extract them and obtain onsets and offsets 
    immobility = df.loc[df[bp, 'motion'] == 'Immobility'][bp, 'motion']
    immobility_onsets = []
    immobility_offsets = [] 
    sindexes = immobility.index.to_list()
    #Given each immobility interval is at least 10 seconds, 
    #   Each onset will not be consecutive with past interval
    #   Each offset will not be consecutive with next interval 

    for i, index in enumerate(sindexes):
        try:
            if sindexes[i] + 1 != sindexes[i+1]:
                immobility_offsets.append(sindexes[i]/10)
                immobility_onsets.append(sindexes[i+1]/10)
            elif i == 0: 
                immobility_onsets.append(sindexes[0]/10)
        except:
            if i == len(sindexes)-1: 
                immobility_offsets.append(sindexes[i]/10)
            else:
                print('error occurred in calculating immobility onsets/offsets')
    
    #Wiggles are defined as < 10 seconds of motion in between immobility offsets and onsets 
    #Then remove all immobility onsets/offsets that are wiggle offsets/onsets 
    wiggle_onsets = []
    wiggle_offsets = []
    remove_immobilityonset = []
    remove_immobilityoffset = []
    for i in range(len(immobility_offsets)-1):
        if immobility_onsets[i+1] - immobility_offsets[i] < time_for_wiggles:
            wiggle_onsets.append(immobility_offsets[i])
            remove_immobilityoffset.append(immobility_offsets[i])
            wiggle_offsets.append(immobility_onsets[i+1])
            remove_immobilityonset.append(immobility_onsets[i+1])
    old_immobility_onsets = immobility_onsets
    old_immobility_offsets = immobility_offsets
    immobility_onsets = [x for x in immobility_onsets if x not in remove_immobilityonset]
    immobility_offsets = [x for x in immobility_offsets if x not in remove_immobilityoffset]
    #Note: wiggles are not included in dataframe, mobility is labeled as is 


    #Obtain onsets and offsets of long-duration activity, marked as 'mobile' in df 
    mobility = df.loc[df[bp, 'motion'] == 'Mobile'][bp, 'motion']
    mobile_onsets = [] 
    mobile_offsets = [] 
    mindexes = mobility.index.to_list()

    for i, index in enumerate(mindexes):
        try:
            if mindexes[i] + 1 != mindexes[i+1]:
                mobile_offsets.append(mindexes[i]/10)
                mobile_onsets.append(mindexes[i+1]/10)
            elif i == 0: 
                mobile_onsets.append(mindexes[0]/10)
        except:
            if i == len(mindexes)-1: 
                mobile_offsets.append(mindexes[i]/10)
            else:
                print('error occurred in calculating mobility onsets/offsets')


    #Reorganize dataframe 
    df = df.reindex(columns=bodyparts, level=0)

    return df, immobility_onsets, immobility_offsets, wiggle_onsets, wiggle_offsets, mobile_onsets, mobile_offsets, old_immobility_onsets, old_immobility_offsets

def calculate_eatdrink_behaviors(df, roi_output, bodyparts, bp='tip_of_head', time_to_drink=10, time_to_feed = 50):
    '''
    Uses CSV file from ROI data, obtained by DLC_ROI_tool
    Classifies drinking events as >1 second of frames within ROI
    Classifies *potentially* feeding as >5 seconds of framese within ROI (ROI is set to be big usually)
    Returns input dataframe with a 'eat/drink' column under bodypart of choice (default: tip_of_head)
    Returns list of onsets and offsets for both drinking and feeding 
    '''

    df[bp, 'eat/drink'] = ''

    #Obtain all drinking ROI frames 
    roidf = pd.read_csv(roi_output)
    df[bp, 'temp_eat/drink'] = roidf[bp]
    
    drink = df.loc[df[bp, 'temp_eat/drink'] == 'Drinking'][bp, 'temp_eat/drink']

    drink_onsets = []
    drink_offsets = [] 

    #Obtain indexes of drinking frames
    dindexes = drink.index.to_list()
    dindexrange = range(len(dindexes))
    
    #For drinking frames, find frames not in sequence to previous, check if all frames 1 seconds from then are together (1 frame apart), then iterate to find end of drinking event
    for frame in dindexrange:
        if frame == 0: #Begin sequence with first frame 
            begin = dindexes[frame] 
        elif dindexes[frame] -1 != dindexes[frame-1]: #If drinking frame is not in sequence with past frame, begin new sequence 
            begin = dindexes[frame]
        elif frame >= len(dindexes)-time_to_drink: #If end of drinking frames has been reached, break 
            break 
        else: # If drinking frame is not a new sequence, next loop 
            continue 
        if all((dindexes[frame + requiredframes] == begin + requiredframes) for requiredframes in range(time_to_drink)):
            try:
                end = begin + next(i for i in range(len(dindexes)-frame) if frame+i < len(dindexes) and dindexes[frame + i] != begin + i) -1
                drink_onsets.append(begin/10)
                drink_offsets.append(end/10)
                df.loc[begin:end, (bp,'eat/drink')] = 'Drinking'
            except StopIteration:
                end = dindexes[-1]
                drink_onsets.append(begin/10)
                drink_offsets.append(end/10)
                df.loc[begin:end, (bp,'eat/drink')] = 'Drinking'


    feed = df.loc[df[bp, 'temp_eat/drink'] == 'FED_Area'][bp, 'temp_eat/drink']
    #Must be in feeding region for at least 5 seconds to be counted as possibly feeding 
    feed_onsets = []
    feed_offsets = []

    #Obtain indexes of feeding frames
    findexes = feed.index.to_list() 
    findexrange = range(len(findexes))

    #For feeding frames, find frames not in sequence to previous, check if all frames 10 seconds from then are together (1 frame apart), then iterate to find end of fed event
    for frame in findexrange:
        if frame == 0: #Begin sequence with first frame 
            begin = findexes[frame] 
        elif findexes[frame] -1 != findexes[frame-1]: #If fed frame is not in sequence with past frame, begin new sequence 
            begin = findexes[frame]
        elif frame >= len(findexes)-time_to_feed: #If end of fed frames has been reached, break 
            break 
        else: # If fed frame is not a new sequence, next loop 
            continue 
        if all((findexes[frame + requiredframes] == begin + requiredframes) for requiredframes in range(time_to_feed)):
            try:
                end = begin + next(i for i in range(len(findexes)-frame) if frame+i < len(findexes) and findexes[frame + i] != begin + i) -1
                feed_onsets.append(begin/10)
                feed_offsets.append(end/10)
                df.loc[begin:end, (bp,'eat/drink')] = 'Possibly Feeding'
            except StopIteration:
                end = findexes[-1]
                feed_onsets.append(begin/10)
                feed_offsets.append(end/10)
                df.loc[begin:end, (bp,'eat/drink')] = 'Possibly Feeding'
    
    #Drop temporary column
    df = df.drop('temp_eat/drink', axis=1, level=1)
    #Reorganize df
    df = df.reindex(columns=bodyparts, level=0)

    return df, drink_onsets, drink_offsets, feed_onsets, feed_offsets

def behaviorplotting(path, plot_title, roi=False, manualdrinks=[], manualfeeds=[], manualimmobilityonsets=[], manualimmobilityoffsets=[], manualwiggles=[]): 
    '''
    Plots all behavioral annotations, able to compare to manual annotations by passing arrays of times in seconds 
    Returns PDF file with all various behaviors plotted
    '''

    with open(path + f'DLCbehaviors.yml', 'r') as file:
        behaviors=yaml.load(file, Loader=yaml.FullLoader)
    
    fig = plt.figure(figsize=(10,4))
    ax = fig.add_subplot(111)

    #Drink onsets/offsets 
    if roi == True:
        for idx, val in enumerate(behaviors['drink_onsets']):
            rect = patches.Rectangle((val/3600, -5), width=(behaviors['drink_offsets'][idx]-val)/3600, height=1, color='salmon', alpha=0.4)
            ax = plt.gca()
            ax.add_patch(rect)

    #Drink annotations 
    for drinkevents in manualdrinks:
        ax.vlines((drinkevents)/3600, ymin=-4, ymax=-3, linewidth=1,color='red', alpha=1)

    if roi == True:
    #FED onsets/offsets 
        for idx, val in enumerate(behaviors['feed_onsets']):
            rect = patches.Rectangle((val/3600, -2), width=(behaviors['feed_offsets'][idx]-val)/3600, height=1, color='lightgreen', alpha=0.4)
            ax = plt.gca()
            ax.add_patch(rect)

    #FED annotations 
    for fedevents in manualfeeds:
        ax.vlines((fedevents)/3600, ymin=-1, ymax=0, linewidth=1,color='black', alpha=1)

    #immobility onsets/offsets from DLC 
    try:
        for idx, val in enumerate(behaviors['immobility_onsets']):
            rect = patches.Rectangle((val/3600, 1), width=(behaviors['immobility_offsets'][idx]-val)/3600, height=1, color='blue', alpha=0.4)
            ax = plt.gca()
            ax.add_patch(rect)
    except:
        for i in range(len(behaviors['immobility_offsets'])):
            print(behaviors['immobility_onsets'][i], behaviors['immobility_offsets'][i])
        raise Exception('Error in immobility onsets/offsets. Debug code.')

    #immobility onsets/offsets from annotations 
    for idx, val in enumerate(manualimmobilityonsets):
        rect = patches.Rectangle((val/3600, 2), width=(manualimmobilityoffsets[idx]-val)/3600, height=1, color='plum', alpha=0.4)
        ax = plt.gca()
        ax.add_patch(rect)

    #Wiggle onsets/offsets from DLC 
    for idx, val in enumerate(behaviors['wiggle_onsets']):
        rect = patches.Rectangle((val/3600,3), width=(behaviors['wiggle_offsets'][idx]-val)/3600, height=1, color='orange', alpha=0.4)
        ax = plt.gca()
        ax.add_patch(rect)

    #Wiggle annotations 
    for wiggle in manualwiggles:
        ax.vlines((wiggle)/3600, ymin=4, ymax=5,linewidth=1, color='brown',alpha=1)

    plt.xlabel('Time in Hours')
    plt.ylabel('Behaviors')
    plt.title(f'Behavioral Annotations for {plot_title}')

    ax.relim()
    ax.autoscale_view()
    ax.set_yticks([-4.5, -3.5, -1.5, -0.5, 1.5, 2.5, 3.5, 4.5])
    ax.set_yticklabels(['DLC Drink', 'Manual Drink', 'DLC Feed', 'Manual Feed', 'DLC Immobility', 'Manual Immobility', 'DLC Wiggles', 'Manual Wiggles'])
    plt.ylim(-6, 8)

    plt.savefig(path + f'variousbehaviors.pdf')


if __name__ == '__main__':
    main()




