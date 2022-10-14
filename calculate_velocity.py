import numpy as np 
import pandas as pd 
import math 
import dlc2kinematics
import warnings 
from scipy.signal import savgol_filter
from scipy.ndimage import uniform_filter1d



def calc_vel(df,  bodyparts, scorer, pixel_per_mm=1, method='DLC', bodypart_of_choice='tip_of_head', backup_bp='tail_base'):
    """Calculate velocity from direct DLC output dataframe 
    Modified from DLC2Kinematics functions in order to integrate pixel_per_mm into the calculation
    Note: math and DLC methods are not the same, but they produce roughly similar velocities (DLC is far faster in computation)
    Parameters
    ----------
    df : DataFrame
        Dataframe containing DLC output data
    pixel_per_mm : float
        Number of pixels per mm (e.g. can be obtained from SimBA)
    bodyparts : list
        List of all bodyparts
    scorer : str
        Name of DLC scorer 
    method : str
        Method to use for velocity calculation ('math' or 'DLC')
    bodypart_of_choice : str
        Bodypart to calculate velocity for 
    backup_bp : str
        Second bodypart to calculate velocity for, used in cases where bodypart_of_choice has low prediction likelihood

    Returns
    -------
    df : DataFrame
        DataFrame with appended velocity data for each bodypart
    average_velocities: list 
        List of average velocities for bodypart_of_choice
    """
    df = df.copy()
    
    if method == 'math':
        warnings.filterwarnings('ignore', category=RuntimeWarning) #prevent warning from distance calculation due to NaN values
        df = smooth_trajectory(df)
        distance = lambda x1, y1, x2, y2: math.sqrt((x1-x2)**2 + (y1-y2)**2)
        dflen = len(df.index) - 1 

        for bp in [bodypart_of_choice, backup_bp]: 
            dist = [distance(
                df[scorer, bp, 'x'][i], df[scorer, bp, 'y'][i], df[scorer, bp, 'x'][i+1], df[scorer, bp, 'y'][i+1]
                ) for i in range(dflen)]
            speed = pd.Series(dist) / pixel_per_mm  #convert to cm/s
            df[scorer, bp, 'speed'] = speed #.shift(periods=1) #shift by one frame since first frame can't be calculated

    elif method == 'DLC':
        df_speed = dlc2kinematics.compute_speed(df,bodyparts=['all'])
        for bp in bodyparts: 
                df[scorer, bp, 'speed'] = df_speed[scorer, bp, 'speed'] / pixel_per_mm
    
    try: #Drop scorer column level, not needed anymore
        df.columns = df.columns.droplevel('scorer')
    except KeyError: 
        pass

    avgvelsarray = [] 
    for bodypart in [bodypart_of_choice, backup_bp]:
        individual_vels = df[bodypart, 'speed'].to_numpy()
        #Average speed over each second, and average remaining X frames in the last 0.X second 
        #average_vel_per_second = np.nanmean(np.pad(individual_vels.astype(float), ( 0, ((10 - individual_vels.size%10) % 10) ), mode='constant', constant_values=np.NaN).reshape(-1, 10),axis=1)

        #Average speed with a moving average filter, length of 10 frames (1 second), then record every 10th speed as the average speed for that second
        average_vel_per_second = uniform_filter1d(individual_vels, size=10)[::10]

        #Expand average velocities to fit each frame in df (duplicate each element 10x and cut off last X frames)
        averagespeeds = [s for s in average_vel_per_second for i in range(10)]
        averagespeeds = averagespeeds[:len(df)]

        #Append average speeds
        df[bodypart,'avg'] = averagespeeds
        avgvelsarray.append(average_vel_per_second)
    average_velocities = avgvelsarray[0]

    df = df.reindex(columns=df.columns.get_level_values('bodyparts').unique(), level=0).dropna() #reorder columns to match bodyparts 


    return df, average_velocities #returns dataframe with velocity columns and list of average velocities per second for bodypart of choice


def smooth_trajectory(df, filter_window=3, order=1, deriv=0):
    # Smooth dataframe coordinates - all bodyparts 
    xy = df.columns.get_level_values('coords') != 'likelihood'
    columns_to_smooth = xy & np.ones(df.shape[1], dtype=bool)
    df.loc[:, columns_to_smooth] = savgol_filter(df.loc[:, columns_to_smooth], filter_window, order, deriv, axis=0)

    return df














