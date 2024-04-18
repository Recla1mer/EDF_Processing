"""
Author: Johannes Peter Knoll

This file contains functions that are used to check if the data used in the project is valid.
"""

import numpy as np


def eval_thresholds_for_check_ecg(
        data: dict, 
        detection_intervals: list,
        threshold_multiplier: float,
        threshold_dezimal_places: int,
        relevant_key = "ECG",
    ):
    """
    Evaluate useful thresholds (std_threshold, distance_to_std_threshold) for the 
    check_ecg function.

    We usually have a large standard deviation when the signal is good. Therefore, we will
    estimate the standard deviation for different valuable signals. From this we can create
    an interval for the standard deviation that is considered good.
    Of course when the signal is bad, and all we have is noise, the standard deviation will
    be high too and can be similar to that of a good signal. 
    However, the distance between the maximum and minimum value will in this case
    be about the same as the standard deviation. Therefore, we will also calculate the distance
    between the maximum and minimum value and divide it by the standard deviation, to see
    what this ratio looks like for valuable signals. Anything less will be considered as 
    invalid.

    ARGUMENTS:
    --------------------------------
    data: dict
        dictionary containing the ECG data among other signals
    detection_intervals: list
        list of detection intervals
    threshold_multiplier: float between 0 and 1
        multiplier that is either Multiplier or Divisor for the threshold values
        (because valid data could also differ slightly from the test intervals used)
    threshold_dezimal_places: int
        number of decimal places for the threshold values
    relevant_key: str
        key of the ECG data in the data dictionary
    
    RETURNS:
    --------------------------------
    check_ecg_std_min_threshold: float
        minimum threshold for the standard deviation
    check_ecg_std_max_threshold: float
        maximum threshold for the standard deviation
    check_ecg_distance_std_ratio_threshold: float
        threshold for the distance to standard deviation ratio
    """
    if threshold_multiplier <= 0 or threshold_multiplier > 1:
        raise ValueError("threshold_multiplier must be between 0 and 1.")

    # calculate the standard deviation and std max-min-distance ratio for each detection interval
    std_values = []
    max_min_distance_values = []
    
    for interval in detection_intervals:
        std_values.append(np.std(data[relevant_key][interval[0]:interval[1]]))
        max_min_distance_values.append(np.max(data[relevant_key][interval[0]:interval[1]]) - np.min(data[relevant_key][interval[0]:interval[1]]))
    
    std_to_max_min_distance_ratios = 0.5 * np.array(max_min_distance_values) / np.array(std_values)
    
    # calculate the thresholds
    max_std = np.max(std_values)
    min_std = np.min(std_values)
    min_std_distance_ratio = np.min(std_to_max_min_distance_ratios)
    
    check_ecg_std_min_threshold = round(min_std*threshold_multiplier,threshold_dezimal_places)
    check_ecg_std_max_threshold = round(max_std/threshold_multiplier,threshold_dezimal_places)
    check_ecg_distance_std_ratio_threshold = round(min_std_distance_ratio*threshold_multiplier,threshold_dezimal_places)

    return check_ecg_std_min_threshold, check_ecg_std_max_threshold, check_ecg_distance_std_ratio_threshold


def check_ecg(
        data: dict, 
        frequency: dict,
        check_ecg_std_min_threshold: float, 
        check_ecg_std_max_threshold: float, 
        check_ecg_distance_std_ratio_threshold: float,
        time_interval_seconds: int, 
        min_valid_length_minutes: int,
        allowed_invalid_region_length_seconds: int,
        relevant_key = "ECG"
    ):
    """
    Check where the ECG data is valid.

    ARGUMENTS:
    --------------------------------
    data: dict
        dictionary containing the ECG data among other signals
    frequency: dict
        dictionary containing the frequency of the signals
    check_ecg_std_min_threshold: float
        minimum threshold for the standard deviation
    check_ecg_std_max_threshold: float
        maximum threshold for the standard deviation
    check_ecg_distance_std_ratio_threshold: float
        threshold for the distance to standard deviation ratio
    time_interval_seconds: int
        time interval length to be checked for validity in seconds
    min_valid_length_minutes: int
        minimum length of valid data in minutes
    allowed_invalid_region_length_seconds: int
        allowed length of invalid data in seconds
    relevant_key: str
        key of the ECG data in the data dictionary

    RETURNS:
    --------------------------------
    valid_regions: list
        list of tuples containing the start and end indices of the valid regions
    """
    #check if the ECG data is in the data dictionary
    if relevant_key not in data:
        raise ValueError("ECG data not found in the data dictionary.")
    
    #check if the ECG data is a 1D numpy array
    if not isinstance(data[relevant_key], np.ndarray):
        raise ValueError("ECG data is not a numpy array.")
    if len(data[relevant_key].shape) != 1:
        raise ValueError("ECG data is not a 1D numpy array.")
    
    #check if the frequency is in the frequency dictionary
    if relevant_key not in frequency:
        raise ValueError("ECG frequency not found in the frequency dictionary.")
    
    #check if the frequency is a positive integer
    if not isinstance(frequency[relevant_key], float):
        raise ValueError("ECG frequency is not an integer.")
    if frequency[relevant_key] <= 0:
        raise ValueError("ECG frequency is not a positive integer.")

    # calculate the number of iterations from time and frequency
    time_interval_iterations = int(time_interval_seconds * frequency[relevant_key])

    # check condition for given time intervals and add regions (multiple time intervals) to a list if number of invalid intervals is sufficiently low
    valid_regions = []
    current_valid_intervals = 0 # counts valid intervals
    total_intervals = 0 # counts intervals, set to 0 when region is completed (valid or invalid)
    lower_border = 0 # lower border of the region
    skip_interval = 0 # skip intervals if region is valid but total intervals not max (= intervals_per_region)
    min_valid_intervals = int((min_valid_length_minutes * 60 - allowed_invalid_region_length_seconds) / time_interval_seconds) # minimum number of valid intervals in a region
    intervals_per_region = int(min_valid_length_minutes * 60 / time_interval_seconds) # number of intervals in a region
    valid_total_ratio = min_valid_intervals / intervals_per_region # ratio of valid intervals in a region, for the last region that might be too short
    print("Variables: ", time_interval_iterations, min_valid_intervals, intervals_per_region)

    for i in np.arange(0, len(data[relevant_key]), time_interval_iterations):
        # if region met condition, but there are still intervals left, skip them
        if skip_interval > 0:
            skip_interval -= 1
            continue
        print("NEW ITERATION: ", i)

        # make sure upper border is not out of bounds
        if i + time_interval_iterations > len(data[relevant_key]):
            upper_border = len(data[relevant_key])
        else:
            upper_border = i + time_interval_iterations
        
        # check if interval is valid
        this_std = np.std(data[relevant_key][i:upper_border])
        this_max = np.max(data[relevant_key][i:upper_border])
        this_min = np.min(data[relevant_key][i:upper_border])
        max_min_distance = this_max - this_min
        std_distance_ratio = 0.5 * max_min_distance / this_std

        if this_std >= check_ecg_std_min_threshold and this_std <= check_ecg_std_max_threshold and std_distance_ratio >= check_ecg_distance_std_ratio_threshold:
            current_valid_intervals += 1
            print("VALID")
        
        # increase total intervals
        total_intervals += 1
        
        # check if the region is valid
        if current_valid_intervals >= min_valid_intervals:
            print("VALID REGION: ", lower_border, lower_border + intervals_per_region*time_interval_iterations)
            valid_regions.append((lower_border,lower_border + intervals_per_region*time_interval_iterations))
            lower_border += intervals_per_region*time_interval_iterations
            skip_interval = intervals_per_region - total_intervals
            current_valid_intervals = 0
            total_intervals = 0
            continue
        
        #check if region is invalid
        if total_intervals >= intervals_per_region:
            lower_border += intervals_per_region*time_interval_iterations
            total_intervals = 0
            current_valid_intervals = 0
        
        # check if the last region is valid
        if upper_border == len(data[relevant_key]):
            if current_valid_intervals / total_intervals >= valid_total_ratio:
                print("VALID REGION: ", lower_border, upper_border)
                valid_regions.append((lower_border,upper_border))
    
    return valid_regions