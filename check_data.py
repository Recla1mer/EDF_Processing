"""
Author: Johannes Peter Knoll

Python implementation of ECG data validation
"""

# IMPORTS
import numpy as np

# LOCAL IMPORTS
import read_edf
from side_functions import *


def eval_std_thresholds(
        data: dict, 
        detection_intervals: list,
        threshold_multiplier: float,
        threshold_dezimal_places: int,
        ecg_key: str,
    ):
    """
    Evaluate useful thresholds (check_ecg_std_min_threshold, check_ecg_std_max_threshold,
    check_ecg_distance_std_ratio_threshold) for the check_ecg function from given data.

    We will estimate the standard deviation for different valuable signals. From this we
    can create an interval for the standard deviation that is considered good.
    Of course when the signal is bad, and all we have is noise, the standard deviation can
    be similar to that of a good signal. 
    However, the distance between the maximum and minimum value will in this case
    be about the same as twice the standard deviation. Therefore, we will also calculate
    the distance between the maximum and minimum value and divide it by twice the standard
    deviation, to see what this ratio looks like for valuable signals. 
    Anything less will be considered as invalid.

    ARGUMENTS:
    --------------------------------
    data: dict
        dictionary containing the data arrays
    detection_intervals: list
        list of detection intervals, in which the data is considered valid
    threshold_multiplier: float between 0 and 1
        multiplier that is either Multiplier or Divisor for the threshold values
        (because valid data could also differ slightly from the detection intervals used)
    threshold_dezimal_places: int
        number of decimal places for the threshold values
    ecg_key: str
        key of the ECG data in the data dictionary
    
    RETURNS:
    --------------------------------
    check_ecg_std_min_threshold: float
        minimum threshold for the standard deviation
    check_ecg_std_max_threshold: float
        maximum threshold for the standard deviation
    check_ecg_distance_std_ratio_threshold: float
        threshold for the max-min distance to (2 * standard deviation) ratio
    """

    # create lists to save the standard deviation and the max-min distance values
    std_values = []
    max_min_distance_values = []
    
    # calculate the standard deviation and std max-min distance for the detection intervals
    for interval in detection_intervals:
        std_values.append(np.std(data[ecg_key][interval[0]:interval[1]]))
        max_min_distance_values.append(np.max(data[ecg_key][interval[0]:interval[1]]) - np.min(data[ecg_key][interval[0]:interval[1]]))
    
    # calculate the ratios
    std_to_max_min_distance_ratios = 0.5 * np.array(max_min_distance_values) / np.array(std_values)
    
    # calculate the thresholds (take values that will include most datapoints)
    max_std = np.max(std_values)
    min_std = np.min(std_values)
    min_std_distance_ratio = np.min(std_to_max_min_distance_ratios)
    
    # apply the threshold multiplier and round the values
    check_ecg_std_min_threshold = round(min_std*threshold_multiplier,threshold_dezimal_places)
    check_ecg_std_max_threshold = round(max_std/threshold_multiplier,threshold_dezimal_places)
    check_ecg_distance_std_ratio_threshold = round(min_std_distance_ratio*threshold_multiplier,threshold_dezimal_places)

    return check_ecg_std_min_threshold, check_ecg_std_max_threshold, check_ecg_distance_std_ratio_threshold


def create_ecg_thresholds(
        ecg_calibration_file_path: str, 
        ecg_calibration_intervals: list,
        ecg_thresholds_multiplier: float,
        ecg_thresholds_dezimal_places: int,
        ecg_key: str,
        ecg_thresholds_save_path: str
    ):
    """
    This function provides the data and calculates the thresholds for ecg data validation.

    Please note that the intervals will be chosen manually and might need to be adjusted,
    if you can't use the calibration data. In this case, you can use the option:
    'show_calibration_data' in the main.py file to show what the calibration data should 
    look like.

    ARGUMENTS:
    --------------------------------
    ecg_calibration_file_path: str
        path to the EDF file for ecg threshold calibration
     ecg_calibration_intervals: list
        list of tuples containing the start and end indices of the calibration intervals
    ecg_thresholds_multiplier: float
        multiplier for the thresholds (see 'eval_std_thresholds()')
    ecg_thresholds_dezimal_places: int
        number of dezimal places for the ecg thresholds (see 'eval_std_thresholds()')
    ecg_key: str
        key of the ECG data in the data dictionary
    ecg_thresholds_save_path: str
        path to the pickle file where the thresholds are saved

    RETURNS:
    --------------------------------
    None, but the thresholds are saved as dictionary to a pickle file with the following
    format:
        {
            "check_ecg_std_min_threshold": check_ecg_std_min_threshold,
            "check_ecg_std_max_threshold": check_ecg_std_max_threshold,
            "check_ecg_distance_std_ratio_threshold": check_ecg_distance_std_ratio_threshold
        }
    """

    # check if ecg thresholds already exist and if yes: ask for permission to override
    user_answer = ask_for_permission_to_override(file_path = ecg_thresholds_save_path, 
        message = "\nThresholds for ECG validation (see check_data.check_ecg()) already exist in " + ecg_thresholds_save_path + ".")
    
    # Load the data
    sigbufs, sigfreqs, sigdims, duration = read_edf.get_edf_data(ecg_calibration_file_path)

    # cancel if user does not want to override
    if user_answer == "n":
        return

    # Calculate and save the thresholds for check_ecg() function
    threshold_values = eval_std_thresholds(
        sigbufs, 
        ecg_calibration_intervals,
        threshold_multiplier = ecg_thresholds_multiplier,
        threshold_dezimal_places = ecg_thresholds_dezimal_places,
        ecg_key = ecg_key,
        )
    
    # write the thresholds to a dictionary and save them
    check_ecg_thresholds = dict()
    check_ecg_thresholds["check_ecg_std_min_threshold"] = threshold_values[0]
    check_ecg_thresholds["check_ecg_std_max_threshold"] = threshold_values[1]
    check_ecg_thresholds["check_ecg_distance_std_ratio_threshold"] = threshold_values[2]
    
    save_to_pickle(check_ecg_thresholds, ecg_thresholds_save_path)


def check_ecg_blocks(
        data: dict, 
        frequency: dict,
        check_ecg_std_min_threshold: float, 
        check_ecg_std_max_threshold: float, 
        check_ecg_distance_std_ratio_threshold: float,
        time_interval_seconds: int, 
        min_valid_length_minutes: int,
        allowed_invalid_region_length_seconds: int,
        ecg_key: str
    ):
    """
    This function won't be used in the project. It was a first draft, but the final function:
    check_ecg() is more useful.

    Check where the ECG data is valid.
    (Checks blocks of x minutes for validity)

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
    ecg_key: str
        key of the ECG data in the data dictionary

    RETURNS:
    --------------------------------
    valid_regions: list
        list of tuples containing the start and end indices of the valid regions
    """

    # calculate the number of iterations from time and frequency
    time_interval_iterations = int(time_interval_seconds * frequency[ecg_key])

    # check condition for given time intervals and add regions (multiple time intervals) to a list if number of invalid intervals is sufficiently low
    valid_regions = []
    current_valid_intervals = 0 # counts valid intervals
    total_intervals = 0 # counts intervals, set to 0 when region is completed (valid or invalid)
    lower_border = 0 # lower border of the region
    skip_interval = 0 # skip intervals if region is valid but total intervals not max (= intervals_per_region)
    min_valid_intervals = int((min_valid_length_minutes * 60 - allowed_invalid_region_length_seconds) / time_interval_seconds) # minimum number of valid intervals in a region
    intervals_per_region = int(min_valid_length_minutes * 60 / time_interval_seconds) # number of intervals in a region
    valid_total_ratio = min_valid_intervals / intervals_per_region # ratio of valid intervals in a region, for the last region that might be too short
    # print("Variables: ", time_interval_iterations, min_valid_intervals, intervals_per_region)

    for i in np.arange(0, len(data[ecg_key]), time_interval_iterations):
        # if region met condition, but there are still intervals left, skip them
        if skip_interval > 0:
            skip_interval -= 1
            continue
        # print("NEW ITERATION: ", i)

        # make sure upper border is not out of bounds
        if i + time_interval_iterations > len(data[ecg_key]):
            upper_border = len(data[ecg_key])
        else:
            upper_border = i + time_interval_iterations
        
        # check if interval is valid
        this_std = np.std(data[ecg_key][i:upper_border])
        this_max = np.max(data[ecg_key][i:upper_border])
        this_min = np.min(data[ecg_key][i:upper_border])
        max_min_distance = this_max - this_min
        std_distance_ratio = 0.5 * max_min_distance / this_std

        if this_std >= check_ecg_std_min_threshold and this_std <= check_ecg_std_max_threshold and std_distance_ratio >= check_ecg_distance_std_ratio_threshold:
            current_valid_intervals += 1
            # print("VALID")
        
        # increase total intervals
        total_intervals += 1
        
        # check if the region is valid
        if current_valid_intervals >= min_valid_intervals:
            # print("VALID REGION: ", lower_border, lower_border + intervals_per_region*time_interval_iterations)
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
        if upper_border == len(data[ecg_key]):
            try:
                if current_valid_intervals / total_intervals >= valid_total_ratio:
                    # print("VALID REGION: ", lower_border, upper_border)
                    valid_regions.append((lower_border,upper_border))
            except:
                continue
    
    print("Valid regions ratio: ", round(len(valid_regions) / (len(data[ecg_key]) / int(min_valid_length_minutes * 60 * frequency[ecg_key])), 5)*100, "%")
    
    return valid_regions


def check_ecg(
        data: dict, 
        frequency: dict,
        check_ecg_std_min_threshold: float, 
        check_ecg_std_max_threshold: float, 
        check_ecg_distance_std_ratio_threshold: float,
        time_interval_seconds: int, 
        min_valid_length_minutes: int,
        allowed_invalid_region_length_seconds: int,
        ecg_key: str
    ):
    """
    Check where the ECG data is valid.
    (valid regions must be x minutes long, invalid regions can be as short as possible)

    ARGUMENTS:
    --------------------------------
    data: dict
        dictionary containing the data arrays
    frequency: dict
        dictionary containing the frequency of the signals
    check_ecg_std_min_threshold: float
        minimum threshold for the standard deviation
    check_ecg_std_max_threshold: float
        maximum threshold for the standard deviation
    check_ecg_distance_std_ratio_threshold: float
        threshold for the max-min distance to twice the standard deviation ratio
    time_interval_seconds: int
        time interval length in seconds to be checked for validity
    min_valid_length_minutes: int
        minimum length of valid data in minutes
    allowed_invalid_region_length_seconds: int
        length of data in seconds that is allowed to be invalid in a valid region of size min_valid_length_minutes
    ecg_key: str
        key of the ECG data in the data dictionary

    RETURNS:
    --------------------------------
    valid_regions: list
        list of lists containing the start and end indices of the valid regions: valid_regions[i] = [start, end] of region i
    """

    # calculate the number of iterations from time and frequency
    time_interval_iterations = int(time_interval_seconds * frequency[ecg_key])

    # check condition for given time intervals and add regions (multiple time intervals) to a list if number of invalid intervals is sufficiently low
    valid_regions = []
    current_valid_intervals = 0 # counts valid intervals
    total_intervals = 0 # counts intervals, set to 0 when region is completed (valid or invalid)
    lower_border = 0 # lower border of the region
    min_length_reached = False # check if the minimum length is reached
    min_valid_intervals = int((min_valid_length_minutes * 60 - allowed_invalid_region_length_seconds) / time_interval_seconds) # minimum number of valid intervals in a region
    intervals_per_region = int(min_valid_length_minutes * 60 / time_interval_seconds) # number of intervals in a region
    valid_total_ratio = min_valid_intervals / intervals_per_region # ratio of valid intervals in a region, for the last region that might be too short
    # print("Variables: ", time_interval_iterations, min_valid_intervals, intervals_per_region)

    for i in np.arange(0, len(data[ecg_key]), time_interval_iterations):

        # make sure upper border is not out of bounds
        if i + time_interval_iterations > len(data[ecg_key]):
            upper_border = len(data[ecg_key])
        else:
            upper_border = i + time_interval_iterations
        
        # calc std and max-min-distance ratio
        this_std = np.std(data[ecg_key][i:upper_border])
        this_max = np.max(data[ecg_key][i:upper_border])
        this_min = np.min(data[ecg_key][i:upper_border])
        max_min_distance = this_max - this_min
        std_distance_ratio = 0.5 * max_min_distance / this_std

        if min_length_reached:
            # check if interval is valid
            if this_std >= check_ecg_std_min_threshold and this_std <= check_ecg_std_max_threshold and std_distance_ratio >= check_ecg_distance_std_ratio_threshold:
                valid_regions[-1][1] = upper_border
            else:
                min_length_reached = False
                lower_border = upper_border
        else:
            # check if interval is valid
            if this_std >= check_ecg_std_min_threshold and this_std <= check_ecg_std_max_threshold and std_distance_ratio >= check_ecg_distance_std_ratio_threshold:
                current_valid_intervals += 1
            
            # increase total intervals
            total_intervals += 1
            
            # check if the region is valid
            if current_valid_intervals >= min_valid_intervals:
                valid_regions.append([lower_border,lower_border + intervals_per_region*time_interval_iterations])
                lower_border += intervals_per_region*time_interval_iterations
                skip_interval = intervals_per_region - total_intervals
                current_valid_intervals = 0
                total_intervals = 0
                min_length_reached = True
                continue
            
            #check if region is invalid
            if total_intervals >= intervals_per_region:
                lower_border += intervals_per_region*time_interval_iterations
                total_intervals = 0
                current_valid_intervals = 0
            
            # check if the last region is valid
            if upper_border == len(data[ecg_key]):
                try:
                    if current_valid_intervals / total_intervals >= valid_total_ratio:
                        valid_regions.append([lower_border,upper_border])
                except:
                    continue
    
    return valid_regions


def determine_valid_ecg_regions(
        data_directory: str,
        valid_file_types: list,
        check_ecg_std_min_threshold: float, 
        check_ecg_std_max_threshold: float, 
        check_ecg_distance_std_ratio_threshold: float,
        check_ecg_time_interval_seconds: int, 
        check_ecg_min_valid_length_minutes: int,
        check_ecg_allowed_invalid_region_length_seconds: int,
        ecg_key: str,
        valid_ecg_regions_path: str
    ):
    """
    Determine the valid ECG regions for all valid file types in the given data directory.

    ARGUMENTS:
    --------------------------------
    data_directory: str
        directory where the data is stored
    valid_file_types: list
        valid file types in the data directory
    valid_ecg_regions_path: str
        path to the pickle file where the valid regions are saved
    others: see check_ecg()

    RETURNS:
    --------------------------------
    None, but the valid regions are saved as dictionary to a pickle file in the following
    format:
        {
            "file_name_1": valid_regions_1,
            "file_name_2": valid_regions_2,
            ...
        }
    See check_ecg() for the format of the valid_regions.
    """

    # check if valid regions already exist and if yes: ask for permission to override
    user_answer = ask_for_permission_to_override(file_path = valid_ecg_regions_path,
                            message = "\nValid regions for the ECG data already exist in " + valid_ecg_regions_path + ".")
    
    # cancel if user does not want to override
    if user_answer == "n":
        return
    
    # get all valid files
    all_files = os.listdir(data_directory)
    valid_files = [file for file in all_files if get_file_type(file) in valid_file_types]

    # create variables to track progress
    total_files = len(valid_files)
    progressed_files = 0

    # create dictionary to save the valid regions
    valid_regions = dict()

    print("\nCalculating valid regions for the ECG data in %i files:" % total_files)

    for file in valid_files:
        # show progress
        progress_bar(progressed_files, total_files)

        # load the data
        sigbufs, sigfreqs, sigdims, duration = read_edf.get_edf_data(data_directory + file)

        # calculate the valid regions
        valid_regions[file] = check_ecg(
            sigbufs, 
            sigfreqs, 
            check_ecg_std_min_threshold = check_ecg_std_min_threshold, 
            check_ecg_std_max_threshold = check_ecg_std_max_threshold, 
            check_ecg_distance_std_ratio_threshold = check_ecg_distance_std_ratio_threshold,
            time_interval_seconds = check_ecg_time_interval_seconds, 
            min_valid_length_minutes = check_ecg_min_valid_length_minutes,
            allowed_invalid_region_length_seconds = check_ecg_allowed_invalid_region_length_seconds,
            ecg_key = ecg_key
            )
        progressed_files += 1

    progress_bar(progressed_files, total_files)
    
    # save the valid regions
    save_to_pickle(valid_regions, valid_ecg_regions_path)


def valid_total_ratio(data: dict, valid_regions: list, ecg_key: str):
    """
    Calculate the ratio of valid to total ecg data.

    ARGUMENTS:
    --------------------------------
    data: dict
        dictionary containing the data arrays
    valid_regions: list
        list of lists containing the start and end indices of the valid regions
    ecg_key: str
        key of the ECG data in the data dictionary

    RETURNS:
    --------------------------------
    valid_ratio: float
        ratio of valid to total ecg data
    """
    valid_data = 0
    for region in valid_regions:
        valid_data += region[1] - region[0]
    valid_ratio = valid_data / len(data[ecg_key])
    return valid_ratio