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
    Evaluate useful thresholds (check_ecg_std_min_threshold, check_ecg_distance_std_ratio_threshold) 
    for the check_ecg function from given data.

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

    return check_ecg_std_min_threshold, check_ecg_distance_std_ratio_threshold


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
    check_ecg_thresholds["check_ecg_distance_std_ratio_threshold"] = threshold_values[1]
    
    save_to_pickle(check_ecg_thresholds, ecg_thresholds_save_path)


def check_ecg_blocks(
        data: dict, 
        frequency: dict,
        check_ecg_std_min_threshold: float, 
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

        if this_std >= check_ecg_std_min_threshold and std_distance_ratio >= check_ecg_distance_std_ratio_threshold:
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


def check_ecg_more_mistakes(
        data: dict, 
        frequency: dict,
        check_ecg_std_min_threshold: float, 
        check_ecg_distance_std_ratio_threshold: float,
        time_interval_seconds: int, 
        min_valid_length_minutes: int,
        allowed_invalid_region_length_seconds: int,
        ecg_key: str
    ):
    """
    This function won't be used in the project. It was the second draft, but the final function:
    check_ecg() is more useful.

    (The function checks in non overlapping intervals. It appends a valid region if its
    long enough with few enough errors. Afterwards it expands the upper border if the following 
    interval is also valid. Problem 1: All errors could be right at the start of a region, without
    excluding them. Problem 2: It won't append an almost long enough region to a valid region, if
    they are separated by a short invalid region, however short it will be.)

    ARGUMENTS:
    --------------------------------
    data: dict
        dictionary containing the data arrays
    frequency: dict
        dictionary containing the frequency of the signals
    check_ecg_std_min_threshold: float
        minimum threshold for the standard deviation
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
            if this_std >= check_ecg_std_min_threshold and std_distance_ratio >= check_ecg_distance_std_ratio_threshold:
                valid_regions[-1][1] = upper_border
            else:
                min_length_reached = False
                lower_border = upper_border
        else:
            # check if interval is valid
            if this_std >= check_ecg_std_min_threshold and std_distance_ratio >= check_ecg_distance_std_ratio_threshold:
                current_valid_intervals += 1
            
            # increase total intervals
            total_intervals += 1
            
            # check if the region is valid
            if current_valid_intervals >= min_valid_intervals:
                valid_regions.append([lower_border,lower_border + intervals_per_region*time_interval_iterations])
                lower_border += intervals_per_region*time_interval_iterations
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


def check_ecg(
        data: dict, 
        frequency: dict,
        check_ecg_std_min_threshold: float, 
        check_ecg_distance_std_ratio_threshold: float,
        time_interval_seconds: int, 
        min_valid_length_minutes: int,
        allowed_invalid_region_length_seconds: int,
        ecg_key: str
    ):
    """
    Check where the ECG data is valid.
    (valid regions must be x minutes long, invalid regions can be as short as possible)

    Data will be checked in overlapping intervals. Those will be concatenated afterwards.
    Then the gaps between the regions will be checked. If its useful to connect them (regions
    long enough compared to gap, but not too long that they already fulfill the
    min_valid_length_minutes condition), they will be connected.

    ARGUMENTS:
    --------------------------------
    data: dict
        dictionary containing the data arrays
    frequency: dict
        dictionary containing the frequency of the signals
    check_ecg_std_min_threshold: float
        minimum threshold for the standard deviation
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
    overlapping_valid_regions = []
    
    interval_steps = int(time_interval_iterations/5)
    was_valid = False

    for i in np.arange(0, len(data[ecg_key]), interval_steps):

        # make sure upper border is not out of bounds
        if i + time_interval_iterations > len(data[ecg_key]):
            upper_border = len(data[ecg_key])
        else:
            upper_border = i + time_interval_iterations
        
        # calc std and max-min-distance ratio
        this_std = abs(np.std(data[ecg_key][i:upper_border]))
        this_max = np.max(data[ecg_key][i:upper_border])
        this_min = np.min(data[ecg_key][i:upper_border])
        max_min_distance = abs(this_max - this_min)

        if this_std == 0:
            std_distance_ratio = check_ecg_distance_std_ratio_threshold + 1
        else:
            std_distance_ratio = 0.5 * max_min_distance / this_std

        # check if interval is valid
        if this_std >= check_ecg_std_min_threshold and std_distance_ratio >= check_ecg_distance_std_ratio_threshold:
            overlapping_valid_regions.append([i,upper_border])
            was_valid = True
        else:
            if was_valid:
                limit = upper_border - time_interval_iterations/5
                for j in range(len(overlapping_valid_regions)-1, -1, -1):
                    if overlapping_valid_regions[j][1] >= limit:
                        del overlapping_valid_regions[j]
                    else:
                        break
            was_valid = False

    # concatenate neighbouring intervals
    concatenated_intervals = []
    this_interval = [overlapping_valid_regions[0][0], overlapping_valid_regions[0][1]]
    for i in range(1, len(overlapping_valid_regions)):
        if overlapping_valid_regions[i][0] < this_interval[1]:
            this_interval[1] = overlapping_valid_regions[i][1]
        else:
            concatenated_intervals.append(this_interval)
            del this_interval
            this_interval = [overlapping_valid_regions[i][0], overlapping_valid_regions[i][1]]
    concatenated_intervals.append(this_interval)

    # return concatenated_intervals

    del overlapping_valid_regions

    # calculate thresholds from other units in iterations
    iterations_per_region = int(min_valid_length_minutes * 60 * frequency[ecg_key])
    allowed_invalid_iterations = int(allowed_invalid_region_length_seconds * frequency[ecg_key])
    allowed_wrong_total_ratio = allowed_invalid_region_length_seconds / min_valid_length_minutes / 60

    # collect possible connections (gap between regions is smaller than allowed_invalid_iterations)
    possible_connections = []
    
    for i in range(1, len(concatenated_intervals)):
        if concatenated_intervals[i][0] - concatenated_intervals[i-1][1] < allowed_invalid_iterations:
            possible_connections.append([i-1, i])
    
    # check which connections are useful to make
    current_invalid_iterations = 0
    total_iterations = 0

    validated_connections = []
    last_connection = -1

    for i in range(len(possible_connections)):
        # calculate the length of the regions and the distance between them
        len_region_1 = concatenated_intervals[possible_connections[i][0]][1] - concatenated_intervals[possible_connections[i][0]][0]
        len_region_2 = concatenated_intervals[possible_connections[i][1]][1] - concatenated_intervals[possible_connections[i][1]][0]
        distance = concatenated_intervals[possible_connections[i][1]][0] - concatenated_intervals[possible_connections[i][0]][1]

        # check if the regions are not too short and not too long
        if len_region_1 > iterations_per_region and len_region_2 > iterations_per_region:
            continue
        
        if len_region_1 < 2*allowed_invalid_iterations or len_region_2 < 2*allowed_invalid_iterations:
            continue
        
        # check if last connection was established and check if you can append to it
        if last_connection == possible_connections[i][0]:
            current_invalid_iterations += distance
            total_iterations += len_region_2 + distance
            if current_invalid_iterations / total_iterations <= allowed_wrong_total_ratio:
                validated_connections[-1].append(possible_connections[i][1])
                # we don't want to add to much errors, should the interval be long enough, therefore we just compare the last two connections (reset values)
                current_invalid_iterations = distance
                total_iterations = len_region_1 + len_region_2 + distance
                last_connection = possible_connections[i][1]
                continue
        
        # check if you can establish a new connection
        current_invalid_iterations = distance
        total_iterations = len_region_1 + len_region_2 + distance
        if current_invalid_iterations / total_iterations <= allowed_wrong_total_ratio:
            validated_connections.append(possible_connections[i])
            last_connection = possible_connections[i][1]
        else:
            last_connection = -1

    # create the intervals for the valid connections
    connected_regions = []
    for con in validated_connections:
        connected_regions.append([concatenated_intervals[con[0]][0], concatenated_intervals[con[-1]][1]])

    # replace the connections in the concatenated intervals
    replaced_connections = []
    for i in range(len(concatenated_intervals)-1, -1, -1):
        for con_index in range(len(validated_connections)):
            if i in validated_connections[con_index]:
                if con_index in replaced_connections:
                    del concatenated_intervals[i]
                    break
                else:
                    concatenated_intervals[i] = connected_regions[con_index]
                    replaced_connections.append(con_index)
                    break
    
    # append long enough regions to the valid regions
    valid_regions = []
    for region in concatenated_intervals:
        if region[1] - region[0] >= iterations_per_region:
            valid_regions.append(region)
    
    return valid_regions


def determine_valid_ecg_regions(
        data_directory: str,
        valid_file_types: list,
        check_ecg_std_min_threshold: float, 
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


"""
Following code won't be used for the final implementation, but is useful for testing and
comparing the results of the ECG validation, as interval classification is available for
the GIF data. They might or might not have been calculated automatically and
later checked manually.

They are stored in .txt files. Therefore we need to implement functions to compare the
results of the validation and to read the intervals from the .txt files.
They are stored in the following format: "integer integer" after a file header containing
various information separated by a line of "-".

The first integer is the index in the ECG data and the second integer is the classification
of the data point (0: valid, 1: invalid).
"""


def ecg_validation_txt_string_evaluation(string: str):
    """
    Appearence of string entries in the .txt files: "integer integer"
    The first integer is the index in the ECG data and the second integer is the classification:
    (0: valid, 1: invalid)

    ARGUMENTS:
    --------------------------------
    string: str
        string to be evaluated
    
    RETURNS:
    --------------------------------
    datapoint: int
        index of the data point
    classification: str
        classification of the data point
    """

    # set default values if the integer or the letter do not exist
    datapoint = " "
    classification = " "

    was_number = False
    for i in range(len(string)):
        if string[i].isdigit():
            if datapoint != " ":
                classification = string[i]
                break
            if not was_number:
                start = i
            was_number = True
        else:
            if was_number:
                datapoint = int(string[start:i])
            was_number = False
    
    return datapoint, classification


def get_ecg_classification_from_txt_file(file_path: str):
    """
    Get the ECG classification from a .txt file.

    ARGUMENTS:
    --------------------------------
    file_path: str
        path to the .txt file containing the ECG classification
    
    RETURNS:
    --------------------------------
    ecg_classification: dict
        dictionary containing the ECG classification in the following format:
        {
            "0": [valid_datapoint_1, valid_datapoint_2, ...],
            "1": [invalid_datapoint_1, invalid_datapoint_2, ...]
        }
    """

    # read the txt file
    with open(file_path, 'r') as file:
        txt_lines = file.readlines()
    
    # start of ecg validation is separated by a line of "-" from the file header in the .txt files
    # retrieve the start of the R-peaks in the file
    for i in range(len(txt_lines)):
        count_dash = 0
        for j in range(len(txt_lines[i])):
            if txt_lines[i][j] == "-":
                count_dash += 1
        if count_dash/len(txt_lines[i]) > 0.9:
            start = i + 1
            break
    
    # create dictionary to save the classification
    ecg_classification = dict()

    # determine valid datapoints from the txt file
    for i in range(start, len(txt_lines)):
        datapoint, classification = ecg_validation_txt_string_evaluation(txt_lines[i])
        
        if isinstance(datapoint, int) and classification.isdigit():
            if classification in ecg_classification:
                ecg_classification[classification].append(datapoint)
            else:
                ecg_classification[classification] = [datapoint]
    
    return ecg_classification


def compare_ecg_validations(
        validated_intervals: list, 
        ecg_classification: dict,
    ):
    """
    Compare the validated intervals with the ECG classification.

    ARGUMENTS:
    --------------------------------
    validated_intervals: list
        list of tuples containing the start and end indices of the intervals considered valid
    ecg_classification: dict
        dictionary containing the ECG classification
    
    RETURNS:
    --------------------------------
    correct_valid_ratio: float
        ratio of correctly classified valid points
    correct_invalid_ratio: float
        ratio of correctly classified invalid points
    valid_wrong_ratio: float
        ratio of valid points classified as invalid
    invalid_wrong_ratio: float
        ratio of invalid points classified as valid
    """

    # get points considered valid and invalid by the ECG classification
    classification_invalid_points = ecg_classification["1"]
    classification_valid_points = ecg_classification["0"]

    # create lists to save the intersecting points and the wrong classified points
    intersecting_invalid_points = []
    intersecting_valid_points = []

    invalid_points_wrong = []
    valid_points_wrong = []

    # check if point classified as valid is in the validated intervals
    # append to the corresponding list depending on the result
    for point in classification_valid_points:
        appended = False
        for interval in validated_intervals:
            if point >= interval[0] and point <= interval[1]:
                intersecting_valid_points.append(point)
                appended = True
                break
        if not appended:
            valid_points_wrong.append(point)
    
    # check if point classified as invalid is outside the validated intervals
    # append to the corresponding list depending on the result
    for point in classification_invalid_points:
        appended = False
        for interval in validated_intervals:
            if point >= interval[0] and point <= interval[1]:
                invalid_points_wrong.append(point)
                appended = True
                break
        if not appended:
            intersecting_invalid_points.append(point)

    # calculate the ratios and return them
    correct_valid_ratio = len(intersecting_valid_points) / len(classification_valid_points)
    correct_invalid_ratio = len(intersecting_invalid_points) / len(classification_invalid_points)

    valid_wrong_ratio = len(valid_points_wrong) / len(classification_valid_points)
    invalid_wrong_ratio = len(invalid_points_wrong) / len(classification_invalid_points)

    return correct_valid_ratio, correct_invalid_ratio, valid_wrong_ratio, invalid_wrong_ratio


def ecg_validation_comparison(
        ecg_classification_values_directory: str,
        ecg_classification_file_types: list,
        ecg_validation_comparison_evaluation_path: str,
        valid_ecg_regions_path: str
    ):
    """
    Compare the ECG validation with the ECG classification values.

    ARGUMENTS:
    --------------------------------
    ecg_classification_values_directory: str
        directory where the ECG classification values are stored
    ecg_classification_file_types: list
        valid file types for the ECG classification values
    ecg_validation_comparison_evaluation_path: str
        path to the pickle file where the evaluation is saved
    valid_ecg_regions_path: str
        path to the pickle file where the valid regions are stored
    
    RETURNS:
    --------------------------------
    None, but the evaluation is saved as dictionary to a pickle file in the following
    format:
        {
            "file_name_1": [correct_valid_ratio, correct_invalid_ratio, valid_wrong_ratio, invalid_wrong_ratio],
            "file_name_2": [correct_valid_ratio, correct_invalid_ratio, valid_wrong_ratio, invalid_wrong_ratio],
            ...
        }
    """
    
    # check if the evaluation already exists and if yes: ask for permission to override
    user_answer = ask_for_permission_to_override(file_path = ecg_validation_comparison_evaluation_path,
                        message = "\nEvaluation of ECG validation comparison already exists in " + ecg_validation_comparison_evaluation_path + ".")
    
    # cancel if user does not want to override
    if user_answer == "n":
        return

    # get all determined ECG Validation files
    determined_ecg_validation_dictionary = load_from_pickle(valid_ecg_regions_path)

    # get all ECG classification files
    all_classification_files = os.listdir(ecg_classification_values_directory)
    valid_classification_files = [file for file in all_classification_files if get_file_type(file) in ecg_classification_file_types]

    # create variables to track progress
    total_data_files = len(determined_ecg_validation_dictionary)
    progressed_data_files = 0

    # create dictionary to store the ECG Validation comparison values for all files
    all_files_ecg_validation_comparison = dict()
    
    # calculate the R peak comparison values
    print("\nCalculating ECG validation comparison values for %i files:" % total_data_files)
    for file_key in determined_ecg_validation_dictionary:
        # show progress
        progress_bar(progressed_data_files, total_data_files)
        progressed_data_files += 1

        # get the file name without the file type
        this_file_name = os.path.splitext(file_key)[0]

        # get corresponding ECG classification file name for this file
        for clfc_file in valid_classification_files:
            if this_file_name in clfc_file:
                this_classification_file = clfc_file
        try:
            ecg_classification_dictionary = get_ecg_classification_from_txt_file(ecg_classification_values_directory + this_classification_file)
        except ValueError:
            print("ECG classification is missing for %s. Skipping this file." % file_key)
            continue
        
        # compare the differnt ECG validations
        this_file_comparison_values = compare_ecg_validations(
            validated_intervals = determined_ecg_validation_dictionary[file_key],
            ecg_classification = ecg_classification_dictionary
            )
        
        # save the comparison values for this file to the dictionary
        all_files_ecg_validation_comparison[file_key] = [comparison_value for comparison_value in this_file_comparison_values]
    
    progress_bar(progressed_data_files, total_data_files)
    
    # save the comparison values to a pickle file
    save_to_pickle(all_files_ecg_validation_comparison, ecg_validation_comparison_evaluation_path)


def ecg_validation_comparison_report(
        ecg_validation_comparison_report_path: str,
        ecg_validation_comparison_evaluation_path: str,
        ecg_validation_comparison_report_dezimal_places: int,
    ):
    """
    Create a report for the ECG Validation comparison.

    ARGUMENTS:
    --------------------------------
    ecg_validation_comparison_report_path: str
        path to the file where the report is saved
    ecg_validation_comparison_evaluation_path: str
        path to the pickle file where the evaluation is stored
    ecg_validation_comparison_report_dezimal_places: int
        number of decimal places for the report
    
    RETURNS:
    --------------------------------
    None, but the report is saved to a file as a table
    """

    # check if the report already exists and if yes: ask for permission to override
    user_answer = ask_for_permission_to_override(file_path = ecg_validation_comparison_report_path,
                        message = "\nECG validation comparison report already exists in " + ecg_validation_comparison_report_path + ".")

    # cancel if user does not want to override
    if user_answer == "n":
        return

    # open the file to write the report to
    comparison_file = open(ecg_validation_comparison_report_path, "w")

    # load the data
    all_files_ecg_validation_comparison = load_from_pickle(ecg_validation_comparison_evaluation_path)

    # write the file header
    message = "ECG VALIDATION COMPARISON REPORT"
    comparison_file.write(message + "\n")
    comparison_file.write("=" * len(message) + "\n\n\n")

    # set the table captions
    CORRECT_VALID_CAPTION = "Correct Valid"
    CORRECT_INVALID_CAPTION = "Correct Invalid"
    FILE_CAPTION = "File"
    WRONG_AS_VALID_CAPTION = "Wrong Valid"
    WRONG_AS_INVALID_CAPTION = "Wrong Invalid"

    MEAN_ROW_CAPTION = "Mean values"

    # create lists to collect all acccuracy values and print the mean of them
    correct_valid_values = []
    correct_invalid_values = []
    wrong_as_valid_values = []
    wrong_as_invalid_values = []

    # collect all comparison values
    for file in all_files_ecg_validation_comparison:
        all_files_ecg_validation_comparison[file][0] = round(all_files_ecg_validation_comparison[file][0], ecg_validation_comparison_report_dezimal_places)
        all_files_ecg_validation_comparison[file][1] = round(all_files_ecg_validation_comparison[file][1], ecg_validation_comparison_report_dezimal_places)
        all_files_ecg_validation_comparison[file][2] = round(all_files_ecg_validation_comparison[file][2], ecg_validation_comparison_report_dezimal_places)
        all_files_ecg_validation_comparison[file][3] = round(all_files_ecg_validation_comparison[file][3], ecg_validation_comparison_report_dezimal_places)

        correct_valid_values.append(all_files_ecg_validation_comparison[file][0])
        correct_invalid_values.append(all_files_ecg_validation_comparison[file][1])
        wrong_as_valid_values.append(all_files_ecg_validation_comparison[file][2])
        wrong_as_invalid_values.append(all_files_ecg_validation_comparison[file][3])
    
    # calculate mean of them
    mean_correct_valid = round(np.mean(correct_valid_values), ecg_validation_comparison_report_dezimal_places)
    mean_correct_invalid = round(np.mean(correct_invalid_values), ecg_validation_comparison_report_dezimal_places)
    mean_wrong_as_valid = round(np.mean(wrong_as_valid_values), ecg_validation_comparison_report_dezimal_places)
    mean_wrong_as_invalid = round(np.mean(wrong_as_invalid_values), ecg_validation_comparison_report_dezimal_places)

    quick_items = list(all_files_ecg_validation_comparison.items())
    quick_items.insert(0, (MEAN_ROW_CAPTION, [mean_correct_valid, mean_correct_invalid, mean_wrong_as_valid, mean_wrong_as_invalid]))
    all_files_ecg_validation_comparison = dict(quick_items)

    # calcualte max lengths of table columns
    all_file_lengths = [len(key) for key in all_files_ecg_validation_comparison]
    max_file_length = max(len(FILE_CAPTION), max(all_file_lengths)) + 3

    all_correct_valid_lengths = []
    all_correct_invalid_lengths = []
    all_wrong_as_valid_lengths = []
    all_wrong_as_invalid_lengths = []
    for file in all_files_ecg_validation_comparison:
        all_correct_valid_lengths.append(len(str(all_files_ecg_validation_comparison[file][0])))
        all_correct_invalid_lengths.append(len(str(all_files_ecg_validation_comparison[file][1])))
        all_wrong_as_valid_lengths.append(len(str(all_files_ecg_validation_comparison[file][2])))
        all_wrong_as_invalid_lengths.append(len(str(all_files_ecg_validation_comparison[file][3])))

    all_correct_valid_lengths = np.array(all_correct_valid_lengths)
    all_correct_invalid_lengths = np.array(all_correct_invalid_lengths)
    all_wrong_as_valid_lengths = np.array(all_wrong_as_valid_lengths)
    all_wrong_as_invalid_lengths = np.array(all_wrong_as_invalid_lengths)

    max_correct_valid_length = max(len(CORRECT_VALID_CAPTION), max(all_correct_valid_lengths))
    max_correct_invalid_length = max(len(CORRECT_INVALID_CAPTION), max(all_correct_invalid_lengths))
    max_wrong_as_valid_length = max(len(WRONG_AS_VALID_CAPTION), max(all_wrong_as_valid_lengths))
    max_wrong_as_invalid_length = max(len(WRONG_AS_INVALID_CAPTION), max(all_wrong_as_invalid_lengths))

    # write the legend for the table
    message = "Legend:"
    comparison_file.write(message + "\n")
    comparison_file.write("-" * len(message) + "\n\n")
    comparison_file.write(CORRECT_VALID_CAPTION + "... Matching valid regions ratio\n")
    comparison_file.write(CORRECT_INVALID_CAPTION + "... Matching invalid regions ratio\n")
    comparison_file.write(WRONG_AS_VALID_CAPTION + "... valid (detected) / invalid (gif) ratio\n")
    comparison_file.write(WRONG_AS_INVALID_CAPTION + "... invalid (detected) / valid (gif) ratio\n\n\n")

    message = "Table with comparison values for each file:"
    comparison_file.write(message + "\n")
    comparison_file.write("-" * len(message) + "\n\n")

    # create table header
    comparison_file.write(print_in_middle(FILE_CAPTION, max_file_length) + " | ")
    comparison_file.write(print_in_middle(CORRECT_VALID_CAPTION, max_correct_valid_length) + " | ")
    comparison_file.write(print_in_middle(CORRECT_INVALID_CAPTION, max_correct_invalid_length) + " | ")
    comparison_file.write(print_in_middle(WRONG_AS_VALID_CAPTION, max_wrong_as_valid_length) + " | ")
    comparison_file.write(print_in_middle(WRONG_AS_INVALID_CAPTION, max_wrong_as_invalid_length) + "\n")
    total_length = max_file_length + max_correct_valid_length + max_correct_invalid_length + max_wrong_as_valid_length + max_wrong_as_invalid_length + 3*4 + 1
    comparison_file.write("-" * total_length + "\n")

    # write the data
    for file in all_files_ecg_validation_comparison:
        comparison_file.write(print_in_middle(file, max_file_length) + " | ")
        comparison_file.write(print_in_middle(str(all_files_ecg_validation_comparison[file][0]), max_correct_valid_length) + " | ")
        comparison_file.write(print_in_middle(str(all_files_ecg_validation_comparison[file][1]), max_correct_invalid_length) + " | ")
        comparison_file.write(print_in_middle(str(all_files_ecg_validation_comparison[file][2]), max_wrong_as_valid_length) + " | ")
        comparison_file.write(print_in_middle(str(all_files_ecg_validation_comparison[file][3]), max_wrong_as_invalid_length) + "\n")
        comparison_file.write("-" * total_length + "\n")

    comparison_file.close()